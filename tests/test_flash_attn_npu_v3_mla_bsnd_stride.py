"""Independent BSND stride attention, with v1/v2/v5 and fixed-source/math checks.

Source inputs are materialized once before timing; arbitrary FA3 strides have
no matching source ABI. This suite does not open production stride support.
"""

import ctypes as ct
import hashlib
import itertools
import json
import os
from pathlib import Path

import pytest
import torch
import torch_npu  # noqa: F401

from tests.mla_dense_build import ROOT, build_probes, sha256
from tests.mla_metadata_build import build_probe
from tests.mla_metadata_error_probe import forbid_host_readback
from tests.test_flash_attn_npu_v3_mla_dense import Schedule
from tests.test_flash_attn_npu_v3_mla_metadata import load_probe
from tests.test_flash_attn_npu_v3_mla_metadata_consumer import Context as DenseContext
from tests.test_flash_attn_npu_v3_mla_bsnd_metadata import Args, arguments
from tests.test_flash_attn_npu_v3_mla_bsnd_metadata_aicpu import enqueue, host_record
from tests.test_flash_attn_npu_v3_mla_bsnd_metadata_consumer import context_for, expected_schedule
from tests.test_flash_attn_npu_v3_mla_bsnd_schedule import length_case
from tests.test_flash_attn_npu_v3_mla_used_attention import (
    Array,
    Context,
    Inputs,
    SENTINEL,
    UsedSchedule,
    check_math,
    configure_producer as configure_used,
    effective,
    input_address,
    make_case as make_used_case,
    references,
    used_case,
    valid_outputs,
)

NAMES = ("q", "qv", "k", "v")
I64_MAX = (1 << 63) - 1


class View(ct.Structure):
    _fields_ = [
        ("sizes", ct.c_int64 * 4),
        ("strides", ct.c_int64 * 4),
        ("offset", ct.c_int64),
        ("storage_bytes", ct.c_uint64),
    ]


def describe(tensor):
    return View(
        (ct.c_int64 * 4)(*tensor.shape),
        (ct.c_int64 * 4)(*tensor.stride()),
        tensor.storage_offset(),
        tensor.untyped_storage().nbytes(),
    )


def spec(version=5, profile="independent", shape=(2, 7, 16, 33), uq=None, uk=None, aic=None):
    return dict(version=version, profile=profile, shape=shape, uq=uq, uk=uk, aic=aic)


# Fixed before sampling, with both compact controls and real strided inputs.
PERFORMANCE_CASES = [
    spec(1, "contiguous", (2, 1, 8, 257)),
    spec(1, "bnsd", (2, 7, 16, 513)),
    spec(2, "independent", (2, 2, 128, 1025), uk=[511, 513]),
    spec(5, "independent", uq=[3, 1], uk=[17, 0]),  # U01
    spec(5, "bnsd", (5, 9, 171, 1025), [3, 0, 1, 7, 0], [17, 65, 513, 0, 1], 1),
    spec(5, "independent", (2, 3, 8, 16384), [1, 2], [16384, 8193]),
    spec(1, "matrix_65536", (1, 7, 8, 33)),
    spec(1, "head_65536", (1, 7, 8, 33)),
    spec(1, "broadcast", (2, 7, 8, 33)),
]
LAYOUT_CASES = [
    spec(
        version, profile, uq=[3, 1] if version == 5 else None, uk=[17, 0] if version != 1 else None
    )
    for version, profile in itertools.product((1, 2, 5), ("contiguous", "bnsd", "independent"))
] + [spec(1, profile, (2, 7, 8, 33)) for profile in ("broadcast", "overlap")]
BOUNDARY_CASES = [
    spec(5, f"{axis}_{stride}", (2, 5, 1 if axis == "row" else 5, 17), [5, 3], [17, 7])
    for axis, stride in itertools.product(("row", "matrix", "head"), (65535, 65536, 65537))
] + [
    spec(5, "singleton", (1, 1, 1, 1), [1], [1]),
    spec(5, "singleton", (0, 1, 1, 1), [], []),
    spec(5, "singleton", (2, 0, 1, 0), [0, 0], [0, 0]),
    spec(5, "independent", (2, 7, 16, 0), [3, 1], [0, 0]),
    spec(5, "independent", (2, 7, 16, 33), [0, 0], [17, 0]),
    spec(5, "independent", (2, 7, 16, 33), [3, 1], [0, 0]),
]
WIDE_CASES = [
    spec(5, f"wide_{name}_{axis}", (2, 2, 2, 2), [2, 1], [2, 1])
    for name in NAMES
    for axis in ((0, 1, 2) if name in ("q", "qv") else (0, 1))
]


def configure_producer(producer):
    configure_used(producer)
    for name in ("launch_metadata", "launch_bsnd_metadata"):
        getattr(producer["lib"], name).argtypes = [ct.c_void_p] * 4 + [ct.POINTER(Args)]
    producer["lib"].metadata_host.argtypes = [ct.POINTER(Args), ct.c_void_p]
    producer["lib"].bsnd_metadata_host.argtypes = [ct.POINTER(Args), ct.c_void_p]
    producer["lib"].bsnd_metadata_host.restype = ct.c_uint32


def load_attention(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "build-manifest.json").read_text())
    assert manifest["stride"] and manifest["ndebug"]
    for name, digest in manifest["sources"].items():
        assert sha256(ROOT / name) == digest, name
    libraries = {}
    for name in ("oracle", "migrated"):
        row = manifest["libraries"][name]
        assert sha256(Path(row["path"])) == row["sha256"]
        lib = ct.CDLL(row["path"])
        lib.launch_bsnd_stride.argtypes = [ct.c_void_p] * 10 + [
            ct.c_uint32,
            ct.POINTER(DenseContext),
            ct.POINTER(Context),
            ct.POINTER(Inputs),
            ct.POINTER(UsedSchedule),
            ct.c_float,
            ct.c_void_p,
            ct.c_void_p,
            ct.c_bool,
            ct.c_uint32,
            ct.c_uint32,
            ct.POINTER(View),
        ]
        lib.launch_bsnd_stride.restype = ct.c_int
        libraries[name] = lib
    libraries["dtype"] = torch.float16 if manifest["dtype"] == "fp16" else torch.bfloat16
    return libraries, directory


@pytest.fixture(scope="module")
def stride_producer(tmp_path_factory):
    directory = Path(
        os.environ.get("MLA_STRIDE_PRODUCER_DIR", tmp_path_factory.mktemp("stride-producer"))
    )
    if os.environ.get("MLA_STRIDE_REUSE") != "1":
        build_probe(directory, bsnd=True, used=True)
    manifest = json.loads((directory / "build-manifest.json").read_text())
    assert sha256(Path(manifest["library"])) == manifest["library_sha256"]
    for name, digest in manifest["sources"].items():
        assert sha256(ROOT / name) == digest, name
    with load_probe(manifest["library"], directory) as producer:
        configure_producer(producer)
        yield producer


@pytest.fixture(scope="module", params=("fp16", "bf16"))
def stride_attention(tmp_path_factory, request):
    directory = (
        Path(os.environ.get("MLA_STRIDE_DIR", tmp_path_factory.mktemp("stride-attention")))
        / request.param
    )
    if os.environ.get("MLA_STRIDE_REUSE") != "1":
        build_probes(directory, request.param, stride=True)
    return load_attention(directory)


def strides_for(shape, profile, name):
    b, s, h, d = shape
    if profile == "contiguous":
        return (s * h * d, h * d, d, 1)
    if profile == "singleton":
        return tuple(I64_MAX if size <= 1 else s * h * d for size in (b, s, h)) + (1,)
    if profile == "bnsd" or (profile == "independent" and name == "q"):
        return (h * s * d + 97, d, s * d, 1)
    if profile == "broadcast":
        return (0, 0 if name in ("q", "k") else d, 0 if name == "qv" else d, 1)
    if profile == "overlap":
        return (19, 17, 32, 1)
    if profile.startswith(("row_", "matrix_", "head_", "combined_")):
        axis, value = profile.split("_")
        value = int(value)
        row, head = (h * d + 13, value) if axis == "head" else (value, d)
        if axis == "combined":
            head = s * row + d + 29
        return (s * row + h * head + 31, row, head, 1)
    padding = NAMES.index(name) * 2 + 3
    head = 2 * d + padding
    row = (h + 1) * head + padding
    strides = [(s + 1) * row + padding, row, head, 1]
    if profile.startswith("wide_"):
        _, target, axis = profile.split("_")
        if name == target:
            strides[int(axis)] = (1 << 32) + 65537
    return tuple(strides)


def make_view(data, strides, offset, *, overlap=False):
    """Preserve physical storage on H2D; wide tests allocate real >8 GiB spans.

    Wide holes are not copied or initialized. Active rows, redzones and the
    addresses that uint32 truncation would reach are populated independently.
    """
    shape = tuple(data.shape)
    span = 0 if not data.numel() else 1 + sum((n - 1) * s for n, s in zip(shape, strides))
    count = offset + span + 17
    checks = []
    if span < (1 << 28):
        before = torch.full((count,), 123.0, dtype=data.dtype)
        host = before.as_strided(shape, strides, offset)
        if overlap:
            gen = torch.Generator().manual_seed(20261010 + offset)
            before[offset : offset + span] = torch.randn(span, generator=gen).to(data.dtype)
        else:
            before[offset : offset + span] = torch.nan
            host.copy_(data)
        base = before.npu()
        checks.append((before, base))
        logical = host.contiguous()
    else:
        assert not overlap
        base = torch.empty(count, dtype=data.dtype, device="npu")
        logical = data.contiguous()
        coords = list(itertools.product(*(range(n) for n in shape[:3])))
        # Poison compact and uint32-truncated addresses before writing real rows.
        poison = torch.full((max(2048, data.numel()),), torch.nan, dtype=data.dtype).npu()
        base[: poison.numel()].copy_(poison)
        for coord in coords:
            address = offset + sum(c * s for c, s in zip(coord, strides))
            truncated = offset + sum(c * (s & 0xFFFFFFFF) for c, s in zip(coord, strides))
            base[truncated : truncated + shape[-1]].copy_(poison[: shape[-1]])
        for coord in coords:
            address = offset + sum(c * s for c, s in zip(coord, strides))
            expected = data[coord].contiguous()
            base[address : address + shape[-1]].copy_(expected.npu())
            checks.append((expected, base[address : address + shape[-1]]))
        for start, size in ((0, offset), (offset + span, 17)):
            expected = torch.full((size,), 123.0, dtype=data.dtype)
            base[start : start + size].copy_(expected.npu())
            checks.append((expected, base[start : start + size]))
    return base.as_strided(shape, strides, offset), logical, checks


def make_case(probes, producer, config, *, scale=None):
    b, sq, heads, sk = config["shape"]
    item = used_case(0, [sq] * b, [sk] * b, config["uq"], config["uk"], heads=heads, sq=sq, sk=sk)
    if config["aic"] is not None:
        item["aic"] = config["aic"]
    case = make_used_case(probes, producer, item, scale=scale)
    case.update(config=config, dense_context=DenseContext(), input_checks=[])
    case["guards"] = case["guards"][8:]  # Replace all four input storages on both sides.
    for name, offset in zip(NAMES, (5, 11, 19, 29)):
        shape = (
            b,
            sq if name in ("q", "qv") else sk,
            heads if name in ("q", "qv") else 1,
            64 if name in ("q", "k") else 512,
        )
        data = case["host"][name].reshape(shape)
        device, logical, checks = make_view(
            data,
            strides_for(shape, config["profile"], name),
            offset,
            overlap=config["profile"] in ("broadcast", "overlap"),
        )
        case["devices"]["migrated"][name] = device
        case["host"][name] = logical.reshape(-1, shape[2], shape[3])
        source, _, source_checks = make_view(
            logical, strides_for(shape, "contiguous", name), offset + 2
        )
        case["devices"]["oracle"][name] = source
        case["input_checks"] += checks + source_checks
    case["desc"] = {
        name: torch.tensor([24, 1 << 32, 0xFFFFFFFF, data["v"].data_ptr()], dtype=torch.int64).npu()
        for name, data in case["devices"].items()
    }
    version = config["version"]
    if version != 5:
        assert config["uq"] is None and (version != 1 or config["uk"] is None)
        p = producer["platform"]
        values = length_case(
            config["uk"],
            batch=b,
            sq=sq,
            heads=heads,
            sk=sk,
            aic=config["aic"] or p.aic,
            ratio=2,
            libapi=p.libapi,
            **{n: getattr(p.memory, n) for n in ("l1", "ub", "l0a", "l0b", "l0c")},
        )
        kind = 1 if probes[0]["dtype"] == torch.float16 else 2
        args = arguments(values, kind)
        if version == 1:
            args.words = 401
            raw = (ct.c_uint32 * 401)()
            producer["lib"].metadata_host(ct.byref(args), raw)
            row = list(raw)
        else:
            row = host_record(producer, values, kind)
        case["dense_context"] = context_for(row, values, kind)
        case["schedule"].tiling = Schedule.from_buffer_copy(
            (ct.c_uint32 * 89)(*expected_schedule(row))
        )
        assert case["context"].workspace >= case["dense_context"].workspace
        case["metadata_storage"] = torch.full(
            (len(row) + 2,), SENTINEL, dtype=torch.int32, device="npu"
        )
        case["metadata"] = case["metadata_storage"][1:-1]
        case["expected"] = row
        with forbid_host_readback():
            enqueue(producer, args, case["metadata_storage"], case["inputs"][3], dense=version == 1)
    case["views"] = {
        name: (View * 4)(*[describe(data[k]) for k in NAMES])
        for name, data in case["devices"].items()
    }
    return case


def launch(probes, case, name="migrated", *, views=None):
    c = case["context"]
    arrays = Inputs(
        *[Array(input_address(t), 0 if t is None else t.numel()) for t in case["inputs"]]
    )
    pointers = [case["devices"][name][k] for k in ("qv", "q", "v", "k")] + [
        case["desc"][name],
        *case["outputs"][name],
        case["workspace"][name],
        case["metadata"],
    ]
    stream = torch.npu.current_stream()
    for tensor in [*pointers, *case["inputs"], case["source_q"], case["source_k"]]:
        if tensor is not None:
            tensor.record_stream(stream)
    uq, uk = effective(case["item"])
    blocks = c.block_dim if case["config"]["version"] == 5 else case["dense_context"].block_dim
    return probes[0][name].launch_bsnd_stride(
        stream.npu_stream,
        *(t.data_ptr() for t in pointers),
        case["metadata"].numel(),
        ct.byref(case["dense_context"]),
        ct.byref(c),
        ct.byref(arrays),
        ct.byref(case["schedule"]),
        case["scale"],
        case["source_q"].data_ptr(),
        case["source_k"].data_ptr(),
        any(q != case["schedule"].tiling.sq for q in uq) or any(k == 0 for k in uk),
        blocks,
        case["config"]["version"],
        case["views"][name] if views is None else views,
    )


def check_case(probes, case, *, repeats=3):
    uq, uk = effective(case["item"])
    empty = torch.tensor([k == 0 for q, k in zip(uq, uk) for _ in range(q)], dtype=torch.bool)
    source_ok = bool(sum(uq) and sum(case["item"]["kv"]))
    first = None
    for _ in range(repeats):
        with forbid_host_readback():
            if source_ok:
                assert launch(probes, case, "oracle") == 0
            assert launch(probes, case) == 0
        actual = valid_outputs(case, [t.cpu() for t in case["outputs"]["migrated"]])
        if source_ok:
            source = valid_outputs(case, [t.cpu() for t in case["outputs"]["oracle"]], source=True)
            torch.testing.assert_close(actual[0], source[0], rtol=0, atol=0)
            torch.testing.assert_close(actual[1][:, ~empty], source[1][:, ~empty], rtol=0, atol=0)
            assert torch.isposinf(source[1][:, empty]).all()
        if first is not None:
            for a, b in zip(actual, first):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        first = actual
    errors = check_math(first, references(case))
    if not sum(case["item"]["kv"]):
        assert (case["outputs"]["migrated"][0].cpu() == 0).all()
        assert torch.isposinf(case["outputs"]["migrated"][1].cpu()).all()
    for before, after in case["input_checks"]:
        torch.testing.assert_close(after.cpu(), before, rtol=0, atol=0, equal_nan=True)
    for before, after, prefix, _ in case["guards"]:
        got = after.cpu()
        assert torch.equal(got[:prefix], before[:prefix]) and torch.equal(got[-17:], before[-17:])
    for buffer in case["ws_storage"].values():
        assert (buffer[:512].cpu() == 205).all() and (buffer[-512:].cpu() == 205).all()
    assert [w & 0xFFFFFFFF for w in case["metadata_storage"].cpu().tolist()] == [
        SENTINEL,
        *case["expected"],
        SENTINEL,
    ]
    report = dict(
        config=case["config"],
        dtype=str(probes[0]["dtype"]),
        scale=case["scale"],
        errors=errors,
        source_compared=source_ok,
        m_base=case["schedule"].tiling.m_base,
        inputs={
            n: dict(
                shape=list(t.shape),
                strides=list(t.stride()),
                offset=t.storage_offset(),
                storage_bytes=t.untyped_storage().nbytes(),
            )
            for n, t in case["devices"]["migrated"].items()
        },
        output_sha256=hashlib.sha256(
            b"".join(t.contiguous().view(torch.uint8).numpy().tobytes() for t in first)
        ).hexdigest(),
    )
    with (probes[1] / "accuracy.jsonl").open("a") as out:
        out.write(json.dumps(report) + "\n")
    return report


@pytest.mark.parametrize("config", LAYOUT_CASES + BOUNDARY_CASES + PERFORMANCE_CASES)
def test_stride_attention(stride_attention, stride_producer, config):
    check_case(stride_attention, make_case(stride_attention, stride_producer, config))


@pytest.mark.parametrize("config", WIDE_CASES)
def test_wide_active_stride(stride_attention, stride_producer, config):
    torch.npu.empty_cache()
    case = make_case(stride_attention, stride_producer, config)
    check_case(stride_attention, case)


@pytest.mark.parametrize(
    "config",
    [spec(5, f"combined_{stride}", (2, 5, 5, 17), [5, 3], [17, 7]) for stride in (65536, 65537)]
    + [spec(5, "independent", uq=[3, 1])],
)
def test_combined_fallback_and_default_kv(stride_attention, stride_producer, config):
    check_case(stride_attention, make_case(stride_attention, stride_producer, config))


@pytest.mark.parametrize("scale", (0.0, -0.125, 0.23))
def test_scale_and_input_reuse(stride_attention, stride_producer, scale):
    case = make_case(stride_attention, stride_producer, spec(uq=[3, 1], uk=[17, 0]), scale=scale)
    check_case(stride_attention, case)
    # Reuse the exact record with a different scale and fresh four-way views.
    other = make_case(
        stride_attention,
        stride_producer,
        spec(profile="bnsd", uq=[3, 1], uk=[17, 0]),
        scale=scale + 0.125,
    )
    assert other["expected"] == case["expected"]
    other["metadata"], other["metadata_storage"] = case["metadata"], case["metadata_storage"]
    check_case(stride_attention, other)


@pytest.mark.parametrize("field", ("stride", "storage", "shape", "offset"))
def test_invalid_view_rejected_before_launch(stride_attention, stride_producer, field):
    case = make_case(stride_attention, stride_producer, spec(uq=[3, 1], uk=[17, 0]))
    for index in range(4):
        views = (View * 4).from_buffer_copy(case["views"]["migrated"])
        if field == "stride":
            views[index].strides[1] = -1
        elif field == "storage":
            views[index].storage_bytes = 0
        elif field == "shape":
            views[index].sizes[0] += 1
        else:
            views[index].offset = I64_MAX
        assert launch(stride_attention, case, views=views) == -3
    assert (case["outputs"]["migrated"][0].cpu() == 12345).all()
