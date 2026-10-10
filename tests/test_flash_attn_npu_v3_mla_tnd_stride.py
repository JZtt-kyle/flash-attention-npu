"""Independent v3/v4/v5 TND strides; source materialization is outside timing.

The fixed CANN non-FD oracle receives independently packed effective inputs.
The migrated Kernel reads original allocation views. Production entry coverage
is maintained separately in test_flash_attn_npu_v3_mla_tnd_stride_production.py.
"""

import ctypes as ct
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
from tests.test_flash_attn_npu_v3_mla_metadata import load_probe
from tests.test_flash_attn_npu_v3_mla_bsnd_stride import (
    I64_MAX,
    NAMES,
    View,
    check_case as check_bsnd_case,
    describe,
    make_view,
    strides_for,
)
from tests.test_flash_attn_npu_v3_mla_used_attention import (
    Array,
    Context,
    Inputs,
    Schedule,
    UsedSchedule,
    configure_producer as configure_used,
    effective,
    input_address,
    make_case as make_used_case,
    used_case,
)
from tests.test_flash_attn_npu_v3_mla_tnd_attention import configure_producer as configure_tnd
from tests.test_flash_attn_npu_v3_mla_tnd_bsnd_attention import (
    configure_producer as configure_mixed,
)
from tests import test_flash_attn_npu_v3_mla_tnd_metadata_aicpu as tnd_metadata
from tests import test_flash_attn_npu_v3_mla_tnd_bsnd_metadata_aicpu as mixed_metadata
from tests.test_flash_attn_npu_v3_mla_tnd_schedule import tnd_case
from tests.test_flash_attn_npu_v3_mla_tnd_bsnd_schedule import mixed_case
from tests.test_flash_attn_npu_v3_mla_bsnd_metadata_consumer import expected_schedule


def spec(
    version=5,
    layout=1,
    profile="independent",
    q=(3, 0, 1),
    kv=None,
    uq=None,
    uk=None,
    heads=16,
    aic=None,
):
    if kv is None:
        kv = (17, 0, 33) if layout == 1 else (64, 64, 64)
    return dict(
        version=version,
        profile=profile,
        item=used_case(
            layout, q, kv, uq, uk, heads=heads, sq=max(q, default=0), sk=max(kv, default=0)
        ),
        aic=aic,
    )


# Fixed before sampling. Includes both compact controls and physical stride views.
PERFORMANCE_CASES = [
    spec(3, profile="contiguous"),
    spec(3, profile="ntd", q=(7, 0, 3), kv=(513, 65, 17), heads=128),
    spec(4, 2, "independent", uk=(17, 0, 33)),
    spec(4, 2, "ntd", q=(7, 0, 3), kv=(1025,) * 3, uk=(513, 0, 17), heads=128),
    spec(5, uq=(2, 0, 1), uk=(7, 0, 17)),
    spec(5, 2, uq=(2, 0, 1), uk=(17, 0, 33)),
    spec(
        5,
        1,
        "ntd",
        q=(7, 0, 3, 9, 1),
        kv=(33, 65, 513, 7, 1),
        uq=(3, 0, 1, 7, 0),
        uk=(17, 65, 513, 0, 1),
        heads=171,
        aic=1,
    ),
    spec(
        5,
        2,
        q=(7, 0, 3, 9, 1),
        kv=(1025,) * 5,
        uq=(3, 0, 1, 7, 0),
        uk=(17, 65, 513, 0, 1),
        heads=171,
        aic=1,
    ),
    spec(5, q=(3, 2), kv=(16384, 8193), uq=(1, 2), uk=(16384, 8193), heads=8),
    spec(5, 2, q=(3, 2), kv=(16384, 16384), uq=(1, 2), uk=(16384, 8193), heads=8),
    spec(3, profile="matrix_65536", q=(7, 1), kv=(33, 17), heads=8),
    spec(4, 2, "broadcast", uk=(17, 0, 33), heads=8),
]
LAYOUT_CASES = [
    spec(
        version,
        layout,
        profile,
        uq=(2, 0, 1) if version == 5 else None,
        uk=(7, 0, 17) if version != 3 else None,
    )
    for (version, layout), profile in itertools.product(
        ((3, 1), (4, 2), (5, 1), (5, 2)),
        ("contiguous", "ntd", "independent", "broadcast", "overlap"),
    )
]
BOUNDARY_CASES = [
    spec(
        version,
        layout,
        f"{axis}_{stride}",
        q=(5, 1),
        kv=(17, 17),
        uq=(3, 1) if version == 5 else None,
        uk=(17, 7) if version != 3 else None,
        heads=1 if axis == "row" else 5,
    )
    for (version, layout), axis, stride in itertools.product(
        ((3, 1), (4, 2), (5, 1), (5, 2)),
        ("row", "matrix", "head", "combined"),
        (65535, 65536, 65537),
    )
]
EMPTY_CASES = [
    spec(
        version,
        layout,
        "singleton" if not q or sum(q) <= 1 else "independent",
        q=q,
        kv=k,
        heads=1,
        uq=uq if version == 5 else None,
        uk=uk if version != 3 else None,
    )
    for version, layout in ((3, 1), (4, 2), (5, 1), (5, 2))
    for q, k, uq, uk in (
        ((), (), (), ()),
        ((0, 0), (17, 17), (0, 0), (17, 0)),
        ((3, 1), (0, 0), (2, 1), (0, 0)),
        ((1,), (1,), (1,), (1,)),
        ((3, 1), (17, 17), (0, 0), (17, 0)),
        ((3, 1), (17, 17), (2, 1), (0, 0)),
    )
]
WIDE_CASES = [
    spec(version, layout, f"wide_{name}_{axis}", q=(1, 1), kv=(1, 1), heads=2)
    for version, layout in ((3, 1), (4, 2), (5, 1), (5, 2))
    for name in NAMES
    for axis in ((1, 2) if name in ("q", "qv") else (1,) if layout == 1 else (0,))
]


def configure_producer(producer):
    configure_used(producer)
    configure_tnd(producer)
    configure_mixed(producer)


def load_attention(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "build-manifest.json").read_text())
    assert manifest["tnd_stride"] and manifest["ndebug"]
    for name, digest in manifest["sources"].items():
        assert sha256(ROOT / name) == digest, name
    libraries = {}
    for name in ("oracle", "migrated"):
        row = manifest["libraries"][name]
        assert sha256(Path(row["path"])) == row["sha256"]
        lib = ct.CDLL(row["path"])
        lib.launch_tnd_stride.argtypes = [ct.c_void_p] * 10 + [
            ct.c_uint32,
            ct.POINTER(Context),
            ct.POINTER(Inputs),
            ct.POINTER(UsedSchedule),
            ct.c_float,
            ct.c_void_p,
            ct.c_void_p,
            ct.c_bool,
            ct.c_uint32,
            ct.POINTER(View),
        ]
        lib.launch_tnd_stride.restype = ct.c_int
        libraries[name] = lib
    libraries["dtype"] = torch.float16 if manifest["dtype"] == "fp16" else torch.bfloat16
    return libraries, directory


@pytest.fixture(scope="module")
def stride_producer(tmp_path_factory):
    directory = Path(
        os.environ.get(
            "MLA_TND_STRIDE_PRODUCER_DIR", tmp_path_factory.mktemp("tnd-stride-producer")
        )
    )
    if os.environ.get("MLA_TND_STRIDE_REUSE") != "1":
        build_probe(directory, tnd=True, mixed=True, used=True)
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
        Path(os.environ.get("MLA_TND_STRIDE_DIR", tmp_path_factory.mktemp("tnd-stride-attention")))
        / request.param
    )
    if os.environ.get("MLA_TND_STRIDE_REUSE") != "1":
        build_probes(directory, request.param, tnd_stride=True)
    return load_attention(directory)


def make_case(probes, producer, config, *, scale=None, full_launch=False):
    item = dict(config["item"])
    if config["aic"] is not None:
        item["aic"] = config["aic"]
    if full_launch:
        item["aic"] = producer["platform"].aic
        item["block_dim"] = item["aic"] if sum(item["q"]) and sum(item["kv"]) else 0
    case = make_used_case(probes, producer, item, scale=scale)
    item = case["item"]
    case.update(config=config, input_checks=[])
    case["guards"] = case["guards"][8:]
    uq, uk = effective(item)
    for name, offset in zip(NAMES, (5, 11, 19, 29)):
        is_q = name in ("q", "qv")
        is_tnd = is_q or item["layout"] == 1
        alloc, used = (item["q"], uq) if is_q else (item["kv"], uk)
        h, d = (item["heads"] if is_q else 1), (64 if name in ("q", "k") else 512)
        shape = (1, sum(alloc), h, d) if is_tnd else (len(alloc), item["sk"], h, d)
        profile = "bnsd" if config["profile"] == "ntd" else config["profile"]
        device, logical, checks = make_view(
            case["host"][name].reshape(shape),
            strides_for(shape, profile, name),
            offset,
            overlap=profile in ("broadcast", "overlap"),
        )
        case["devices"]["migrated"][name] = device[0] if is_tnd else device
        case["host"][name] = logical.reshape(sum(alloc), h, d)
        # Pack effective TND segments directly from test inputs, never metadata.
        data = case["host"][name]
        pos, chunks = 0, []
        for capacity, length in zip(alloc, used):
            chunks.append(data[pos : pos + length])
            pos += capacity
        source = torch.cat(chunks) if is_tnd and chunks else data
        source = source.reshape(1, -1, h, d) if is_tnd else source.reshape(shape)
        source_device, _, source_checks = make_view(
            source,
            strides_for(source.shape, "contiguous", name),
            offset + 2,
        )
        case["devices"]["oracle"][name] = source_device
        case["input_checks"] += checks + source_checks
    case["desc"] = {
        n: torch.tensor([24, 1 << 32, 0xFFFFFFFF, d["v"].data_ptr()], dtype=torch.int64).npu()
        for n, d in case["devices"].items()
    }
    case["views"] = {
        side: (View * 4)(
            *[describe(t.unsqueeze(0) if t.ndim == 3 else t) for t in (devices[n] for n in NAMES)]
        )
        for side, devices in case["devices"].items()
    }
    if config["version"] != 5:
        assert item["uq"] is None and (config["version"] != 3 or item["uk"] is None)
        kind = 1 if probes[0]["dtype"] == torch.float16 else 2
        shape = {
            key: item[key]
            for key in (
                "heads",
                "sq",
                "sk",
                "aic",
                "ratio",
                "libapi",
                "l1",
                "ub",
                "l0a",
                "l0b",
                "l0c",
            )
        }
        if full_launch:
            shape["block_dim"] = item["block_dim"]
        if config["version"] == 3:
            module = tnd_metadata
            values = tnd_case(item["q"], item["kv"], **shape)
            k = case["inputs"][1]
        else:
            module = mixed_metadata
            values = mixed_case(item["q"], uk, **shape)
            k = torch.tensor(uk, dtype=torch.int32).npu()
            case["inputs"][3] = k
        row = module.host_record(producer, values, kind)
        assert row[5:8] == [0, 0, 0]
        # Common schedule layout is unchanged across protocol versions.
        case["schedule"].tiling = Schedule.from_buffer_copy(
            (ct.c_uint32 * 89)(*expected_schedule(row))
        )
        c = case["context"]
        c.block_dim, c.workspace = row[21], row[385] | row[386] << 32
        case["ws_storage"] = {
            n: torch.full((c.workspace + 1024,), 205, dtype=torch.uint8, device="npu")
            for n in case["devices"]
        }
        case["workspace"] = {n: t[512:-512] for n, t in case["ws_storage"].items()}
        args = module.arguments(values, kind)
        case["metadata_storage"] = module.storage_for(args)
        case["metadata"] = case["metadata_storage"][1:-1]
        case["expected"] = row
        with forbid_host_readback():
            module.enqueue(producer, args, case["metadata_storage"], case["inputs"][0], k)
    return case


def launch(probes, case, name="migrated", *, views=None):
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
    for t in [*pointers, *case["inputs"], case["source_q"], case["source_k"]]:
        if t is not None:
            t.record_stream(stream)
    uq, uk = effective(case["item"])
    return probes[0][name].launch_tnd_stride(
        stream.npu_stream,
        *(t.data_ptr() for t in pointers),
        case["metadata"].numel(),
        ct.byref(case["context"]),
        ct.byref(arrays),
        ct.byref(case["schedule"]),
        case["scale"],
        case["source_q"].data_ptr(),
        case["source_k"].data_ptr(),
        any(q != case["schedule"].tiling.sq for q in uq) or any(k == 0 for k in uk),
        case["config"]["version"],
        case["views"][name] if views is None else views,
    )


def check_case(probes, case, *, repeats=3):
    return check_bsnd_case(probes, case, repeats=repeats, launch_fn=launch)


@pytest.mark.parametrize("config", LAYOUT_CASES + BOUNDARY_CASES + EMPTY_CASES + PERFORMANCE_CASES)
def test_stride_attention(stride_attention, stride_producer, config):
    check_case(stride_attention, make_case(stride_attention, stride_producer, config))


@pytest.mark.parametrize("config", WIDE_CASES)
def test_wide_active_stride(stride_attention, stride_producer, config):
    torch.npu.empty_cache()
    check_case(stride_attention, make_case(stride_attention, stride_producer, config))


@pytest.mark.parametrize("version,layout", ((3, 1), (4, 2), (5, 1), (5, 2)))
@pytest.mark.parametrize("scale", (0.0, -0.125, 0.23))
def test_scale_and_view_reuse(stride_attention, stride_producer, version, layout, scale):
    config = spec(version, layout)
    first = make_case(stride_attention, stride_producer, config, scale=scale)
    check_case(stride_attention, first)
    other = make_case(
        stride_attention, stride_producer, {**config, "profile": "ntd"}, scale=scale + 0.125
    )
    assert other["expected"] == first["expected"]
    other["metadata"], other["metadata_storage"] = first["metadata"], first["metadata_storage"]
    check_case(stride_attention, other)


@pytest.mark.parametrize("version,layout", ((3, 1), (4, 2), (5, 1), (5, 2)))
@pytest.mark.parametrize("field", ("stride", "last_stride", "storage", "shape", "offset"))
def test_invalid_view_rejected_before_launch(
    stride_attention, stride_producer, version, layout, field
):
    case = make_case(stride_attention, stride_producer, spec(version, layout))
    for index in range(4):
        views = (View * 4).from_buffer_copy(case["views"]["migrated"])
        if field == "stride":
            views[index].strides[1] = -1
        elif field == "last_stride":
            views[index].strides[3] = 2
        elif field == "storage":
            views[index].storage_bytes = 0
        elif field == "shape":
            views[index].sizes[1] += 1
        else:
            views[index].offset = I64_MAX
        assert launch(stride_attention, case, views=views) == -3
    assert (case["outputs"]["migrated"][0].cpu() == 12345).all()
