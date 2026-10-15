"""Shared helpers extracted from tests/mla/unit/used/test_flash_attn_npu_v3_mla_used_attention.py.

v5 attention: physical allocations, independent effective source inputs and math.
"""

import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import pytest
import torch

import torch_npu  # noqa: F401 -- register the NPU backend before torch.npu helpers
from tests.common.compare import assert_fa_close
from tests.mla.support.mla_ref import mla_reference_pair
from tests.mla.support.mla_dense_build import ROOT, build_probes, sha256
from tests.mla.support.mla_metadata_error_probe import forbid_host_readback
from tests.mla.support.components.dense import Schedule
from tests.mla.support.components.used_metadata_aicpu import (
    used_probe as used_probe,
    UsedArgs,
    arguments,
    device_inputs,
    enqueue,
    host_record,
    storage_for,
)
from tests.mla.support.components.used_metadata_consumer import (
    Allocation,
    Array,
    Context,
    Inputs,
    SENTINEL,
    context_for,
    expected_schedule,
    used_case,
)
from tests.mla.support.components.used_metadata import input_arrays
from tests.mla.support.components.used_metadata_device import input_address

# Fixed before sampling: allocation Q/KV, optional used Q/KV, heads and platform.
CASES = [
    used_case(uq=[3, 1], uk=[17, 0], heads=128),  # U01
    used_case(1, [3, 0, 1], [17, 0, 33], [2, 0, 1], [7, 0, 17], heads=128),  # T01
    used_case(2, [3, 0, 1], [64] * 3, [2, 0, 1], [17, 0, 33], heads=128),
    used_case(0, [9] * 5, [1025] * 5, [3, 0, 1, 7, 0], [17, 65, 513, 0, 1], heads=171, aic=1),
    used_case(
        1,
        [7, 0, 3, 9, 1],
        [33, 65, 513, 7, 1],
        [3, 0, 1, 7, 0],
        [17, 65, 513, 0, 1],
        heads=171,
        aic=1,
    ),
    used_case(
        2, [7, 0, 3, 9, 1], [1025] * 5, [3, 0, 1, 7, 0], [17, 65, 513, 0, 1], heads=171, aic=1
    ),
    used_case(0, [3] * 10, [511] * 10, [2] * 10, [511] + [0] * 9, heads=128, aic=15),
    used_case(1, [3] * 10, [511] * 10, [2] * 10, [511] + [0] * 9, heads=128, aic=15),
    used_case(2, [3] * 10, [511] * 10, [2] * 10, [511] + [0] * 9, heads=128, aic=15),
    used_case(1, [4, 2, 7], [16384, 8193, 17], [1, 2, 3], [16384, 8193, 17], heads=8),
    used_case(2, [4, 2, 7], [16384] * 3, [1, 2, 3], [16384, 8193, 17], heads=8),
    used_case(0, [3] * 3, [16384] * 3, [1, 2, 3], [16384, 8193, 17], heads=8),
    used_case(1, [260, 3, 1], [33, 513, 17], [257, 1, 0], [33, 513, 0], heads=1),
    used_case(2, [3] * 43, [513] * 43, [2, 0] * 21 + [1], [0, 513] * 21 + [0], heads=171),
    used_case(1, [3, 2, 5], [31, 32, 33], [1, 1, 2], [31, 32, 33], heads=15),
    used_case(2, [3, 2, 5], [513] * 3, [1, 1, 2], [127, 128, 129], heads=255),
    used_case(0, [3] * 3, [513] * 3, [2, 0, 1], [512, 0, 513], heads=256),
    used_case(1, [2, 3, 1], [513, 33, 7], [0, 0, 0], [513, 33, 7], heads=129),
    used_case(1, [3, 7], [33, 513], [1, 2], [17, 257], heads=16, sq=64, sk=1025),
]

PERFORMANCE_CASES = CASES[:14]

EMPTY_CASES = [
    used_case(layout, q, k, uq, uk, heads=171)
    for layout in range(3)
    for q, k, uq, uk in (
        ([], [], [], []),
        ([0, 0], [33, 33], [0, 0], [17, 0]),
        ([3, 3], [0, 0], [2, 1], [0, 0]),
        ([0, 0], [0, 0], [0, 0], [0, 0]),
        ([3, 3], [33, 33], [2, 1], [0, 0]),
        ([3, 3], [33, 33], [0, 0], [0, 0]),
    )
]


class UsedSchedule(ct.Structure):
    _fields_ = [
        ("tiling", Schedule),
        ("query", Allocation),
        ("kv", Allocation),
        ("flags", ct.c_uint32),
    ]


def load_attention(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "build-manifest.json").read_text())
    assert manifest["used"] and manifest["ndebug"]
    for name, digest in manifest["sources"].items():
        assert sha256(ROOT / name) == digest, name
    libraries = {}
    for name in ("oracle", "migrated"):
        row = manifest["libraries"][name]
        assert sha256(Path(row["path"])) == row["sha256"]
        lib = ct.CDLL(row["path"])
        lib.launch_used_attention.argtypes = [ct.c_void_p] * 10 + [
            ct.c_uint32,
            ct.POINTER(Context),
            ct.POINTER(Inputs),
            ct.POINTER(UsedSchedule),
            ct.c_float,
            ct.c_void_p,
            ct.c_void_p,
            ct.c_bool,
            ct.c_void_p,
            ct.c_uint32,
            ct.c_bool,
        ]
        lib.launch_used_attention.restype = ct.c_int
        libraries[name] = lib
    libraries["dtype"] = torch.float16 if manifest["dtype"] == "fp16" else torch.bfloat16
    return libraries, directory


@pytest.fixture(scope="module", params=("fp16", "bf16"))
def used_attention(tmp_path_factory, request):
    directory = (
        Path(
            os.environ.get("MLA_USED_ATTENTION_DIR", tmp_path_factory.mktemp("mla-used-attention"))
        )
        / request.param
    )
    if os.environ.get("MLA_USED_ATTENTION_REUSE") != "1":
        build_probes(directory, request.param, used=True)
    return load_attention(directory)


def configure_producer(producer):
    lib = producer["lib"]
    lib.launch_used_metadata.argtypes = [ct.c_void_p] * 4 + [ct.POINTER(UsedArgs)]
    lib.launch_used_metadata.restype = ct.c_int
    lib.used_metadata_host.argtypes = [ct.POINTER(UsedArgs), ct.c_void_p]
    lib.used_metadata_host.restype = ct.c_uint32


def effective(item):
    return [
        item[used] if item[used] is not None else item[capacity]
        for used, capacity in (("uq", "q"), ("uk", "kv"))
    ]


def make_case(probes, producer, spec, *, scale=None, offsets=(1, 3, 5, 7), amplitude=1.0, worker=0):
    item = dict(spec)
    p = producer["platform"]
    item.update(
        aic=item.get("aic", p.aic),
        ratio=2,
        libapi=p.libapi,
        **{n: getattr(p.memory, n) for n in ("l1", "ub", "l0a", "l0b", "l0c")},
    )
    dtype = probes[0]["dtype"]
    kind = 1 if dtype == torch.float16 else 2
    row = host_record(producer, item, kind)
    assert row[5:8] == [0, 0, 0]
    context = context_for(row, item, kind)
    schedule = UsedSchedule.from_buffer_copy((ct.c_uint32 * 98)(*expected_schedule(row)))
    qalloc, kalloc = item["q"], item["kv"]
    uq, uk = effective(item)
    heads, layout = context.heads, item["layout"]
    gen = torch.Generator().manual_seed(20261010)
    host, source_host = {}, {}
    for name, allocations, used, h, dim, compact in (
        ("qv", qalloc, uq, heads, 512, layout != 0),
        ("q", qalloc, uq, heads, 64, layout != 0),
        ("v", kalloc, uk, 1, 512, layout == 1),
        ("k", kalloc, uk, 1, 64, layout == 1),
    ):
        data = torch.randn(sum(allocations), h, dim, generator=gen).to(dtype)
        if name in ("q", "qv"):
            data.mul_(amplitude)
        start, chunks = 0, []
        for capacity, length in zip(allocations, used):
            data[start + length : start + capacity] = torch.nan
            chunks.append(data[start : start + length])
            start += capacity
        host[name] = data
        # Independent source adapter: never consult v5 spans or parsed output.
        source_host[name] = torch.cat(chunks) if compact and chunks else data
    guards = []

    def guarded(t, prefix, readonly=False):
        before = torch.full((prefix + t.numel() + 17,), 123.0, dtype=t.dtype)
        before[prefix : prefix + t.numel()] = t.flatten()
        device = before.npu()
        guards.append((before, device, prefix, readonly))
        return device[prefix : prefix + t.numel()].view(t.shape)

    devices = {
        name: {key: guarded(t, offset, True) for (key, t), offset in zip(values.items(), offsets)}
        for name, values in (("migrated", host), ("oracle", source_host))
    }
    outputs = {}
    for name in devices:
        total = sum(uq) if name == "oracle" and layout != 0 else sum(qalloc)
        lse_shape = (
            ((total, heads) if name == "oracle" else (heads, total))
            if layout != 0
            else (len(qalloc), heads, context.query.maximum)
        )
        outputs[name] = [
            guarded(torch.full((total, heads, 512), 12345.0, dtype=dtype), 3),
            guarded(torch.full(lse_shape, 12345.0), 5),
        ]
    ws_storage = {
        name: torch.full((context.workspace + 1024,), 205, dtype=torch.uint8, device="npu")
        for name in devices
    }
    desc = {
        name: torch.tensor([24, 1 << 32, 0xFFFFFFFF, d["v"].data_ptr()], dtype=torch.int64).npu()
        for name, d in devices.items()
    }
    source_q, source_k = [torch.tensor(lengths, dtype=torch.int64) for lengths in (uq, uk)]
    if layout != 0:
        # arch22 chooses both length parser modes from Q layout, even mixed KV.
        source_q, source_k = source_q.cumsum(0), source_k.cumsum(0)
    args = arguments(item, kind)
    storage, arrays = storage_for(args), device_inputs(item)
    with forbid_host_readback():
        enqueue(producer, args, storage, arrays, worker)
    return dict(
        item=item,
        host=host,
        devices=devices,
        outputs=outputs,
        guards=guards,
        ws_storage=ws_storage,
        workspace={n: t[512:-512] for n, t in ws_storage.items()},
        metadata_storage=storage,
        metadata=storage[1:-1],
        inputs=arrays,
        source_q=source_q.npu(),
        source_k=source_k.npu(),
        desc=desc,
        context=context,
        schedule=schedule,
        expected=row,
        args=args,
        scale=576**-0.5 if scale is None else scale,
        status=torch.full((3 * context.block_dim * 16,), SENTINEL, dtype=torch.int32, device="npu"),
    )


def launch(
    probes, case, name="migrated", *, checked=True, words=None, descriptors=None, context=None
):
    c = case["context"] if context is None else context
    arrays = Inputs(
        *[Array(input_address(t), 0 if t is None else t.numel()) for t in case["inputs"]]
    )
    for key, changes in (descriptors or {}).items():
        for field, value in changes.items():
            setattr(getattr(arrays, key), field, value)
    pointers = [case["devices"][name][k] for k in ("qv", "q", "v", "k")] + [
        case["desc"][name],
        *case["outputs"][name],
        case["workspace"][name],
        case["metadata"],
    ]
    stream = torch.npu.current_stream()
    for tensor in [*pointers, *case["inputs"], case["source_q"], case["source_k"], case["status"]]:
        if tensor is not None:
            tensor.record_stream(stream)
    uq, uk = effective(case["item"])
    result = probes[0][name].launch_used_attention(
        stream.npu_stream,
        *(t.data_ptr() for t in pointers),
        case["metadata"].numel() if words is None else words,
        ct.byref(c),
        ct.byref(arrays),
        ct.byref(case["schedule"]),
        case["scale"],
        case["source_q"].data_ptr(),
        case["source_k"].data_ptr(),
        any(q != case["schedule"].tiling.sq for q in uq) or any(k == 0 for k in uk),
        case["status"].data_ptr(),
        case["context"].block_dim,
        checked,
    )
    assert result == 0, result


def valid_outputs(case, outputs, *, source=False):
    """Select only defined used rows; unused query slots have no asserted value."""
    item = case["item"]
    uq, _ = effective(item)
    oslices, lslices, pos = [], [], 0
    for b, (capacity, length) in enumerate(zip(item["q"], uq)):
        oslices.append(outputs[0][pos : pos + length])
        lslices.append(
            outputs[1][b, :, :length]
            if item["layout"] == 0
            else outputs[1][pos : pos + length].T
            if source
            else outputs[1][:, pos : pos + length]
        )
        pos += length if source and item["layout"] != 0 else capacity
    heads = case["context"].heads
    return [
        torch.cat(oslices) if oslices else outputs[0].reshape(0, heads, 512),
        torch.cat(lslices, dim=1) if lslices else outputs[1].reshape(heads, 0),
    ]


def references(case):
    uq, uk = effective(case["item"])
    results = [[], [], [], []]
    qp = kp = 0
    heads = case["context"].heads
    for qa, ka, q, k in zip(case["item"]["q"], case["item"]["kv"], uq, uk):
        if q:
            if k:
                data = {
                    name: t[qp : qp + q].unsqueeze(0)
                    if name in ("q", "qv")
                    else t[kp : kp + k].unsqueeze(0)
                    for name, t in case["host"].items()
                }
                refs = [t[0] for t in mla_reference_pair(**data, softmax_scale=case["scale"])]
            else:
                out = torch.zeros((q, heads, 512), dtype=case["host"]["v"].dtype)
                lse = torch.full((heads, q), -torch.inf if sum(case["item"]["kv"]) else torch.inf)
                refs = [out, lse, out, lse]
            for dst, ref in zip(results, refs):
                dst.append(ref)
        qp += qa
        kp += ka
    return [
        torch.cat(r, dim=i % 2)
        if r
        else torch.empty(
            (heads, 0) if i % 2 else (0, heads, 512),
            dtype=torch.float32 if i % 2 else case["host"]["v"].dtype,
        )
        for i, r in enumerate(results)
    ]


def check_math(actual, refs):
    errors = []
    for i, a in enumerate(actual):
        assert torch.equal(torch.isneginf(a), torch.isneginf(refs[i]))
        assert torch.equal(torch.isposinf(a), torch.isposinf(refs[i]))
        assert not torch.isnan(a).any()
        finite = torch.isfinite(refs[i])
        if finite.any():
            assert_fa_close(
                a[finite],
                refs[i][finite],
                refs[i + 2][finite],
                name="used-out" if i == 0 else "used-lse",
            )
        errors.append(float((a[finite] - refs[i][finite]).abs().max()) if finite.any() else 0)
    return errors


def check_case(probes, case, *, repeats=3):
    uq, uk = effective(case["item"])
    empty = torch.tensor([k == 0 for q, k in zip(uq, uk) for _ in range(q)], dtype=torch.bool)
    source_ok = bool(sum(uq) and sum(case["item"]["kv"]))
    first = None
    for _ in range(repeats):
        with forbid_host_readback():
            if source_ok:
                launch(probes, case, "oracle")
            launch(probes, case)
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
    for before, after, prefix, readonly in case["guards"]:
        got = after.cpu()
        if readonly:
            torch.testing.assert_close(got, before, rtol=0, atol=0, equal_nan=True)
        else:
            assert torch.equal(got[:prefix], before[:prefix])
            assert torch.equal(got[-17:], before[-17:])
    for buffer in case["ws_storage"].values():
        assert (buffer[:512].cpu() == 205).all() and (buffer[-512:].cpu() == 205).all()
    assert [w & 0xFFFFFFFF for w in case["metadata_storage"].cpu().tolist()] == [
        SENTINEL,
        *case["expected"],
        SENTINEL,
    ]
    for tensor, (values, present, _) in zip(case["inputs"], input_arrays(case["item"])):
        assert (tensor is not None) == present
        if present:
            assert tensor.cpu().tolist() == values
    report = dict(
        item=case["item"],
        scale=case["scale"],
        dtype=str(probes[0]["dtype"]),
        errors=errors,
        source_compared=source_ok,
        m_base=case["schedule"].tiling.m_base,
        output_sha256=hashlib.sha256(
            b"".join(t.contiguous().view(torch.uint8).numpy().tobytes() for t in first)
        ).hexdigest(),
    )
    with (probes[1] / "accuracy.jsonl").open("a") as out:
        out.write(json.dumps(report) + "\n")
    return report


def check_rejected(case, status):
    assert (case["status"][::16].cpu() == status).all()
    assert (case["status"].view(-1, 16)[:, 1:].cpu() == SENTINEL).all()
    for t in case["outputs"]["migrated"]:
        assert (t.cpu() == t.new_tensor(12345).cpu()).all()
    assert (case["ws_storage"]["migrated"].cpu() == 205).all()


def fault_case(probes, producer, fault):
    item = used_case(1, [7, 7], [33, 33], [3, 1], [17, 0], heads=8)
    if fault == "empty_q":
        item.update(q=[0, 0], uq=[0, 0])
    if fault == "empty_kv":
        item.update(kv=[0, 0], uk=[0, 0])
    if fault == "empty_batch":
        item.update(q=[], kv=[], uq=[], uk=[])
    case = make_case(probes, producer, item)
    mutations = dict(
        v4=(1, 4, 2),
        wide_start=(411, 1, 12),
        allocation=(414, 1, 12),
        flags=(409, 0, 5),
        workspace=(386, -1, 9),
        endpoint=(23, 99, 8),
    )
    options = {}
    if fault in mutations:
        word, value, status = mutations[fault]
        case["metadata"][word] = value
    elif fault in ("count_uk", "align_uq"):
        options = dict(
            descriptors={"used_k": {"count": 1}}
            if fault == "count_uk"
            else {"used_q": {"address": 1}}
        )
        status = 13
    else:
        axis = {
            "stale_q": 0,
            "stale_k": 1,
            "stale_uq": 2,
            "stale_uk": 3,
            "empty_q": 3,
            "empty_kv": 2,
            "empty_batch": 0,
            "generation": 2,
        }[fault]
        case["inputs"][axis][0] = -1
        status = 12 if axis < 2 else 13
        if fault == "generation":
            enqueue(producer, case["args"], case["metadata_storage"], case["inputs"])
            status = 3
    return case, options, status
