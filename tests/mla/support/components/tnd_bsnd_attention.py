"""Shared helpers extracted from tests/mla/unit/mixed/test_flash_attn_npu_v3_mla_tnd_bsnd_attention.py.

TND-Q/BSND-KV v4 attention: fixed CANN, independent math and guarded storage.
"""

import ctypes as ct
import hashlib
import json
from pathlib import Path
import torch

import torch_npu  # noqa: F401 -- register the NPU backend before torch.npu helpers
from tests.common.compare import assert_fa_close
from tests.mla.support.mla_ref import mla_reference_pair
from tests.mla.support.mla_dense_build import ROOT, sha256
from tests.mla.support.mla_metadata_error_probe import forbid_host_readback
from tests.mla.support.components.dense import Schedule
from tests.mla.support.components.tnd_bsnd_metadata_aicpu import (
    MixedArgs,
    arguments,
    device_inputs,
    enqueue,
    host_record,
    storage_for,
)
from tests.mla.support.components.tnd_bsnd_metadata_consumer import (
    Context,
    SENTINEL,
    context_for,
    expected_schedule,
    mixed_case,
)

# Q lengths, KV lengths, physical Sk, Hq, optional synthetic AIC count.
# Physical cache rows never move when a preceding sample has a short/zero KV.
CASES = [
    ([3, 0, 1], [17, 0, 33], 64, 128, None),
    ([1, 7, 2], [1, 513, 31], 1025, 16, None),
    ([0, 3, 1, 0, 7], [33, 511, 0, 513, 1025], 2048, 171, None),
    ([1, 2, 3], [16384, 8193, 17], 16384, 8, None),
    ([1, 1, 1], [17, 129, 257], 513, 129, None),
    ([2] * 10, [511] + [0] * 9, 511, 128, 15),  # omitted zero-cost suffix
    ([3, 0, 1, 7, 0], [17, 65, 513, 0, 1], 1025, 171, 1),  # cross-sample task ring
    ([3, 0] * 21, [0, 513] * 21, 513, 171, None),  # nonempty, zero total cost
    ([257, 1, 0, 3], [33, 513, 17, 1], 513, 1, None),
    ([1, 1, 1, 1], [31, 32, 33, 65], 65, 15, None),
    ([1, 1, 1, 1], [127, 128, 129, 511], 511, 255, None),
    ([2, 0, 1], [512, 0, 513], 513, 256, None),
    ([0, 1, 0, 0], [0, 0, 0, 1], 17, 1, None),
    ([3, 0, 1], [0, 0, 0], 64, 171, None),  # actual zero is not empty capacity
    ([2] * 10, [511] * 10, 511, 128, 20),  # mixed M=128 vs compact TND M=256
    ([1] * 43, [17] * 43, 33, 16, None),  # physical batches beyond core count
]

PERFORMANCE_CASES = [CASES[i] for i in (0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14)]


class MixedSchedule(ct.Structure):
    _fields_ = [("tiling", Schedule), ("total_q", ct.c_uint32), ("total_k", ct.c_uint64)]


def load_attention(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "build-manifest.json").read_text())
    assert manifest["mixed"] and manifest["ndebug"]
    for name, digest in manifest["sources"].items():
        assert sha256(ROOT / name) == digest, name
    libraries = {}
    for name in ("oracle", "migrated"):
        row = manifest["libraries"][name]
        assert sha256(Path(row["path"])) == row["sha256"]
        lib = ct.CDLL(row["path"])
        lib.launch_mixed_attention.argtypes = [ct.c_void_p] * 10 + [
            ct.c_uint32,
            ct.c_void_p,
            ct.c_uint64,
            ct.c_void_p,
            ct.c_uint64,
            ct.POINTER(Context),
            ct.POINTER(MixedSchedule),
            ct.c_float,
            ct.c_void_p,
            ct.c_void_p,
            ct.c_bool,
            ct.c_void_p,
            ct.c_bool,
        ]
        lib.launch_mixed_attention.restype = ct.c_int
        libraries[name] = lib
    libraries["dtype"] = torch.float16 if manifest["dtype"] == "fp16" else torch.bfloat16
    return libraries, directory


def configure_producer(producer):
    lib = producer["lib"]
    lib.launch_mixed_metadata.argtypes = [ct.c_void_p] * 4 + [ct.POINTER(MixedArgs)]
    lib.launch_mixed_metadata.restype = ct.c_int
    lib.mixed_metadata_host.argtypes = [ct.POINTER(MixedArgs), ct.c_void_p]
    lib.mixed_metadata_host.restype = ct.c_uint32


def make_case(probes, producer, spec, *, scale=None, offsets=(1, 3, 5, 7), amplitude=1.0, worker=0):
    qs, ks, capacity, heads, aic = spec
    p = producer["platform"]
    values = mixed_case(
        qs,
        ks,
        heads=heads,
        sq=max(qs, default=0) + 3,
        capacity=capacity,
        has_k=bool(ks),
        aic=aic or p.aic,
        ratio=2,
        libapi=p.libapi,
        **{n: getattr(p.memory, n) for n in ("l1", "ub", "l0a", "l0b", "l0c")},
    )
    dtype = probes[0]["dtype"]
    kind = 1 if dtype == torch.float16 else 2
    row = host_record(producer, values, kind)
    context = context_for(row, values, kind)
    schedule = MixedSchedule.from_buffer_copy((ct.c_uint32 * 92)(*expected_schedule(row)))
    gen = torch.Generator().manual_seed(20261009)
    inputs = {
        name: torch.randn(tokens, h, dim, generator=gen).to(dtype)
        for name, tokens, h, dim in [
            ("qv", sum(qs), heads, 512),
            ("q", sum(qs), heads, 64),
            ("v", len(ks) * capacity, 1, 512),
            ("k", len(ks) * capacity, 1, 64),
        ]
    }
    qpos = kpos = 0
    for b, (q, k) in enumerate(zip(qs, ks)):
        if not k:
            for name in ("qv", "q"):
                inputs[name][qpos : qpos + q] = torch.nan
        for name in ("v", "k"):
            inputs[name][b * capacity + (k if q else 0) : (b + 1) * capacity] = torch.nan
        qpos += q
        kpos += capacity
    inputs["qv"].mul_(amplitude)
    inputs["q"].mul_(amplitude)
    guards = []

    def guarded(t, prefix, readonly=False):
        before = torch.full((t.numel() + prefix + 17,), 123.0, dtype=t.dtype)
        before[prefix : prefix + t.numel()] = t.flatten()
        storage = before.npu()
        guards.append((before, storage, prefix, readonly))
        return storage[prefix : prefix + t.numel()].view(t.shape)

    device = {name: guarded(t, prefix, True) for (name, t), prefix in zip(inputs.items(), offsets)}
    outputs = {
        name: [
            guarded(torch.full((sum(qs), heads, 512), 12345.0, dtype=dtype), 3),
            guarded(
                torch.full((sum(qs), heads) if name == "oracle" else (heads, sum(qs)), 12345.0), 5
            ),
        ]
        for name in ("oracle", "migrated")
    }
    ws_storage = {
        name: torch.full((context.workspace + 1024,), 205, dtype=torch.uint8, device="npu")
        for name in outputs
    }
    metadata_storage = storage_for(arguments(values, kind))
    query, kv = device_inputs(values)
    # Fixed arch22 uses Q layout to choose the KV length parser: numerical
    # int64 cumulative ends feed ACCUM, while its BSH KV format uses b*Sk.
    # This adapter is independent of both v4 spans and CPU scheduling inputs.
    source_q = torch.tensor(qs, dtype=torch.int64).cumsum(0).npu()
    source_k = torch.tensor(ks, dtype=torch.int64).cumsum(0).npu()
    desc = torch.tensor([24, 1 << 32, 0xFFFFFFFF, device["v"].data_ptr()], dtype=torch.int64).npu()
    args = arguments(values, kind)
    with forbid_host_readback():
        enqueue(producer, args, metadata_storage, query, kv, worker)
    return dict(
        spec=spec,
        inputs=inputs,
        device=device,
        outputs=outputs,
        guards=guards,
        ws_storage=ws_storage,
        workspace={n: t[512:-512] for n, t in ws_storage.items()},
        metadata_storage=metadata_storage,
        metadata=metadata_storage[1:-1],
        query=query,
        kv=kv,
        source_q=source_q,
        source_k=source_k,
        desc=desc,
        context=context,
        schedule=schedule,
        expected=row,
        values=values,
        args=args,
        scale=576**-0.5 if scale is None else scale,
        status=torch.full((3 * context.block_dim * 16,), SENTINEL, dtype=torch.int32, device="npu"),
    )


def launch(
    probes,
    case,
    name="migrated",
    *,
    checked=True,
    words=None,
    qcount=None,
    kcount=None,
    qaddress=None,
):
    d = case["device"]
    pointers = [d[n] for n in ("qv", "q", "v", "k")] + [
        case["desc"],
        *case["outputs"][name],
        case["workspace"][name],
        case["metadata"],
    ]
    stream = torch.npu.current_stream()
    for t in [
        *pointers,
        case["query"],
        case["kv"],
        case["source_q"],
        case["source_k"],
        case["status"],
    ]:
        if t is not None:
            t.record_stream(stream)
    result = probes[0][name].launch_mixed_attention(
        stream.npu_stream,
        *(t.data_ptr() for t in pointers),
        case["metadata"].numel() if words is None else words,
        case["query"].data_ptr() if qaddress is None else qaddress,
        case["query"].numel() if qcount is None else qcount,
        0 if case["kv"] is None else case["kv"].data_ptr(),
        (0 if case["kv"] is None else case["kv"].numel()) if kcount is None else kcount,
        ct.byref(case["context"]),
        ct.byref(case["schedule"]),
        case["scale"],
        case["source_q"].data_ptr(),
        case["source_k"].data_ptr(),
        # Fixed source ProcessQActualSeqLoop / ProcessKVActualSeqLoop.
        any(q != max(case["spec"][0], default=0) for q in case["spec"][0])
        or any(k == 0 for k in case["spec"][1]),
        case["status"].data_ptr(),
        checked,
    )
    assert result == 0, result


def references(case):
    qs, ks, capacity, heads, _ = case["spec"]
    all_refs = [[], [], [], []]
    qpos = kpos = 0
    for q, k in zip(qs, ks):
        if q:
            if k:
                inputs = {
                    name: t[
                        (qpos if name in ("q", "qv") else kpos) : (
                            qpos + q if name in ("q", "qv") else kpos + k
                        )
                    ].unsqueeze(0)
                    for name, t in case["inputs"].items()
                }
                raw = mla_reference_pair(**inputs, softmax_scale=case["scale"])
                refs = [raw[0][0], raw[1][0], raw[2][0], raw[3][0]]
            else:
                out = torch.zeros((q, heads, 512), dtype=case["inputs"]["v"].dtype)
                lse = torch.full((heads, q), -torch.inf if capacity else torch.inf)
                refs = [out, lse, out, lse]
            for dst, ref in zip(all_refs, refs):
                dst.append(ref)
        qpos += q
        kpos += capacity
    return [
        torch.cat(r, dim=i % 2)
        if r
        else torch.empty(
            (heads, 0) if i % 2 else (0, heads, 512),
            dtype=torch.float32 if i % 2 else case["inputs"]["v"].dtype,
        )
        for i, r in enumerate(all_refs)
    ]


def check_case(probes, case, *, repeats=3):
    qs, ks, capacity, heads, _ = case["spec"]
    empty = (
        torch.cat([torch.full((q,), k == 0, dtype=torch.bool) for q, k in zip(qs, ks)])
        if qs
        else torch.empty(0, dtype=torch.bool)
    )
    source_ok = bool(sum(qs) and capacity)
    first = None
    for _ in range(repeats):
        with forbid_host_readback():
            if source_ok:
                launch(probes, case, "oracle")
            launch(probes, case)
        actual = [t.cpu() for t in case["outputs"]["migrated"]]
        if source_ok:
            source = [t.cpu() for t in case["outputs"]["oracle"]]
            torch.testing.assert_close(actual[0], source[0], rtol=0, atol=0)
            torch.testing.assert_close(actual[1][:, ~empty], source[1][~empty].T, rtol=0, atol=0)
            assert torch.isposinf(source[1][empty]).all()
        if first is not None:
            for a, b in zip(actual, first):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        first = actual
    refs = references(case)
    errors = []
    for i in range(2):
        if i == 0:
            if first[i].numel():
                assert_fa_close(first[i], refs[i], refs[i + 2], name="mixed-out")
            finite = torch.ones_like(refs[i], dtype=torch.bool)
        else:
            assert torch.equal(torch.isneginf(first[i]), torch.isneginf(refs[i]))
            assert torch.equal(torch.isposinf(first[i]), torch.isposinf(refs[i]))
            finite = torch.isfinite(refs[i])
            if finite.any():
                assert_fa_close(
                    first[i][finite], refs[i][finite], refs[i + 2][finite], name="mixed-lse"
                )
        errors.append(
            float((first[i][finite] - refs[i][finite]).abs().max()) if finite.any() else 0
        )
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
    cq = torch.tensor([0, *torch.tensor(qs).cumsum(0).tolist()], dtype=torch.int32)
    assert torch.equal(case["query"].cpu(), cq)
    if case["kv"] is not None:
        assert case["kv"].cpu().tolist() == ks
    report = dict(
        spec=case["spec"],
        scale=case["scale"],
        dtype=str(probes[0]["dtype"]),
        errors=errors,
        source_compared=source_ok,
        m_base=case["schedule"].tiling.m_base,
        output_sha256=hashlib.sha256(
            b"".join(t.contiguous().view(torch.uint8).numpy().tobytes() for t in first)
        ).hexdigest(),
    )
    with (probes[1] / "accuracy.jsonl").open("a") as f:
        f.write(json.dumps(report) + "\n")
    return report
