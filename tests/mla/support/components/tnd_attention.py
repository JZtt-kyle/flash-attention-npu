"""Shared helpers extracted from tests/mla/unit/tnd/test_flash_attn_npu_v3_mla_tnd_attention.py.

Compact TND v3 attention: fixed CANN, independent math and guarded storage.
"""

import ctypes as ct
import json
from pathlib import Path
import torch

import torch_npu  # noqa: F401 -- register the NPU backend before torch.npu helpers
from tests.mla.support.mla_ref import mla_reference_pair
from tests.mla.support.mla_dense_build import ROOT, sha256
from tests.mla.support.mla_metadata_error_probe import forbid_host_readback
from tests.mla.support.components.dense import Schedule
from tests.mla.support.components.tnd_metadata_aicpu import (
    TndArgs,
    arguments,
    device_inputs,
    enqueue,
    host_record,
    storage_for,
)
from tests.mla.support.components.tnd_metadata_consumer import (
    Context,
    SENTINEL,
    context_for,
    expected_schedule,
    tnd_case,
)


class TndSchedule(ct.Structure):
    _fields_ = [("tiling", Schedule), ("total_q", ct.c_uint32), ("total_k", ct.c_uint32)]


def load_attention(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "build-manifest.json").read_text())
    assert manifest["tnd"] and manifest["ndebug"]
    for name, digest in manifest["sources"].items():
        assert sha256(ROOT / name) == digest, name
    libraries = {}
    for name in ("oracle", "migrated"):
        row = manifest["libraries"][name]
        assert sha256(Path(row["path"])) == row["sha256"]
        lib = ct.CDLL(row["path"])
        lib.launch_tnd_attention.argtypes = [ct.c_void_p] * 10 + [
            ct.c_uint32,
            ct.c_void_p,
            ct.c_uint64,
            ct.c_void_p,
            ct.c_uint64,
            ct.POINTER(Context),
            ct.POINTER(TndSchedule),
            ct.c_float,
            ct.c_void_p,
            ct.c_void_p,
            ct.c_bool,
            ct.c_void_p,
            ct.c_bool,
        ]
        lib.launch_tnd_attention.restype = ct.c_int
        libraries[name] = lib
    libraries["dtype"] = torch.float16 if manifest["dtype"] == "fp16" else torch.bfloat16
    return libraries, directory


def configure_producer(producer):
    lib = producer["lib"]
    lib.launch_tnd_metadata.argtypes = [ct.c_void_p] * 4 + [ct.POINTER(TndArgs)]
    lib.launch_tnd_metadata.restype = ct.c_int
    lib.tnd_metadata_host.argtypes = [ct.POINTER(TndArgs), ct.c_void_p]
    lib.tnd_metadata_host.restype = ct.c_uint32


def make_case(probes, producer, spec, *, scale=None, offsets=(1, 3, 5, 7), amplitude=1.0, worker=0):
    qs, ks, heads, aic = spec
    p = producer["platform"]
    values = tnd_case(
        qs,
        ks,
        heads=heads,
        sq=max(qs, default=0) + 3,
        sk=min(16384, max(ks, default=0) + 7),
        aic=aic or p.aic,
        ratio=2,
        libapi=p.libapi,
        **{n: getattr(p.memory, n) for n in ("l1", "ub", "l0a", "l0b", "l0c")},
    )
    dtype = probes[0]["dtype"]
    kind = 1 if dtype == torch.float16 else 2
    row = host_record(producer, values, kind)
    context = context_for(row, values, kind)
    schedule = TndSchedule.from_buffer_copy((ct.c_uint32 * 91)(*expected_schedule(row)))
    gen = torch.Generator().manual_seed(20261009)
    inputs = {
        name: torch.randn(tokens, h, dim, generator=gen).to(dtype)
        for name, tokens, h, dim in [
            ("qv", sum(qs), heads, 512),
            ("q", sum(qs), heads, 64),
            ("v", sum(ks), 1, 512),
            ("k", sum(ks), 1, 64),
        ]
    }
    qpos = kpos = 0
    for q, k in zip(qs, ks):
        if not k:
            for name in ("qv", "q"):
                inputs[name][qpos : qpos + q] = torch.nan
        if not q:
            for name in ("v", "k"):
                inputs[name][kpos : kpos + k] = torch.nan
        qpos += q
        kpos += k
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
    # Independent source representation: int64 cumulative ends without leading 0.
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


def launch(probes, case, name="migrated", *, checked=True, words=None, qcount=None, kcount=None):
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
        t.record_stream(stream)
    result = probes[0][name].launch_tnd_attention(
        stream.npu_stream,
        *(t.data_ptr() for t in pointers),
        case["metadata"].numel() if words is None else words,
        case["query"].data_ptr(),
        case["query"].numel() if qcount is None else qcount,
        case["kv"].data_ptr(),
        case["kv"].numel() if kcount is None else kcount,
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
    qs, ks, heads, _ = case["spec"]
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
                lse = torch.full((heads, q), -torch.inf if sum(ks) else torch.inf)
                refs = [out, lse, out, lse]
            for dst, ref in zip(all_refs, refs):
                dst.append(ref)
        qpos += q
        kpos += k
    return [
        torch.cat(r, dim=i % 2)
        if r
        else torch.empty(
            (heads, 0) if i % 2 else (0, heads, 512),
            dtype=torch.float32 if i % 2 else case["inputs"]["v"].dtype,
        )
        for i, r in enumerate(all_refs)
    ]
