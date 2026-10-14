"""Compact TND v3 attention: fixed CANN, independent math and guarded storage."""

import ctypes as ct
import hashlib
import json
import os
from pathlib import Path

import pytest
import torch
import torch_npu  # noqa: F401

from tests.common.compare import assert_fa_close
from tests.mla.support.mla_ref import mla_reference_pair
from tests.mla.support.mla_dense_build import ROOT, build_probes, sha256
from tests.mla.support.mla_metadata_error_probe import forbid_host_readback
from tests.mla.unit.dense.test_flash_attn_npu_v3_mla_dense import Schedule
from tests.mla.unit.tnd.test_flash_attn_npu_v3_mla_tnd_metadata_aicpu import (
    tnd_probe as tnd_probe,
    TndArgs,
    arguments,
    device_inputs,
    enqueue,
    host_record,
    storage_for,
)
from tests.mla.unit.tnd.test_flash_attn_npu_v3_mla_tnd_metadata_consumer import (
    tnd_artifacts as tnd_artifacts,
    tnd_oracle as tnd_oracle,
    Context,
    SENTINEL,
    TND_CORRUPTIONS,
    check_records,
    context_for,
    expected_schedule,
    tnd_case,
)


# Q lengths, KV lengths, Hq, optional synthetic AIC count. The input capacity is
# the sum of the given segments, never B * max(lengths). Include total KV >16K.
CASES = [
    ([3, 0, 1], [17, 0, 33], 128, None),  # T01
    ([1, 7, 2], [1, 513, 31], 16, None),
    ([0, 3, 1, 0, 7], [33, 511, 0, 513, 1025], 171, None),
    ([1, 2, 3], [16384, 8193, 17], 8, None),
    ([1, 1, 1], [17, 129, 257], 129, None),
    ([2] * 10, [511] + [0] * 9, 128, 15),  # omitted zero-cost suffix
    ([3, 0, 1, 7, 0], [17, 65, 513, 0, 1], 171, 1),  # task ring crosses samples
    ([3, 0] * 21, [0, 513] * 21, 171, None),  # nonempty, zero total cost
    ([257, 1, 0, 3], [33, 513, 17, 1], 1, None),
    ([1, 1, 1, 1], [31, 32, 33, 65], 15, None),
    ([1, 1, 1, 1], [127, 128, 129, 511], 255, None),
    ([2, 0, 1], [512, 0, 513], 256, None),
    ([0, 1, 0, 0], [0, 0, 0, 1], 1, None),
]
PERFORMANCE_CASES = [CASES[i] for i in (0, 1, 2, 3, 4, 5, 6, 7, 8, 12)]
EMPTY_CASES = [
    ([], [], 128, None),
    ([0, 0], [0, 513], 128, None),
    ([3, 0, 1], [0, 0, 0], 171, None),
    ([0], [0], 1, None),
]


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


@pytest.fixture(scope="module", params=("fp16", "bf16"))
def tnd_attention(tmp_path_factory, request):
    directory = (
        Path(os.environ.get("MLA_TND_ATTENTION_DIR", tmp_path_factory.mktemp("mla-tnd-attention")))
        / request.param
    )
    if os.environ.get("MLA_TND_ATTENTION_REUSE") != "1":
        build_probes(directory, request.param, tnd=True)
    return load_attention(directory)


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


def check_case(probes, case, *, repeats=3):
    qs, ks, heads, _ = case["spec"]
    empty = (
        torch.cat([torch.full((q,), k == 0, dtype=torch.bool) for q, k in zip(qs, ks)])
        if qs
        else torch.empty(0, dtype=torch.bool)
    )
    source_ok = bool(sum(qs) and sum(ks))
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
                assert_fa_close(first[i], refs[i], refs[i + 2], name="tnd-out")
            finite = torch.ones_like(refs[i], dtype=torch.bool)
        else:
            assert torch.equal(torch.isneginf(first[i]), torch.isneginf(refs[i]))
            assert torch.equal(torch.isposinf(first[i]), torch.isposinf(refs[i]))
            finite = torch.isfinite(refs[i])
            if finite.any():
                assert_fa_close(
                    first[i][finite], refs[i][finite], refs[i + 2][finite], name="tnd-lse"
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


@pytest.mark.parametrize("spec", CASES + EMPTY_CASES)
def test_accuracy(tnd_attention, tnd_probe, spec):
    check_case(tnd_attention, make_case(tnd_attention, tnd_probe, spec))


@pytest.mark.parametrize("scale", [0.0, -0.03125, 0.125])
def test_scales_and_max_changes(tnd_attention, tnd_probe, scale):
    check_case(
        tnd_attention, make_case(tnd_attention, tnd_probe, CASES[2], scale=scale, amplitude=3.0)
    )


def test_source_schedules(tnd_probe, tnd_oracle, tmp_path):
    dummy = ({"dtype": torch.float16}, tmp_path)
    cases = [make_case(dummy, tnd_probe, spec) for spec in CASES]
    check_records(
        [c["expected"] for c in cases],
        [c["values"] for c in cases],
        tnd_oracle,
        tmp_path,
        "attention",
        1,
    )


def test_corruption_precedes_attention(tnd_attention, tnd_probe):
    case = make_case(tnd_attention, tnd_probe, ([1, 1], [257, 257], 8, None))
    good = case["metadata"].clone()
    for word, value, status in TND_CORRUPTIONS:
        case["metadata"].copy_(good)
        case["metadata"][word] = value if value < 2**31 else value - 2**32
        with forbid_host_readback():
            launch(tnd_attention, case, checked=False)
        assert (case["status"][::16].cpu() == status).all()
        assert (case["status"].view(-1, 16)[:, 1:].cpu() == SENTINEL).all()
        for t in case["outputs"]["migrated"]:
            assert (t.cpu() == 12345).all()
        assert (case["ws_storage"]["migrated"].cpu() == 205).all()
    case["metadata"].copy_(good)
    # Same allocation and same total capacity with changed segment boundaries.
    case["query"][1] = 0
    launch(tnd_attention, case, checked=False)
    assert (case["status"][::16].cpu() == 12).all()
    for t in case["outputs"]["migrated"]:
        assert (t.cpu() == 12345).all()
    assert (case["ws_storage"]["migrated"].cpu() == 205).all()


def test_streams_clone_and_regeneration(tnd_attention, tnd_probe):
    streams = [torch.npu.Stream(), torch.npu.Stream()]
    original = torch.npu.current_stream()
    case = make_case(tnd_attention, tnd_probe, ([3, 0, 1], [17, 0, 33], 128, None))
    for i, stream in enumerate(streams):
        stream.wait_stream(original)
        with torch.npu.stream(stream):
            clone = case["metadata"].clone()
            other = dict(case, metadata=clone)
            with forbid_host_readback():
                launch(tnd_attention, other)
            del clone, other
            torch.empty_like(case["metadata"]).fill_(-1)
        original.wait_stream(stream)
    check_case(tnd_attention, case)
    # Update boundaries while retaining totals; all inputs are regenerated from
    # the new independent logical segments (old poison must not become data).
    updated = make_case(tnd_attention, tnd_probe, ([1, 2, 1], [16, 17, 17], 128, None), worker=1)
    updated["metadata_storage"] = case["metadata_storage"]
    updated["metadata"] = case["metadata"]
    with forbid_host_readback():
        enqueue(
            tnd_probe,
            updated["args"],
            updated["metadata_storage"],
            updated["query"],
            updated["kv"],
            1,
        )
        launch(tnd_attention, updated)
    check_case(tnd_attention, updated)


def test_device_schedule_and_prefix_alias(tnd_attention, tnd_probe):
    case = make_case(tnd_attention, tnd_probe, ([1, 3], [1, 3], 129, None))
    case["kv"] = case["query"]
    check_case(tnd_attention, case)
    expected = [t.clone() for t in case["outputs"]["migrated"]]
    case["schedule"] = TndSchedule()  # Only the fixed-source driver may use this.
    for t in case["outputs"]["migrated"]:
        t.fill_(12345)
    with forbid_host_readback():
        launch(tnd_attention, case)
    for actual, want in zip(case["outputs"]["migrated"], expected):
        torch.testing.assert_close(actual.cpu(), want.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize(
    "fault",
    [
        "v1",
        "v2",
        "words",
        "span",
        "workspace",
        "stale_q",
        "stale_k",
        "empty_stale",
        "empty_batch",
        "empty_kv",
        "suffix",
        "variable_m",
        "generation",
    ],
)
def test_runtime_rejection(tnd_attention, tnd_probe, fault):
    import subprocess
    import sys

    directory = tnd_attention[1] / ("trap-" + fault)
    directory.mkdir(exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "tests.mla.support.mla_tnd_attention_error_probe",
        str(tnd_attention[1]),
        str(tnd_probe["directory"] / "build-manifest.json"),
        fault,
        str(directory),
    ]
    (directory / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    with (directory / "process.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=90)
    assert result.returncode == 0, (directory / "process.log").read_text()[-4000:]
    report = json.loads((directory / "runtime-error.json").read_text())
    assert report["elapsed_seconds"] < 15
