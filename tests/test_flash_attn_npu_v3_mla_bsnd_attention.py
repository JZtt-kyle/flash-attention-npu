"""M06b3: standalone v2 BSND attention, pinned CANN and FA3 empty-row semantics."""

import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

import pytest
import torch
import torch_npu  # noqa: F401

from tests.common.compare import assert_fa_close
from tests.common.mla_ref import mla_reference_pair
from tests.mla_dense_build import ROOT, build_probes, sha256
from tests.mla_metadata_error_probe import forbid_host_readback
from tests.test_flash_attn_npu_v3_mla_bsnd_metadata import Args, arguments, check_source
from tests.test_flash_attn_npu_v3_mla_bsnd_metadata_aicpu import (
    bsnd_probe as bsnd_probe,
    device_lengths,
    enqueue,
    host_record,
)
from tests.test_flash_attn_npu_v3_mla_bsnd_metadata_consumer import (
    context_for,
    expected_schedule,
)
from tests.test_flash_attn_npu_v3_mla_bsnd_schedule import (
    length_artifacts as length_artifacts,
    length_case,
    length_oracle as length_oracle,
)
from tests.test_flash_attn_npu_v3_mla_dense import Schedule, load_probes, make_case
from tests.test_flash_attn_npu_v3_mla_metadata_consumer import Context, SENTINEL
from tests.test_flash_attn_npu_v3_mla_metadata import load_probe


# (B, Sq, Hq, KV capacity), actual lengths, optional synthetic core count.
# The 15-core case preserves the source endpoint before the final zero sample.
CASES = [
    ((2, 7, 16, 33), [17, 0], None),
    ((2, 1, 8, 257), [17, 129], None),
    ((1, 7, 16, 65), [33], None),
    ((2, 2, 128, 1025), [511, 513], None),
    ((6, 3, 171, 1025), [0, 1, 511, 512, 513, 1024], None),
    ((2, 1, 8, 16384), [8193, 16383], None),
    ((16, 1, 1, 513), [0, 1, 511, 512, 513] * 3 + [0], 15),
    ((5, 3, 171, 513), [0, 17, 0, 513, 0], 1),
    ((1, 1, 1, 1), [0], None),
    ((41, 3, 171, 65), [0] * 41, None),
    ((41, 2, 129, 65), [0, 17, 0, 33] * 10 + [0], None),
    ((2, 1, 8, 257), None, None),
    ((2, 1, 8, 257), [257, 257], None),
    ((1, 7, 16, 65), [65], None),
    ((1, 1, 128, 512), [512], None),
    ((2, 2, 128, 511), [511, 511], None),
    ((2, 1, 8, 8193), [8193, 8193], None),
    ((1, 1, 128, 16384), [16384], None),
]
PERFORMANCE_CASES = [CASES[i] for i in (0, 1, 2, 3, 5, 6, 8, 9, 11, 14, 16, 17)]


def configure(libraries, directory, dtype):
    probes = load_probes(libraries, directory, dtype)
    for name in ("oracle", "migrated"):
        probes[0][name].launch_dense.argtypes = [ct.c_void_p] * 10 + [
            ct.c_float,
            ct.c_void_p,
            ct.c_void_p,
            ct.c_uint32,
            ct.POINTER(Context),
            ct.c_bool,
        ]
    return probes


def load_attention(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "build-manifest.json").read_text())
    assert manifest["bsnd"]
    for name, digest in manifest["sources"].items():
        assert sha256(ROOT / name) == digest, name
    for row in manifest["libraries"].values():
        assert sha256(Path(row["path"])) == row["sha256"]
    dtype = torch.float16 if manifest["dtype"] == "fp16" else torch.bfloat16
    return configure({k: v["path"] for k, v in manifest["libraries"].items()}, directory, dtype)


def configure_producer(producer):
    producer["lib"].launch_bsnd_metadata.argtypes = [ct.c_void_p] * 4 + [ct.POINTER(Args)]
    producer["lib"].launch_bsnd_metadata.restype = ct.c_int
    producer["lib"].bsnd_metadata_host.argtypes = [ct.POINTER(Args), ct.c_void_p]
    producer["lib"].bsnd_metadata_host.restype = ct.c_uint32


@pytest.fixture(scope="module", params=("fp16", "bf16"))
def bsnd_attention(tmp_path_factory, request):
    directory = (
        Path(
            os.environ.get("MLA_BSND_ATTENTION_DIR", tmp_path_factory.mktemp("mla-bsnd-attention"))
        )
        / request.param
    )
    directory.mkdir(parents=True, exist_ok=True)
    if os.environ.get("MLA_BSND_ATTENTION_REUSE") == "1":
        return load_attention(directory)
    libraries = build_probes(directory, request.param, bsnd=True)
    dtype = torch.float16 if request.param == "fp16" else torch.bfloat16
    return configure(libraries, directory, dtype)


def values_for(producer, shape, lengths, aic=None):
    p = producer["platform"]
    return length_case(
        lengths,
        batch=shape[0],
        sq=shape[1],
        heads=shape[2],
        sk=shape[3],
        aic=aic or p.aic,
        ratio=2,
        libapi=p.libapi,
        **{name: getattr(p.memory, name) for name in ("l1", "ub", "l0a", "l0b", "l0c")},
    )


def make_bsnd_case(probes, producer, spec, *, scale=None, poison=True, worker=0):
    shape, lengths, aic = spec
    values = values_for(producer, shape, lengths, aic)
    dtype = 1 if probes[0]["dtype"] == torch.float16 else 2
    row = host_record(producer, values, dtype)
    context = context_for(row, values, dtype)
    schedule = Schedule.from_buffer_copy((ct.c_uint32 * 89)(*expected_schedule(row)))

    def wide(index):
        return row[index] | row[index + 1] << 32

    details = [
        wide(385),
        wide(383),
        context.aic,
        context.aiv,
        *[getattr(context.memory, key) for key in ("l1", "ub", "l0a", "l0b", "l0c")],
        context.libapi,
        wide(375),
        wide(377),
        wide(379),
        wide(381),
        row[22],
        row[103],
    ]
    case = make_case(probes, *shape, scale=scale, offset=1, schedule_details=(schedule, details))
    logical = [shape[3]] * shape[0] if lengths is None else lengths
    if poison:
        for b, length in enumerate(logical):
            for name in ("k", "v"):
                case["inputs"][name][b, length:] = torch.nan
            if length == 0:
                for name in ("q", "qv"):
                    case["inputs"][name][b] = torch.nan
        for name, tensor in case["device"].items():
            tensor.copy_(case["inputs"][name])
        for name, (before, _, prefix) in zip(case["inputs"], case["storage"][:4]):
            before[prefix : prefix + case["inputs"][name].numel()] = case["inputs"][name].flatten()
    storage = torch.full((len(row) + 2,), SENTINEL, dtype=torch.int32, device="npu")
    current = device_lengths(values)
    # Fixed CANN consumes numerical int64 lengths; migrated v2 consumes int32.
    source_lengths = (
        torch.tensor(logical, dtype=torch.int64, device="npu") if lengths is not None else None
    )
    case.update(
        metadata_storage=storage,
        metadata=storage[1:-1],
        lengths=current,
        source_lengths=source_lengths,
        logical_lengths=logical,
        context=context,
        expected=row,
        values=values,
        poison=poison,
        args=arguments(values, dtype),
    )
    with forbid_host_readback():
        enqueue(producer, case["args"], storage, current, worker)
    return case


def launch_bsnd(probes, case, name="migrated"):
    d = case["device"]
    tensors = [d[n] for n in ("qv", "q", "v", "k")]
    tensors += [case["desc"], *case["outputs"][name], case["workspace"][name]]
    lengths = case["source_lengths"] if name == "oracle" else case["lengths"]
    stream = torch.npu.current_stream()
    for tensor in [*tensors, case["metadata"], *([] if lengths is None else [lengths])]:
        tensor.record_stream(stream)
    result = probes[0][name].launch_dense(
        stream.npu_stream,
        *(t.data_ptr() for t in tensors),
        ct.byref(case["schedule"]),
        case["scale"],
        0 if lengths is None else lengths.data_ptr(),
        case["metadata"].data_ptr(),
        case["metadata"].numel(),
        ct.byref(case["context"]),
        any(length != case["shape"][3] for length in case["logical_lengths"]),
    )
    assert result == 0, result


def references(case):
    batch, sq, heads, _ = case["shape"]
    outputs = [[], [], [], []]
    for b, length in enumerate(case["logical_lengths"]):
        if length == 0:
            # Capacity is nonzero: FA3 kernel empty row has -inf LSE. Calling
            # dense reference with a sliced-empty K would test its +inf host case.
            out = torch.zeros((1, sq, heads, 512), dtype=case["inputs"]["v"].dtype)
            lse = torch.full((1, heads, sq), -torch.inf)
            refs = (out, lse, out, lse)
        else:
            inputs = {
                name: tensor[b : b + 1, : length if name in ("k", "v") else sq]
                for name, tensor in case["inputs"].items()
            }
            refs = mla_reference_pair(**inputs, softmax_scale=case["scale"])
        for result, ref in zip(outputs, refs):
            result.append(ref)
    return [torch.cat(result) for result in outputs]


def check_bsnd_case(probes, case, *, repeats=3):
    first = None
    empty = torch.tensor(case["logical_lengths"]) == 0
    for _ in range(repeats):
        with forbid_host_readback():
            launch_bsnd(probes, case, "oracle")
            launch_bsnd(probes, case)
        source = [t.cpu() for t in case["outputs"]["oracle"]]
        actual = [t.cpu() for t in case["outputs"]["migrated"]]
        torch.testing.assert_close(actual[0], source[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[1][~empty], source[1][~empty], rtol=0, atol=0)
        assert torch.isposinf(source[1][empty]).all()
        assert torch.isneginf(actual[1][empty]).all()
        assert (actual[0][empty] == 0).all()
        if first is not None:
            for value, old in zip(actual, first):
                torch.testing.assert_close(value, old, rtol=0, atol=0)
        first = actual
    refs = references(case)
    errors = []
    for i, name in enumerate(("out", "lse")):
        assert_fa_close(first[i], refs[i], refs[i + 2], name=name)
        finite = torch.isfinite(refs[i])
        errors.append(
            float((first[i][finite] - refs[i][finite]).abs().max()) if finite.any() else 0.0
        )
    for i, (before, tensor, prefix) in enumerate(case["storage"]):
        after = tensor.cpu()
        if i < 4:
            # Bitwise preservation includes poison NaNs.
            assert torch.equal(after.view(torch.uint8), before.view(torch.uint8))
        else:
            assert torch.equal(after[:prefix], before[:prefix])
            assert torch.equal(after[-16:], before[-16:])
    for ws in case["ws_storage"].values():
        assert (ws[:512].cpu() == 205).all() and (ws[-512:].cpu() == 205).all()
        assert (ws[512 + case["details"][13] :].cpu() == 205).all()
    assert [x & 0xFFFFFFFF for x in case["metadata_storage"].cpu().tolist()] == [
        SENTINEL,
        *case["expected"],
        SENTINEL,
    ]
    if case["lengths"] is not None:
        assert case["lengths"].cpu().tolist() == case["logical_lengths"]
    s = case["schedule"]
    report = dict(
        dtype=case["dtype"],
        shape=case["shape"],
        lengths=case["logical_lengths"],
        scale=case["scale"],
        poison=case["poison"],
        errors=errors,
        repeats=repeats,
        m_base=s.m_base,
        block_dim=s.block_dim,
        cores=s.cores,
        endpoints=list(zip(s.b_end[: s.cores], s.m_end[: s.cores], s.s_end[: s.cores])),
        source_lse_empty="+inf",
        fa3_lse_empty="-inf",
        sha256=hashlib.sha256(
            b"".join(t.contiguous().view(torch.uint8).numpy().tobytes() for t in first)
        ).hexdigest(),
    )
    with (probes[1] / "accuracy.jsonl").open("a") as f:
        f.write(json.dumps(report) + "\n")
    return report


@pytest.mark.parametrize("spec", CASES)
def test_accuracy(bsnd_attention, bsnd_probe, spec):
    case = make_bsnd_case(bsnd_attention, bsnd_probe, spec)
    if spec[2] == 15:
        assert case["schedule"].b_end[case["schedule"].cores - 1] == 15
    check_bsnd_case(bsnd_attention, case)


@pytest.mark.parametrize("scale", [0.0, -0.23, 0.23])
@pytest.mark.parametrize("spec", [CASES[0], CASES[3], CASES[7]])
def test_scales(bsnd_attention, bsnd_probe, scale, spec):
    check_bsnd_case(bsnd_attention, make_bsnd_case(bsnd_attention, bsnd_probe, spec, scale=scale))


@pytest.mark.parametrize(
    "length", [1, 15, 16, 17, 31, 32, 33, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024]
)
def test_tail_lengths(bsnd_attention, bsnd_probe, length):
    spec = ((2, 1, 17, 1025), [length, 0], 1)
    check_bsnd_case(bsnd_attention, make_bsnd_case(bsnd_attention, bsnd_probe, spec))


def test_source_schedules(bsnd_probe, length_oracle):
    cases = [values_for(bsnd_probe, *spec) for spec in CASES]
    rows = [host_record(bsnd_probe, values, 1) for values in cases]
    check_source(rows, cases, length_oracle, bsnd_probe["directory"], "attention-schedules")


def test_clone_streams_and_regenerate(bsnd_attention, bsnd_probe):
    case = make_bsnd_case(bsnd_attention, bsnd_probe, CASES[3])
    for i, lengths in enumerate(([513, 0], [0, 0], [0, 1025], [1024, 511])):
        # New input contents as well as lengths, with independent buffers.
        current = make_bsnd_case(
            bsnd_attention,
            bsnd_probe,
            (CASES[3][0], lengths, None),
            scale=(i - 1) * 0.23,
            worker=i % 2,
        )
        case["lengths"].copy_(current["lengths"])
        case["args"] = arguments(current["values"], case["context"].dtype)
        with forbid_host_readback():
            enqueue(bsnd_probe, case["args"], case["metadata_storage"], case["lengths"], i % 2)
            cloned = case["metadata"].clone()
            ready = torch.npu.Event()
            ready.record()
            stream = torch.npu.Stream()
            with torch.npu.stream(stream):
                stream.wait_event(ready)
                current.update(metadata=cloned, lengths=case["lengths"])
                launch_bsnd(bsnd_attention, current)
            done = torch.npu.Event()
            done.record(stream)
            torch.npu.current_stream().wait_event(done)
        refs = references(current)
        for index, out in enumerate(current["outputs"]["migrated"]):
            assert_fa_close(out.cpu(), refs[index], refs[index + 2], name=("out", "lse")[index])
        assert [x & 0xFFFFFFFF for x in cloned.cpu().tolist()] == current["expected"]
    check_bsnd_case(bsnd_attention, current)


@pytest.mark.parametrize("fault", ("stale", "header", "suffix"))
def test_runtime_rejection(bsnd_attention, bsnd_probe, fault):
    directory = bsnd_attention[1] / f"error-{fault}"
    directory.mkdir(exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "tests.test_flash_attn_npu_v3_mla_bsnd_attention",
        str(bsnd_attention[1]),
        str(bsnd_probe["directory"] / "build-manifest.json"),
        fault,
        str(directory),
    ]
    (directory / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    with (directory / "process.log").open("w") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            code = process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            pytest.fail(f"v2 attention rejection hung: {directory}")
    assert code == 0, directory
    report = json.loads((directory / "runtime-error.json").read_text())
    assert report["stage"] == "synchronize" and report["seconds"] < 10


def error_probe(directory, producer_manifest, fault, evidence):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    probes = load_attention(directory)
    manifest = json.loads(Path(producer_manifest).read_text())
    for name, digest in manifest["sources"].items():
        assert sha256(ROOT / name) == digest
    assert sha256(Path(manifest["library"])) == manifest["library_sha256"]
    with load_probe(manifest["library"], Path(evidence) / "producer") as producer:
        configure_producer(producer)
        case = make_bsnd_case(probes, producer, ((5, 3, 171, 513), [1, 0, 7, 0, 0], None))
        check_bsnd_case(probes, case, repeats=1)
        if fault == "stale":
            case["lengths"][2] = 0
        elif fault == "header":
            case["metadata"][1] = 1
        else:
            for index, value in ((20, 1), (23, 1), (49, 0), (75, 0)):
                case["metadata"][index] = value
        torch.npu.synchronize()
        stage = "launch"
        start = time.monotonic()
        try:
            with forbid_host_readback():
                launch_bsnd(probes, case)
            stage = "synchronize"
            torch.npu.synchronize()
        except RuntimeError as error:
            report = dict(
                fault=fault,
                stage=stage,
                seconds=time.monotonic() - start,
                error=str(error),
                device=torch.npu.get_device_name(0),
            )
            (Path(evidence) / "runtime-error.json").write_text(json.dumps(report, indent=2) + "\n")
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
        raise AssertionError("invalid BSND attention record did not Trap")


if __name__ == "__main__":
    error_probe(*sys.argv[1:])
