"""M06b3: standalone v2 BSND attention, pinned CANN and FA3 empty-row semantics."""

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
from tests.mla.support.mla_dense_build import ROOT, build_probes, sha256
from tests.mla.support.mla_metadata_error_probe import forbid_host_readback
from tests.mla.support.components.bsnd_metadata import arguments
from tests.mla.unit.bsnd.test_flash_attn_npu_v3_mla_bsnd_metadata import check_source
from tests.mla.unit.bsnd.test_flash_attn_npu_v3_mla_bsnd_metadata_aicpu import (
    bsnd_probe as bsnd_probe,
)
from tests.mla.support.components.bsnd_metadata_aicpu import enqueue, host_record
from tests.mla.unit.bsnd.test_flash_attn_npu_v3_mla_bsnd_schedule import (
    length_artifacts as length_artifacts,
    length_oracle as length_oracle,
)
from tests.mla.support.components.metadata import load_probe

from tests.mla.support.components.bsnd_attention import (
    CASES as CASES,
    PERFORMANCE_CASES as PERFORMANCE_CASES,
    check_bsnd_case as check_bsnd_case,
    configure as configure,
    configure_producer as configure_producer,
    launch_bsnd as launch_bsnd,
    load_attention as load_attention,
    make_bsnd_case as make_bsnd_case,
    references as references,
    values_for as values_for,
)


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
        "tests.mla.unit.bsnd.test_flash_attn_npu_v3_mla_bsnd_attention",
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
