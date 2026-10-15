"""TND PA causal average public E2E and representative source component samples."""

import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

from tests.mla.support.native_package import native_vendor

import pytest
import torch

from tests.mla.support.mla_dense_build import build_probes
from tests.mla.support.mla_native_dispatch import parse_dispatch, sha256
from tests.mla.support.mla_validation_policy import check_public_performance
from tests.mla.support.mla_public_performance import summarize_mixed
from tests.mla.support.mla_causal_tnd_profile import sample_kernels
from tests.mla.support.public_cases.causal_paged_tnd_public import (
    PERFORMANCE_CASES,
    REPRESENTATIVES,
    make_case,
    compare,
    call,
    forward_args,
    component,
    paged_probe as paged_probe,
)


@pytest.fixture(scope="module")
def profile_probes():
    result = {}
    for dtype in ("fp16", "bf16"):
        directory = Path(os.environ["MLA_VALIDATION_RESULT"]) / "probes" / dtype
        build_probes(directory, dtype, paged=True, paged_causal=True)
        result[dtype] = component.load_attention(directory)
    return result


def producer_manifest():
    return Path(os.environ["MLA_PAGED_METADATA_LIBRARY"]).parent / "build-manifest.json"


def native_profile(dtype, probes, producer_manifest):
    directory = Path(os.environ["MLA_VALIDATION_RESULT"]) / "public-profile" / dtype
    directory.parent.mkdir(parents=True, exist_ok=True)
    vendor = native_vendor()
    env = dict(os.environ, ASCEND_CUSTOM_OPP_PATH=str(vendor), MLA_CANN_VENDOR=str(vendor))
    env["LD_LIBRARY_PATH"] = str(vendor / "op_api/lib") + ":" + env.get("LD_LIBRARY_PATH", "")
    diagnostic = directory.with_name(dtype + "-dispatch")
    log_path = directory.with_suffix(".dispatch.log")
    command = [
        sys.executable,
        "-m",
        "tests.mla.support.mla_causal_paged_tnd_profile",
        dtype,
        str(diagnostic),
        str(vendor),
        str(probes),
        str(producer_manifest),
    ]
    with log_path.open("w") as log:
        result = subprocess.run(
            command,
            env=dict(env, ASCEND_GLOBAL_LOG_LEVEL="0", ASCEND_SLOG_PRINT_TO_STDOUT="1"),
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=600,
        )
    assert result.returncode == 0, str(log_path)
    proof = parse_dispatch(log_path, dtype, len(PERFORMANCE_CASES), vendor)
    accuracy = diagnostic / "accuracy.json"
    environment = diagnostic / "environment.json"
    proof["inputs"] = json.loads(accuracy.read_text())["inputs"]
    for path in (accuracy, environment):
        proof["artifacts"][str(path)] = sha256(path)
    proof_path = directory.with_suffix(".dispatch.json")
    proof_path.write_text(json.dumps(proof, indent=2) + "\n")
    command = [
        sys.executable,
        "-m",
        "tests.mla.support.mla_causal_paged_tnd_profile",
        dtype,
        str(directory),
        str(vendor),
        str(probes),
        str(producer_manifest),
        str(proof_path),
    ]
    affinity = os.sched_getaffinity(0)
    os.sched_setaffinity(0, {min(affinity)})
    try:
        with directory.with_suffix(".timing.log").open("w") as log:
            result = subprocess.run(
                command,
                env=dict(env, ASCEND_GLOBAL_LOG_LEVEL="3", ASCEND_SLOG_PRINT_TO_STDOUT="0"),
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=600,
            )
    finally:
        os.sched_setaffinity(0, affinity)
    assert result.returncode == 0, str(directory.with_suffix(".timing.log"))
    return json.loads((directory / "profile.json").read_text())


def test_native_public_profile(profile_probes, paged_probe):
    reports = [
        native_profile(dtype, profile_probes[dtype][1], producer_manifest())
        for dtype in ("fp16", "bf16")
    ]
    result = summarize_mixed(reports, PERFORMANCE_CASES)
    (Path(os.environ["MLA_VALIDATION_RESULT"]) / "public-profile/average.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    check_public_performance(result)


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_component_profile(profile_probes, paged_probe, dtype):
    probes = profile_probes[dtype]
    output = Path(os.environ["MLA_VALIDATION_RESULT"]) / "component" / dtype
    output.mkdir(parents=True)
    cases = [make_case(probes, paged_probe, s) for s in REPRESENTATIVES]
    for c in cases:
        compare(probes, c, [call(forward_args(c), "op")])

    def run(index, name):
        c = cases[index]
        if name == "production":
            return call(forward_args(c), "op")
        assert component.launch(probes, c, "oracle") == 0
        return c["outputs"]["oracle"]

    names = ("production", "oracle")
    affinity = os.sched_getaffinity(0)
    os.sched_setaffinity(0, {min(affinity)})
    try:
        for i in range(len(cases)):
            for _ in range(5):
                for name in names:
                    result = run(i, name)
                    torch.npu.synchronize()
        del result
        samples, paths = sample_kernels(
            run,
            len(cases),
            output,
            names,
            "mla_paged_tnd_",
            "cann_paged_attention_",
            "ComputeMlaCausalPagedTndMetadata",
        )
        rows = []
        for i, case in enumerate(cases):
            values = {
                n: statistics.median(
                    r["kernel_us"] for r in samples if r["case"] == i and r["name"] == n
                )
                for n in names
            }
            rows.append(
                dict(spec=case["config"], **values, ratio=values["production"] / values["oracle"])
            )
        (output / "summary.json").write_text(
            json.dumps(
                dict(warmup=5, rounds=5, profiler_iterations=10, results=rows, **paths), indent=2
            )
            + "\n"
        )
    finally:
        os.sched_setaffinity(0, affinity)
