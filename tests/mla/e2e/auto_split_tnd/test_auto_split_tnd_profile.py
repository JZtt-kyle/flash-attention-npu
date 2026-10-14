"""Automatic split TND native dispatch and fixed public R=1/4 average E2E."""

from tests.mla.support.native_package import native_vendor


import json
import os
from pathlib import Path
import subprocess
import sys

from tests.mla.support.mla_native_dispatch import parse_dispatch, sha256
from tests.mla.support.mla_validation_policy import check_public_performance
from tests.mla.support.mla_public_performance import summarize_varlen
from tests.mla.support.mla_auto_split_tnd_cases import PERFORMANCE_CASES


def native_profile(dtype):
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
        "tests.mla.support.mla_auto_split_tnd_profile",
        dtype,
        str(diagnostic),
        str(vendor),
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
        "tests.mla.support.mla_auto_split_tnd_profile",
        dtype,
        str(directory),
        str(vendor),
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
    report = json.loads((directory / "profile.json").read_text())
    assert report["automatic_split"]
    assert report["window"] and not report["causal"]
    assert any(row["native_call"]["sparse_mode"] == 4 for row in report["results"])
    return report


def test_native_public_profile():
    reports = [native_profile(dtype) for dtype in ("fp16", "bf16")]
    result = summarize_varlen(reports, PERFORMANCE_CASES, include_native_fd=True)
    (Path(os.environ["MLA_VALIDATION_RESULT"]) / "public-profile/average.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    check_public_performance(result)
