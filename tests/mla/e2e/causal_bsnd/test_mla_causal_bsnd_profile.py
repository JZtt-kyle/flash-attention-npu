"""Public causal E2E on the fixed native CANN package, with R fixed at 1/4."""

import json
import os
from pathlib import Path
import subprocess
import sys

from tests.mla.support.native_package import native_vendor

import pytest

from tests.mla.support.mla_native_dispatch import parse_dispatch, sha256
from tests.mla.support.mla_validation_policy import check_public_performance
from tests.mla.support.mla_public_performance import summarize
from tests.mla.support.public_cases.causal_public import NATIVE_PREFLIGHT_CASES, PERFORMANCE_CASES
from tests.mla.unit.causal_bsnd.test_flash_attn_npu_v3_mla_causal_attention import (
    causal_attention as causal_attention,
)
from tests.mla.unit.causal_bsnd.test_flash_attn_npu_v3_mla_causal_metadata import (
    metadata_driver as metadata_driver,
)
from tests.mla.unit.causal_bsnd.test_flash_attn_npu_v3_mla_causal_schedule import (
    causal_artifacts as causal_artifacts,
)
from tests.mla.support.components.used_schedule import (
    used_artifacts as used_artifacts,
    used_oracle as used_oracle,
)


def native_profile(dtype, preflight=False):
    directory = (
        Path(os.environ["MLA_VALIDATION_RESULT"])
        / ("native-preflight" if preflight else "public-profile")
        / dtype
    )
    vendor = native_vendor()
    env = dict(
        os.environ,
        ASCEND_CUSTOM_OPP_PATH=str(vendor),
        MLA_CANN_VENDOR=str(vendor),
        MLA_FORWARD_ARTIFACT_DIR=str(directory),
        MLA_FORWARD_DTYPE=dtype,
        MLA_FORWARD_CAUSAL="1",
        MLA_FORWARD_METADATA_REUSE="1",
    )
    if preflight:
        env["MLA_FORWARD_CAUSAL_PREFLIGHT"] = "1"
    env["LD_LIBRARY_PATH"] = str(vendor / "op_api/lib") + ":" + env.get("LD_LIBRARY_PATH", "")
    affinity = os.sched_getaffinity(0)
    try:
        log_path = directory.with_suffix(".dispatch.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostic = directory.with_name(directory.name + "-dispatch")
        diagnostic_env = dict(
            env,
            MLA_FORWARD_DISPATCH_ONLY="1",
            MLA_FORWARD_ARTIFACT_DIR=str(diagnostic),
            ASCEND_GLOBAL_LOG_LEVEL="0",
            ASCEND_SLOG_PRINT_TO_STDOUT="1",
        )
        with log_path.open("w") as log:
            result = subprocess.run(
                [sys.executable, "-m", "tests.mla.support.mla_forward_profile"],
                env=diagnostic_env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        assert result.returncode == 0, str(log_path)
        count = len(NATIVE_PREFLIGHT_CASES if preflight else PERFORMANCE_CASES)
        proof = parse_dispatch(log_path, dtype, count, vendor)
        accuracy = diagnostic / "accuracy.json"
        proof["inputs"] = [
            {k: v for k, v in item.items() if k != "comparisons"}
            for item in json.loads(accuracy.read_text())
        ]
        proof["artifacts"][str(accuracy)] = sha256(accuracy)
        environment = diagnostic / "environment.json"
        proof["artifacts"][str(environment)] = sha256(environment)
        proof_path = directory.with_suffix(".dispatch.json")
        proof_path.write_text(json.dumps(proof, indent=2) + "\n")
        env.update(
            MLA_FORWARD_NATIVE_DISPATCH=str(proof_path),
            ASCEND_GLOBAL_LOG_LEVEL="3",
            ASCEND_SLOG_PRINT_TO_STDOUT="0",
        )
        # The diagnostic child has populated the reference cache with all CPU
        # cores available. Pin before starting the timed child so its runtime
        # and worker threads inherit the same single CPU as the original E2E.
        os.sched_setaffinity(0, {min(affinity)})
        result = subprocess.run(
            [sys.executable, "-m", "tests.mla.support.mla_forward_profile"], env=env
        )
    finally:
        os.sched_setaffinity(0, affinity)
    assert result.returncode == 0
    report = json.loads((directory / "profile.json").read_text())
    assert report["causal"] and len(report["results"]) == count
    assert report["warmup"] == 5 and report["rounds"] == 5 and report["wall_iterations"] == 20
    assert report["metadata_lifecycle"]["complete_segments"] == [1, 4]
    # Native FD is retained and reported separately from the migrated non-FD path.
    # Exact native dispatch classification is checked from the recorded kernels.
    for item in report["results"]:
        assert item["native_call"]["sparse_mode"] == 3

    return report


def test_native_public_profile():
    reports = [native_profile(dtype) for dtype in ("fp16", "bf16")]
    result = summarize(reports, len(PERFORMANCE_CASES))
    path = Path(os.environ["MLA_VALIDATION_RESULT"]) / "public-profile" / "average.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    check_public_performance(result)


@pytest.mark.parametrize("dtype", ("fp16", "bf16"))
def test_native_preflight(dtype):
    native_profile(dtype, preflight=True)


def test_used_component_profile(causal_attention, metadata_driver, used_oracle):
    from tests.mla.support.mla_causal_bsnd_used_profile import profile

    profile(causal_attention, metadata_driver, used_oracle, causal_attention[1] / "used-profile")
