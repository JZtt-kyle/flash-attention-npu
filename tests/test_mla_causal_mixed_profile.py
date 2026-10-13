"""Causal mixed public average E2E and representative source component performance."""

import json
import os
from pathlib import Path
import subprocess
import sys

from tests.mla_dense_build import ROOT
from tests.mla_native_dispatch import parse_dispatch, sha256
from tests.mla_validation_policy import check_public_performance
from tests.mla_public_performance import summarize_mixed
from tests.test_flash_attn_npu_v3_mla_causal_mixed_public import PERFORMANCE_CASES
from tests.test_flash_attn_npu_v3_mla_causal_attention import causal_attention as causal_attention
from tests.test_flash_attn_npu_v3_mla_causal_mixed_metadata import (
    mixed_metadata_driver as mixed_metadata_driver,
)
from tests.test_flash_attn_npu_v3_mla_causal_schedule import causal_artifacts as causal_artifacts
from tests.test_flash_attn_npu_v3_mla_used_schedule import (
    used_artifacts as used_artifacts,
    used_oracle as used_oracle,
)


def native_profile(dtype):
    directory = Path(os.environ["MLA_VALIDATION_RESULT"]) / "public-profile" / dtype
    directory.parent.mkdir(parents=True, exist_ok=True)
    vendor = ROOT / "build/mla-m06c3a-varlen/vendor/mla_ca18aee_transformer"
    env = dict(os.environ, ASCEND_CUSTOM_OPP_PATH=str(vendor), MLA_CANN_VENDOR=str(vendor))
    env["LD_LIBRARY_PATH"] = str(vendor / "op_api/lib") + ":" + env.get("LD_LIBRARY_PATH", "")
    diagnostic = directory.with_name(dtype + "-dispatch")
    log_path = directory.with_suffix(".dispatch.log")
    command = [
        sys.executable,
        "-m",
        "tests.mla_causal_mixed_profile",
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
        "tests.mla_causal_mixed_profile",
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
    return json.loads((directory / "profile.json").read_text())


def test_native_public_profile():
    reports = [native_profile(dtype) for dtype in ("fp16", "bf16")]
    result = summarize_mixed(reports, PERFORMANCE_CASES)
    (Path(os.environ["MLA_VALIDATION_RESULT"]) / "public-profile/average.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    check_public_performance(result)


def test_component_profile(causal_attention, mixed_metadata_driver, used_oracle):
    from tests.mla_causal_mixed_component_profile import profile

    profile(
        causal_attention,
        mixed_metadata_driver,
        used_oracle,
        causal_attention[1] / "mixed-component-profile",
    )
