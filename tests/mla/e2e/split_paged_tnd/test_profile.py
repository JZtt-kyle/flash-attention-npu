"""Explicit split paged native dispatch and fixed public R=1/4 average E2E."""

from tests.mla.support.native_package import native_vendor


from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from tests.mla.support.mla_native_dispatch import parse_dispatch, sha256
from tests.mla.support.mla_validation_policy import check_public_performance
from tests.mla.support.mla_public_performance import summarize_mixed
from tests.mla.support.mla_split_paged_tnd_cases import PERFORMANCE_CASES


@contextmanager
def timing_output(directory):
    """Archive a single dtype's capture before releasing temporary storage."""
    scratch = Path(tempfile.mkdtemp(prefix="mla-profile-", dir=os.environ.get("TMPDIR")))
    try:
        yield scratch
    finally:
        files = {str(p.relative_to(scratch)): sha256(p) for p in scratch.rglob("*") if p.is_file()}
        # Keep the temporary capture if copying or verification fails.
        shutil.copytree(scratch, directory)
        assert all(sha256(directory / name) == digest for name, digest in files.items())
        profile = directory / "profile.json"
        if profile.exists():
            report = json.loads(profile.read_text())
            profile.rename(directory / "profile-capture.json")
            for name in ("kernel_csv", "raw_csv"):
                relative = Path(report[name]).relative_to(scratch)
                assert str(relative) in files
                report[name] = str(directory / relative)
            profile.write_text(json.dumps(report, indent=2) + "\n")
        (directory / "timing-storage.json").write_text(
            json.dumps(dict(scratch=str(scratch), files=files), indent=2) + "\n"
        )
        shutil.rmtree(scratch)


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
        "tests.mla.support.mla_split_paged_tnd_profile",
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
        "tests.mla.support.mla_split_paged_tnd_profile",
        dtype,
        str(directory),
        str(vendor),
        str(proof_path),
    ]
    with timing_output(directory) as scratch:
        command[4] = str(scratch)
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
    assert report["window"] and not report["causal"]
    assert any(row["native_call"]["sparse_mode"] == 4 for row in report["results"])
    return report


def test_native_public_profile():
    reports = [native_profile(dtype) for dtype in ("fp16", "bf16")]
    result = summarize_mixed(reports, json.loads(json.dumps(PERFORMANCE_CASES)))
    (Path(os.environ["MLA_VALIDATION_RESULT"]) / "public-profile/average.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    check_public_performance(result)
