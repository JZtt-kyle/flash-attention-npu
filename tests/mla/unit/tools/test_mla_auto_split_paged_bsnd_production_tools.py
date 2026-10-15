"""Paged split module selection, source identity, failure propagation and split timing."""

import pytest
from tests.mla.support import mla_validation as runner
from tests.mla.unit.tools.test_mla_validation import (
    args,
    test_success_failure_interruption_and_resume as check_states,
)
from tests.mla.unit.tools.test_organization import (
    test_native_package_requires_intact_content as check_native_package,
)


@pytest.mark.parametrize(
    "suite",
    [
        "auto-split-paged-bsnd-tools",
        "auto-split-paged-bsnd-public",
        "auto-split-paged-bsnd-trap",
        "auto-split-paged-bsnd-profile",
    ],
)
def test_selection_and_failure(tmp_path, monkeypatch, suite):
    assert runner.SUITES[suite]
    assert runner.environment(args(tmp_path), tmp_path, suite)["MLA_VALIDATION_NPU"] == (
        "0" if suite.endswith("tools") else "1"
    )
    if suite.endswith("profile"):
        assert suite in runner.PERFORMANCE_SUITES
    else:
        monkeypatch.setattr(runner, "CPU_SUITES", runner.CPU_SUITES | {suite})
        monkeypatch.setattr(runner, "validate_causal_builds", lambda *a, **k: None)
        check_states(tmp_path, monkeypatch, suite)


def test_ownership():
    from tests.mla.catalog import E2E, is_e2e

    assert is_e2e(E2E["auto_split_paged_bsnd"])
    assert not is_e2e(runner.SUITES["unit-auto_split_paged_bsnd_protocol"])
    assert "auto-split-paged-bsnd-profile" in runner.PERFORMANCE_SUITES
    assert len(E2E["auto_split_paged_bsnd"]) == 4


def test_native_package_identity(tmp_path, monkeypatch):
    check_native_package(monkeypatch, tmp_path)


def test_profile_summary_json_identity(tmp_path, monkeypatch):
    import copy
    import json

    from tests.mla.e2e.auto_split_paged_bsnd import test_profile as profile
    from tests.mla.unit.tools.test_mla_validation_policy import mixed_reports

    _, reports = mixed_reports()
    for report in reports:
        template = report["results"][0]
        report.update(causal=False, window=True, automatic_split=True)
        report["results"] = [
            dict(copy.deepcopy(template), case=i, spec=s)
            for i, s in enumerate(profile.PERFORMANCE_CASES)
        ]
    captured = {r["dtype"]: r for r in json.loads(json.dumps(reports))}
    monkeypatch.setattr(profile, "native_profile", lambda dtype: captured[dtype])
    monkeypatch.setenv("MLA_VALIDATION_RESULT", str(tmp_path))
    (tmp_path / "public-profile").mkdir()
    profile.test_native_public_profile()
    result = json.loads((tmp_path / "public-profile/average.json").read_text())
    assert result["passed"] and result["measured_cases"] == 2 * len(profile.PERFORMANCE_CASES)
    captured["fp16"]["results"][0]["spec"]["window"][0] += 1
    with pytest.raises(ValueError, match="input matrix"):
        profile.test_native_public_profile()


@pytest.mark.parametrize(
    "fault", [None, "missing", "extra", "stage", "dtype", "overlap", "device", "metadata"]
)
@pytest.mark.parametrize("mangled", [False, True])
def test_profile_boundary(fault, mangled):
    from tests.mla.support.mla_split_tnd_profile_rows import production_sample

    names = (
        "ComputeMlaAutoSplitPagedBsndMetadata",
        "mla_auto_split_paged_bsnd_init_fp16",
        "mla_auto_split_paged_bsnd_partial_fp16",
        "mla_auto_split_paged_bsnd_merge_fp16",
    )
    rows = [
        {
            "Op Name": n,
            "Device_id": "6",
            "Task Start Time(us)": str(4 * i),
            "Task Duration(us)": "2",
        }
        for i, n in enumerate(names)
    ]
    if fault == "missing":
        rows.pop()
    elif fault == "extra":
        rows.append(dict(rows[-1]))
    elif fault == "stage":
        rows[1], rows[2] = rows[2], rows[1]
    elif fault == "dtype":
        rows[2]["Op Name"] = "mla_auto_split_paged_bsnd_merge_bf16"
    elif fault == "overlap":
        rows[2]["Task Start Time(us)"] = "5"
    elif fault == "device":
        rows[2]["Device_id"] = "5"
    elif fault == "metadata":
        rows[0]["Op Name"] = "ComputeMlaSplitBsndMetadata"
    if mangled:
        for row in rows[1:]:
            name = row["Op Name"]
            row["Op Name"] = f"_ZN10flash_attn3mla{len(name)}{name}Ev"
    stages = (
        "mla_auto_split_paged_bsnd_init_",
        "mla_auto_split_paged_bsnd_partial_",
        "mla_auto_split_paged_bsnd_merge_",
    )
    if fault:
        with pytest.raises(ValueError):
            production_sample(rows, names[0], stages)
    else:
        assert production_sample(rows, names[0], stages)["Task Duration(us)"] == "6"


@pytest.mark.parametrize("mode", ["success", "failure", "timeout"])
def test_isolated_process_failure_propagation(tmp_path, mode):
    import os
    import sys
    from tests.mla.support.mla_trap_process import run_isolated

    script = "import time\nprint('child probe', flush=True)\n"
    if mode == "failure":
        script += "raise RuntimeError('expected child failure')\n"
    if mode == "timeout":
        script += "time.sleep(10)\n"
    log = tmp_path / "child.log"
    result = run_isolated(
        [sys.executable, "-c", script], log, timeout=0.3 if mode == "timeout" else 5
    )
    assert result["pid"] != os.getpid()
    assert result["timed_out"] == (mode == "timeout")
    assert (result["returncode"] == 0) == (mode == "success")
    assert "child probe" in log.read_text()
    if mode == "failure":
        assert "expected child failure" in log.read_text()
    with pytest.raises(ChildProcessError):
        os.waitpid(result["pid"], os.WNOHANG)
