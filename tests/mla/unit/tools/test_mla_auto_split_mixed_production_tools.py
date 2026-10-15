"""Automatic mixed selection, failure evidence, stage pairing and native FD average."""

import copy
import pytest
from tests.mla.support import mla_validation as runner
from tests.mla.support.mla_public_performance import summarize_mixed
from tests.mla.support.mla_split_tnd_profile_rows import production_sample
from tests.mla.unit.tools.test_mla_validation import (
    args,
    test_success_failure_interruption_and_resume as check_states,
    test_evidence_missing_changed_failed_or_different_identity_cannot_resume as check_evidence,
)
from tests.mla.unit.tools.test_mla_validation_policy import mixed_reports
from tests.mla.unit.tools.test_mla_split_tnd_tools import test_profile_boundary as check_explicit


@pytest.mark.parametrize(
    "suite",
    [
        "auto-split-mixed-check",
        "unit-auto_split_mixed",
        "auto-split-mixed-public",
        "auto-split-mixed-trap",
        "auto-split-mixed-profile",
        "auto-split-mixed-tools",
    ],
)
def test_selection_and_failure(tmp_path, monkeypatch, suite):
    env = runner.environment(args(tmp_path), tmp_path, suite)
    assert env["MLA_VALIDATION_NPU"] == ("0" if suite.endswith("tools") else "1")
    if suite == "unit-auto_split_mixed":
        assert env["MLA_SPLIT_KV_MERGE_DIR"] == str(tmp_path / "probes")
    if suite.endswith("profile"):
        assert suite in runner.PERFORMANCE_SUITES
    else:
        monkeypatch.setattr(runner, "CPU_SUITES", runner.CPU_SUITES | {suite})
        monkeypatch.setattr(runner, "validate_causal_builds", lambda *a, **k: None)
        check_states(tmp_path, monkeypatch, suite)


def test_ownership_and_evidence(tmp_path):
    from tests.mla.catalog import E2E, is_e2e

    assert is_e2e(E2E["auto_split_mixed"])
    assert not is_e2e(runner.SUITES["unit-auto_split_mixed"])
    assert all("auto_split_mixed/" in path for path in E2E["auto_split_mixed"])
    check_evidence(tmp_path)


@pytest.mark.parametrize(
    "fault", [None, "missing", "extra", "stage", "dtype", "overlap", "device", "metadata"]
)
@pytest.mark.parametrize("mangled", [False, True])
def test_three_phases_and_explicit_compatibility(fault, mangled):
    check_explicit(fault, mangled)
    stages = (
        "mla_auto_split_mixed_init_",
        "mla_auto_split_mixed_partial_",
        "mla_auto_split_mixed_merge_",
    )
    names = ("ComputeMlaAutoSplitMixedMetadata", *(s + "fp16" for s in stages))
    rows = [
        dict(
            **{
                "Op Name": name,
                "Device_id": "6",
                "Task Start Time(us)": str(4 * i),
                "Task Duration(us)": "2",
            }
        )
        for i, name in enumerate(names)
    ]
    if fault == "missing":
        rows.pop(1)
    elif fault == "extra":
        rows.append(dict(rows[-1]))
    elif fault == "stage":
        rows[1], rows[2] = rows[2], rows[1]
    elif fault == "dtype":
        rows[3]["Op Name"] = stages[2] + "bf16"
    elif fault == "overlap":
        rows[3]["Task Start Time(us)"] = "9"
    elif fault == "device":
        rows[3]["Device_id"] = "5"
    elif fault == "metadata":
        rows[0]["Op Name"] = "ComputeMlaSplitTndMetadata"
    if mangled:
        for row in rows[1:]:
            name = row["Op Name"]
            row["Op Name"] = f"_ZN10flash_attn3mla{len(name)}{name}Ev"
    if fault:
        with pytest.raises(ValueError):
            production_sample(rows, names[0], stages)
    else:
        assert production_sample(rows, names[0], stages)["Task Duration(us)"] == "6"


def test_native_fd_average():
    specs, reports = mixed_reports()
    original = summarize_mixed(reports, specs)
    assert not original["includes_native_fd"]
    for s in specs:
        s["splits"] = 0
    for report in reports:
        report["automatic_split"] = True
        for i, row in enumerate(report["results"]):
            row["spec"] = copy.deepcopy(specs[i])
            row["native_fd"] = bool(i)
    result = summarize_mixed(reports, specs, include_native_fd=True)
    assert result["includes_native_fd"] and result["applicable_cases"] == 4
    assert result["mean_ratio"] == pytest.approx(original["mean_ratio"])
    with pytest.raises(ValueError, match="classification"):
        summarize_mixed(reports, specs)
    reports[0]["automatic_split"] = False
    with pytest.raises(ValueError, match="automatic split"):
        summarize_mixed(reports, specs, include_native_fd=True)
    reports[0]["automatic_split"] = True
    specs[0]["splits"] = 2
    with pytest.raises(ValueError, match="automatic split"):
        summarize_mixed(reports, specs, include_native_fd=True)


@pytest.mark.parametrize("fault", ["input", "dispatch", "rounds", "ratio", "incomplete"])
def test_invalid_performance_evidence(fault):
    specs, reports = mixed_reports()
    for s in specs:
        s["splits"] = 0
    for report in reports:
        report["automatic_split"] = True
        for i, row in enumerate(report["results"]):
            row["spec"] = copy.deepcopy(specs[i])
            row["native_fd"] = True
    row = reports[0]["results"][0]
    if fault == "input":
        row["spec"]["splits"] = 2
    elif fault == "dispatch":
        row["e2e_applicable"] = False
    elif fault == "rounds":
        reports[0]["rounds"] = 4
    elif fault == "ratio":
        row["entries"]["reuse"]["r4"]["ratio"] = float("nan")
    else:
        reports[0]["results"].pop()
    with pytest.raises(ValueError):
        summarize_mixed(reports, specs, include_native_fd=True)
