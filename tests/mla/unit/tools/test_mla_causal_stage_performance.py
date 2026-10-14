"""Phase-wide budgeting, evidence binding and the global average gate."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tests.mla.support.mla_causal_stage_performance import load_evidence, summarize_stage
from tests.mla.support.mla_dense_build import sha256
from tests.mla.support.mla_public_performance import summarize, summarize_mixed, summarize_varlen
from tests.mla.support.mla_validation import environment, run
from tests.mla.support.mla_validation_policy import (
    CAUSAL_STAGE_GROUPS,
    CAUSAL_STAGE_POLICY,
    CAUSAL_STAGE_SUITES,
    check_public_performance,
    public_cases,
    stage_selection,
)
from tests.mla.unit.tools.test_mla_validation_policy import reports, varlen_reports, mixed_reports


def test_stage_cap_stratification_and_order_independence(monkeypatch, tmp_path):
    monkeypatch.setenv("MLA_CAUSAL_STAGE_PERFORMANCE", "1")
    monkeypatch.setenv("MLA_VALIDATION_RESULT", str(tmp_path))
    total = 0
    for group in CAUSAL_STAGE_GROUPS:
        specs = [
            dict(page=p, stride=s, lengths=[i], shape=[1, i, 8, 257])
            for p in (16, 48, 112, 128, 528, 1024)
            for s in ("contiguous", "independent")
            for i in range(4)
        ]
        chosen = public_cases(specs, group=group)
        assert len(chosen) == 10
        assert sorted(chosen, key=str) == sorted(
            stage_selection(specs[::-1], group)["selected"], key=str
        )
        if group.startswith("paged-"):
            assert {s["page"] for s in chosen} == {s["page"] for s in specs}
        else:
            assert {s["stride"] for s in chosen} == {"contiguous", "independent"}
        saved = json.loads((tmp_path / f"public-stage-selection-{group}.json").read_text())
        assert saved == stage_selection(specs, group)
        assert len(saved["selected"]) + len(saved["deselected"]) == len(specs)
        total += 2 * len(chosen)
    assert total == 100


def test_default_inputs_and_performance_gate_unchanged(monkeypatch, tmp_path):
    monkeypatch.delenv("MLA_CAUSAL_STAGE_PERFORMANCE", raising=False)
    specs = [dict(shape=[1, i, 8, 257]) for i in range(120)]
    baseline = public_cases(specs)
    assert len(baseline) == 50
    assert all(public_cases(specs, group=g) == baseline for g in CAUSAL_STAGE_GROUPS)
    with pytest.raises(AssertionError):
        check_public_performance(dict(passed=False, mean_ratio=1.1))
    monkeypatch.setenv("MLA_CAUSAL_STAGE_PERFORMANCE", "1")
    monkeypatch.setenv("MLA_VALIDATION_RESULT", str(tmp_path))
    assert public_cases(specs) == baseline
    check_public_performance(dict(passed=False, mean_ratio=1.1))
    with pytest.raises(ValueError):
        stage_selection([specs[0], specs[0]], "bsnd")


def test_stage_environment_is_explicit_and_only_for_performance(monkeypatch, tmp_path):
    monkeypatch.setenv("MLA_CAUSAL_STAGE_PERFORMANCE", "1")
    args = SimpleNamespace(
        device="6", golden="cache", artifacts=tmp_path, causal_stage_performance=False
    )
    assert "MLA_CAUSAL_STAGE_PERFORMANCE" not in environment(args, tmp_path, "causal-bsnd-profile")
    args.causal_stage_performance = True
    assert environment(args, tmp_path, "causal-bsnd-profile")["MLA_CAUSAL_STAGE_PERFORMANCE"] == "1"
    assert "MLA_CAUSAL_STAGE_PERFORMANCE" not in environment(args, tmp_path, "causal-bsnd-public")
    args.suite = ["causal-bsnd-public"]
    with pytest.raises(ValueError, match="performance suites"):
        run(args)


def stage_evidence(tmp_path, ratio=0.8):
    groups = {}
    for suite, group in CAUSAL_STAGE_SUITES.items():
        directory = tmp_path / suite
        directory.mkdir()
        if group == "bsnd":
            values = reports((ratio, ratio))
            specs = [
                dict(shape=r["shape"], scale=r["scale"], lengths=r["lengths"])
                for r in values[0]["results"]
            ]
            average = summarize(values, len(specs))
        else:
            specs, values = (varlen_reports if group == "tnd" else mixed_reports)((ratio, ratio))
            if group.startswith("paged-"):
                for spec in specs:
                    spec["page"] = 16
            average = (summarize_varlen if group == "tnd" else summarize_mixed)(values, specs)
        files = {
            "sources.json": {"shared.py": "a" * 64, f"{group}.py": "b" * 64},
            f"public-stage-selection-{group}.json": stage_selection(specs, group),
            "public-profile/average.json": average,
            **{f"public-profile/{r['dtype']}/profile.json": r for r in values},
        }
        for relative, value in files.items():
            p = directory / relative
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(value))
        groups[suite] = dict(
            status="passed",
            identity=suite,
            directory=str(directory),
            evidence={str(directory / p): sha256(directory / p) for p in files},
        )
    summary = dict(
        causal_stage_performance=True,
        causal_stage_policy=CAUSAL_STAGE_POLICY,
        runtime_identity={"extension_sha256": "test"},
        groups=groups,
    )
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary))
    return path


def test_evidence_global_weighting_and_layout_regression(tmp_path):
    path = stage_evidence(tmp_path)
    result = load_evidence([path])
    assert result["passed"] and result["measured_cases"] == 20
    assert result["source_identity"]["files"] == 6
    averages = {g: summarize(reports((0.5, 0.5)), 2) for g in CAUSAL_STAGE_GROUPS}
    averages["bsnd"] = summarize(reports((2.0,)), 1)
    result = summarize_stage(averages)
    assert result["passed"] and result["mean_ratio"] == pytest.approx(2 / 3)
    assert result["regressed_cases"] == 2
    assert not result["layouts"]["bsnd"]["passed"]
    assert not summarize_stage({g: summarize(reports((1.1, 1.1)), 2) for g in CAUSAL_STAGE_GROUPS})[
        "passed"
    ]


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "duplicate",
        "failed",
        "damaged",
        "unbound",
        "selection",
        "average",
        "runtime",
        "mode",
        "sources",
        "source_conflict",
    ],
)
def test_incomplete_or_invalid_stage_evidence_rejected(tmp_path, fault):
    path = stage_evidence(tmp_path)
    summary = json.loads(path.read_text())
    record = summary["groups"]["causal-bsnd-profile"]
    paths = [path]
    if fault == "missing":
        del summary["groups"]["causal-bsnd-profile"]
    elif fault == "duplicate":
        paths.append(path)
    elif fault == "failed":
        record["status"] = "failed"
    elif fault == "mode":
        summary["causal_stage_performance"] = False
    elif fault == "runtime":
        other = tmp_path / "other.json"
        changed = deepcopy(summary)
        changed["runtime_identity"] = {"extension_sha256": "changed"}
        other.write_text(json.dumps(changed))
        paths.append(other)
    elif fault in {"sources", "source_conflict"}:
        file = Path(record["directory"]) / "sources.json"
        value = {} if fault == "sources" else {"shared.py": "c" * 64}
        file.write_text(json.dumps(value))
        record["evidence"][str(file)] = sha256(file)
    else:
        name = (
            "public-profile/average.json"
            if fault == "average"
            else "public-stage-selection-bsnd.json"
        )
        file = Path(record["directory"]) / name
        if fault == "unbound":
            del record["evidence"][str(file)]
        else:
            value = json.loads(file.read_text())
            value["mean_ratio" if fault == "average" else "policy"] = 999
            file.write_text(json.dumps(value))
            if fault != "damaged":
                record["evidence"][str(file)] = sha256(file)
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        load_evidence(paths)


@pytest.mark.parametrize("fault", ["cap", "nan", "zero", "count", "mean"])
def test_invalid_aggregate_rejected(fault):
    averages = {g: summarize(reports((0.8, 0.8)), 2) for g in CAUSAL_STAGE_GROUPS}
    result = averages["bsnd"]
    if fault == "cap":
        result["cases"] *= 6
        result["measured_cases"] = len(result["cases"])
    elif fault in {"nan", "zero"}:
        result["cases"][0]["mean_ratio"] = float("nan") if fault == "nan" else 0
    else:
        result["measured_cases" if fault == "count" else "mean_ratio"] = 999
    with pytest.raises(ValueError):
        summarize_stage(averages)


@pytest.mark.parametrize("ratio, code", [(0.8, 0), (1.1, 1)])
def test_stage_cli_propagates_acceptance(tmp_path, ratio, code):
    path = stage_evidence(tmp_path, ratio)
    output = tmp_path / "stage.json"
    command = [
        sys.executable,
        "-m",
        "tests.mla.support.mla_causal_stage_performance",
        "--evidence",
        str(path),
        "--output",
        str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == code, result.stdout + result.stderr
    assert json.loads(output.read_text())["passed"] == (code == 0)
    path.unlink()
    assert subprocess.run(command, capture_output=True).returncode == 1
    assert "error" in json.loads(output.read_text())
