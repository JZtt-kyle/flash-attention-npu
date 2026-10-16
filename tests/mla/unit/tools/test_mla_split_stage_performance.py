"""M10 subset identity, native FD classification and complete stage evidence."""

from copy import deepcopy
import importlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tests.mla.support.mla_dense_build import sha256
from tests.mla.support.mla_split_stage_cases import (
    SPLIT_STAGE_POLICY,
    SPLIT_STAGE_SUITES,
    check_split_stage_dispatch,
    split_stage_cases,
    split_stage_selection,
)
from tests.mla.support.mla_split_stage_performance import load_evidence, recompute
from tests.mla.support.mla_validation import environment, run, SUITES, PERFORMANCE_SUITES
from tests.mla.support.mla_validation import pytest_command
from tests.mla.support.mla_validation_policy import check_public_performance
from tests.mla.unit.tools.test_mla_validation_policy import reports, varlen_reports, mixed_reports


def candidate_module(suite):
    stem = suite.removesuffix("-profile").replace("-", "_")
    if stem == "window_bsnd":
        stem = "window_public"
    return "tests.mla.support.mla_" + stem + "_cases"


def candidates(suite):
    return json.loads(
        json.dumps(importlib.import_module(candidate_module(suite)).PERFORMANCE_CASES)
    )


def test_fixed_inputs_cover_modes_masks_and_default_is_unchanged(monkeypatch, tmp_path):
    monkeypatch.delenv("MLA_SPLIT_STAGE_PERFORMANCE", raising=False)
    total = 0
    for suite, (layout, mode, indices) in SPLIT_STAGE_SUITES.items():
        specs = candidates(suite)
        assert split_stage_cases(specs, suite) == specs
        selected = split_stage_selection(specs, suite)
        assert selected["selected"] == [specs[i] for i in indices]
        assert len(selected["selected"]) + len(selected["deselected"]) == len(specs)
        if mode != "single":
            assert {tuple(s["window"]) for s in selected["selected"]} >= {(-1, -1)}
            assert any(s.get("causal") for s in selected["selected"])
            assert any(s["window"][0] >= 0 for s in selected["selected"])
            assert all((s["splits"] == 0) == (mode == "automatic") for s in selected["selected"])
        assert suite in SUITES and suite in PERFORMANCE_SUITES
        total += len(selected["selected"]) * 2
        with pytest.raises(ValueError):
            split_stage_selection(specs[:1], suite)
        with pytest.raises(ValueError):
            split_stage_selection(specs + specs, suite)
    assert total == 92 <= 100
    # Import-time selection is exercised in a fresh process, as in the actual
    # dispatch and timing children. Correctness lists must be identical.
    script = """
import importlib, json
from tests.mla.unit.tools.test_mla_split_stage_performance import candidate_module
from tests.mla.support.mla_split_stage_cases import SPLIT_STAGE_SUITES
result = {}
for suite in SPLIT_STAGE_SUITES:
    module = importlib.import_module(candidate_module(suite))
    result[suite] = {k:v for k,v in vars(module).items() if k.endswith('CASES') or k == 'REPRESENTATIVES'}
print(json.dumps(result))
"""
    baseline = json.loads(subprocess.check_output([sys.executable, "-c", script], text=True))
    monkeypatch.setenv("MLA_SPLIT_STAGE_PERFORMANCE", "1")
    monkeypatch.setenv("MLA_VALIDATION_RESULT", str(tmp_path))
    selected = json.loads(subprocess.check_output([sys.executable, "-c", script], text=True))
    for suite in SPLIT_STAGE_SUITES:
        selection = split_stage_selection(baseline[suite]["PERFORMANCE_CASES"], suite)
        assert (
            json.loads((tmp_path / f"split-stage-selection-{suite}.json").read_text()) == selection
        )
        assert selected[suite].pop("PERFORMANCE_CASES") == selection["selected"]
        baseline[suite].pop("PERFORMANCE_CASES")
        assert selected[suite] == baseline[suite]


def test_stage_environment_and_gate_are_explicit(monkeypatch, tmp_path):
    monkeypatch.setenv("MLA_SPLIT_STAGE_PERFORMANCE", "1")
    args = SimpleNamespace(device="6", golden="cache", artifacts=tmp_path)
    assert "MLA_SPLIT_STAGE_PERFORMANCE" not in environment(args, tmp_path, "split-bsnd-profile")
    args.split_stage_performance = True
    assert environment(args, tmp_path, "split-bsnd-profile")["MLA_SPLIT_STAGE_PERFORMANCE"] == "1"
    assert (
        environment(args, tmp_path, "split-bsnd-profile")["MLA_SPLIT_STAGE_SUITE"]
        == "split-bsnd-profile"
    )
    assert "MLA_SPLIT_STAGE_PERFORMANCE" not in environment(args, tmp_path, "e2e")
    args.suite = ["e2e"]
    with pytest.raises(ValueError, match="M10 performance suites"):
        run(args)
    args.suite = ["split-bsnd-profile"]
    args.causal_stage_performance = True
    with pytest.raises(ValueError, match="M10 performance suites"):
        run(args)
    command = pytest_command("window-paged-profile", tmp_path, split_stage=True)
    assert command[command.index("-k") + 1] == "native_public_profile"
    check_public_performance(dict(passed=False, mean_ratio=1.2))
    monkeypatch.delenv("MLA_SPLIT_STAGE_PERFORMANCE")
    with pytest.raises(AssertionError):
        check_public_performance(dict(passed=False, mean_ratio=1.2))


def stage_evidence(tmp_path, ratio=0.8):
    groups = {}
    for suite, (layout, mode, _) in SPLIT_STAGE_SUITES.items():
        case_ratio = ratio.get(suite, 0.8) if isinstance(ratio, dict) else ratio
        selection = split_stage_selection(candidates(suite), suite)
        specs = selection["selected"]
        if layout == "bsnd":
            values = reports((case_ratio,) * len(specs))
            for report in values:
                report.update(window=True, automatic_split=mode == "automatic")
                for row, spec in zip(report["results"], specs):
                    row.update(
                        shape=spec["shape"],
                        scale=spec.get("scale"),
                        lengths=spec.get("lengths"),
                        stride_profile=spec.get("stride", "contiguous"),
                        window=spec["window"],
                        causal=spec.get("causal", False),
                        native_fd=mode == "automatic",
                    )
                    if mode != "single":
                        row["num_splits"] = spec["splits"]
        else:
            _, values = (varlen_reports if layout == "tnd" else mixed_reports)(
                (case_ratio, case_ratio)
            )
            for report in values:
                report.update(window=True, automatic_split=mode == "automatic")
                template = report["results"][0]
                report["results"] = [
                    dict(deepcopy(template), case=i, spec=s, native_fd=mode == "automatic")
                    for i, s in enumerate(specs)
                ]
        average = recompute(suite, values, specs)
        directory = tmp_path / suite
        files = {
            "sources.json": {"shared.py": "a" * 64},
            f"split-stage-selection-{suite}.json": selection,
            "public-profile/average.json": average,
            **{f"public-profile/{r['dtype']}/profile.json": r for r in values},
        }
        for name, value in files.items():
            path = directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value))
        groups[suite] = dict(
            status="passed",
            identity=suite,
            directory=str(directory),
            evidence={str(directory / p): sha256(directory / p) for p in files},
        )
    summary = dict(
        split_stage_performance=True,
        split_stage_policy=SPLIT_STAGE_POLICY,
        runtime_identity={"extension_sha256": "c" * 64},
        groups=groups,
    )
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary))
    return path


def test_full_stage_recomputes_equal_input_weights_and_native_fd(tmp_path):
    path = stage_evidence(tmp_path)
    result = load_evidence([path])
    assert result["passed"] and result["mean_ratio"] == pytest.approx(0.8)
    assert result["measured_cases"] == result["applicable_cases"] == 92
    assert len(result["groups"]) == 15
    assert all(
        r["includes_native_fd"] == s.startswith("auto-") for s, r in result["groups"].items()
    )
    # A single bad layout group does not override the equal configuration gate.
    one_slow = stage_evidence(tmp_path / "one-slow", {"auto-split-paged-tnd-profile": 2.0})
    combined = load_evidence([one_slow])
    assert combined["passed"] and combined["mean_ratio"] == pytest.approx((86 * 0.8 + 6 * 2) / 92)
    assert combined["regressed_cases"] == 6
    changed = stage_evidence(tmp_path / "slow", ratio=1.1)
    slow = load_evidence([changed])
    assert not slow["passed"] and slow["regressed_cases"] == 92


def test_selected_groups_keep_configuration_weights_and_full_stage_boundary(tmp_path):
    suites = [
        "split-bsnd-profile",
        "auto-split-bsnd-profile",
        "auto-split-tnd-profile",
        "auto-split-mixed-profile",
    ]
    path = stage_evidence(tmp_path, {"auto-split-bsnd-profile": 1.3})
    summary = json.loads(path.read_text())
    summary["groups"] = {s: summary["groups"][s] for s in suites}
    path.write_text(json.dumps(summary))
    result = load_evidence([path], suites)
    assert result["scope"] == "selected-groups" and result["selected_suites"] == suites
    assert result["measured_cases"] == result["applicable_cases"] == 26
    assert result["passed"] and result["mean_ratio"] == pytest.approx((20 * 0.8 + 6 * 1.3) / 26)
    assert result["regressed_cases"] == 6
    assert not result["groups"]["auto-split-bsnd-profile"]["passed"]
    assert set(result["groups"]) == set(suites)
    with pytest.raises(ValueError, match="All fifteen"):
        load_evidence([path])


@pytest.mark.parametrize("suites", [[], ["unknown"], ["split-bsnd-profile"] * 2])
def test_invalid_selected_groups_rejected(suites):
    with pytest.raises(ValueError, match="distinct known"):
        load_evidence([], suites)


@pytest.mark.parametrize("mode", ["single", "explicit", "automatic"])
def test_dispatch_applicability_checked_before_timing(monkeypatch, mode):
    fd = [dict(native_template=5, native_fd=True)]
    monkeypatch.delenv("MLA_SPLIT_STAGE_PERFORMANCE", raising=False)
    check_split_stage_dispatch(fd)
    monkeypatch.setenv("MLA_SPLIT_STAGE_PERFORMANCE", "1")
    with pytest.raises(ValueError, match="Missing M10 performance suite"):
        check_split_stage_dispatch(fd)
    suite = next(s for s, (_, m, _) in SPLIT_STAGE_SUITES.items() if m == mode)
    monkeypatch.setenv("MLA_SPLIT_STAGE_SUITE", suite)
    if mode == "automatic":
        check_split_stage_dispatch(fd)
    else:
        with pytest.raises(ValueError, match="before timing"):
            check_split_stage_dispatch(fd)
    check_split_stage_dispatch(fd + [dict(native_template=5, native_fd=False)])
    for cases in ([], [dict(native_template=0, native_fd=False)]):
        with pytest.raises(ValueError, match="before timing"):
            check_split_stage_dispatch(cases)


@pytest.mark.parametrize("subset", [False, True])
@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "failed",
        "hash",
        "unbound",
        "selection",
        "input",
        "average",
        "protocol",
        "source",
        "policy",
        "duplicate",
        "runtime",
    ],
)
def test_incomplete_or_modified_stage_evidence_rejected(tmp_path, fault, subset):
    path = stage_evidence(tmp_path)
    summary = json.loads(path.read_text())
    suite = "auto-split-paged-tnd-profile"
    record = summary["groups"][suite]
    paths = [path]
    relative = {
        "selection": f"split-stage-selection-{suite}.json",
        "input": "public-profile/fp16/profile.json",
        "average": "public-profile/average.json",
        "protocol": "public-profile/fp16/profile.json",
        "source": "sources.json",
        "hash": "sources.json",
        "unbound": "sources.json",
    }.get(fault)
    if relative:
        file = Path(record["directory"]) / relative
        value = json.loads(file.read_text())
        if fault == "selection":
            value["selected"].reverse()
        elif fault == "input":
            value["results"][0]["spec"]["heads"] += 1
        elif fault == "average":
            value["mean_ratio"] = 0.1
        elif fault == "protocol":
            value["rounds"] = 1
        else:
            value["shared.py"] = "b" * 64
        file.write_text(json.dumps(value))
        if fault == "unbound":
            del record["evidence"][str(file)]
        elif fault != "hash":
            record["evidence"][str(file)] = sha256(file)
    elif fault == "missing":
        del summary["groups"][suite]
    elif fault == "failed":
        record["status"] = "failed"
    elif fault == "policy":
        summary["split_stage_policy"] += 1
    elif fault == "duplicate":
        paths.append(path)
    elif fault == "runtime":
        extra = deepcopy(summary)
        extra["runtime_identity"] = {"extension_sha256": "d" * 64}
        extra["groups"] = {}
        second = tmp_path / "different-runtime.json"
        second.write_text(json.dumps(extra))
        paths.append(second)
    path.write_text(json.dumps(summary))
    with pytest.raises((ValueError, KeyError)):
        load_evidence(paths, [suite, "split-bsnd-profile"] if subset else None)


@pytest.mark.parametrize("subset", [False, True])
@pytest.mark.parametrize("ratio,code", [(0.8, 0), (1.1, 1)])
def test_stage_cli_propagates_gate(tmp_path, ratio, code, subset):
    path = stage_evidence(tmp_path, ratio)
    output = tmp_path / "result.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.mla.support.mla_split_stage_performance",
            "--evidence",
            str(path),
            "--output",
            str(output),
            *(["--suite", "split-bsnd-profile", "auto-split-tnd-profile"] if subset else []),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == code, result.stdout + result.stderr
    assert json.loads(output.read_text())["passed"] == (code == 0)
