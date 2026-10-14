"""TND PA window reference mapping, stage selection and evidence integrity."""

import json
from types import SimpleNamespace
from pathlib import Path
import pytest
import torch
from tests.mla.support import mla_validation as runner
from tests.mla.support.mla_validation_policy import WINDOW_STAGE_SUITES, window_public_cases
from tests.mla.support.mla_causal_stage_performance import load_evidence
from tests.mla.support.mla_public_performance import summarize, summarize_varlen, summarize_mixed
from tests.mla.support.mla_dense_build import sha256
from tests.mla.unit.tools.test_mla_validation import args
from tests.mla.unit.tools.test_mla_causal_stage_performance import stage_evidence
from tests.mla.support.mla_window_paged_tnd_cases import (
    PERFORMANCE_CASES,
    REPRESENTATIVES,
    COST_CASES,
    ACCURACY_CASES,
)


def test_new_module_selection_and_environment(tmp_path, monkeypatch):
    assert all(s in REPRESENTATIVES + COST_CASES + ACCURACY_CASES for s in PERFORMANCE_CASES)
    assert 2 * len(PERFORMANCE_CASES) <= 100
    assert len(runner.E2E["window_paged_tnd"]) == 8
    for suite in (
        "unit-window_paged_tnd",
        "window-paged-tnd-cpu",
        "window-paged-tnd-sanitize",
        "window-paged-tnd-tools",
    ):
        assert runner.environment(args(tmp_path), tmp_path, suite)["MLA_VALIDATION_NPU"] == "0"
    monkeypatch.setenv("MLA_WINDOW_STAGE_PERFORMANCE", "1")
    a = args(tmp_path)
    assert "MLA_WINDOW_STAGE_PERFORMANCE" not in runner.environment(
        a, tmp_path, "window-paged-tnd-profile"
    )
    a.window_stage_performance = True
    assert (
        runner.environment(a, tmp_path, "window-paged-tnd-profile")["MLA_WINDOW_STAGE_PERFORMANCE"]
        == "1"
    )
    a.suite = ["window-paged-tnd-public"]
    with pytest.raises(ValueError, match="performance suites"):
        runner.run(a)
    monkeypatch.setenv("MLA_VALIDATION_RESULT", str(tmp_path))
    assert window_public_cases(PERFORMANCE_CASES, "paged-tnd") == window_public_cases(
        PERFORMANCE_CASES, "paged-tnd"
    )
    selection = json.loads((tmp_path / "public-stage-selection-paged-tnd.json").read_text())
    assert {s["page"] for s in selection["selected"]} == {s["page"] for s in PERFORMANCE_CASES}
    assert len(selection["selected"]) == 10


def window_evidence(tmp_path):
    path = stage_evidence(tmp_path)
    summary = json.loads(path.read_text())
    summary.update(window_stage_performance=True, causal_stage_performance=False)
    groups = {}
    for suite, group in WINDOW_STAGE_SUITES.items():
        record = summary["groups"][suite.replace("window", "causal")]
        directory = Path(record["directory"])
        selection_path = directory / f"public-stage-selection-{group}.json"
        selection = json.loads(selection_path.read_text())
        for collection in ("candidates", "selected", "deselected"):
            for spec in selection[collection]:
                spec.update(window=[2, 1], causal=False)
        selection_path.write_text(json.dumps(selection))
        reports = []
        for dtype in ("fp16", "bf16"):
            report_path = directory / f"public-profile/{dtype}/profile.json"
            report = json.loads(report_path.read_text())
            report.update(window=True, causal=False)
            for row in report["results"]:
                (row if group == "bsnd" else row["spec"]).update(window=[2, 1], causal=False)
            report_path.write_text(json.dumps(report))
            reports.append(report)
        average = (
            summarize(reports, len(selection["selected"]))
            if group == "bsnd"
            else (summarize_varlen if group == "tnd" else summarize_mixed)(
                reports, selection["selected"]
            )
        )
        (directory / "public-profile/average.json").write_text(json.dumps(average))
        record["evidence"] = {p: sha256(Path(p)) for p in record["evidence"]}
        groups[suite] = record
    summary["groups"] = groups
    path.write_text(json.dumps(summary))
    return path


@pytest.mark.parametrize(
    "fault",
    (None, "missing", "damaged", "failed", "mode", "window", "average", "runtime", "duplicate"),
)
def test_window_stage_evidence(tmp_path, fault):
    path = window_evidence(tmp_path)
    summary = json.loads(path.read_text())
    record = summary["groups"]["window-paged-tnd-profile"]
    paths = [path]
    if fault == "missing":
        del summary["groups"]["window-paged-tnd-profile"]
    elif fault == "failed":
        record["status"] = "failed"
    elif fault == "mode":
        summary["window_stage_performance"] = False
    elif fault == "runtime":
        extra = dict(summary, runtime_identity={"extension_sha256": "changed"})
        other = tmp_path / "other.json"
        other.write_text(json.dumps(extra))
        paths.append(other)
    elif fault == "duplicate":
        paths.append(path)
    elif fault in ("damaged", "window", "average"):
        target = Path(record["directory"]) / (
            "public-profile/average.json"
            if fault == "average"
            else "public-profile/fp16/profile.json"
        )
        value = json.loads(target.read_text())
        if fault == "average":
            value["mean_ratio"] = 999
        else:
            value["results"][0]["spec"]["window"][0] += 1
        target.write_text(json.dumps(value))
        if fault != "damaged":
            record["evidence"][str(target)] = sha256(target)
    path.write_text(json.dumps(summary))
    if fault:
        with pytest.raises(ValueError):
            load_evidence(paths, mask="window")
    else:
        result = load_evidence(paths, mask="window")
        assert result["passed"] and result["measured_cases"] == 20


def test_tnd_reference_uses_allocation_gaps_and_cached_inputs(tmp_path, monkeypatch):
    from tests.mla.support import mla_window_paged_reference as reference

    monkeypatch.setenv("GOLDEN_CACHE_DIR", str(tmp_path / "golden"))
    monkeypatch.setenv("GOLDEN_CACHE_MODE", "cache")
    monkeypatch.setenv("GOLDEN_CACHE_REFRESH", "0")
    host = {
        n: torch.zeros((5 if n in ("q", "qv") else 16, 1, d), dtype=torch.float16)
        for n, d in (("q", 64), ("qv", 512), ("k", 64), ("v", 512))
    }
    for n in host:
        host[n] = host[n].unsqueeze(0)
    host["v"].fill_(3)
    host["q"][:, 1:3].fill_(float("nan"))
    host["qv"][:, 1:3].fill_(float("nan"))
    case = dict(
        config=dict(
            layout=2, q=[3, 0, 2], lengths=[2, 1, 2], capacity=32, page=16, heads=1, window=(0, 0)
        ),
        item=dict(q=[3, 0, 2], uq=[1, 0, 2]),
        host=host,
        table_host=torch.tensor([[0, -1]] * 3),
        scale=1 / 24,
        context=SimpleNamespace(heads=1),
    )
    result = reference.references(case)
    assert result[0].shape == (3, 1, 512) and result[0].eq(3).all()
    assert result[1].eq(0).all()
    monkeypatch.setattr(
        reference, "uncached_references", lambda c: pytest.fail("recomputed identical reference")
    )
    for a, b in zip(result, reference.references(case)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
