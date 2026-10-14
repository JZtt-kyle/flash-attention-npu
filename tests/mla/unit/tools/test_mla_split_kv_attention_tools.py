"""Partial runner identity, source boundaries and independent reference checks."""

import json

import pytest
import torch

from tests.mla.support import mla_validation as runner
from tests.mla.support.mla_split_kv_attention_cases import (
    CHECK_CASES,
    COST_CASES,
    BOUNDARY_CASES,
    EMPTY_CASES,
    PERFORMANCE_CASES,
)
from tests.mla.support.mla_split_kv_attention_reference import fragments, partial_math
from tests.mla.unit.tools.test_mla_validation import (
    args,
    test_success_failure_interruption_and_resume as check_states,
    test_performance_always_executes_on_resume as check_performance,
)


@pytest.mark.parametrize("suite", sorted(runner.SPLIT_KV_DEVICE_SUITES))
def test_selection_and_actual_device_identity(tmp_path, suite):
    env = runner.environment(args(tmp_path), tmp_path, suite)
    assert env["MLA_VALIDATION_NPU"] == "1"
    assert env["MLA_SPLIT_KV_ATTENTION_DIR"] == str(tmp_path / "probes")
    assert not runner.is_e2e(runner.SUITES[suite])
    assert all("/split_kv_attention/" in p for p in runner.SUITES[suite] if p.endswith(".py"))


def test_tools_failures_and_unit_no_inheritance(tmp_path, monkeypatch):
    check_states(tmp_path, monkeypatch, "split-kv-attention-tools")


def test_performance_is_executed_on_resume(tmp_path, monkeypatch):
    check_performance(tmp_path, monkeypatch, "split-kv-attention-profile")


def test_input_purposes_and_check_partition():
    all_cases = CHECK_CASES + COST_CASES + BOUNDARY_CASES + EMPTY_CASES
    assert len({s["id"] for s in all_cases}) == len(all_cases)
    assert all(s in all_cases for s in PERFORMANCE_CASES)
    assert {s["splits"] for s in all_cases} == {1, 2, 8, 256}
    assert runner.SUITES["split-kv-attention-check"][-1] == "representative or cost_boundary"
    assert runner.SUITES["split-kv-attention"][-1] == "not representative and not cost_boundary"


@pytest.mark.parametrize("mask", [{}, {"causal": True}, {"window": (1, 0)}])
def test_independent_partial_reference_matches_scalar(mask):
    spec = dict(shape=(1, 3, 2, 5), splits=2, **mask)
    # Scale zero makes the expected partial output an exact rational mean.
    inputs = dict(
        q=torch.zeros(1, 3, 2, 64, dtype=torch.float16),
        qv=torch.zeros(1, 3, 2, 512, dtype=torch.float16),
        k=torch.zeros(1, 5, 1, 64, dtype=torch.float16),
        v=torch.arange(5, dtype=torch.float16).reshape(1, 5, 1, 1).expand(1, 5, 1, 512),
    )
    for high in (True, False):
        out, lse, maximum, total = partial_math(inputs, spec, 4, 2, high)
        for i, (_, _, visible) in enumerate(fragments(spec, 4, 2)):
            for row, keys in enumerate(visible):
                selected = keys.nonzero().flatten()
                if len(selected):
                    assert abs(float(out[i, row, 0]) - float(selected.float().mean())) < 0.002
                    torch.testing.assert_close(
                        lse[i, row], torch.tensor(float(len(selected))).log()
                    )
                    assert maximum[i, row] == 0 and total[i, row] == len(selected)
                else:
                    assert (out[i, row] == 0).all() and torch.isneginf(lse[i, row])


def test_partial_build_evidence_rejects_changed_dependencies(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    source = tmp_path / "native.hpp"
    source.write_text("source")
    for dtype in ("fp16", "bf16"):
        base = tmp_path / "probes" / dtype
        base.mkdir(parents=True)
        library = base / "probe.so"
        library.write_text("compiled")
        libs = {"migrated": dict(path=str(library), sha256=runner.sha256(library))}
        manifest = dict(
            split_kv=True,
            dtype=dtype,
            sources={"native.hpp": runner.sha256(source)},
            dependencies={str(source): runner.sha256(source)},
            libraries=libs,
        )
        (base / "build-manifest.json").write_text(json.dumps(manifest))
        (base / "loaded.json").write_text(json.dumps(libs))
    runner.validate_causal_builds(tmp_path, "split_kv")
    source.write_text("changed")
    with pytest.raises(RuntimeError, match="changed"):
        runner.validate_causal_builds(tmp_path, "split_kv")
