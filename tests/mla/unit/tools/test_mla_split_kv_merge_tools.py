"""Merge selection, reference and source/build evidence cannot silently drift."""

import json
import pytest
import torch

from tests.mla.support import mla_validation as runner
from tests.mla.support.mla_split_kv_merge_reference import merge_math
from tests.mla.support.mla_split_kv_merge_cases import (
    CHECK_CASES,
    COST_CASES,
    BOUNDARY_CASES,
    SYNTHETIC_CASES,
)
from tests.mla.unit.tools.test_mla_validation import (
    args,
    test_success_failure_interruption_and_resume as check_states,
    test_performance_always_executes_on_resume as check_performance,
)


@pytest.mark.parametrize("suite", sorted(runner.SPLIT_KV_MERGE_SUITES))
def test_selection_identity(tmp_path, suite):
    env = runner.environment(args(tmp_path), tmp_path, suite)
    assert env["MLA_VALIDATION_NPU"] == "1"
    assert env["MLA_SPLIT_KV_MERGE_DIR"] == str(tmp_path / "probes")
    assert not runner.is_e2e(runner.SUITES[suite])
    assert all("/split_kv_merge/" in p for p in runner.SUITES[suite] if p.endswith(".py"))


def test_no_inheritance_and_failures(tmp_path, monkeypatch):
    check_states(tmp_path, monkeypatch, "split-kv-merge-tools")


def test_profile_executes_on_resume(tmp_path, monkeypatch):
    check_performance(tmp_path, monkeypatch, "split-kv-merge-profile")


def test_purposes_and_check_partition():
    cases = CHECK_CASES + COST_CASES + BOUNDARY_CASES + SYNTHETIC_CASES
    assert len({s["id"] for s in cases}) == len(cases)
    assert {1, 2, 8, 256} <= {s["splits"] for s in cases}
    assert runner.SUITES["split-kv-merge-check"][-1] == "representative or cost_boundary"
    assert runner.SUITES["split-kv-merge"][-1] == "not representative and not cost_boundary"


def test_math_uses_normalized_o_and_ignores_empty_poison():
    o = torch.tensor([2.0, 6.0, torch.nan]).reshape(3, 1, 1).expand(3, 1, 512)
    sums = torch.tensor([2.0, 3.0, 7.0]).reshape(3, 1, 1).expand(3, 1, 8)
    maxima = torch.tensor([0.0, 0.0, 1e8]).reshape(3, 1, 1).expand(3, 1, 8)
    valid = torch.tensor([[True], [True], [False]])
    for high in (False, True):
        out, lse = merge_math(o, sums, maxima, valid, torch.float32, high)
        torch.testing.assert_close(out, torch.full_like(out, 4.4))
        torch.testing.assert_close(lse, torch.tensor([5.0]).log())
        out, lse = merge_math(o, sums, maxima, torch.zeros_like(valid), torch.float32, high)
        assert (out == 0).all() and torch.isneginf(lse).all()


def test_evidence_checks_actual_load_and_dependencies(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    source = tmp_path / "native.hpp"
    source.write_text("source")
    for dtype in ("fp16", "bf16"):
        base = tmp_path / "probes" / dtype
        base.mkdir(parents=True)
        lib = base / "merge.so"
        lib.write_text("fresh build")
        libs = {"migrated": dict(path=str(lib), sha256=runner.sha256(lib))}
        manifest = dict(
            split_kv_merge=True,
            dtype=dtype,
            sources={"native.hpp": runner.sha256(source)},
            dependencies={str(source): runner.sha256(source)},
            libraries=libs,
        )
        (base / "build-manifest.json").write_text(json.dumps(manifest))
        (base / "loaded.json").write_text(json.dumps(libs))
    runner.validate_causal_builds(tmp_path, "split_kv_merge")
    (tmp_path / "probes/bf16/loaded.json").write_text("{}")
    with pytest.raises(RuntimeError, match="changed"):
        runner.validate_causal_builds(tmp_path, "split_kv_merge")
