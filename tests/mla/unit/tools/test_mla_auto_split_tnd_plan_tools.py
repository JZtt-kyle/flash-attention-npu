"""Plan suite ownership, fixed-source identity and failure propagation."""

import pytest

from tests.mla.support import mla_validation as runner
from tests.mla.support.mla_split_kv_build import split_toolchain_identity
from tests.mla.unit.tools.test_mla_validation import (
    args,
    test_success_failure_interruption_and_resume as check_states,
    test_causal_dependencies_changing_during_execution_fail as check_dependencies,
    test_causal_driver_checks_protocol_and_final_sanitizer_exit as check_protocol,
)


@pytest.mark.parametrize(
    "suite",
    [
        "unit-auto_split_tnd_plan",
        "auto-split-tnd-plan-sanitize",
        "auto-split-tnd-plan-tools",
        "auto-split-tnd-plan-check",
    ],
)
def test_selection_environment_and_failure(tmp_path, monkeypatch, suite):
    env = runner.environment(args(tmp_path), tmp_path, suite)
    assert env["MLA_VALIDATION_NPU"] == env["TORCH_DEVICE_BACKEND_AUTOLOAD"] == "0"
    assert not runner.is_e2e(runner.SUITES[suite])
    if not suite.endswith("tools"):
        assert runner.SUITES[suite][0] == "tests/mla/unit/auto_split_tnd_plan/test_plan.py"
        assert runner.SUITES[suite][1:] == (
            ["-k", "representative or capacity_cost_boundary"] if suite.endswith("check") else []
        )
        assert env["MLA_SCHEDULE_SANITIZE"] == str(int(suite.endswith("sanitize")))
        assert env["MLA_SCHEDULE_ARTIFACT_DIR"] == str(tmp_path / "schedule")
    check_states(tmp_path, monkeypatch, suite)


@pytest.mark.parametrize("suite", ["unit-auto_split_tnd_plan", "auto-split-tnd-plan-sanitize"])
def test_pinned_identity_and_midrun_changes(tmp_path, monkeypatch, suite):
    assert runner.suite_dependencies(suite) == split_toolchain_identity(suite.endswith("sanitize"))
    check_dependencies(tmp_path, monkeypatch, suite)


@pytest.mark.parametrize("mode", ["ok", "exit", "stderr", "eof", "malformed", "extra"])
def test_cpu_protocol_and_final_sanitizer_exit(tmp_path, mode):
    check_protocol(tmp_path, mode, False)


def test_existing_commands_and_e2e_are_unchanged():
    assert (
        runner.SUITES["unit-split_kv"]
        == runner.SUITES["split-kv-sanitize"]
        == ["tests/mla/unit/split_kv/test_split_kv.py"]
    )
    assert all("auto_split_tnd_plan" not in path for path in runner.SUITES["e2e"])
    assert '"mla_auto_split_tnd_plan.cpp"' not in (runner.ROOT / "setup.py").read_text()
