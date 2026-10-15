"""TND PA native auto protocol ownership, fixed source identity and runner failure propagation."""

import pytest

from tests.mla.support import mla_validation as runner
from tests.mla.support.mla_auto_split_paged_tnd_build import paged_split_toolchain_identity
from tests.mla.unit.tools.test_mla_validation import (
    args,
    test_success_failure_interruption_and_resume as check_states,
    test_causal_dependencies_changing_during_execution_fail as check_dependencies,
    test_causal_driver_checks_protocol_and_final_sanitizer_exit as check_protocol,
)


@pytest.mark.parametrize(
    "suite",
    [
        "unit-auto_split_paged_tnd_protocol",
        "auto-split-paged-tnd-protocol-sanitize",
        "auto-split-paged-tnd-protocol-check",
        "auto-split-paged-tnd-protocol-tools",
    ],
)
def test_selection_environment_and_failure(tmp_path, monkeypatch, suite):
    env = runner.environment(args(tmp_path), tmp_path, suite)
    assert env["MLA_VALIDATION_NPU"] == env["TORCH_DEVICE_BACKEND_AUTOLOAD"] == "0"
    assert not runner.is_e2e(runner.SUITES[suite])
    if not suite.endswith("tools"):
        assert set(p for p in runner.SUITES[suite] if p.endswith(".py")) == {
            "tests/mla/unit/auto_split_paged_tnd_protocol/test_plan.py",
            "tests/mla/unit/auto_split_paged_tnd_protocol/test_metadata.py",
        }
        assert env["MLA_SCHEDULE_SANITIZE"] == str(int(suite.endswith("sanitize")))
        assert env["MLA_SCHEDULE_ARTIFACT_DIR"] == str(tmp_path / "schedule")
    check_states(tmp_path, monkeypatch, suite)


@pytest.mark.parametrize(
    "suite",
    [
        "unit-auto_split_paged_tnd_protocol",
        "auto-split-paged-tnd-protocol-sanitize",
        "auto-split-paged-tnd-protocol-check",
    ],
)
def test_pinned_identity_and_midrun_changes(tmp_path, monkeypatch, suite):
    assert runner.suite_dependencies(suite) == paged_split_toolchain_identity(
        suite.endswith("sanitize")
    )
    check_dependencies(tmp_path, monkeypatch, suite)


@pytest.mark.parametrize("mode", ["ok", "exit", "stderr", "eof", "malformed", "extra"])
def test_cpu_protocol_and_final_sanitizer_exit(tmp_path, mode):
    check_protocol(tmp_path, mode, False)


def test_existing_commands_and_production_registration():
    assert runner.SUITES["unit-split_tnd_plan"] == runner.SUITES["split-tnd-plan-sanitize"]
    assert runner.SUITES["split-tnd-cpu"] == runner.SUITES["split-tnd-sanitize"]
    assert all("auto_split_paged_tnd_protocol" not in path for path in runner.SUITES["e2e"])
    setup = (runner.ROOT / "setup.py").read_text()
    assert '"mla_auto_split_paged_tnd_plan.cpp"' in setup
    assert '"mla_auto_split_paged_tnd_metadata.aicpu"' in setup
    assert '"mla_auto_split_paged_tnd_metadata.cpp"' not in setup
