"""Split CPU suite ownership, immutable sources, failure and evidence propagation."""

from pathlib import Path
import pytest

from tests.mla.support import mla_validation as runner
from tests.mla.support import mla_split_kv_build as builder
from tests.mla.unit.tools.test_mla_validation import (
    args,
    test_success_failure_interruption_and_resume as check_runner_states,
    test_causal_dependencies_changing_during_execution_fail as check_dependency_failure,
    test_causal_driver_checks_protocol_and_final_sanitizer_exit as check_cpu_protocol,
)


@pytest.mark.parametrize("suite", ["unit-split_kv", "split-kv-sanitize", "split-kv-tools"])
def test_selection_cpu_environment_and_no_inheritance(tmp_path, monkeypatch, suite):
    config = args(tmp_path)
    env = runner.environment(config, tmp_path, suite)
    assert env["MLA_VALIDATION_NPU"] == env["TORCH_DEVICE_BACKEND_AUTOLOAD"] == "0"
    assert not runner.is_e2e(runner.SUITES[suite])
    if suite != "split-kv-tools":
        assert runner.SUITES[suite] == ["tests/mla/unit/split_kv/test_split_kv.py"]
        assert env["MLA_SCHEDULE_SANITIZE"] == str(int(suite.endswith("sanitize")))
        assert env["MLA_SCHEDULE_ARTIFACT_DIR"] == str(tmp_path / "schedule")
    check_runner_states(tmp_path, monkeypatch, suite)


@pytest.mark.parametrize("suite", ["unit-split_kv", "split-kv-sanitize"])
def test_dependency_identity_and_execution_failure(tmp_path, monkeypatch, suite):
    check_dependency_failure(tmp_path, monkeypatch, suite)


@pytest.mark.parametrize("mode", ["ok", "exit", "stderr", "eof", "malformed", "extra"])
def test_cpu_protocol_exit_is_checked(tmp_path, mode):
    check_cpu_protocol(tmp_path, mode, False)


def test_fixed_source_and_snippet_identity(tmp_path, monkeypatch):
    original = builder.blobs()
    actual = builder.split_toolchain_identity(False)
    assert actual["fa3_commit"] == builder.FA3_COMMIT
    assert actual["source_commit"] == builder.CANN_COMMIT
    assert set(actual["split_source_blobs"]) == set(original)
    for name in ("flash_api.cpp", "block.h"):
        (tmp_path / name).write_bytes(original[name])
    monkeypatch.setenv("MLA_FA3_SOURCE_ROOT", str(tmp_path))
    assert builder.split_toolchain_identity(False) == actual
    (tmp_path / "block.h").write_bytes(original["block.h"] + b"// altered\n")
    with pytest.raises(RuntimeError, match="block.h SHA256"):
        builder.split_toolchain_identity(False)
    (tmp_path / "block.h").write_bytes(original["block.h"])
    (tmp_path / "flash_api.cpp").write_bytes(original["flash_api.cpp"] + b"// altered\n")
    with pytest.raises(RuntimeError, match="flash_api.cpp SHA256"):
        builder.split_toolchain_identity(False)
    # The completed protocol is registered in the FA3 production build.
    setup = Path(runner.ROOT / "setup.py").read_text()
    assert '"mla_split_kv.cpp"' in setup
