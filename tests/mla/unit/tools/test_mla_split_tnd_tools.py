"""TND module selection, source identity, failure propagation and split timing."""

import pytest
from tests.mla.support import mla_validation as runner
from tests.mla.support.mla_split_kv_build import split_toolchain_identity
from tests.mla.unit.tools.test_mla_validation import (
    args,
    test_success_failure_interruption_and_resume as check_states,
    test_causal_dependencies_changing_during_execution_fail as check_dependencies,
)
from tests.mla.unit.tools.test_organization import (
    test_native_package_requires_intact_content as check_native_package,
)


@pytest.mark.parametrize("suite", ["split-tnd-cpu", "split-tnd-sanitize", "split-tnd-tools"])
def test_protocol(tmp_path, monkeypatch, suite):
    env = runner.environment(args(tmp_path), tmp_path, suite)
    assert env["MLA_VALIDATION_NPU"] == "0"
    if not suite.endswith("tools"):
        assert env["MLA_SCHEDULE_SANITIZE"] == str(int(suite.endswith("sanitize")))
        assert runner.suite_dependencies(suite) == split_toolchain_identity(
            suite.endswith("sanitize")
        )
        with monkeypatch.context() as patch:
            check_dependencies(tmp_path, patch, suite)
    check_states(tmp_path, monkeypatch, suite)


def test_ownership():
    from tests.mla.catalog import E2E, is_e2e

    assert is_e2e(E2E["split_tnd"])
    assert not is_e2e(runner.SUITES["split-tnd"])
    assert "split-tnd" in runner.SPLIT_KV_MERGE_SUITES
    assert all("split_tnd/" in path for path in E2E["split_tnd"])


def test_native_package_identity(tmp_path, monkeypatch):
    check_native_package(monkeypatch, tmp_path)


@pytest.mark.parametrize(
    "fault", [None, "missing", "extra", "stage", "dtype", "overlap", "device", "metadata"]
)
@pytest.mark.parametrize("mangled", [False, True])
def test_profile_boundary(fault, mangled):
    from tests.mla.support.mla_split_tnd_profile_rows import production_sample

    names = ("ComputeMlaSplitTndMetadata", "mla_split_tnd_partial_fp16", "mla_split_tnd_merge_fp16")
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
        rows[2]["Op Name"] = "mla_split_tnd_merge_bf16"
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
    stages = ("mla_split_tnd_partial_", "mla_split_tnd_merge_")
    if fault:
        with pytest.raises(ValueError):
            production_sample(rows, names[0], stages)
    else:
        assert production_sample(rows, names[0], stages)["Task Duration(us)"] == "4"
