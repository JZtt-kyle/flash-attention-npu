"""Selection, pinned dependencies, execution failures and evidence invalidation."""

import pytest
from tests.mla.support import mla_validation as runner
from tests.mla.support.mla_split_kv_build import split_toolchain_identity
from tests.mla.unit.tools.test_mla_validation import (
    args,
    test_success_failure_interruption_and_resume as check_states,
    test_causal_dependencies_changing_during_execution_fail as check_dependencies,
)


@pytest.mark.parametrize("suite", ["split-bsnd-cpu", "split-bsnd-sanitize", "split-bsnd-tools"])
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


@pytest.mark.parametrize(
    "fault", [None, "missing", "extra", "order", "dtype", "overlap", "device", "native"]
)
def test_profile_phases(fault):
    from tests.mla.support.mla_split_bsnd_profile_rows import pair_split_kernels

    rows = [
        dict(Name=name, **{"Device_id": "6", "Start Time(us)": str(start), "Duration(us)": "2"})
        for name, start in (
            ("mla_split_bsnd_partial_fp16", 1),
            ("mla_split_bsnd_merge_fp16", 5),
            ("FusedInferAttentionScore", 9),
        )
    ]
    labels = [{"name": "fa3"}, {"name": "native_cann"}]
    if fault == "missing":
        rows.pop(1)
    elif fault == "extra":
        rows.append(dict(rows[-1]))
    elif fault == "order":
        rows[0], rows[1] = rows[1], rows[0]
    elif fault == "dtype":
        rows[1]["Name"] = "mla_split_bsnd_merge_bf16"
    elif fault == "overlap":
        rows[1]["Start Time(us)"] = "2"
    elif fault == "device":
        rows[1]["Device_id"] = "5"
    elif fault == "native":
        rows[-1]["Name"] = "unexpected"
    if fault:
        with pytest.raises(ValueError):
            pair_split_kernels(rows, labels, "fp16")
    else:
        result = pair_split_kernels(rows, labels, "fp16")
        assert result[0]["Duration(us)"] == "4"
        assert result[0]["Split End Time(us)"] == "7"
        assert result[0]["Split Segments"] == rows[:2]


def test_public_split_identity():
    from tests.mla.support.mla_public_performance import input_identity, summarize
    from tests.mla.unit.tools.test_mla_validation_policy import reports

    row = dict(
        shape=[1, 3, 2, 513],
        scale=None,
        lengths=None,
        stride_profile="contiguous",
        window=[2, 1],
        causal=False,
    )
    original = input_identity(row)
    a = input_identity(dict(row, num_splits=2))
    b = input_identity(dict(row, num_splits=8))
    assert a != b and a["num_splits"] == 2
    assert "num_splits" not in original
    old = summarize(reports(), 2)
    data = reports()
    for report in data:
        for item in report["results"]:
            item["num_splits"] = 8
    current = summarize(data, 2)
    assert current["mean_ratio"] == old["mean_ratio"]
    assert current["cases"][0]["num_splits"] == 8
    data[1]["results"][0]["num_splits"] = 2
    with pytest.raises(ValueError, match="Dtype input matrices differ"):
        summarize(data, 2)


@pytest.mark.parametrize("suite", ["split-bsnd", "split-bsnd-source", "unit-split_kv_bsnd"])
def test_source_evidence_is_outside_scratch(tmp_path, suite):
    env = runner.environment(args(tmp_path), tmp_path, suite)
    assert env["MLA_SPLIT_KV_MERGE_DIR"] == str(tmp_path / "probes")
    assert suite in runner.SPLIT_KV_MERGE_SUITES
