"""Prefill public selection, native classification and failure propagation."""

from copy import deepcopy
from types import SimpleNamespace
import pytest
from tests.mla.support import mla_validation as driver
from tests.mla.support.mla_native_dispatch import e2e_applicable
from tests.mla.support.mla_public_performance import summarize
from tests.mla.unit.tools.test_mla_validation_policy import reports


def test_suite_ownership_and_timing_environment(tmp_path):
    assert all("/unit/prefill_bsnd/" in p for p in driver.SUITES["unit-prefill_bsnd"])
    assert driver.is_e2e(driver.SUITES["prefill-bsnd-public"])
    assert "prefill-bsnd-profile" in driver.PERFORMANCE_SUITES
    assert "test_daily" in " ".join(driver.SUITES["e2e"])
    args = SimpleNamespace(device="6", golden="cache", artifacts=tmp_path)
    assert (
        driver.environment(args, tmp_path, "prefill-bsnd-profile-check")[
            "MLA_PREFILL_BSND_PROFILE_CHECK"
        ]
        == "1"
    )
    assert (
        driver.environment(args, tmp_path, "prefill-bsnd-profile")["MLA_PREFILL_BSND_PROFILE_CHECK"]
        == "0"
    )


@pytest.mark.parametrize(
    "suite",
    ["prefill-bsnd-tools", "unit-prefill_bsnd", "prefill-bsnd-public", "prefill-bsnd-profile"],
)
def test_success_failure_interruption(tmp_path, monkeypatch, suite):
    from tests.mla.unit.tools.test_mla_validation import (
        test_success_failure_interruption_and_resume,
    )

    monkeypatch.setattr(driver, "validate_causal_builds", lambda *a, **k: None)
    monkeypatch.setattr(driver, "CPU_SUITES", driver.CPU_SUITES | {suite})
    test_success_failure_interruption_and_resume(tmp_path, monkeypatch, suite)


@pytest.mark.parametrize("template", [0, 3, 4, 5])
@pytest.mark.parametrize("fd", [False, True])
def test_representation_specific_dispatch(template, fd):
    assert e2e_applicable(template, fd) == (template == 5 and not fd)
    assert e2e_applicable(template, fd, representation="prefill") == (template in (3, 4) and not fd)
    assert e2e_applicable(template, fd, representation="prefill", include_native_fd=True) == (
        template in (3, 4)
    )


def test_prefill_average_mixed_split_native_fd():
    data = deepcopy(reports((0.6, 1.2)))
    for report in data:
        report["representation"] = "prefill"
        for index, row in enumerate(report["results"]):
            row.update(
                native_template=3 + index, num_splits=1 if index == 0 else 0, native_fd=bool(index)
            )
    result = summarize(data, 2)
    assert result["passed"] and result["mean_ratio"] == pytest.approx(0.9)
    assert result["applicable_cases"] == 4 and result["regressed_cases"] == 2
    assert result["includes_native_fd"]
    data[0]["results"][0]["native_template"] = 5
    with pytest.raises(ValueError, match="classification"):
        summarize(data, 2)


def test_merge_ctypes_has_full_device_record():
    # C++ pads the uint16 interval arrays to an 8-byte aligned 1096-byte record.
    import ctypes
    from tests.mla.support.components.split_kv_merge import Params

    assert ctypes.sizeof(Params) == 1096
    assert Params().kv_heads == 1 and Params.kv_heads.offset == 1088


@pytest.mark.parametrize("fault", [None, "missing", "phase", "metadata", "trailing", "overlap"])
def test_prefill_profiler_boundaries(fault):
    from tests.mla.support.mla_prefill_bsnd_profile_rows import partition_kernels

    def row(name, index):
        return {
            "Name": name,
            "Device_id": "0",
            "Start Time(us)": str(index * 10),
            "Duration(us)": "2",
        }

    rows = [
        row("Fill", 0),
        row("MlaPrefillBsndFp16Attention", 1),
        row("MlaPrefillBsndFp16Merge", 2),
        row("aclnnFusedInferAttentionScore_Slice", 3),
        row("aclnnFusedInferAttentionScore_FusedInferAttentionScore", 4),
    ]
    labels = [dict(case=0, name="fa3"), dict(case=0, name="native_cann")]
    if fault == "missing":
        rows.pop()
    elif fault == "phase":
        rows[1]["Name"] = "wrong"
    elif fault == "metadata":
        rows[0]["Name"] = "ComputeMlaPrefillBsndMetadata"
    elif fault == "trailing":
        rows.append(row("unused", 5))
    elif fault == "overlap":
        rows[0]["Duration(us)"] = "11"
    if fault:
        with pytest.raises(ValueError):
            partition_kernels(rows, labels, [dict(splits=0)], "fp16")
    else:
        samples = partition_kernels(rows, labels, [dict(splits=0)], "fp16")
        assert [r["kernel_us"] for r in samples] == [4, 2]
        assert [r["adaptation_us"] for r in samples] == [2, 2]
