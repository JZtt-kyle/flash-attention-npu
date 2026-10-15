"""TND ownership, fixed native recipe and evidence/failure propagation."""

from copy import deepcopy
from types import SimpleNamespace
import pytest
from tests.mla.support import mla_validation as driver
from tests.mla.support.native_package import LAYOUT_KEYS
from tests.mla.support.mla_public_performance import summarize_varlen
from tests.mla.support.mla_prefill_tnd_cases import PERFORMANCE_CASES


def test_selection_and_environment(tmp_path):
    assert all("/unit/prefill_tnd/" in p for p in driver.SUITES["unit-prefill_tnd"])
    assert driver.is_e2e(driver.SUITES["prefill-tnd-public"])
    assert "prefill-tnd-profile" in driver.PERFORMANCE_SUITES
    assert any("prefill_tnd/test_public.py::test_daily" in p for p in driver.SUITES["e2e"])
    a = SimpleNamespace(device="6", golden="cache", artifacts=tmp_path)
    assert (
        driver.environment(a, tmp_path, "prefill-tnd-profile-check")[
            "MLA_PREFILL_TND_PROFILE_CHECK"
        ]
        == "1"
    )
    assert (
        driver.environment(a, tmp_path, "prefill-tnd-profile")["MLA_PREFILL_TND_PROFILE_CHECK"]
        == "0"
    )
    assert driver.environment(a, tmp_path, "prefill-tnd-sanitize")["MLA_SCHEDULE_SANITIZE"] == "1"


def test_native_recipe_covers_generic_tnd():
    assert {103000000030000003, 104000000030000003} <= set(LAYOUT_KEYS)
    assert len(PERFORMANCE_CASES) * 2 <= 100
    assert {s["mask"] for s in PERFORMANCE_CASES} == {"none", "causal", "window"}
    assert {s["splits"] for s in PERFORMANCE_CASES} == {0, 1, 2}
    assert all(
        s.get("used_q") is None for s in PERFORMANCE_CASES
    )  # kernel parser has no post-attention restore


@pytest.mark.parametrize(
    "suite", ["prefill-tnd-tools", "unit-prefill_tnd", "prefill-tnd-public", "prefill-tnd-profile"]
)
def test_failure_and_evidence(tmp_path, monkeypatch, suite):
    from tests.mla.unit.tools.test_mla_validation import (
        test_success_failure_interruption_and_resume,
    )

    monkeypatch.setattr(driver, "validate_causal_builds", lambda *a, **k: None)
    monkeypatch.setattr(driver, "CPU_SUITES", driver.CPU_SUITES | {suite})
    test_success_failure_interruption_and_resume(tmp_path, monkeypatch, suite)


def test_varlen_prefill_classification():
    specs = [dict(splits=1, window=[-1, -1]), dict(splits=0, window=[7, 2])]
    reports = [
        dict(
            dtype=dt,
            representation="prefill",
            causal=False,
            window=True,
            warmup=5,
            rounds=5,
            wall_iterations=20,
            repeats=[1, 4],
            results=[
                dict(
                    case=i,
                    spec=s,
                    native_template=3 + i,
                    native_fd=True,
                    e2e_applicable=i == 1,
                    r1=dict(fa3=1, cann=2, ratio=0.5),
                    r4=dict(fa3=3, cann=2, ratio=1.5),
                )
                for i, s in enumerate(specs)
            ],
        )
        for dt in ("fp16", "bf16")
    ]
    result = summarize_varlen(reports, specs)
    assert result["passed"] and result["mean_ratio"] == 1
    assert result["applicable_cases"] == 2 and result["excluded_cases"] == 2
    assert result["regressed_segments"] == 2 and result["includes_native_fd"]
    bad = deepcopy(reports)
    bad[0]["results"][1]["native_template"] = 5
    with pytest.raises(ValueError, match="classification"):
        summarize_varlen(bad, specs)
    bad = deepcopy(reports)
    bad[0]["results"][1]["r4"]["ratio"] = 1
    with pytest.raises(ValueError, match="ratio"):
        summarize_varlen(bad, specs)


def test_native_compiler_uses_external_working_directory(tmp_path, monkeypatch):
    from tests.mla.support import native_package

    seen = []
    monkeypatch.setattr(native_package.subprocess, "run", lambda *a, **k: seen.append(k))
    native_package.run(["compiler"], tmp_path, "compile")
    assert seen[0]["cwd"] == tmp_path


@pytest.mark.parametrize("splits", [1, 0])
@pytest.mark.parametrize("fault", [None, "missing", "duplicate", "order", "dtype"])
def test_tnd_initialization_timing(splits, fault):
    from tests.mla.support.mla_prefill_bsnd_profile_rows import partition_kernels

    names = ["MlaPrefillTndFp16Init", "MlaPrefillBsndFp16Attention"]
    if splits != 1:
        names.append("MlaPrefillBsndFp16Merge")
    if fault == "missing":
        names.pop(0)
    elif fault == "duplicate":
        names.insert(0, names[0])
    elif fault == "order":
        names[0], names[1] = names[1], names[0]
    elif fault == "dtype":
        names[0] = "MlaPrefillTndBf16Init"
    rows = [
        {"Name": name, "Device_id": "0", "Start Time(us)": str(i * 10), "Duration(us)": "2"}
        for i, name in enumerate(names)
    ]
    args = (rows, [dict(case=0, name="fa3")], [dict(q=[7, 1], splits=splits)], "fp16")
    if fault:
        with pytest.raises(ValueError):
            partition_kernels(*args)
    else:
        (sample,) = partition_kernels(*args)
        assert sample["adaptation_us"] == 2  # same initialization boundary as zero/fill
        assert sample["kernel_us"] == (2 if splits == 1 else 4)
