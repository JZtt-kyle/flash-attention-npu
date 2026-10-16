"""PA prefill ownership, fixed native recipe and evidence/failure propagation."""

from types import SimpleNamespace
import pytest
from tests.mla.support import mla_validation as driver
from tests.mla.support.native_package import LAYOUT_KEYS
from tests.mla.support.mla_prefill_paged_tnd_cases import PERFORMANCE_CASES


def test_selection_and_environment(tmp_path):
    assert all("/unit/prefill_paged_tnd/" in p for p in driver.SUITES["unit-prefill_paged_tnd"])
    assert driver.is_e2e(driver.SUITES["prefill-paged-tnd-public"])
    assert "prefill-paged-tnd-profile" in driver.PERFORMANCE_SUITES
    assert any("prefill_paged_tnd/test_public.py::test_daily" in p for p in driver.SUITES["e2e"])
    a = SimpleNamespace(device="6", golden="cache", artifacts=tmp_path)
    assert (
        driver.environment(a, tmp_path, "prefill-paged-tnd-profile-check")[
            "MLA_PREFILL_PAGED_TND_PROFILE_CHECK"
        ]
        == "1"
    )
    assert (
        driver.environment(a, tmp_path, "prefill-paged-tnd-profile")[
            "MLA_PREFILL_PAGED_TND_PROFILE_CHECK"
        ]
        == "0"
    )
    assert (
        driver.environment(a, tmp_path, "prefill-paged-tnd-sanitize")["MLA_SCHEDULE_SANITIZE"]
        == "1"
    )


def test_native_recipe_covers_generic_paged_tnd():
    assert {103000000010200003, 104000000010200003} <= set(LAYOUT_KEYS)
    assert len(PERFORMANCE_CASES) * 2 <= 100
    assert {s["mask"] for s in PERFORMANCE_CASES} == {"none", "causal", "window"}
    assert {s["splits"] for s in PERFORMANCE_CASES} == {0, 1, 2}
    assert all(
        s.get("used_q") is None for s in PERFORMANCE_CASES
    )  # kernel parser has no post-attention restore


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("layout", ["page", "q"])
@pytest.mark.parametrize("fault", [None, "missing", "duplicate", "after", "dtype", "dense"])
def test_device_initializer_accounting(dtype, layout, fault):
    from tests.mla.support.mla_prefill_bsnd_profile_rows import partition_kernels

    name = {"fp16": "Fp16", "bf16": "Bf16"}[dtype]
    init = "MlaPrefillTnd" + name + "Init"
    attention = "MlaPrefillBsnd" + name + "Attention"
    merge = "MlaPrefillBsnd" + name + "Merge"
    names = [init, attention, merge]
    spec = dict(splits=0, **{layout: 16})
    if fault == "missing":
        names.pop(0)
    elif fault == "duplicate":
        names.insert(0, init)
    elif fault == "after":
        names = [attention, init, merge]
    elif fault == "dtype":
        names[0] = "MlaPrefillTnd" + ("Bf16" if dtype == "fp16" else "Fp16") + "Init"
    elif fault == "dense":
        spec.pop(layout)
    rows = [
        dict(Name=n, Device_id="0", **{"Start Time(us)": str(i * 10), "Duration(us)": "2"})
        for i, n in enumerate(names)
    ]
    args = (rows, [dict(case=0, name="fa3")], [spec], dtype)
    if fault:
        with pytest.raises(ValueError):
            partition_kernels(*args)
    else:
        (result,) = partition_kernels(*args)
        assert result["tasks"] == rows
        assert result["kernel_us"] == 4 and result["adaptation_us"] == 2


@pytest.mark.parametrize(
    "suite",
    [
        "prefill-paged-tnd-tools",
        "unit-prefill_paged_tnd",
        "prefill-paged-tnd-public",
        "prefill-paged-tnd-profile",
    ],
)
def test_failure_and_evidence(tmp_path, monkeypatch, suite):
    from tests.mla.unit.tools.test_mla_validation import (
        test_success_failure_interruption_and_resume,
    )

    monkeypatch.setattr(driver, "validate_causal_builds", lambda *a, **k: None)
    monkeypatch.setattr(driver, "CPU_SUITES", driver.CPU_SUITES | {suite})
    test_success_failure_interruption_and_resume(tmp_path, monkeypatch, suite)


def test_keys_exist_in_pinned_source():
    import os
    from pathlib import Path
    import subprocess
    from tests.mla.paths import ROOT
    from tests.mla.support.mla_dense_build import COMMIT

    root = Path(os.environ.get("MLA_CANN_SOURCE_ROOT", ROOT.parent / "ops-transformer"))
    source = subprocess.check_output(
        [
            "git",
            "-C",
            str(root),
            "show",
            COMMIT
            + ":attention/fused_infer_attention_score/op_kernel/arch22/fused_infer_attention_score_v3.cpp",
        ],
        text=True,
        timeout=30,
    )
    for base in (103000000010200003, 104000000010200003):
        for dtype in (0, 22220):
            for fd in (0, 100000):
                assert f"TILING_KEY_IS({base + dtype + fd})" in source
