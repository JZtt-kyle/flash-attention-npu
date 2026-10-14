"""Automatic source selection, three-stage timing, FD inclusion and failure evidence."""

import copy
import pytest
from tests.mla.support import mla_validation as runner
from tests.mla.support.mla_split_bsnd_profile_rows import pair_split_kernels
from tests.mla.support.mla_public_performance import summarize
from tests.mla.unit.tools.test_mla_validation import (
    args,
    test_success_failure_interruption_and_resume as check_states,
)
from tests.mla.unit.tools.test_mla_validation_policy import reports


@pytest.mark.parametrize(
    "suite",
    [
        "auto-split-bsnd-check",
        "auto-split-bsnd",
        "auto-split-bsnd-trap",
        "auto-split-bsnd-profile",
        "auto-split-bsnd-tools",
    ],
)
def test_selection_and_failures(tmp_path, monkeypatch, suite):
    env = runner.environment(args(tmp_path), tmp_path, suite)
    assert env["MLA_VALIDATION_NPU"] == ("0" if suite.endswith("tools") else "1")
    if suite == "auto-split-bsnd":
        assert env["MLA_SPLIT_KV_MERGE_DIR"] == str(tmp_path / "probes")
    if suite.endswith("profile"):
        assert suite in runner.PERFORMANCE_SUITES
    else:
        # Exercise subprocess/result handling without acquiring a real device.
        monkeypatch.setattr(runner, "CPU_SUITES", runner.CPU_SUITES | {suite})
        monkeypatch.setattr(runner, "validate_causal_builds", lambda *a, **k: None)
        check_states(tmp_path, monkeypatch, suite)


@pytest.mark.parametrize("fault", [None, "missing", "order", "dtype", "overlap", "device", "extra"])
def test_three_phases(fault):
    rows = [
        dict(
            Name=f"mla_auto_split_bsnd_{stage}_fp16",
            Device_id="6",
            **{"Start Time(us)": str(i * 4), "Duration(us)": "2"},
        )
        for i, stage in enumerate(("init", "partial", "merge"))
    ]
    rows.append(
        dict(
            Name="FusedInferAttentionScore",
            Device_id="6",
            **{"Start Time(us)": "16", "Duration(us)": "2"},
        )
    )
    if fault == "missing":
        rows.pop(1)
    elif fault == "order":
        rows[0], rows[1] = rows[1], rows[0]
    elif fault == "dtype":
        rows[2]["Name"] = "mla_auto_split_bsnd_merge_bf16"
    elif fault == "overlap":
        rows[2]["Start Time(us)"] = "5"
    elif fault == "device":
        rows[2]["Device_id"] = "5"
    elif fault == "extra":
        rows.append(rows[-1])
    labels = [dict(name="fa3"), dict(name="native_cann")]
    if fault:
        with pytest.raises(ValueError):
            pair_split_kernels(rows, labels, "fp16", automatic=True)
    else:
        paired = pair_split_kernels(rows, labels, "fp16", automatic=True)
        assert paired[0]["Duration(us)"] == "6" and paired[0]["Split End Time(us)"] == "10"
        assert paired[0]["Split Segments"] == rows[:3]


def test_native_fd_acceptance():
    data = reports()
    for report in data:
        report["automatic_split"] = True
        for i, row in enumerate(report["results"]):
            row.update(num_splits=0, native_fd=bool(i))
    summary = summarize(data, 2, include_native_fd=True)
    assert summary["applicable_cases"] == 4 and summary["mean_ratio"] == pytest.approx(0.9)
    assert summary["includes_native_fd"]
    with pytest.raises(ValueError, match="classification"):
        summarize(data, 2)
    invalid = copy.deepcopy(data)
    invalid[1].pop("automatic_split")
    with pytest.raises(ValueError, match="automatic split"):
        summarize(invalid, 2, include_native_fd=True)
    old = reports()
    original = summarize(old, 2)
    assert original["applicable_cases"] == 4 and not original["includes_native_fd"]


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize("window,causal", [((-1, -1), False), ((-1, -1), True), ((2, 1), False)])
def test_reference_restricts_only_invisible_keys(dtype_name, window, causal):
    import torch
    from tests.mla.support.mla_auto_split_bsnd_reference import references
    from tests.mla.support.mla_ref import mla_reference, mla_mask

    dtype = getattr(torch, dtype_name)
    g = torch.Generator().manual_seed(713)
    raw = {
        name: torch.randn(
            (1, 3 if name in ("q", "qv") else 17, 2 if name in ("q", "qv") else 1, dim), generator=g
        ).to(dtype)
        for name, dim in (("q", 64), ("qv", 512), ("k", 64), ("v", 512))
    }
    result = references(raw, [3], [17], [(0, 0, 1), (1, 0, 0)], 8, 4, causal, window, 0.125)
    for precision in ("high", "low"):
        expected = mla_reference(
            **raw,
            softmax_scale=0.125,
            causal=causal,
            window_size=window,
            upcast=precision == "high",
        )
        for value, name in zip(expected, ("out", "lse")):
            torch.testing.assert_close(result[name + "_" + precision], value, rtol=1e-5, atol=1e-6)
    # Independent scalar row/key oracle for every real native fragment. Full
    # masks precede fragment clipping; omitted slots must retain 0/-inf.
    visible = ~mla_mask(3, 17, causal=causal, window_size=window)
    splits = [(0, 4), (4, 17)] if visible[:, :4].any() else [(4, 17)]
    for qi in range(3):
        for hi in range(2):
            for split, (lo, end) in enumerate(splits):
                keys = [ki for ki in range(lo, end) if visible[qi, ki]]
                got = result["accum_high"][split, 0, hi, qi]
                lse = result["accum_lse_high"][split, 0, hi, qi]
                if not keys:
                    assert (got == 0).all() and lse.isneginf()
                    continue
                logits = (
                    torch.stack(
                        [
                            (raw["q"][0, qi, hi].double() * raw["k"][0, ki, 0].double()).sum()
                            + (raw["qv"][0, qi, hi].double() * raw["v"][0, ki, 0].double()).sum()
                            for ki in keys
                        ]
                    )
                    * 0.125
                )
                expected = (
                    (logits.softmax(0)[:, None] * raw["v"][0, keys, 0].double()).sum(0).float()
                )
                torch.testing.assert_close(got, expected, rtol=1e-6, atol=1e-6)
                torch.testing.assert_close(lse, logits.logsumexp(0).float(), rtol=1e-6, atol=1e-6)
    assert (result["accum_high"][len(splits) :] == 0).all()
    assert result["accum_lse_high"][len(splits) :].isneginf().all()
