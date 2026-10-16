"""New TND selection, independent allocation math and profiler failure propagation."""

from pathlib import Path
import argparse
import pytest
import torch
from tests.mla.support import mla_validation as driver
from tests.mla.support.mla_softcap_tnd_profile import PERFORMANCE, MODES, partition_rows
from tests.mla.support.public_cases.softcap_tnd import CASES, DAILY, reference


@pytest.mark.parametrize(
    "suite",
    [
        "softcap-tnd-tools",
        "unit-softcap_tnd",
        "softcap-tnd-sanitize",
        "detail-softcap_tnd",
        "softcap-tnd-profile-check",
        "softcap-tnd-profile",
    ],
)
def test_driver_failure_integrity_and_resume(tmp_path, monkeypatch, suite):
    from tests.mla.unit.tools.test_mla_validation import (
        test_success_failure_interruption_and_resume as check,
    )

    monkeypatch.setattr(driver, "validate_causal_builds", lambda *a, **k: None)
    monkeypatch.setattr(driver, "CPU_SUITES", driver.CPU_SUITES | {suite})
    check(tmp_path, monkeypatch, suite)


def test_ownership_and_fixed_coverage():
    assert all("/unit/softcap_tnd/" in p for p in driver.SUITES["unit-softcap_tnd"])
    assert "softcap-tnd-profile" in driver.PERFORMANCE_SUITES
    assert all("public_profile.py" not in p for p in driver.SUITES["detail-softcap_tnd"])
    assert DAILY <= {s["id"] for s in CASES}
    assert {(s["splits"], s.get("mask", "none")) for s in CASES} >= {
        (1, "none"),
        (1, "causal"),
        (1, "window"),
        (0, "none"),
        (0, "causal"),
        (0, "window"),
        (3, "none"),
        (8, "causal"),
        (4, "window"),
    }
    assert len(PERFORMANCE) * 2 + 60 + 30 == 100
    assert MODES == ["metadata", "public_1", "public_4"]
    assert (
        driver.environment(
            argparse.Namespace(device="6", golden="cache", artifacts=Path("/tmp/mla-tools")),
            Path("/tmp/mla-tools"),
            "softcap-tnd-sanitize",
        )["MLA_SCHEDULE_SANITIZE"]
        == "1"
    )


@pytest.mark.parametrize(
    "representation,splits", [(r, s) for r in ("compressed", "prefill") for s in (0, 1, 3)]
)
def test_profiler_phase_boundaries(representation, splits):
    if representation == "prefill":
        phases = [
            "ComputeMlaPrefillBsndMetadata",
            "MlaPrefillTndFp16Init",
            "MlaPrefillBsndFp16Attention",
        ] + ([] if splits == 1 else ["MlaPrefillBsndFp16Merge"])
    else:
        prefix = "mla_auto_split_tnd" if splits == 0 else "mla_split_tnd"
        phases = [
            "ComputeMlaAutoSplitTndMetadata" if splits == 0 else "ComputeMlaSplitTndMetadata"
        ] + [
            f"{prefix}_{stage}_fp16"
            for stage in (["init"] if splits == 0 else []) + ["partial", "merge"]
        ]
    rows = [
        dict(Name=n, Device_id="6", **{"Start Time(us)": str(i * 5), "Duration(us)": "2"})
        for i, n in enumerate(phases)
    ]

    def parse(value, dtype="fp16"):
        raw = [
            {
                "Op Name": r["Name"],
                "Device_id": r["Device_id"],
                "Task Start Time(us)": r["Start Time(us)"],
                "Task Duration(us)": r["Duration(us)"],
            }
            for r in value
        ]
        return partition_rows(
            value,
            raw,
            [dict(case=0, name="fa3")],
            representation,
            dtype,
            dict(splits=splits, q=[1]),
        )

    assert len(parse(rows)) == 1
    for invalid in (
        rows[:-1],
        rows + rows,
        rows[1:],
        [rows[1], rows[0], *rows[2:]],
        [dict(rows[0], Device_id="7"), *rows[1:]],
    ):
        with pytest.raises(ValueError):
            parse(invalid)
    with pytest.raises(ValueError):
        parse(rows, "bf16")


@pytest.mark.parametrize("representation", ["compressed", "prefill"])
@pytest.mark.parametrize("cap", [0, 15, 30])
def test_reference_allocation_and_used_gaps(representation, cap):
    prefill = representation == "prefill"
    dq, dv = (192, 128) if prefill else (64, 512)
    raw = dict(
        q=torch.zeros(3, 2, dq, dtype=torch.float16),
        k=torch.zeros(5, 1, dq, dtype=torch.float16),
        v=torch.zeros(5, 1, dv, dtype=torch.float16),
    )
    raw["v"][:2] = 2
    raw["v"][2] = 99
    raw["v"][3:] = 6
    if not prefill:
        raw["qv"] = torch.zeros(3, 2, 512, dtype=torch.float16)
    c = dict(
        representation=representation,
        shape=(2, 2, 2, 3, 1),
        q=[2, 1],
        k=[3, 2],
        qlens=[1, 1],
        klens=[2, 2],
        raw=raw,
        options=dict(
            softcap=cap, softmax_scale=0, num_splits=1, causal=False, window_size=(-1, -1)
        ),
    )
    rec = torch.zeros(800, dtype=torch.int32)
    rec[13 if prefill else 15] = 16
    rec[14 if prefill else 16] = 32
    r = reference(c, rec)
    for precision in ("high", "low"):
        assert torch.equal(
            r["out_" + precision][[0, 2]],
            torch.tensor([2, 6], dtype=torch.float16)[:, None, None].expand(2, 2, dv),
        )
        assert r["out_" + precision].shape == (3, 2, dv)
