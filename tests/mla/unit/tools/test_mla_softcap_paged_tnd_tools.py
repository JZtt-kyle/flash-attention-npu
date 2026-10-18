"""TND PA softcap selection, profiler boundaries and driver failure propagation."""

from pathlib import Path
import argparse
import pytest
from tests.mla.support import mla_validation as driver
from tests.mla.support.mla_softcap_paged_tnd_profile import PERFORMANCE, MODES, partition_rows
from tests.mla.support.public_cases.softcap_paged_tnd import CASES, DAILY


@pytest.mark.parametrize(
    "suite",
    [
        "softcap-paged-tnd-tools",
        "unit-softcap_paged_tnd",
        "softcap-paged-tnd-sanitize",
        "detail-softcap_paged_tnd",
        "softcap-paged-tnd-profile-check",
        "softcap-paged-tnd-profile",
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
    assert all("/unit/softcap_paged_tnd/" in p for p in driver.SUITES["unit-softcap_paged_tnd"])
    assert "softcap-paged-tnd-profile" in driver.PERFORMANCE_SUITES
    assert all("public_profile.py" not in p for p in driver.SUITES["detail-softcap_paged_tnd"])
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
    assert len(PERFORMANCE) * 2 + 24 + 36 == 70
    assert MODES == ["metadata"] + [
        f"{entry}_{r}" for entry in ("implicit", "generate", "reuse") for r in (1, 4)
    ]
    assert (
        driver.environment(
            argparse.Namespace(device="6", golden="cache", artifacts=Path("/tmp/mla-tools")),
            Path("/tmp/mla-tools"),
            "softcap-paged-tnd-sanitize",
        )["MLA_SCHEDULE_SANITIZE"]
        == "1"
    )


@pytest.mark.parametrize(
    "representation,splits,mangled",
    [("compressed", s, m) for s in (0, 1, 3) for m in (False, True)]
    + [("prefill", s, False) for s in (0, 1, 3)],
)
def test_profiler_phase_boundaries(representation, splits, mangled):
    if representation == "prefill":
        phases = [
            "ComputeMlaPrefillBsndMetadata",
            "MlaPrefillTndFp16Init",
            "MlaPrefillBsndFp16Attention",
        ] + ([] if splits == 1 else ["MlaPrefillBsndFp16Merge"])
    else:
        prefix = "mla_auto_split_paged_tnd" if splits == 0 else "mla_split_paged_tnd"
        phases = [
            "ComputeMlaAutoSplitPagedTndMetadata"
            if splits == 0
            else "ComputeMlaSplitPagedTndMetadata"
        ] + [f"{prefix}_{stage}_fp16" for stage in ["init", "partial", "merge"]]
    if mangled:
        phases = [f"_Z{len(phases[0])}{phases[0]}Pv"] + [
            f"_ZN10flash_attn3mla{len(n)}{n}EmPhS1_f" for n in phases[1:]
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
            dict(splits=splits, page=48),
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
    if mangled:
        bad = dict(rows[1], Name=rows[1]["Name"].replace("3mla", "9mla", 1))
        assert bad != rows[1]
        with pytest.raises(ValueError):
            parse([rows[0], bad, *rows[2:]])
