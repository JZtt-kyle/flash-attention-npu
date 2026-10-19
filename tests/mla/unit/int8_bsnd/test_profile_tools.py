"""Cheap success/failure fixtures for the INT8 public sampling adapter."""

import pytest

from tests.mla.support.mla_int8_bsnd_profile_rows import (
    PERFORMANCE,
    selected,
    modes,
    summarize,
    partition_kernels,
)


def samples():
    return [
        dict(
            case=0,
            name=name,
            mode=mode,
            round=r,
            iteration=it,
            e2e_us=2.0 if name.startswith("native_cann") else 1.0,
        )
        for name in modes(PERFORMANCE[0])
        for mode in ("eager", "graph")
        for r in range(5)
        for it in range(20)
    ]


def test_fixed_selection_and_weighting():
    assert len(PERFORMANCE) == 24
    assert len({(s["index"], s["view"]) for s in PERFORMANCE}) == 24
    assert len(selected(True)) == 4
    reports = summarize(samples(), PERFORMANCE[:1])
    assert all(r["passed"] and r["mean_ratio"] == 0.5 for r in reports.values())
    bad = samples()
    for row in bad:
        if not row["name"].startswith("native_cann"):
            row["e2e_us"] = 3.0
    assert not summarize(bad, PERFORMANCE[:1])["graph"]["passed"]


@pytest.mark.parametrize("fault", ["missing", "duplicate", "nan", "zero", "mode"])
def test_invalid_samples(fault):
    rows = samples()
    if fault == "missing":
        rows.pop()
    elif fault == "duplicate":
        rows.append(rows[0])
    elif fault == "mode":
        rows[0]["mode"] = "capture"
    else:
        rows[0]["e2e_us"] = float("nan") if fault == "nan" else 0
    with pytest.raises(ValueError):
        summarize(rows, PERFORMANCE[:1])


def test_profiler_identity():
    labels = [dict(case=0, name="fa3"), dict(case=0, name="native_cann")]
    rows = [
        dict(Name=name, **{"Duration(us)": "2"}, model_id=i)
        for i, name in enumerate(("mla_int8_bsnd", "x_FusedInferAttentionScore"))
    ]
    assert len(partition_kernels(rows, labels, PERFORMANCE[:1], "int8")) == 2
    rows[1]["model_id"] = 0
    with pytest.raises(ValueError):
        partition_kernels(rows, labels, PERFORMANCE[:1], "int8")
    rows[1]["Name"] = "foreign"
    with pytest.raises(ValueError):
        partition_kernels(rows, labels, PERFORMANCE[:1], "int8")
