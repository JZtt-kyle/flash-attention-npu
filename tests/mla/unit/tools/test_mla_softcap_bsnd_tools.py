"""Public softcap reference, fixed coverage, profiler framing and evidence checks."""

import math
import pytest
import torch
from tests.mla.support import mla_validation as driver
from tests.mla.support.mla_auto_split_bsnd_reference import references
from tests.mla.support.mla_softcap_bsnd_profile import PERFORMANCE, MODES
from tests.mla.support.mla_split_bsnd_profile_rows import pair_split_kernels
from tests.mla.support.mla_prefill_bsnd_profile_rows import partition_kernels
from tests.mla.support.public_cases.softcap_bsnd import CASES, DAILY


def inputs():
    q = torch.zeros(1, 1, 2, 64, dtype=torch.float16)
    k = torch.zeros(1, 3, 1, 64, dtype=torch.float16)
    qv = torch.zeros(1, 1, 2, 512, dtype=torch.float16)
    v = torch.zeros(1, 3, 1, 512, dtype=torch.float16)
    q[..., 0], qv[..., 0] = 16, 32
    k[0, :, 0, 0] = torch.tensor([48, 1, -47])
    v[0, :, 0, 0] = torch.tensor([-24, 0, 24])
    return dict(q=q, k=k, qv=qv, v=v)


def reference(raw, cap):
    return references(raw, [1], [3], [], 16, 32, False, (-1, -1), 0.25, num_splits=1, softcap=cap)


@pytest.mark.parametrize("cap", [0, 15, 30])
def test_total_score_before_cap_matches_scalar(cap):
    result = reference(inputs(), cap)
    scores = [0.0, 4.0, 4.0]
    if cap:
        scores = [cap * math.tanh(s / cap) for s in scores]
    total = sum(math.exp(s) for s in scores)
    value = sum(math.exp(s) * v for s, v in zip(scores, [-24, 0, 24])) / total
    torch.testing.assert_close(
        result["out_high"][..., 0], torch.full((1, 1, 2), value).half(), rtol=0, atol=0
    )
    torch.testing.assert_close(
        result["lse_high"], torch.full((1, 2, 1), math.log(total)), rtol=0, atol=1e-6
    )


def test_reference_cache_cap_and_input_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("GOLDEN_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("GOLDEN_CACHE_MODE", "cache")
    monkeypatch.setenv("GOLDEN_CACHE_REFRESH", "0")
    stats = tmp_path / "events.tsv"
    monkeypatch.setenv("GOLDEN_CACHE_STATS_FILE", str(stats))
    raw = inputs()
    first = reference(raw, 15)
    second = reference(raw, 15)
    assert torch.equal(first["out_high"], second["out_high"])
    assert "hit\t" in stats.read_text().splitlines()[-1]
    reference(raw, 30)
    assert "hit\t" not in stats.read_text().splitlines()[-1]
    raw["q"][..., 0] += 1
    reference(raw, 15)
    assert "hit\t" not in stats.read_text().splitlines()[-1]


@pytest.mark.parametrize(
    "suite",
    [
        "softcap-bsnd-tools",
        "unit-softcap_bsnd",
        "softcap-bsnd-sanitize",
        "detail-softcap_bsnd",
        "softcap-bsnd-profile-check",
        "softcap-bsnd-profile",
    ],
)
def test_driver_failure_integrity_and_resume(tmp_path, monkeypatch, suite):
    from tests.mla.unit.tools.test_mla_validation import (
        test_success_failure_interruption_and_resume as check,
    )

    monkeypatch.setattr(driver, "validate_causal_builds", lambda *a, **k: None)
    monkeypatch.setattr(driver, "CPU_SUITES", driver.CPU_SUITES | {suite})
    check(tmp_path, monkeypatch, suite)


def test_fixed_ownership_and_coverage():
    assert all("/unit/softcap_bsnd/" in p for p in driver.SUITES["unit-softcap_bsnd"])
    assert "softcap-bsnd-profile" in driver.PERFORMANCE_SUITES
    assert all("public_profile.py" not in p for p in driver.SUITES["detail-softcap_bsnd"])
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
    assert {s["splits"] for s in PERFORMANCE} == {0, 1, 3}
    assert len(PERFORMANCE) * 2 * 2 <= 100
    assert set(MODES) == {"generate"} | {
        f"{e}_{r}" for e in ("implicit", "generate", "reuse", "func") for r in (1, 4)
    }


@pytest.mark.parametrize(
    "representation,splits", [(r, s) for r in ("compressed", "prefill") for s in (0, 1, 3)]
)
def test_profiler_phase_boundaries(representation, splits):
    if representation == "prefill":
        phases = ["MlaPrefillBsndFp16Attention"] + (
            [] if splits == 1 else ["MlaPrefillBsndFp16Merge"]
        )
    else:
        prefix = "mla_auto_split_bsnd" if splits == 0 else "mla_split_bsnd"
        phases = [
            f"{prefix}_{stage}_fp16"
            for stage in (["init"] if splits == 0 else []) + ["partial", "merge"]
        ]
    rows = [
        dict(Name=n, Device_id="6", **{"Start Time(us)": str(i * 5), "Duration(us)": "2"})
        for i, n in enumerate(phases)
    ]
    labels = [dict(case=0, name="fa3")]

    def parse(value):
        return (
            partition_kernels(value, labels, [dict(splits=splits)], "fp16")
            if representation == "prefill"
            else pair_split_kernels(value, labels, "fp16", automatic=splits == 0)
        )

    assert len(parse(rows)) == 1
    with pytest.raises(ValueError):
        parse(rows[:-1])
    with pytest.raises(ValueError):
        parse(rows + rows)
