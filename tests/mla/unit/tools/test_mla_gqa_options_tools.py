"""Selection, profile phase/identity and unified failure/invalid-evidence propagation."""

import argparse
import copy
import json
from pathlib import Path
import subprocess
import pytest
from tests.mla.support import mla_validation as driver
from tests.mla.support.mla_gqa_options_profile_rows import (
    PERFORMANCE,
    modes,
    summarize,
    partition_kernels,
)


@pytest.mark.parametrize(
    "suite",
    [
        "unit-gqa_options",
        "gqa-options-sanitize",
        "gqa-options-tools",
        "gqa-options-check",
        "gqa-options-tail-check",
        "softcap-mask-tail-check",
        "gqa-options-source",
        "e2e",
        "gqa-options-profile",
    ],
)
def test_failure_interruption_and_resume(tmp_path, monkeypatch, suite):
    from tests.mla.unit.tools.test_mla_validation import (
        test_success_failure_interruption_and_resume,
    )

    monkeypatch.setattr(driver, "CPU_SUITES", driver.CPU_SUITES | {suite})
    if suite == "softcap-mask-tail-check":
        # This child only simulates exit/evidence behavior; it builds no probes.
        monkeypatch.setattr(driver, "validate_causal_builds", lambda *a, **kw: None)
    test_success_failure_interruption_and_resume(tmp_path, monkeypatch, suite)


def test_ownership_and_matrix():
    assert driver.SUITES["unit-gqa_options"] == driver.SUITES["gqa-options-sanitize"]
    assert driver.is_e2e(driver.SUITES["e2e"])
    assert driver.is_e2e(driver.SUITES["gqa-options-source"])
    assert driver.is_e2e(driver.SUITES["gqa-options-boundaries"])
    assert not driver.is_e2e(driver.SUITES["unit-gqa_options"])
    assert any("gqa_options" in p for p in driver.SUITES["e2e"])
    assert not any("unit/gqa" in p for p in driver.SUITES["e2e"])
    assert len(PERFORMANCE) * 4 <= 100
    assert len({json.dumps(s, sort_keys=True) for s in PERFORMANCE}) == len(PERFORMANCE)
    for suite in (
        "unit-gqa_options",
        "gqa-options-sanitize",
        "gqa-options-tools",
        "gqa-options-profile-check",
    ):
        env = driver.environment(
            argparse.Namespace(device="7", golden="cache", artifacts=Path("/tmp/gqa")),
            Path("/tmp/gqa"),
            suite,
        )
        assert env["MLA_VALIDATION_NPU"] == str(int(suite not in driver.CPU_SUITES))
    assert (
        driver.environment(
            argparse.Namespace(device="7", golden="cache", artifacts=Path("/tmp/gqa")),
            Path("/tmp/gqa"),
            "gqa-options-profile-check",
        )["MLA_GQA_OPTIONS_PROFILE_CHECK"]
        == "1"
    )


def reports():
    result = []
    for rep in ("compressed", "prefill"):
        for dtype in ("fp16", "bf16"):
            rows = []
            for spec in PERFORMANCE:
                row = dict(
                    spec=json.loads(json.dumps(spec)),
                    native_template=5 if rep == "compressed" else 3,
                    native_fd=False,
                    e2e_applicable=not bool(spec.get("cap", 0)),
                )
                row.update(
                    {n: dict(median_e2e_us=2 if n.startswith("native") else 1) for n in modes(spec)}
                )
                rows.append(row)
            result.append(
                dict(
                    representation=rep,
                    dtype=dtype,
                    warmup=5,
                    rounds=5,
                    wall_iterations=20,
                    profiler_iterations=10,
                    results=rows,
                )
            )
    return result


def test_average_and_degradation():
    source = reports()
    first = source[0]["results"][0]
    first["fa3"]["median_e2e_us"] = 100
    result = summarize(source)
    assert result["measured_cases"] == 80 and result["applicable_cases"] == 60
    assert result["regressed_cases"] == 1 and result["passed"]
    assert result["regressed_segments"] == 1
    assert result["segment_means"]["reuse_1"] == pytest.approx((50 + 47 * 0.5) / 48)
    assert result["segment_means"]["public_4"] == 0.5
    for r in source:
        for row in r["results"]:
            for n in modes(row["spec"]):
                if not n.startswith("native"):
                    row[n]["median_e2e_us"] = 100
    assert not summarize(source)["passed"]


@pytest.mark.parametrize(
    "fault", ["missing", "duplicate", "input", "classification", "timing", "nan"]
)
def test_profile_identity_failure(fault):
    source = reports()
    if fault == "missing":
        source.pop()
    elif fault == "duplicate":
        source[1] = copy.deepcopy(source[0])
    elif fault == "input":
        source[0]["results"][0]["spec"]["margin"] = 25
    elif fault == "classification":
        source[0]["results"][0]["e2e_applicable"] = False
    elif fault == "timing":
        source[0]["warmup"] = 1
    else:
        source[0]["results"][0]["fa3"]["median_e2e_us"] = float("nan")
    with pytest.raises(ValueError):
        summarize(source)


@pytest.mark.parametrize("layout", ["bsnd", "tnd"])
def test_kernel_phases(layout):
    names = (["ComputeMlaGqaMetadata"] if layout == "tnd" else []) + [
        "MlaGqaFp16" + s for s in ("Init", "Attention", "Merge")
    ]
    rows = [
        dict(Name=n, Device_id="0", **{"Start Time(us)": str(i * 3), "Duration(us)": "1"})
        for i, n in enumerate(names)
    ]
    labels = [dict(case=0, name="fa3")]
    specs = [dict(layout=layout)]
    assert partition_kernels(rows, labels, specs, "fp16")[0]["kernel_us"] == 2
    for bad in (rows[:-1], rows + rows, rows[::-1]):
        with pytest.raises(ValueError):
            partition_kernels(bad, labels, specs, "fp16")


@pytest.mark.parametrize("failure", [False, True])
def test_all_fixtures_before_sampling(tmp_path, monkeypatch, failure):
    from tests.mla.e2e.gqa_options import public_profile as profile

    monkeypatch.setenv("MLA_VALIDATION_RESULT", str(tmp_path))
    monkeypatch.delenv("MLA_GQA_OPTIONS_PROFILE_CHECK", raising=False)
    monkeypatch.setattr(profile, "native_vendor", lambda: tmp_path / "native")
    monkeypatch.setattr(profile, "parse_dispatch", lambda *a, **kw: {})
    monkeypatch.setattr(profile.os, "sched_setaffinity", lambda *a: None)
    events = []

    def run(cmd, **kwargs):
        timing = "--dispatch" in cmd
        events.append("timing" if timing else "fixture")
        if failure and len(events) == 3:
            raise subprocess.CalledProcessError(1, cmd)
        directory = Path(cmd[cmd.index("--output") + 1])
        directory.mkdir(exist_ok=True)
        if timing:
            (directory / "profile.json").write_text("{}")

    monkeypatch.setattr(profile.subprocess, "run", run)
    monkeypatch.setattr(profile, "summarize", lambda reports, specs: {"count": len(reports)})
    monkeypatch.setattr(profile, "check_public_performance", lambda result: events.append(result))
    if failure:
        with pytest.raises(subprocess.CalledProcessError):
            profile.test_native_public_profile()
        assert events == ["fixture"] * 3
        assert not (tmp_path / "public-profile/average.json").exists()
    else:
        profile.test_native_public_profile()
        assert events == ["fixture"] * 4 + ["timing"] * 4 + [{"count": 4}]
