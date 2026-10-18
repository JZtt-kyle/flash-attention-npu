"""Module ownership and driver/evidence checks for the isolated GQA component."""

import argparse
from pathlib import Path
import pytest
from tests.mla.support import mla_validation as driver
from tests.mla.support.mla_gqa_schedule_build import fa3_options_source
from tests.mla.support.mla_window_build import fa3_blob


@pytest.mark.parametrize(
    "suite",
    ["unit-gqa_schedule", "gqa-schedule-check", "gqa-schedule-sanitize", "gqa-schedule-tools"],
)
def test_failure_interruption_and_resume(tmp_path, monkeypatch, suite):
    from tests.mla.unit.tools.test_mla_validation import (
        test_success_failure_interruption_and_resume as check,
    )

    check(tmp_path, monkeypatch, suite)


def test_selection_ownership_and_sanitizer():
    assert driver.SUITES["unit-gqa_schedule"] == driver.SUITES["gqa-schedule-sanitize"]
    assert driver.SUITES["unit-gqa_schedule"] == ["tests/mla/unit/gqa_schedule/test_schedule.py"]
    assert not driver.is_e2e(driver.SUITES["unit-gqa_schedule"])
    for suite in ("unit-gqa_schedule", "gqa-schedule-sanitize", "gqa-schedule-check"):
        assert suite in driver.CPU_SUITES and suite not in driver.PERFORMANCE_SUITES
        env = driver.environment(
            argparse.Namespace(device="7", golden="cache", artifacts=Path("/tmp/gqa")),
            Path("/tmp/gqa"),
            suite,
        )
        assert env["MLA_SCHEDULE_SANITIZE"] == str(int(suite.endswith("sanitize")))
        assert env["MLA_VALIDATION_NPU"] == "0"


def test_oracle_preserves_fixed_source_statements():
    source = fa3_options_source()
    api = fa3_blob().decode()
    for token in ("params.pack_gqa = pack_gqa_.", "params.num_sm = "):
        line = next(line for line in api.splitlines() if token in line)
        assert line in source
