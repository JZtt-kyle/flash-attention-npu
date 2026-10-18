"""Explicit component performance; excluded from correctness-only discovery."""

import os
from pathlib import Path

from tests.mla.support.mla_int8_profile import PERFORMANCE_CASES, prepare, profile


def test_profile_preflight():
    calls, records = prepare(Path(os.environ["MLA_VALIDATION_RESULT"]) / "preflight")
    assert len(calls) == len(records) == len(PERFORMANCE_CASES)


def test_component_performance():
    report = profile(Path(os.environ["MLA_VALIDATION_RESULT"]) / "profile")
    assert len(report["results"]) == len(PERFORMANCE_CASES)
