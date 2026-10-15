"""Explicit component performance; excluded from correctness-only unit discovery."""

import os
from tests.mla.unit.prefill_attention.test_attention import probes as probes
from tests.mla.support.mla_prefill_profile import profile


def test_component_profile(probes):
    assert profile(
        probes,
        probes[0][1] / "profile",
        preflight=os.environ.get("MLA_PREFILL_PROFILE_CHECK") == "1",
    )["results"]
