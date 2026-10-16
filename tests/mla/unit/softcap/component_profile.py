"""Explicit performance and cheap launch fixtures, outside unit correctness discovery."""

import os
import pytest
from tests.mla.support.mla_softcap_profile import profile, profile_merge


@pytest.mark.parametrize("representation", ("compressed", "prefill"), scope="module")
def test_component_profile(probes):
    assert profile(
        probes,
        probes["directory"] / "profile",
        preflight=os.environ.get("MLA_SOFTCAP_PROFILE_CHECK") == "1",
    )["results"]


@pytest.mark.parametrize("representation", ("compressed", "prefill"), scope="module")
def test_merge_profile(probes):
    assert profile_merge(
        probes,
        probes["directory"] / "merge-profile",
        preflight=os.environ.get("MLA_SOFTCAP_PROFILE_CHECK") == "1",
    )["results"]
