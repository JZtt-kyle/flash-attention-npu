"""Explicit component timing for independent partial attention."""

from tests.mla.unit.split_kv_attention.test_attention import partial_attention as partial_attention
from tests.mla.support.mla_split_kv_attention_profile import profile


def test_component_profile(partial_attention):
    assert profile(partial_attention, partial_attention[1] / "profile")["results"]
