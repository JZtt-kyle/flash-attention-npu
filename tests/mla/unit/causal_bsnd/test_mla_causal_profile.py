"""Run the fixed independent causal performance matrix through the unified entry."""

from tests.mla.unit.causal_bsnd.test_flash_attn_npu_v3_mla_causal_attention import (
    causal_attention as causal_attention,
)
from tests.mla.support.mla_causal_attention_profile import profile


def test_component_profile(causal_attention):
    report = profile(causal_attention, causal_attention[1] / "profile")
    assert report["results"]
