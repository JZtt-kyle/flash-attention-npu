"""Run the fixed independent window performance matrix through the unified entry."""

from tests.mla.unit.window_bsnd.test_flash_attn_npu_v3_mla_window_attention import (
    window_attention as window_attention,
)
from tests.mla.support.mla_window_attention_profile import profile


def test_component_profile(window_attention):
    report = profile(window_attention, window_attention[1] / "profile")
    assert report["results"]
