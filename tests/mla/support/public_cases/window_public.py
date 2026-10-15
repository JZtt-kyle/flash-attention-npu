"""Shared helpers extracted from tests/mla/e2e/window_bsnd/test_flash_attn_npu_v3_mla_window_public.py.

Nonpaged BSND window public/production paths and v12 lifecycle.
"""

from flash_attn_npu_3 import flash_attn_npu_interface as api


def generate(device, lengths, window=(2, 1), causal=False):
    q, k = device["q"], device["k"]
    return api.get_scheduler_metadata(
        q.shape[0],
        q.shape[1],
        k.shape[1],
        q.shape[2],
        1,
        64,
        lengths,
        qkv_dtype=q.dtype,
        headdim_v=512,
        num_splits=1,
        window_size=window,
        causal=causal,
    )


def cached(device, lengths=None, record=None, scale=None, window=(2, 1), causal=False):
    return api.flash_attn_with_kvcache(
        device["q"],
        device["k"],
        device["v"],
        qv=device["qv"],
        cache_seqlens=lengths,
        scheduler_metadata=record,
        softmax_scale=scale,
        num_splits=1,
        window_size=window,
        causal=causal,
        return_softmax_lse=True,
    )
