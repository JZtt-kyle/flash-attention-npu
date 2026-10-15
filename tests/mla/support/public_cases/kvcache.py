"""Shared helpers extracted from tests/mla/e2e/dense/test_flash_attn_npu_v3_mla_kvcache.py.

Public dense MLA metadata and read-only cache: contracts, accuracy and lifetime.
"""

import torch
from flash_attn_npu_3 import flash_attn_npu_interface as api


def generate(device, lengths=None):
    q, k = device["q"], device["k"]
    if lengths is None:
        lengths = torch.full((q.shape[0],), k.shape[1], dtype=torch.int32, device=q.device)
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
    )


def cached(device, metadata=None, lengths=None, scale=None, return_lse=True):
    return api.flash_attn_with_kvcache(
        device["q"],
        device["k"],
        device["v"],
        qv=device["qv"],
        cache_seqlens=lengths,
        scheduler_metadata=metadata,
        softmax_scale=scale,
        num_splits=1,
        return_softmax_lse=return_lse,
    )
