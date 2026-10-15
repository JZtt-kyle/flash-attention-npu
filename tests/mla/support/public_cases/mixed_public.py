"""Shared helpers extracted from tests/mla/e2e/mixed/test_flash_attn_npu_v3_mla_mixed_public.py.

Public mixed kvcache: v4 reuse, source accuracy and stream lifetime.
"""

import itertools
import pytest
import torch
from flash_attn_npu_3 import flash_attn_npu_interface as api


def inputs(dtype=torch.float16, qs=(3, 0, 1), ks=(17, 0, 33), capacity=64, heads=4):
    def offset(values):
        return torch.tensor([123, *values, 456], dtype=torch.int32, device="npu")[1:-1]

    return dict(
        q=torch.zeros((sum(qs), heads, 64), dtype=dtype, device="npu"),
        qv=torch.zeros((sum(qs), heads, 512), dtype=dtype, device="npu"),
        k=torch.zeros((len(qs), capacity, 1, 64), dtype=dtype, device="npu"),
        v=torch.zeros((len(qs), capacity, 1, 512), dtype=dtype, device="npu"),
        cu_seqlens_q=offset(list(itertools.accumulate(qs, initial=0))),
        cache_seqlens=offset(ks),
        max_seqlen_q=max(qs, default=0) + 3,
    )


def metadata(args):
    lengths = args.get("cache_seqlens")
    if lengths is None or isinstance(lengths, int):
        lengths = torch.full(
            (args["k"].shape[0],),
            args["k"].shape[1] if lengths is None else lengths,
            dtype=torch.int32,
            device=args["q"].device,
        )
    return api.get_scheduler_metadata(
        batch_size=args["k"].shape[0],
        max_seqlen_q=args["max_seqlen_q"],
        max_seqlen_k=args["k"].shape[1],
        num_heads_q=args["q"].shape[1],
        num_heads_kv=1,
        headdim=64,
        headdim_v=512,
        qkv_dtype=args["q"].dtype,
        cu_seqlens_q=args["cu_seqlens_q"],
        cache_seqlens=lengths,
        num_splits=1,
    )


def public(args, lse=True):
    args = dict(args)
    args["k_cache"], args["v_cache"] = args.pop("k"), args.pop("v")
    args.setdefault("num_splits", 1)
    return api.flash_attn_with_kvcache(**args, return_softmax_lse=lse)


@pytest.fixture(params=[torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def dtype(request):
    return request.param
