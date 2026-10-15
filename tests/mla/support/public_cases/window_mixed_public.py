"""Shared helpers extracted from tests/mla/e2e/window_mixed/test_flash_attn_npu_v3_mla_window_mixed_public.py.

v14 mixed window cache: physical KV capacity, prefix/used separation and reuse.
"""

import torch
from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.mla.support.public_cases.causal_mixed_public import prepare as prepare_mixed
from tests.mla.support.mla_window_mixed_cases import spec as spec
from tests.mla.support.public_cases.contract import _forward


def prepare(dtype, s, *, poison=True):
    case = prepare_mixed(dtype, s, poison=poison)
    case["args"].update(window_size=s["window"], causal=s.get("causal", False))
    return case


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
        args["k"].shape[0],
        args["max_seqlen_q"],
        args["k"].shape[1],
        args["q"].shape[1],
        1,
        64,
        lengths,
        qkv_dtype=args["q"].dtype,
        headdim_v=512,
        cu_seqlens_q=args["cu_seqlens_q"],
        causal=args["causal"],
        window_size=args["window_size"],
        num_splits=1,
    )


def call(args, entry="public", *, lse=True):
    args = dict(args)
    if entry == "public":
        assert args.pop("seqused_q", None) is None
        args["k_cache"], args["v_cache"] = args.pop("k"), args.pop("v")
        return api.flash_attn_with_kvcache(**args, num_splits=1, return_softmax_lse=lse)
    left, right = args.pop("window_size")
    if entry == "op":
        return torch.ops.flash_attn_npu_3._mla_window_mixed_forward(
            **args, window_left=left, window_right=right
        )
    args["seqused_k"] = args.pop("cache_seqlens", None)
    return _forward(
        entry, dict(args, window_size_left=left, window_size_right=right, num_splits=1)
    )[:2]
