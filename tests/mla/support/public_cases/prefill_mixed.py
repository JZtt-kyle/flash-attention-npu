"""Mixed prefill inputs and public cache/metadata adapters using the TND oracle."""

import torch
from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.mla.support.public_cases import prefill_tnd as tnd
from tests.mla.support.public_cases.prefill_tnd import compare as compare, compact as compact


def make_case(*args, **kwargs):
    c = tnd.make_case(*args, **kwargs)
    b, _, _, sk, hk = c["shape"]
    for name in ("k", "v"):
        src = c["raw"][name]
        raw = src.new_full((b, sk, hk, src.shape[-1]), float("nan"))
        offset = 0
        for i, count in enumerate(c["k"]):
            raw[i, :count] = src[offset : offset + count]
            offset += count
        c["raw"][name] = raw
        value = raw.npu()
        if kwargs.get("strided"):
            d = value.shape[-1]
            storage = torch.empty(
                (b * 2 + 1, sk * 2 + 1, hk * 2 + 1, d + 16), dtype=value.dtype, device="npu"
            )
            view = storage[1 : 1 + 2 * b : 2, 1 : 1 + 2 * sk : 2, 1 : 1 + 2 * hk : 2, 8 : 8 + d]
            view.copy_(value)
            value = view
        c["t"][name] = value
    c["k"] = [sk] * b
    c["lengths"].pop("cu_seqlens_k")
    c["lengths"].pop("max_seqlen_k")
    c["lengths"]["seqused_k"] = torch.tensor(c["klens"], dtype=torch.int32, device="npu")
    return c


def metadata(c):
    b, sq, h, sk, hk = c["shape"]
    o, lengths = c["options"], c["lengths"]
    return torch.ops.flash_attn_npu_3._mla_prefill_mixed_metadata(
        b,
        sq,
        sk,
        h,
        hk,
        c["t"]["q"].dtype,
        lengths["seqused_k"],
        lengths["cu_seqlens_q"],
        o["num_splits"],
        *o["window_size"],
        o["causal"],
        lengths["seqused_q"],
    )


def public_metadata(c):
    b, sq, h, sk, hk = c["shape"]
    o, lengths = c["options"], c["lengths"]
    return api.get_scheduler_metadata(
        b,
        sq,
        sk,
        h,
        hk,
        192,
        lengths["seqused_k"],
        c["t"]["q"].dtype,
        headdim_v=128,
        cu_seqlens_q=lengths["cu_seqlens_q"],
        num_splits=o["num_splits"],
        causal=o["causal"],
        window_size=o["window_size"],
    )


def invoke(c, record=None):
    lengths, o = c["lengths"], c["options"]
    return torch.ops.flash_attn_npu_3._mla_prefill_mixed_forward(
        **c["t"],
        cu_seqlens_q=lengths["cu_seqlens_q"],
        max_seqlen_q=lengths["max_seqlen_q"],
        num_splits=o["num_splits"],
        window_left=o["window_size"][0],
        window_right=o["window_size"][1],
        causal=o["causal"],
        softmax_scale=o["softmax_scale"],
        cache_seqlens=lengths["seqused_k"],
        seqused_q=lengths["seqused_q"],
        scheduler_metadata=record,
    )


def public(c, record=None):
    assert c["lengths"]["seqused_q"] is None
    t, lengths = c["t"], c["lengths"]
    return api.flash_attn_with_kvcache(
        t["q"],
        t["k"],
        t["v"],
        cache_seqlens=lengths["seqused_k"],
        cu_seqlens_q=lengths["cu_seqlens_q"],
        max_seqlen_q=lengths["max_seqlen_q"],
        **c["options"],
        scheduler_metadata=record,
        return_softmax_lse=True,
    )
