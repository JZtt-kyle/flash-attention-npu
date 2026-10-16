"""TND allocation on independent PA storage; reuse the logical TND oracle."""

import itertools
import torch
from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.mla.support.public_cases import prefill_paged_bsnd as paged
from tests.mla.support.public_cases import prefill_tnd as tnd
from tests.mla.support.public_cases.prefill_tnd import compact as compact


def make_case(
    dtype="fp16",
    shape=(2, 17, 8, 1057, 2),
    splits=1,
    mask="none",
    strided=False,
    lengths=None,
    page=112,
    q=None,
    used_q=None,
):
    c = paged.make_case(dtype, shape, splits, mask, strided, lengths, page)
    b, sq, h, cap, _ = c["shape"]
    c["q"] = list(q) if q is not None else [sq] * b
    c["k"] = [cap] * b
    c["qlens"] = list(used_q) if used_q is not None else c["q"]
    c["raw"]["q"] = compact(c, c["raw"]["q"], 0)
    value = c["raw"]["q"].npu()
    if strided:
        storage = value.new_empty((2 * sum(c["q"]) + 1, 2 * h + 1, 208))
        view = storage[1::2, 1::2, 8:200]
        view.copy_(value)
        value = view
    c["t"]["q"] = value
    c["lengths"] = dict(
        cu_seqlens_q=torch.tensor(
            [0, *itertools.accumulate(c["q"])], dtype=torch.int32, device="npu"
        ),
        max_seqlen_q=sq,
        seqused_q=None if used_q is None else torch.tensor(used_q, dtype=torch.int32, device="npu"),
        seqused_k=c["lengths"],
    )
    return c


def metadata(c):
    b, sq, h, cap, hk = c["shape"]
    o, lengths = c["options"], c["lengths"]
    if lengths["seqused_q"] is not None:
        return torch.ops.flash_attn_npu_3._mla_prefill_mixed_metadata(
            b,
            sq,
            cap,
            h,
            hk,
            c["t"]["q"].dtype,
            lengths["seqused_k"],
            lengths["cu_seqlens_q"],
            o["num_splits"],
            *o["window_size"],
            o["causal"],
            lengths["seqused_q"],
            c["page"],
        )
    return api.get_scheduler_metadata(
        b,
        sq,
        cap,
        h,
        hk,
        192,
        lengths["seqused_k"],
        qkv_dtype=c["t"]["q"].dtype,
        headdim_v=128,
        page_size=c["page"],
        cu_seqlens_q=lengths["cu_seqlens_q"],
        **{k: v for k, v in o.items() if k != "softmax_scale"},
    )


def invoke(c, record=None):
    o, lengths = c["options"], c["lengths"]
    if lengths["seqused_q"] is not None:
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
            page_table=c["page_table"],
        )
    return api.flash_attn_with_kvcache(
        c["t"]["q"],
        c["t"]["k"],
        c["t"]["v"],
        cu_seqlens_q=lengths["cu_seqlens_q"],
        max_seqlen_q=lengths["max_seqlen_q"],
        cache_seqlens=lengths["seqused_k"],
        page_table=c["page_table"],
        scheduler_metadata=record,
        return_softmax_lse=True,
        **o,
    )


def compare(c, result, record):
    if c["shape"][3] == 0:
        return paged.compare(c, result, record)
    return tnd.compare(c, result, record)
