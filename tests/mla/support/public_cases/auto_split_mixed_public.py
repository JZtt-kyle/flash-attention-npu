"""Shared helpers extracted from tests/mla/e2e/auto_split_mixed/test_public.py.

Mixed native automatic split, public v22 reuse and real static FP32 slots.
"""

import copy
import torch
from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.mla.support.public_cases.window_mixed_public import prepare as prepare_window
from tests.mla.support.public_cases.auto_split_tnd_public import reference as reference_tnd
from tests.mla.support.public_cases.auto_split_bsnd_public import geometry
from tests.mla.support.public_cases.causal_mixed_public import effective
from tests.mla.support.public_cases.contract import _forward


def prepare(dtype, s, *, poison=True):
    c = prepare_window(dtype, copy.deepcopy(s), poison=poison)
    c["args"]["num_splits"] = s["splits"]
    return c


def metadata(c, entry="public"):
    a = c["args"]
    lengths = a["cache_seqlens"]
    if lengths is None or isinstance(lengths, int):
        lengths = torch.full(
            (a["k"].shape[0],),
            a["k"].shape[1] if lengths is None else lengths,
            device=a["q"].device,
            dtype=torch.int32,
        )
    if entry == "op":
        return torch.ops.flash_attn_npu_3._mla_split_mixed_metadata(
            a["k"].shape[0],
            a["max_seqlen_q"],
            a["k"].shape[1],
            a["q"].shape[1],
            a["q"].dtype,
            lengths,
            a["cu_seqlens_q"],
            *a["window_size"],
            a["causal"],
            a["num_splits"],
        )
    from tests.test_flash_attn_npu_v3_metadata_signature import metadata as generate

    return generate(
        entry,
        dict(
            batch_size=a["k"].shape[0],
            max_seqlen_q=a["max_seqlen_q"],
            max_seqlen_k=a["k"].shape[1],
            num_heads_q=a["q"].shape[1],
            num_heads_kv=1,
            headdim=64,
            headdim_v=512,
            qkv_dtype=a["q"].dtype,
            cache_seqlens=lengths,
            cu_seqlens_q=a["cu_seqlens_q"],
            window_size=a["window_size"],
            causal=a["causal"],
            num_splits=a["num_splits"],
        ),
    )


def call(args, entry="public", record=None, *, lse=True):
    a = dict(args)
    if record is not None:
        a["scheduler_metadata"] = record
    a.setdefault("scheduler_metadata", None)
    if entry == "public":
        assert a.pop("seqused_q", None) is None
        a["k_cache"], a["v_cache"] = a.pop("k"), a.pop("v")
        return api.flash_attn_with_kvcache(**a, return_softmax_lse=lse)
    left, right = a.pop("window_size")
    if entry == "op":
        return torch.ops.flash_attn_npu_3._mla_split_mixed_forward(
            **a, window_left=left, window_right=right
        )
    a["seqused_k"] = a.pop("cache_seqlens")
    return _forward(entry, dict(a, window_size_left=left, window_size_right=right))


def schedule_metadata(c):
    # Internal Q-used always generates its own record. For the independent
    # reference only, an equivalent compact used-Q prefix recovers native
    # endpoints through the public producer; it is never passed to forward.
    # Original allocation origins remain in c and determine all output checks.
    if c["spec"]["uq"] is None:
        return metadata(c)
    qs, _ = effective(c["spec"])
    prefix = [0]
    for q in qs:
        prefix.append(prefix[-1] + q)
    a = dict(c["args"], cu_seqlens_q=torch.tensor(prefix, dtype=torch.int32, device="npu"))
    return metadata(dict(c, args=a))


def reference(c, tile=None):
    if tile is None:
        tile = geometry(schedule_metadata(c).cpu(), len(c["spec"]["q"]))
    mapped = dict(c, args=dict(c["args"], max_seqlen_k=c["spec"]["capacity"]))
    return reference_tnd(mapped, tile)
