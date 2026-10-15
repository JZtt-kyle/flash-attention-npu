"""Shared helpers extracted from tests/mla/e2e/window_tnd/test_flash_attn_npu_v3_mla_window_tnd_public.py.

TND window varlen: allocation gaps, source/math, public boundaries and lifetime.
"""

from concurrent.futures import ThreadPoolExecutor
import itertools
import torch
from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.mla.support.mla_ref import mla_reference_pair
from tests.mla.support.components.bsnd_stride import make_view, strides_for
from tests.mla.support.public_cases.contract import _forward


def effective(s):
    return s["q"] if s["uq"] is None else s["uq"], s["kv"] if s["uk"] is None else s["uk"]


def prepare(dtype, s, *, poison=True):
    gen = torch.Generator().manual_seed(20261013)
    raw, device, guards = {}, {}, []
    uq, uk = effective(s)
    for name, dim, offset in (("q", 64, 5), ("qv", 512, 11), ("k", 64, 19), ("v", 512, 29)):
        query = name in ("q", "qv")
        allocation, used = (s["q"], uq) if query else (s["kv"], uk)
        tensor = torch.randn((sum(allocation), s["heads"] if query else 1, dim), generator=gen).to(
            dtype
        )
        if query:
            tensor.mul_(s.get("amplitude", 1))
        start = 0
        if poison:
            for capacity, length in zip(allocation, used):
                tensor[start + length : start + capacity] = torch.nan
                start += capacity
        if s.get("stride"):
            shape = (1, *tensor.shape)
            profile = "bnsd" if s["stride"] == "ntd" else s["stride"]
            view, logical, checks = make_view(
                tensor.unsqueeze(0),
                strides_for(shape, profile, name),
                offset,
                overlap=profile in ("broadcast", "overlap"),
            )
            device[name], raw[name] = view[0], logical[0]
            guards += checks
        else:
            raw[name], device[name] = tensor, tensor.npu()
    args = dict(
        device,
        cu_seqlens_q=torch.tensor(
            list(itertools.accumulate(s["q"], initial=0)), dtype=torch.int32, device="npu"
        ),
        cu_seqlens_k=torch.tensor(
            list(itertools.accumulate(s["kv"], initial=0)), dtype=torch.int32, device="npu"
        ),
        max_seqlen_q=s.get("sq", max(s["q"], default=0)),
        max_seqlen_k=s.get("sk", max(s["kv"], default=0)),
        seqused_q=None
        if s["uq"] is None
        else torch.tensor(s["uq"], dtype=torch.int32, device="npu"),
        seqused_k=None
        if s["uk"] is None
        else torch.tensor(s["uk"], dtype=torch.int32, device="npu"),
        softmax_scale=s.get("scale"),
        window_size=s["window"],
        causal=s.get("causal", False),
    )
    return dict(spec=s, raw=raw, args=args, guards=guards)


def call(args, entry="public"):
    if entry == "public":
        return api.flash_attn_varlen_func(**args, num_splits=1, return_attn_probs=True)
    if entry == "op":
        options = dict(args)
        left, right = options.pop("window_size")
        return torch.ops.flash_attn_npu_3._mla_window_tnd_forward(
            **options, window_left=left, window_right=right
        )
    options = dict(args, num_splits=1)
    options["window_size_left"], options["window_size_right"] = options.pop("window_size")
    return _forward(entry, options)[:2]


def reference_pairs(case, window):
    """Original allocation/used mapping, bounded CPU-only sample parallelism."""
    spec, raw = case["spec"], case["raw"]
    uq, uk = effective(spec)
    starts_q = list(itertools.accumulate(spec["q"], initial=0))
    starts_k = list(itertools.accumulate(spec["kv"], initial=0))

    def one(sample):
        if not uq[sample]:
            return None
        logical = {}
        for name, tensor in raw.items():
            query = name in ("q", "qv")
            start = starts_q[sample] if query else starts_k[sample]
            length = uq[sample] if query else uk[sample]
            logical[name] = tensor[start : start + length].unsqueeze(0).contiguous()
        refs = mla_reference_pair(
            **logical,
            softmax_scale=spec.get("scale"),
            causal=window.kind == 1,
            window_size=(window.left, window.right) if window.kind == 2 else (-1, -1),
        )
        return logical, refs

    with ThreadPoolExecutor(max_workers=min(4, max(1, len(uq)))) as pool:
        return list(pool.map(one, range(len(uq))))
