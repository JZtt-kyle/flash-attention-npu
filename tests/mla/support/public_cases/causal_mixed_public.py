"""Shared helpers extracted from tests/mla/e2e/causal_mixed/test_flash_attn_npu_v3_mla_causal_mixed_public.py.

v9 mixed causal cache: physical KV capacity, prefix/used separation and reuse.
"""

import itertools
import torch
from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.mla.support.mla_validation_policy import public_cases
from tests.mla.support.components.bsnd_stride import make_view, strides_for
from tests.mla.support.public_cases.causal_tnd_public import compare as compare, effective
from tests.mla.support.public_cases.contract import _forward
from tests.mla.support.components.used_schedule import used_case, execute
from tests.mla.support.components.causal_attention import make_case


def spec(q=(7, 0, 3), k=(3, 0, 17), capacity=33, uq=None, h=16, implicit_k=False, **kw):
    return dict(
        q=list(q),
        kv=[capacity] * len(q),
        uq=uq,
        uk=None if implicit_k else list(k),
        heads=h,
        capacity=capacity,
        **kw,
    )


REPRESENTATIVES = [
    spec(),
    spec(q=(257, 3), k=(257, 513), capacity=513, h=1),
    spec(uq=[5, 0, 3], stride="independent"),
]

PERFORMANCE_CASES = public_cases(
    [
        spec(q=(3, 0, 1), k=(17, 0, 33), capacity=64, h=128),
        spec(q=(1, 7, 2), k=(1, 513, 31), capacity=1025),
        spec(q=(0, 3, 1, 0, 7), k=(33, 511, 0, 513, 1025), capacity=2048, h=128),
        spec(q=(1, 2, 3), k=(16384, 8193, 17), capacity=16384, h=8),
        spec(q=(257, 1, 0, 3), k=(33, 513, 17, 1), capacity=513, h=1),
        spec(q=(3, 0) * 21, k=(0, 513) * 21, capacity=513, h=128),
        *REPRESENTATIVES[:2],
        spec(stride="independent"),
        spec(stride="ntd", h=128),
        spec(q=(5, 1), k=(17, 7), capacity=33, h=5, stride="combined_65536"),
        spec(q=(513, 257), k=(257, 129), capacity=513, h=3),
    ],
    group="mixed",
)


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
        physical = tensor.unsqueeze(0) if query else tensor.view(len(s["q"]), s["capacity"], 1, dim)
        if s.get("stride"):
            profile = "bnsd" if s["stride"] == "ntd" else s["stride"]
            view, logical, checks = make_view(
                physical,
                strides_for(physical.shape, profile, name),
                offset,
                overlap=profile in ("broadcast", "overlap"),
            )
            device[name] = view[0] if query else view
            raw[name] = logical.reshape(tensor.shape)
            guards += checks
        else:
            raw[name], device[name] = tensor, tensor.npu() if query else physical.npu()
    args = dict(
        device,
        cu_seqlens_q=torch.tensor(
            list(itertools.accumulate(s["q"], initial=0)), dtype=torch.int32, device="npu"
        ),
        max_seqlen_q=s.get("sq", max(s["q"], default=0)),
        cache_seqlens=None
        if s["uk"] is None
        else torch.tensor(s["uk"], dtype=torch.int32, device="npu"),
        seqused_q=None
        if s["uq"] is None
        else torch.tensor(s["uq"], dtype=torch.int32, device="npu"),
        softmax_scale=s.get("scale"),
    )
    return dict(spec=s, raw=raw, args=args, guards=guards)


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
        causal=True,
        num_splits=1,
    )


def call(args, entry="public", *, lse=True):
    args = dict(args)
    if entry == "public":
        assert args.pop("seqused_q", None) is None
        args["k_cache"], args["v_cache"] = args.pop("k"), args.pop("v")
        return api.flash_attn_with_kvcache(
            **args, causal=True, num_splits=1, return_softmax_lse=lse
        )
    if entry == "op":
        return torch.ops.flash_attn_npu_3._mla_causal_mixed_forward(**args)
    args["seqused_k"] = args.pop("cache_seqlens", None)
    return _forward(entry, dict(args, causal=True, num_splits=1))[:2]


def source_tile(probes, case, oracle):
    s = case["spec"]
    details = make_case(probes, dict(shape=(1, 1, 1, 1)))["details"]
    item = used_case(
        2,
        s["q"],
        s["kv"],
        s["uq"],
        s["uk"],
        heads=s["heads"],
        aic=int(details[2]),
        ratio=2,
        sq=case["args"]["max_seqlen_q"],
        sk=s["capacity"],
    )
    row = execute(oracle, [item])[0]
    assert row[0] == 0
    return row[1:3]
