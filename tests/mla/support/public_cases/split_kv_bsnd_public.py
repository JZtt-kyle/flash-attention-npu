"""Shared helpers extracted from tests/mla/e2e/split_kv_bsnd/test_public.py.

Public BSND explicit split, allocation/used, strides, empty slots and accumulators.
"""

import torch
from flash_attn_npu_3 import flash_attn_npu_interface as api


def make_case(
    dtype_name="fp16", shape=(2, 7, 2, 33), splits=8, mask="window", strided=False, used=False
):
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
    b, q, h, k = shape
    g = torch.Generator().manual_seed(20261014)
    raw = {
        name: torch.randn(
            (b, q if name in ("q", "qv") else k, h if name in ("q", "qv") else 1, dim), generator=g
        ).to(dtype)
        for name, dim in (("q", 64), ("qv", 512), ("k", 64), ("v", 512))
    }
    qlens = [q] * b
    klens = [k] * b
    if b > 1:
        klens[-1] = 0 if mask == "causal" else min(k, 2)
    if used and b > 1:
        qlens[-1] = max(0, q - 3)
    tensors = {}
    for name, t in raw.items():
        if strided:
            # Independent head/sequence/batch strides and nonzero storage offsets.
            dims = list(t.shape)
            dims[0] += 1
            dims[1] = dims[1] * 2 + 1
            dims[2] *= 2
            base = torch.empty(dims, dtype=dtype, device="npu")
            dev = base[1:, : 2 * t.shape[1] : 2, ::2, :]
            dev.copy_(t)
        else:
            dev = t.npu()
        tensors[name] = dev
    options = dict(
        num_splits=splits,
        causal=mask == "causal",
        window_size=(2, 1) if mask == "window" else (-1, -1),
        softmax_scale=0.125,
    )
    lengths = torch.tensor(klens, dtype=torch.int32, device="npu")
    usedq = torch.tensor(qlens, dtype=torch.int32, device="npu") if used else None
    return dict(
        raw=raw,
        t=tensors,
        options=options,
        lengths=lengths,
        usedq=usedq,
        qlens=qlens,
        klens=klens,
        shape=shape,
    )


def metadata(c):
    b, q, h, k = c["shape"]
    return api.get_scheduler_metadata(
        b,
        q,
        k,
        h,
        1,
        64,
        c["lengths"],
        qkv_dtype=c["t"]["q"].dtype,
        headdim_v=512,
        num_splits=c["options"]["num_splits"],
        causal=c["options"]["causal"],
        window_size=c["options"]["window_size"],
    )


def invoke(c, record=None):
    t = c["t"]
    o = c["options"]
    if c["usedq"] is not None:
        return torch.ops.flash_attn_npu_3._mla_split_bsnd_forward(
            t["q"],
            t["k"],
            t["v"],
            t["qv"],
            o["num_splits"],
            *o["window_size"],
            o["causal"],
            o["softmax_scale"],
            c["lengths"],
            c["usedq"],
            record,
        )
    return api.flash_attn_with_kvcache(
        t["q"],
        t["k"],
        t["v"],
        qv=t["qv"],
        cache_seqlens=c["lengths"],
        scheduler_metadata=record,
        return_softmax_lse=True,
        **o,
    )
