"""Compact prefill inputs; padding exists only in the independent mathematical oracle."""

import itertools
import torch
from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.mla.support.mla_auto_split_bsnd_reference import references
from tests.common.compare import assert_fa_close


def make_case(
    dtype="fp16",
    q=(7, 0, 17),
    k=(1057, 31, 3),
    heads=8,
    kv_heads=2,
    splits=1,
    mask="none",
    strided=False,
    used_q=None,
    used_k=None,
    maxima=None,
):
    dt = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
    gen = torch.Generator().manual_seed(1729)
    raw = {
        name: torch.randn((sum(q if name == "q" else k), h, d), generator=gen).to(dt)
        for name, h, d in (("q", heads, 192), ("k", kv_heads, 192), ("v", kv_heads, 128))
    }
    tensors = {name: x.npu() for name, x in raw.items()}
    if strided:
        for name, x in list(tensors.items()):
            t, h, d = x.shape
            storage = torch.empty((t * 2 + 1, h * 2 + 1, d + 16), dtype=dt, device="npu")
            view = storage[1 : 1 + 2 * t : 2, 1 : 1 + 2 * h : 2, 8 : 8 + d]
            view.copy_(x)
            tensors[name] = view

    def prefix(a):
        return torch.tensor([0, *itertools.accumulate(a)], dtype=torch.int32, device="npu")

    def used(a):
        return None if a is None else torch.tensor(a, dtype=torch.int32, device="npu")

    maxq, maxk = maxima or (max(q, default=0), max(k, default=0))
    return dict(
        q=list(q),
        k=list(k),
        qlens=list(q if used_q is None else used_q),
        klens=list(k if used_k is None else used_k),
        raw=raw,
        t=tensors,
        shape=(len(q), maxq, heads, maxk, kv_heads),
        lengths=dict(
            cu_seqlens_q=prefix(q),
            cu_seqlens_k=prefix(k),
            max_seqlen_q=maxq,
            max_seqlen_k=maxk,
            seqused_q=used(used_q),
            seqused_k=used(used_k),
        ),
        options=dict(
            num_splits=splits,
            causal=mask == "causal",
            window_size=(7, 2) if mask == "window" else (-1, -1),
            softmax_scale=192**-0.5,
        ),
    )


def metadata_args(c):
    o = c["options"]
    return dict(
        **c["t"],
        **c["lengths"],
        num_splits=o["num_splits"],
        window_left=o["window_size"][0],
        window_right=o["window_size"][1],
        causal=o["causal"],
    )


def metadata(c):
    return torch.ops.flash_attn_npu_3._mla_prefill_tnd_metadata(**metadata_args(c))


def invoke(c, record=None):
    return torch.ops.flash_attn_npu_3._mla_prefill_tnd_forward(
        **metadata_args(c), softmax_scale=c["options"]["softmax_scale"], scheduler_metadata=record
    )


def public(c):
    return api.flash_attn_varlen_func(
        **c["t"], **c["lengths"], **c["options"], return_attn_probs=True
    )


def compact(c, tensor, axis):
    # Oracle BSND -> actual allocation token addresses, preserving all unused slots.
    pieces = [tensor.select(axis, b).narrow(axis, 0, length) for b, length in enumerate(c["q"])]
    shape = list(tensor.shape)
    del shape[axis]
    shape[axis] = 0
    return torch.cat(pieces, dim=axis) if pieces else tensor.new_empty(shape)


def reference(c, record):
    b, sq, h, sk, hk = c["shape"]
    g = h // hk
    r = record.cpu().tolist()
    tail = 24 + 8 * b
    ends = list(
        zip(
            r[tail + 4 : tail + 4 + r[tail + 3]],
            r[tail + 30 : tail + 30 + r[tail + 3]],
            r[tail + 56 : tail + 56 + r[tail + 3]],
        )
    )
    padded = {}
    for name, value in c["raw"].items():
        capacities, seq, heads = (c["q"], sq, h) if name == "q" else (c["k"], sk, hk)
        dst = value.new_zeros((b, seq, heads, value.shape[-1]))
        if value.ndim == 4:
            dst.copy_(value)
        else:
            start = 0
            for i, length in enumerate(capacities):
                dst[i, :length] = value[start : start + length]
                start += length
        padded[name] = (
            dst.reshape(b, seq, hk, g if name == "q" else 1, value.shape[-1])
            .permute(0, 2, 1, 3, 4)
            .reshape(b * hk, seq, g if name == "q" else 1, value.shape[-1])
        )
    o = c["options"]
    refs = references(
        padded,
        [x for x in c["qlens"] for _ in range(hk)],
        [x for x in c["klens"] for _ in range(hk)],
        ends,
        r[13],
        r[14],
        o["causal"],
        o["window_size"],
        o["softmax_scale"],
        num_splits=o["num_splits"],
    )
    for precision in ("high", "low"):
        refs["out_" + precision] = compact(
            c,
            refs["out_" + precision]
            .reshape(b, hk, sq, g, 128)
            .permute(0, 2, 1, 3, 4)
            .reshape(b, sq, h, 128),
            0,
        )
        refs["lse_" + precision] = compact(
            c, refs["lse_" + precision].reshape(b, h, sq).permute(1, 0, 2), 1
        )
        capacity = o["num_splits"] or 26
        refs["accum_" + precision] = compact(
            c, refs["accum_" + precision].reshape(capacity, b, h, sq, 128).permute(0, 2, 1, 3, 4), 2
        )
        refs["accum_lse_" + precision] = compact(
            c, refs["accum_lse_" + precision].reshape(capacity, b, h, sq).permute(0, 2, 1, 3), 2
        )
        if not sum(c["k"]):
            refs["lse_" + precision].fill_(torch.inf)
    return refs


def compare(c, result, record):
    refs = reference(c, record)
    for i, name in enumerate(("out", "lse", "accum", "accum_lse")[: len(result)]):
        actual = result[i].cpu()
        if i >= 2 and c["options"]["num_splits"] == 1:
            assert actual.numel() == 0
        else:
            assert actual.shape == refs[name + "_high"].shape
            assert_fa_close(
                actual, refs[name + "_high"], refs[name + "_low"], name="prefill-tnd-" + name
            )
    return refs
