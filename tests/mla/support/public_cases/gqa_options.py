"""Five-layout scheduling options over the established input and math fixtures."""

import hashlib
import itertools
import os
from pathlib import Path

import torch
from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.common.compare import assert_fa_close
from tests.mla.support.public_cases import (
    softcap_bsnd,
    softcap_tnd,
    softcap_mixed,
    softcap_paged_bsnd,
    softcap_paged_tnd,
)
from tests.mla.support.public_cases.contract import metadata_raw_args
from tests.mla.support.public_cases.prefill_tnd import compact
from tests.mla.support.mla_auto_split_bsnd_reference import references

LAYOUTS = dict(
    bsnd=softcap_bsnd,
    tnd=softcap_tnd,
    mixed=softcap_mixed,
    paged_bsnd=softcap_paged_bsnd,
    paged_tnd=softcap_paged_tnd,
)
# Each tuple has a distinct option/split/mask, allocation or numerical boundary.
CASES = [
    dict(id="unpacked-single-none", pack=False, margin=0, splits=1),
    dict(
        id="unpacked-explicit-window",
        pack=False,
        margin=0,
        splits=3,
        mask="window",
        strided=True,
        q=(7, 0, 3),
        k=(513, 31, 3),
        used_k=(511, 0, 2),
        repeat_pages=True,
    ),
    dict(
        id="unpacked-auto-causal",
        pack=False,
        margin=1,
        splits=0,
        mask="causal",
        q=(17, 3),
        k=(513, 3),
    ),
    dict(
        id="unpacked-auto-head-distribution",
        pack=False,
        margin=1,
        splits=0,
        mask="window",
        q=(3, 1),
        k=(4097, 1057),
        window=(256, 2),
    ),
    dict(id="packed-single-window", pack=True, margin=1, splits=1, mask="window"),
    dict(
        id="packed-explicit-causal",
        pack=True,
        margin=2,
        splits=3,
        mask="causal",
        q=(17, 3),
        k=(3, 1),
    ),
    dict(
        id="default-auto-window",
        pack=None,
        margin=1,
        splits=0,
        mask="window",
        q=(33, 3),
        k=(1057, 513),
        window=(256, 2),
    ),
    dict(id="unpacked-softcap-auto", pack=False, margin=2, splits=0, mask="window", cap=15),
    dict(
        id="packed-softcap-explicit",
        pack=True,
        margin=1,
        splits=3,
        mask="causal",
        cap=30,
        scale=-0.23,
    ),
    dict(
        id="max-splits-zero-scale",
        pack=False,
        margin=1,
        splits=256,
        mask="window",
        scale=0,
        q=(1,),
        k=(33,),
        heads=2,
        page=16,
    ),
    dict(
        id="long-kv",
        pack=False,
        margin=1,
        splits=3,
        mask="causal",
        q=(1, 1),
        k=(16384, 513),
        heads=2,
        page=1024,
    ),
    dict(
        id="query-tile-tail",
        pack=False,
        margin=1,
        splits=3,
        mask="window",
        q=(513, 3),
        k=(33, 1),
        heads=4,
        page=16,
    ),
]
DAILY = {
    "unpacked-explicit-window",
    "unpacked-auto-causal",
    "unpacked-auto-head-distribution",
    "packed-single-window",
}
REPRESENTATIVE = DAILY | {
    "unpacked-softcap-auto",
    "long-kv",
    "max-splits-zero-scale",
    "query-tile-tail",
}


def make_case(layout, representation, dtype="fp16", **spec):
    spec = dict(spec)
    q, k = spec.get("q", (7, 3)), spec.get("k", (65, 7))
    spec.setdefault("cap", 0)
    spec.setdefault("scale", (192 if representation == "prefill" else 576) ** -0.5)
    if layout in ("bsnd", "paged_bsnd"):
        spec.setdefault(
            "shape", (len(q), max(q, default=0), spec.get("heads", 8), max(k, default=0))
        )
        spec.setdefault("lengths", spec.get("used_k", k))
    c = LAYOUTS[layout].make_case(representation, dtype, **spec)
    c["layout"] = layout
    c["shape"] = (*c["shape"][:4], c["t"]["k"].shape[-2])
    c["options"].update(pack_gqa=spec.get("pack", False), sm_margin=spec.get("margin", 0))
    return c


def metadata(c, *, entry="public", geometry=False):
    b, sq, h, sk, hk = c["shape"]
    o = c["options"]
    lengths = c["lengths"]
    cuq = lengths.get("cu_seqlens_q") if isinstance(lengths, dict) else None
    usedk = lengths["seqused_k"] if isinstance(lengths, dict) else lengths
    if geometry or c["layout"] == "tnd":
        # Varlen public metadata describes mixed cache allocations. The oracle
        # needs only the scheduling endpoints; effective prefixes retain the
        # original allocation solely in the independent math mapping below.
        usedk = torch.tensor(c["klens"], dtype=torch.int32, device=c["t"]["q"].device)
        if cuq is not None:
            cuq = torch.tensor(
                [0, *itertools.accumulate(c["qlens"])], dtype=torch.int32, device=c["t"]["q"].device
            )
    args = dict(
        batch_size=b,
        max_seqlen_q=sq,
        max_seqlen_k=sk,
        num_heads_q=h,
        num_heads_kv=hk,
        headdim=192 if c["representation"] == "prefill" else 64,
        headdim_v=c["t"]["v"].shape[-1],
        cache_seqlens=usedk,
        qkv_dtype=c["t"]["q"].dtype,
        cu_seqlens_q=cuq,
        page_size=c.get("page"),
        has_softcap=o["softcap"] > 0,
        **{
            name: o[name]
            for name in ("num_splits", "causal", "window_size", "pack_gqa", "sm_margin")
        },
    )
    if entry == "public":
        return api.get_scheduler_metadata(**args)
    if entry == "op":
        return torch.ops.flash_attn_npu_3._mla_gqa_metadata(
            b,
            sq,
            sk,
            h,
            hk,
            args["headdim_v"],
            args["qkv_dtype"],
            usedk,
            cuq,
            o["num_splits"],
            *o["window_size"],
            o["causal"],
            c.get("page", 0),
            o["softcap"] > 0,
            o["pack_gqa"],
            o["sm_margin"],
            lengths.get("seqused_q") if isinstance(lengths, dict) else c.get("usedq"),
        )
    args["window_size_left"], args["window_size_right"] = args.pop("window_size")
    return api.flash_attn_npu_3.get_scheduler_metadata(*metadata_raw_args(args).values())


def invoke(c, record=None, *, entry="public"):
    if entry != "op":
        return LAYOUTS[c["layout"]].invoke(c, record, entry=entry)
    t, o = c["t"], c["options"]
    lengths = c["lengths"] if isinstance(c["lengths"], dict) else dict(seqused_k=c["lengths"])
    return torch.ops.flash_attn_npu_3._mla_gqa_forward(
        t["q"],
        t["k"],
        t["v"],
        t.get("qv"),
        o["num_splits"],
        *o["window_size"],
        o["causal"],
        o["softmax_scale"],
        lengths.get("seqused_q", c.get("usedq")),
        lengths.get("seqused_k"),
        record,
        lengths.get("cu_seqlens_q"),
        lengths.get("cu_seqlens_k"),
        lengths.get("max_seqlen_q", 0),
        lengths.get("max_seqlen_k", 0),
        c.get("page_table"),
        o["softcap"],
        o["pack_gqa"],
        o["sm_margin"],
    )


def reference(c, record):
    b, sq, h, sk, hk = c["shape"]
    o, r = c["options"], record.cpu().tolist()
    assert r[1:3] == [47, 47] and r[5] == 0
    groups = h if o["pack_gqa"] is False else hk
    g = h // groups
    padded = {}
    for name, value in c["raw"].items():
        query = name in ("q", "qv")
        seq, heads = (sq, h) if query else (sk, hk)
        if value.ndim == 3:
            dst = value.new_zeros((b, seq, heads, value.shape[-1]))
            start = 0
            for i, count in enumerate(c["q"] if query else c["k"]):
                dst[i, :count] = value[start : start + count]
                start += count
        else:
            dst = value
        if not query:
            dst = dst.repeat_interleave(groups // hk, dim=2)
        padded[name] = (
            dst.reshape(b, seq, groups, g if query else 1, value.shape[-1])
            .permute(0, 2, 1, 3, 4)
            .reshape(b * groups, seq, g if query else 1, value.shape[-1])
        )
    tail = 32 + 8 * b
    count = r[tail + 3]
    ends = list(
        zip(
            r[tail + 4 : tail + 4 + count],
            r[tail + 30 : tail + 30 + count],
            r[tail + 56 : tail + 56 + count],
        )
    )
    refs = references(
        padded,
        [v for v in c["qlens"] for _ in range(groups)],
        [v for v in c["klens"] for _ in range(groups)],
        ends,
        r[25],
        r[26],
        o["causal"],
        o["window_size"],
        o["softmax_scale"],
        num_splits=o["num_splits"],
        softcap=o["softcap"],
    )
    refs = dict(refs)
    dim, capacity = c["t"]["v"].shape[-1], o["num_splits"] or 26
    for p in ("high", "low"):
        refs["out_" + p] = (
            refs["out_" + p]
            .reshape(b, groups, sq, g, dim)
            .permute(0, 2, 1, 3, 4)
            .reshape(b, sq, h, dim)
        )
        refs["lse_" + p] = refs["lse_" + p].reshape(b, h, sq)
        refs["accum_" + p] = refs["accum_" + p].reshape(capacity, b, h, sq, dim)
        refs["accum_lse_" + p] = refs["accum_lse_" + p].reshape(capacity, b, h, sq)
        if c["t"]["q"].ndim == 3:
            refs["out_" + p] = compact(c, refs["out_" + p], 0)
            refs["lse_" + p] = compact(c, refs["lse_" + p].permute(1, 0, 2), 1)
            refs["accum_" + p] = compact(c, refs["accum_" + p].permute(0, 2, 1, 3, 4), 2)
            refs["accum_lse_" + p] = compact(c, refs["accum_lse_" + p].permute(0, 2, 1, 3), 2)
        if sk == 0 and c.get("page"):
            refs["lse_" + p] = refs["lse_" + p].clone().fill_(-torch.inf)
        if c["layout"] == "tnd" and not sum(c["k"]):
            refs["lse_" + p] = refs["lse_" + p].clone().fill_(torch.inf)
    return refs


def compare(c, result, record):
    refs = reference(c, record)
    actual = [x.cpu() for x in result]
    directory = os.environ.get("MLA_VALIDATION_RESULT")
    if directory:
        key = hashlib.sha256(
            (
                os.environ.get("PYTEST_CURRENT_TEST", "")
                + str(c["options"])
                + c["layout"]
                + c["representation"]
                + str(c["t"]["q"].dtype)
                + str(c["shape"])
            ).encode()
        ).hexdigest()[:20]
        torch.save(
            dict(
                raw=c["raw"],
                record=record.cpu(),
                options=c["options"],
                qlens=c["qlens"],
                klens=c["klens"],
                actual=actual,
                refs=refs,
            ),
            Path(directory) / ("gqa-" + key + ".pt"),
        )
    for i, name in enumerate(("out", "lse", "accum", "accum_lse")[: len(actual)]):
        if i >= 2 and c["options"]["num_splits"] == 1:
            assert actual[i].numel() == 0 and actual[i].dtype == torch.get_default_dtype()
        else:
            assert_fa_close(
                actual[i],
                refs[name + "_high"],
                refs[name + "_low"],
                softcap=c["options"]["softcap"],
                name="gqa-" + name,
            )
    return refs
