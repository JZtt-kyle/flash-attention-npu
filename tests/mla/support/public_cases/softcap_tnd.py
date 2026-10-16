"""TND softcap adapter with independent allocation-to-math mapping."""

import hashlib
import os
from pathlib import Path
import torch
from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.common.compare import assert_fa_close
from tests.mla.support.public_cases.prefill_tnd import make_case as prefill_case, compact
from tests.mla.support.public_cases.window_tnd_public import prepare as compressed_case
from tests.mla.support.mla_auto_split_bsnd_reference import references
from tests.mla.support.public_cases.contract import _forward

CASES = [
    dict(id="none-single", splits=1),
    dict(id="causal-single-empty-rows", splits=1, mask="causal", q=(17, 3), k=(3, 1)),
    dict(id="window-single-negative", splits=1, mask="window", scale=-0.23, cap=30),
    dict(id="none-explicit", splits=3, q=(7, 0, 17), k=(1057, 31, 3)),
    dict(id="causal-empty-splits", splits=8, mask="causal", q=(33, 2), k=(3, 0)),
    dict(
        id="window-explicit-used-stride",
        splits=4,
        mask="window",
        q=(7, 0, 17),
        k=(513, 31, 3),
        used_q=(3, 0, 1),
        used_k=(511, 0, 2),
        strided=True,
        maxima=(33, 1057),
    ),
    dict(id="none-auto", splits=0, q=(3, 1), k=(1057, 513)),
    dict(id="causal-auto", splits=0, mask="causal", q=(17, 1), k=(513, 3), used_k=(513, 0)),
    dict(id="window-auto", splits=0, mask="window", q=(17, 3), k=(1057, 513), window=(256, 2)),
    dict(id="zero-scale-single", splits=1, scale=0, window=(1, 0)),
    dict(id="zero-scale-multi", splits=2, scale=0, window=(1, 0)),
    dict(id="near-zero", splits=1, cap=1e8),
    dict(id="small-cap", splits=0, cap=1e-4),
    dict(id="max-splits", splits=256, mask="window", q=(1,), k=(33,), heads=2),
    dict(id="long-kv", splits=3, mask="causal", q=(1, 1), k=(16384, 513), heads=2),
    dict(id="q-tile-tail", splits=3, mask="window", q=(129, 3), k=(513, 1), heads=4),
    dict(id="zero-used", splits=0, q=(7, 3), k=(33, 7), used_q=(0, 0), used_k=(0, 0)),
]
DAILY = {"none-single", "window-explicit-used-stride", "causal-auto"}
REPRESENTATIVE = DAILY | {"zero-scale-multi", "long-kv"}


def make_case(representation, dtype="fp16", **spec):
    q, k = spec.get("q", (7, 0, 3)), spec.get("k", (65, 7, 3))
    heads = spec.get("heads", 8)
    splits = spec.get("splits", 1)
    mask = spec.get("mask", "none")
    common = dict(
        q=q,
        k=k,
        heads=heads,
        splits=splits,
        mask=mask,
        strided=spec.get("strided", False),
        used_q=spec.get("used_q"),
        used_k=spec.get("used_k"),
        maxima=spec.get("maxima"),
    )
    if representation == "prefill":
        c = prefill_case(dtype, **common)
    else:
        maxq, maxk = spec.get("maxima") or (max(q, default=0), max(k, default=0))
        original = compressed_case(
            getattr(torch, "float16" if dtype == "fp16" else "bfloat16"),
            dict(
                q=q,
                kv=k,
                heads=heads,
                uq=spec.get("used_q"),
                uk=spec.get("used_k"),
                sq=maxq,
                sk=maxk,
                window=(7, 2) if mask == "window" else (-1, -1),
                causal=mask == "causal",
                stride="independent" if common["strided"] else None,
            ),
            poison=True,
        )
        a = original["args"]
        c = dict(
            q=list(q),
            k=list(k),
            qlens=list(q if common["used_q"] is None else common["used_q"]),
            klens=list(k if common["used_k"] is None else common["used_k"]),
            raw=original["raw"],
            t={name: a[name] for name in ("q", "k", "v", "qv")},
            lengths={
                name: a[name]
                for name in (
                    "cu_seqlens_q",
                    "cu_seqlens_k",
                    "max_seqlen_q",
                    "max_seqlen_k",
                    "seqused_q",
                    "seqused_k",
                )
            },
            shape=(len(q), maxq, heads, maxk, 1),
            options=dict(num_splits=splits, causal=mask == "causal", window_size=a["window_size"]),
            guards=original["guards"],
        )
    c["representation"] = representation
    c["options"].update(softcap=spec.get("cap", 15), softmax_scale=spec.get("scale"))
    if "window" in spec:
        c["options"]["window_size"] = spec["window"]
    return c


def op_args(c):
    a = dict(**c["t"], **c["lengths"], **c["options"])
    a["window_left"], a["window_right"] = a.pop("window_size")
    return a


def metadata(c, *, enabled=None):
    a = op_args(c)
    a.pop("softmax_scale")
    cap = a.pop("softcap")
    a["has_softcap"] = cap > 0 if enabled is None else enabled
    op = (
        torch.ops.flash_attn_npu_3._mla_prefill_tnd_metadata
        if c["representation"] == "prefill"
        else torch.ops.flash_attn_npu_3._mla_split_tnd_metadata
    )
    # The uncapped compressed single-split entry uses an older non-split
    # record. A S=2 scheduling record suffices only for reference tile geometry.
    if c["representation"] == "compressed" and a["num_splits"] == 1 and not a["has_softcap"]:
        a["num_splits"] = 2
    return op(**a)


def invoke(c, record=None, *, entry="op"):
    if entry == "public":
        return api.flash_attn_varlen_func(
            **c["t"], **c["lengths"], **c["options"], return_attn_probs=True
        )
    if entry in ("cpp", "raw", "fake"):
        a = dict(**c["t"], **c["lengths"], **c["options"], scheduler_metadata=record)
        a["window_size_left"], a["window_size_right"] = a.pop("window_size")
        return _forward(entry, a)
    op = (
        torch.ops.flash_attn_npu_3._mla_prefill_tnd_forward
        if c["representation"] == "prefill"
        else torch.ops.flash_attn_npu_3._mla_split_tnd_forward
    )
    return op(**op_args(c), scheduler_metadata=record)


def reference(c, record):
    b, sq, h, sk, hk = c["shape"]
    g = h // hk
    prefill = c["representation"] == "prefill"
    r = record.cpu().tolist()
    base = 24 + 8 * b if prefill else 410 + 8 * b
    ends = []
    if prefill or c["options"]["num_splits"] == 0:
        count = r[base + 3]
        ends = list(
            zip(
                r[base + 4 : base + 4 + count],
                r[base + 30 : base + 30 + count],
                r[base + 56 : base + 56 + count],
            )
        )
    padded = {}
    for name, value in c["raw"].items():
        query = name in ("q", "qv")
        capacities, seq, heads = (c["q"], sq, h) if query else (c["k"], sk, hk)
        dst = value.new_zeros((b, seq, heads, value.shape[-1]))
        start = 0
        for i, length in enumerate(capacities):
            dst[i, :length] = value[start : start + length]
            start += length
        padded[name] = (
            dst.reshape(b, seq, hk, g if query else 1, value.shape[-1])
            .permute(0, 2, 1, 3, 4)
            .reshape(b * hk, seq, g if query else 1, value.shape[-1])
        )
    o = c["options"]
    m, n = r[13:15] if prefill else r[15:17]
    scale = o["softmax_scale"]
    scale = (192 if prefill else 576) ** -0.5 if scale is None else scale
    refs = references(
        padded,
        [x for x in c["qlens"] for _ in range(hk)],
        [x for x in c["klens"] for _ in range(hk)],
        ends,
        m or 256,
        n or 512,
        o["causal"],
        o["window_size"],
        scale,
        num_splits=o["num_splits"],
        softcap=o["softcap"],
    )
    dv = 128 if prefill else 512
    capacity = o["num_splits"] or 26
    for precision in ("high", "low"):
        refs["out_" + precision] = compact(
            c,
            refs["out_" + precision]
            .reshape(b, hk, sq, g, dv)
            .permute(0, 2, 1, 3, 4)
            .reshape(b, sq, h, dv),
            0,
        )
        refs["lse_" + precision] = compact(
            c, refs["lse_" + precision].reshape(b, h, sq).permute(1, 0, 2), 1
        )
        refs["accum_" + precision] = compact(
            c, refs["accum_" + precision].reshape(capacity, b, h, sq, dv).permute(0, 2, 1, 3, 4), 2
        )
        refs["accum_lse_" + precision] = compact(
            c, refs["accum_lse_" + precision].reshape(capacity, b, h, sq).permute(0, 2, 1, 3), 2
        )
        if not sum(c["k"]):
            refs["lse_" + precision].fill_(torch.inf)
    return refs


def compare(c, result, record):
    refs = reference(c, record)
    actual = [x.cpu() for x in result]
    directory = os.environ.get("MLA_VALIDATION_RESULT")
    if directory:
        key = hashlib.sha256(
            (os.environ.get("PYTEST_CURRENT_TEST", "") + str(c["options"])).encode()
        ).hexdigest()[:20]
        torch.save(
            dict(
                raw=c["raw"],
                q=c["q"],
                k=c["k"],
                qlens=c["qlens"],
                klens=c["klens"],
                record=record.cpu(),
                options=c["options"],
                actual=actual,
                refs=refs,
            ),
            Path(directory) / ("softcap-tnd-" + key + ".pt"),
        )
    for i, name in enumerate(("out", "lse", "accum", "accum_lse")[: len(result)]):
        if i >= 2 and c["options"]["num_splits"] == 1:
            assert actual[i].numel() == 0 and actual[i].dtype == torch.get_default_dtype()
            continue
        assert actual[i].shape == refs[name + "_high"].shape
        if c["representation"] == "prefill" or c["options"]["num_splits"] == 0:
            assert_fa_close(
                actual[i],
                refs[name + "_high"],
                refs[name + "_low"],
                softcap=c["options"]["softcap"],
                name="softcap-tnd-initialized-" + name,
            )
        start = 0
        for allocation, used in zip(c["q"], c["qlens"]):
            index = (
                (slice(start, start + used),)
                if i == 0
                else (slice(None), slice(start, start + used))
                if i == 1
                else (slice(None), slice(None), slice(start, start + used))
            )
            assert_fa_close(
                actual[i][index],
                refs[name + "_high"][index],
                refs[name + "_low"][index],
                softcap=c["options"]["softcap"],
                name="softcap-tnd-" + name,
            )
            start += allocation
    for before, view in c.get("guards", []):
        torch.testing.assert_close(view.cpu(), before, rtol=0, atol=0, equal_nan=True)
    return refs
