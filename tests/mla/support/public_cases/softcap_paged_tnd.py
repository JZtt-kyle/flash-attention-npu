"""TND PA softcap: compact Q allocation over the existing independent PA fixture."""

import itertools
import torch
from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.mla.support.public_cases import softcap_tnd as tnd
from tests.mla.support.public_cases import softcap_paged_bsnd as paged
from tests.mla.support.public_cases.prefill_tnd import compact
from tests.mla.support.public_cases.contract import _forward, metadata_raw_args

CASES = [dict(s, page=48) for s in tnd.CASES]
for s in CASES:
    # Public cache uses complete Q allocations; Q-used is covered separately at
    # the internal and raw entries, preserving its existing reuse boundary.
    s.pop("used_q", None)
    if s["id"] in ("none-single", "causal-empty-splits", "zero-scale-single"):
        s["page"] = 16
    if s["id"] in ("none-auto", "near-zero", "long-kv"):
        s["page"] = 1024
    if s["id"] in ("none-explicit", "window-explicit-used-stride", "window-auto"):
        s["repeat_pages"] = True
DAILY = tnd.DAILY
REPRESENTATIVE = tnd.REPRESENTATIVE | {"none-auto"}


def make_case(representation, dtype="fp16", **spec):
    q, k = spec.get("q", (7, 0, 3)), spec.get("k", (65, 7, 3))
    sq, sk = spec.get("maxima") or (max(q, default=0), max(k, default=0))
    h = spec.get("heads", 8)
    options = dict(spec, shape=(len(q), sq, h, sk), lengths=spec.get("used_k", k))
    options.pop("used_q", None)
    c = paged.make_case(representation, dtype, **options)
    c["shape"] = (*c["shape"][:4], c["shape"][4] if representation == "prefill" else 1)
    b, _, _, cap = c["shape"][:4]
    c["q"], c["k"] = list(q), [cap] * b
    c["qlens"] = list(spec.get("used_q", q))
    for name in ("q", "qv"):
        if name not in c["raw"]:
            continue
        raw = compact(c, c["raw"][name], 0)
        c["raw"][name] = raw
        value = raw.npu()
        if spec.get("strided"):
            d = raw.shape[-1]
            step = 2 if name == "q" else 3
            storage = value.new_full((step * sum(q) + 1, step * h + 1, d + 16), float("nan"))
            view = storage[1::step, 1::step, 8 : 8 + d]
            view.copy_(value)
            value = view
        c["t"][name] = value
    c["lengths"] = dict(
        cu_seqlens_q=torch.tensor([0, *itertools.accumulate(q)], dtype=torch.int32, device="npu"),
        max_seqlen_q=sq,
        seqused_q=None
        if "used_q" not in spec
        else torch.tensor(spec["used_q"], dtype=torch.int32, device="npu"),
        seqused_k=c["lengths"],
    )
    c["metadata_tail"] = 24 + 8 * b if representation == "prefill" else 408 + 5 * b
    return c


def metadata(c, *, enabled=None, entry="public"):
    b, sq, h, sk, hk = c["shape"]
    o, lengths = c["options"], c["lengths"]
    enabled = o["softcap"] > 0 if enabled is None else enabled
    if entry == "op" and c["representation"] == "prefill":
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
            page_size=c["page"],
            has_softcap=enabled,
        )
    assert lengths["seqused_q"] is None
    function = (
        api.get_scheduler_metadata
        if entry == "public"
        else api.flash_attn_npu_3.get_scheduler_metadata
    )
    keywords = dict(
        headdim_v=128 if c["representation"] == "prefill" else 512,
        cu_seqlens_q=lengths["cu_seqlens_q"],
        has_softcap=enabled,
        page_size=c["page"],
        num_splits=o["num_splits"],
        causal=o["causal"],
    )
    if entry == "public":
        keywords["window_size"] = o["window_size"]
    else:
        keywords.update(window_size_left=o["window_size"][0], window_size_right=o["window_size"][1])
    if entry != "public":
        arguments = metadata_raw_args(
            dict(
                batch_size=b,
                max_seqlen_q=sq,
                max_seqlen_k=sk,
                num_heads_q=h,
                num_heads_kv=hk,
                headdim=192 if c["representation"] == "prefill" else 64,
                cache_seqlens=lengths["seqused_k"],
                qkv_dtype=c["t"]["q"].dtype,
                **keywords,
            )
        )
        return function(*arguments.values())
    return function(
        b,
        sq,
        sk,
        h,
        hk,
        192 if c["representation"] == "prefill" else 64,
        lengths["seqused_k"],
        c["t"]["q"].dtype,
        **keywords,
    )


def invoke(c, record=None, *, entry="op"):
    t, lengths, o = c["t"], c["lengths"], c["options"]
    if entry == "public":
        assert lengths["seqused_q"] is None
        return api.flash_attn_with_kvcache(
            t["q"],
            t["k"],
            t["v"],
            qv=t.get("qv"),
            cache_seqlens=lengths["seqused_k"],
            page_table=c["page_table"],
            cu_seqlens_q=lengths["cu_seqlens_q"],
            max_seqlen_q=lengths["max_seqlen_q"],
            **o,
            scheduler_metadata=record,
            return_softmax_lse=True,
        )
    if entry in ("cpp", "raw", "fake"):
        a = dict(**t, **lengths, **o, scheduler_metadata=record, page_table=c["page_table"])
        a["window_size_left"], a["window_size_right"] = a.pop("window_size")
        return _forward(entry, a)
    a = dict(**t, **lengths, **o, scheduler_metadata=record, page_table=c["page_table"])
    a["cache_seqlens"] = a.pop("seqused_k")
    a["window_left"], a["window_right"] = a.pop("window_size")
    op = (
        torch.ops.flash_attn_npu_3._mla_prefill_mixed_forward
        if c["representation"] == "prefill"
        else torch.ops.flash_attn_npu_3._mla_split_paged_tnd_forward
    )
    return op(**a)


def _reference_case(c):
    # Compact each entire rectangular cache slot, including unused poisoned rows.
    # Allocation lengths remain Sk, so the oracle's KV origins are exactly b*Sk.
    raw = {
        name: value.reshape(-1, value.shape[-2], value.shape[-1]) if name in ("k", "v") else value
        for name, value in c["raw"].items()
    }
    return dict(c, raw=raw)


def reference(c, record):
    return tnd.reference(_reference_case(c), record)


def compare(c, result, record):
    # For cap=0 compressed single split, use an S=2 record only for tile geometry.
    if (
        c["representation"] == "compressed"
        and c["options"]["num_splits"] == 1
        and c["options"]["softcap"] == 0
    ):
        record = metadata(dict(c, options=dict(c["options"], num_splits=2)))
    return tnd.compare(_reference_case(c), result, record)
