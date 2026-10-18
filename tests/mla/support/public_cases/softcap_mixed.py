"""Mixed softcap inputs and public cache/metadata adapters using the TND oracle."""

import torch
from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.mla.support.public_cases import softcap_tnd as tnd
from tests.mla.support.public_cases.contract import _forward, metadata_raw_args

# Keep each independent mask/split, allocation, scale and stride boundary.
CASES = [dict(s) for s in tnd.CASES]
for _case in CASES:
    _case.pop("used_q", None)
DAILY = tnd.DAILY
REPRESENTATIVE = tnd.REPRESENTATIVE


def make_case(representation, dtype="fp16", **spec):
    c = tnd.make_case(representation, dtype, **spec)
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
        if spec.get("strided"):
            d = value.shape[-1]
            step = 2 if name == "k" else 3
            storage = torch.full(
                (b * step + 1, sk * step + 1, hk * step + 1, d + 16),
                float("nan"),
                dtype=value.dtype,
                device="npu",
            )
            view = storage[
                1 : 1 + step * b : step,
                1 : 1 + step * sk : step,
                1 : 1 + step * hk : step,
                8 : 8 + d,
            ]
            view.copy_(value)
            value = view
        c["t"][name] = value
        c.setdefault("guards", []).append((raw.clone(), value))
    c["k"] = [sk] * b
    c["lengths"].pop("cu_seqlens_k")
    c["lengths"].pop("max_seqlen_k")
    c["lengths"]["seqused_k"] = torch.tensor(c["klens"], dtype=torch.int32, device="npu")
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
            cu_seqlens_q=lengths["cu_seqlens_q"],
            max_seqlen_q=lengths["max_seqlen_q"],
            **o,
            scheduler_metadata=record,
            return_softmax_lse=True,
        )
    if entry in ("cpp", "raw", "fake"):
        a = dict(**t, **lengths, **o, scheduler_metadata=record)
        a["window_size_left"], a["window_size_right"] = a.pop("window_size")
        return _forward(entry, a)
    a = dict(**t, **lengths, **o, scheduler_metadata=record)
    a["cache_seqlens"] = a.pop("seqused_k")
    a["window_left"], a["window_right"] = a.pop("window_size")
    op = (
        torch.ops.flash_attn_npu_3._mla_prefill_mixed_forward
        if c["representation"] == "prefill"
        else torch.ops.flash_attn_npu_3._mla_split_mixed_forward
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
