"""BSND PA softcap adapters using existing BSND inputs and cached interval math."""

import torch
from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.mla.support.public_cases import softcap_bsnd as bsnd
from tests.mla.support.public_cases.contract import _forward, metadata_raw_args

CASES = [dict(s, page=48) for s in bsnd.CASES]
for s in CASES:
    if s["id"] in ("none-single", "causal-empty-splits", "zero-scale-single"):
        s["page"] = 16
    if s["id"] in ("none-auto", "near-zero"):
        s["page"] = 1024
    if s["id"] in ("window-explicit-used-stride", "window-auto"):
        s["repeat_pages"] = True
DAILY = bsnd.DAILY
REPRESENTATIVE = bsnd.REPRESENTATIVE | {"none-auto"}


def make_case(representation, dtype="fp16", **spec):
    b, q, h, k = spec.get("shape", (2, 7, 8, 65))
    page = spec.get("page", 48)
    cap = (k + page - 1) // page * page
    options = dict(spec, shape=(b, q, h, cap), lengths=spec.get("lengths", [k] * b))
    c = bsnd.make_case(representation, dtype, **options)
    columns = cap // page
    physical = b * columns + 2 if columns else 0
    table = torch.arange(b * columns, dtype=torch.int32).reshape(b, columns).flip(1) + 1
    c["physical"] = {}
    for name in ("k", "v"):
        raw = c["raw"][name]
        hk, d = raw.shape[-2:]
        pages = raw.new_full((physical, page, hk, d), float("nan"))
        if b * columns:
            pages[table.long()] = raw.reshape(b, columns, page, hk, d)
        c["physical"][name] = pages
    if spec.get("repeat_pages") and columns > 1 and b:
        table[0, 1] = table[0, 0]
    for name, pages in c["physical"].items():
        hk, d = pages.shape[-2:]
        c["raw"][name] = pages[table.long()].reshape(b, cap, hk, d)
        value = pages.npu()
        if spec.get("strided"):
            # Independent page, row and head pitches; nonzero storage origins.
            step = 2 if name == "k" else 3
            storage = value.new_full(
                (physical * step + 1, page * step + 1, hk * step + 1, d + 16), float("nan")
            )
            view = storage[
                1 : 1 + step * physical : step,
                1 : 1 + step * page : step,
                1 : 1 + step * hk : step,
                8 : 8 + d,
            ]
            view.copy_(value)
            value = view
        c["t"][name] = value
    for i, length in enumerate(c["klens"]):
        table[i, (length + page - 1) // page :] = -1
    c["raw_pages"] = table
    value = table.npu()
    if spec.get("strided"):
        storage = torch.full((b + 1, columns + 3), -77, dtype=torch.int32, device="npu")
        view = storage[1:, 1 : columns + 1]
        view.copy_(value)
        value = view
    c.update(
        page=page,
        page_table=value,
        metadata_tail=24 + 2 * b if representation == "prefill" else 408 + 5 * b,
    )
    if "used_q" in spec:
        c["qlens"] = list(spec["used_q"])
        c["usedq"] = torch.tensor(c["qlens"], dtype=torch.int32, device="npu")
    return c


def metadata(c, *, enabled=None, entry="public"):
    b, q, h, cap = c["shape"][:4]
    prefill = c["representation"] == "prefill"
    o = c["options"]
    keywords = dict(
        batch_size=b,
        max_seqlen_q=q,
        max_seqlen_k=cap,
        num_heads_q=h,
        num_heads_kv=c["shape"][4] if prefill else 1,
        headdim=192 if prefill else 64,
        headdim_v=128 if prefill else 512,
        cache_seqlens=c["lengths"],
        qkv_dtype=c["t"]["q"].dtype,
        page_size=c["page"],
        has_softcap=o["softcap"] > 0 if enabled is None else enabled,
        num_splits=o["num_splits"],
        causal=o["causal"],
        window_size=o["window_size"],
    )
    if entry == "public":
        return api.get_scheduler_metadata(**keywords)
    keywords["window_size_left"], keywords["window_size_right"] = keywords.pop("window_size")
    return api.flash_attn_npu_3.get_scheduler_metadata(*metadata_raw_args(keywords).values())


def invoke(c, record=None, *, entry="op"):
    t, o = c["t"], c["options"]
    if entry in ("cpp", "raw", "fake"):
        args = dict(
            **t,
            **o,
            page_table=c["page_table"],
            seqused_k=c["lengths"],
            seqused_q=c.get("usedq"),
            scheduler_metadata=record,
        )
        args["window_size_left"], args["window_size_right"] = args.pop("window_size")
        return _forward(entry, args)
    if entry in ("public", "cache"):
        assert c.get("usedq") is None
        return api.flash_attn_with_kvcache(
            t["q"],
            t["k"],
            t["v"],
            qv=t.get("qv"),
            page_table=c["page_table"],
            cache_seqlens=c["lengths"],
            scheduler_metadata=record,
            return_softmax_lse=True,
            **o,
        )
    args = dict(
        **t,
        **o,
        page_table=c["page_table"],
        cache_seqlens=c["lengths"],
        seqused_q=c.get("usedq"),
        scheduler_metadata=record,
    )
    args["window_left"], args["window_right"] = args.pop("window_size")
    op = (
        torch.ops.flash_attn_npu_3._mla_prefill_bsnd_forward
        if c["representation"] == "prefill"
        else torch.ops.flash_attn_npu_3._mla_split_paged_bsnd_forward
    )
    return op(**args)


def compare(c, result, record):
    # cap=0 single compressed PA uses v6/v10/v15; an S=2 record supplies only geometry.
    if (
        c["representation"] == "compressed"
        and c["options"]["num_splits"] == 1
        and c["options"]["softcap"] == 0
    ):
        record = metadata(dict(c, options=dict(c["options"], num_splits=2)))
    return bsnd.compare(c, result, record)
