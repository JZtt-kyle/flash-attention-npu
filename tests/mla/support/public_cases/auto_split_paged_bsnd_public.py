"""Shared helpers extracted from tests/mla/e2e/auto_split_paged_bsnd/test_public.py.

v24 native automatic PA production and public split; independent raw-page/math mapping.
"""

import torch
from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.mla.support.mla_auto_split_bsnd_reference import references
from tests.mla.support.public_cases.contract import _forward


def prepare(dtype, s):
    b, q, h, p, cap = len(s["lengths"]), s["q"], s["heads"], s["page"], s["capacity"]
    columns = cap // p
    pool = b * columns + 2
    g = torch.Generator().manual_seed(10723)
    raw = {
        n: torch.randn(shape, generator=g, dtype=dtype) * 0.1
        for n, shape in (
            ("q", (b, q, h, 64)),
            ("qv", (b, q, h, 512)),
            ("k", (pool, p, 1, 64)),
            ("v", (pool, p, 1, 512)),
        )
    }
    table = torch.arange(b * columns, dtype=torch.int32).reshape(b, columns).flip(1) + 1
    for bi, length in enumerate(s["lengths"]):
        for token in range(length, cap):
            for name in ("k", "v"):
                raw[name][(bi + 1) * columns - token // p, token % p] = torch.nan
    args = {}
    for i, (name, x) in enumerate(raw.items()):
        if s["stride"]:
            strides = (
                x.shape[1] * (x.shape[2] + 1) * (x.shape[3] + 16) + 32,
                (x.shape[2] + 1) * (x.shape[3] + 16),
                x.shape[3] + 16,
                1,
            )
            count = 1 + sum(max(0, n - 1) * st for n, st in zip(x.shape, strides))
            backing = torch.empty(count + 32, dtype=dtype, device="npu")
            value = backing.as_strided(x.shape, strides, 16)
            value.copy_(x)
        else:
            value = x.to("npu")
        args[name] = value
    if s["stride"]:
        backing = torch.empty((b, columns + 3), dtype=torch.int32, device="npu")
        device_table = backing[:, 1 : columns + 1]
        device_table.copy_(table)
    else:
        device_table = table.to("npu")
    args.update(
        page_table=device_table,
        cache_seqlens=torch.tensor(s["lengths"], dtype=torch.int32, device="npu"),
        seqused_q=None
        if s["used"] is None
        else torch.tensor(s["used"], dtype=torch.int32, device="npu"),
        num_splits=s["splits"],
        causal=s["causal"],
        window_size=s["window"],
        softmax_scale=s["scale"],
    )
    return dict(args=args, raw=raw, table=table, spec=s)


def metadata(c, entry="public"):
    a, s = c["args"], c["spec"]
    lengths = a["cache_seqlens"]
    if lengths is None or isinstance(lengths, int):
        lengths = torch.full(
            (a["q"].shape[0],),
            s["capacity"] if lengths is None else lengths,
            dtype=torch.int32,
            device=a["q"].device,
        )
    if entry == "op":
        return torch.ops.flash_attn_npu_3._mla_paged_metadata(
            a["q"].shape[0],
            s["q"],
            s["capacity"],
            s["heads"],
            s["page"],
            a["q"].dtype,
            lengths,
            None,
            s["causal"],
            *s["window"],
            s["splits"],
        )
    from tests.test_flash_attn_npu_v3_metadata_signature import metadata as generate

    return generate(
        entry,
        dict(
            batch_size=a["q"].shape[0],
            max_seqlen_q=s["q"],
            max_seqlen_k=s["capacity"],
            num_heads_q=s["heads"],
            num_heads_kv=1,
            headdim=64,
            headdim_v=512,
            qkv_dtype=a["q"].dtype,
            cache_seqlens=lengths,
            page_size=s["page"],
            causal=s["causal"],
            window_size=s["window"],
            num_splits=s["splits"],
        ),
    )


def call(a, entry="public", record=None, *, lse=True):
    a = dict(a)
    if record is not None:
        a["scheduler_metadata"] = record
    if entry == "public":
        assert a.pop("seqused_q", None) is None
        a["k_cache"], a["v_cache"] = a.pop("k"), a.pop("v")
        return api.flash_attn_with_kvcache(**a, return_softmax_lse=lse)
    left, right = a.pop("window_size")
    if entry == "op":
        return torch.ops.flash_attn_npu_3._mla_split_paged_bsnd_forward(
            **a, window_left=left, window_right=right
        )
    a["seqused_k"] = a.pop("cache_seqlens")
    return _forward(entry, dict(a, window_size_left=left, window_size_right=right))


def geometry(c):
    s = c["spec"]
    words = metadata(c).cpu().tolist()
    base = 408 + 5 * len(s["lengths"])
    count = words[base + 3]
    page, cap = s["page"], s["capacity"]
    m, n = (
        words[15] or 256,
        words[16] or min(cap, 512 // page * page if page < 512 and 512 % page else 512) or 1,
    )
    ends = list(zip(*(words[base + off : base + off + count] for off in (4, 30, 56))))
    if s["used"] is not None:
        # Allocation controls PA tile geometry; effective Q controls native partitioning.
        from tests.mla.support.mla_auto_split_paged_bsnd_reference import native_schedule

        source = native_schedule(s, m, n, words[17])
        count = source[0]
        ends = list(
            zip(*(source[7 + off : 7 + off + count] for off in (0, words[17], 2 * words[17])))
        )
    return m, n, ends


def reference(c, tile=None):
    s = c["spec"]
    b = len(s["lengths"])
    p = s["page"]
    cap = s["capacity"]
    raw = {n: c["raw"][n] for n in ("q", "qv")}
    for name, d in [("k", 64), ("v", 512)]:
        raw[name] = torch.zeros((b, cap, 1, d), dtype=c["raw"][name].dtype)
        for bi, length in enumerate(s["lengths"]):
            for token in range(length):
                raw[name][bi, token] = c["raw"][name][int(c["table"][bi, token // p]), token % p]
    m, n, ends = geometry(c) if tile is None else tile
    refs = references(
        raw,
        [s["q"]] * b if s["used"] is None else s["used"],
        s["lengths"],
        ends,
        m,
        n,
        s["causal"],
        s["window"],
        576**-0.5 if s["scale"] is None else s["scale"],
    )
    if cap == 0:
        refs = {k: v.clone() for k, v in refs.items()}
        refs["lse_high"].fill_(-torch.inf)
        refs["lse_low"].fill_(-torch.inf)
    return refs
