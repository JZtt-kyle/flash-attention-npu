"""v26 native PA partition references packed into original compact Q allocation."""

import torch
from tests.mla.support.public_cases.split_paged_tnd_public import (
    prepare as prepare,
    metadata as metadata,
    call as call,
)
from tests.mla.support.mla_auto_split_bsnd_reference import references


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
        from tests.mla.support.mla_auto_split_paged_tnd_reference import native_schedule

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
    raw = {}
    for name, d in [("q", 64), ("qv", 512)]:
        raw[name] = torch.zeros((b, s["max_q"], s["heads"], d), dtype=c["raw"][name].dtype)
        start = 0
        for bi, q in enumerate(s["q"]):
            raw[name][bi, :q] = c["raw"][name][start : start + q]
            start += q
    for name, d in [("k", 64), ("v", 512)]:
        raw[name] = torch.zeros((b, cap, 1, d), dtype=c["raw"][name].dtype)
        for bi, length in enumerate(s["lengths"]):
            for token in range(length):
                raw[name][bi, token] = c["raw"][name][int(c["table"][bi, token // p]), token % p]
    m, n, ends = geometry(c) if tile is None else tile
    refs = references(
        raw,
        s["q"] if s["used"] is None else s["used"],
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
    packed = {}
    for name, value in refs.items():
        axis = 0 if name.startswith("out_") else 1 if name.startswith("lse_") else 2
        parts = [
            value[bi, :q] if axis == 0 else value[bi, :, :q] if axis == 1 else value[:, bi, :, :q]
            for bi, q in enumerate(s["q"])
        ]
        shape = (
            (0, s["heads"], 512)
            if axis == 0
            else (s["heads"], 0)
            if axis == 1
            else (26, s["heads"], 0, *(() if name.startswith("accum_lse") else (512,)))
        )
        packed[name] = torch.cat(parts, dim=axis) if parts else value.new_empty(shape)
    return packed
