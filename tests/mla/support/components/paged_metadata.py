"""Shared helpers extracted from tests/mla/unit/paged_bsnd/test_flash_attn_npu_v3_mla_paged_metadata.py.

v6 CPU wire contract, independently decoded against pinned PA source.

The strict test decoder checks every word (including diagnostic fields) against
an independent call context/source oracle. It is not a PA device consumer;
page-table validation and device rejection remain separate migration steps.
"""

import itertools
from tests.mla.support.components.paged_schedule import (
    encode as schedule_encode,
    paged_case as paged_case,
    spans,
)

PREFIX = 408

MAGIC = 0x4D4C4131


def wide(value):
    return [value & 0xFFFFFFFF, (value >> 32) & 0xFFFFFFFF]


def encode(item):
    layout, q = item["layout"], item["q"]
    values = schedule_encode(item)
    result = [
        values[0],
        values[2],
        0 if layout == 0 else 1 if layout == 2 else 99,
        item.get("total_q", sum(q)),
        values[1],
        item["page"],
        values[3],
        *values[4:13],
    ]
    for name, data, present, count_name in (
        (
            "q",
            item.get("cq", list(itertools.accumulate(q, initial=0)) if layout == 2 else []),
            layout == 2,
            "qcount",
        ),
        ("uq", item["uq"] or [], item["uq"] is not None, "uqcount"),
        ("uk", item["uk"] or [], item["uk"] is not None, "ukcount"),
    ):
        result += [
            int(item.get("has_" + name, present)),
            item.get(count_name, len(data)),
            len(data),
            *data,
        ]
    return result


def expected_record(item, dtype, source):
    """Numerical decoding specification; never calls migrated normalization/serialization."""
    values = schedule_encode(item)
    batch, sq, heads, capacity, aic, aiv = values[:6]
    q, kv = item["q"], item["uk"]
    s1 = max(q, default=0) if item["layout"] == 2 else sq
    s2 = (max(kv, default=0) + item["page"] - 1) // item["page"] * item["page"]
    empty = batch == 0 or s1 == 0 or s2 == 0
    assert 16 <= item["page"] <= 1024 and item["page"] % 16 == 0
    assert 0 <= capacity <= 16384 and capacity % item["page"] == 0
    assert len(q) == len(kv) == batch and all(0 <= length <= capacity for length in kv)
    if empty:
        assert source is None
        s = [0] * 356
        s[3], s[16], s[17] = 1, values[12] or 1, 1
    else:
        assert source[0] == 0
        s = source
        assert s[356:-4] == spans(item)
        assert s[-4:] == [item["page"], capacity, s1, s2]
    record = [MAGIC, 6, 6, PREFIX + batch * 5, dtype, 0, 0, 0]
    record += [batch, 1, heads, s1, s2, 512, 64, s[1], s[2], aic, aiv, aiv // aic, s[3], s[16], 1]
    record += s[18:96] + wide(s[4]) + s[5:9] + s[96:356]
    r1, r2 = s[9:11]
    slots = 2 * s[16]
    p = slots * r1 * 4
    pv = p + slots * r1 * 2
    vec2 = pv + slots * r2 * 4
    kernel = vec2 + slots * r2 * 4
    record += list(
        itertools.chain.from_iterable(
            wide(v) for v in [*s[9:13], 0, p, pv, vec2, kernel, *s[13:16]]
        )
    )
    memory = [0] * 5 if empty else [516096, 178176 if aiv == aic else 174080, 65536, 65536, 131072]
    record += list(itertools.chain.from_iterable(wide(v) for v in memory))
    assert len(record) == 401
    record += [
        int(item["layout"] == 2),
        *wide(item.get("total_q", sum(q))),
        sq,
        int(item["uq"] is not None),
        item["page"],
        capacity,
    ]
    query_used = q if item["uq"] is None else item["uq"]
    starts = list(itertools.accumulate(q, initial=0))
    for b in range(batch):
        assert 0 <= query_used[b] <= q[b] <= sq
        record += [*wide(starts[b]), q[b], query_used[b], kv[b]]
    return record


def decode_record(record, item, dtype, source):
    """Strict independent test decoder: full wire equality, not just round-trip."""
    expected = expected_record(item, dtype, source)
    if len(record) != len(expected):
        raise ValueError("word count")
    for index, (actual, want) in enumerate(zip(record, expected)):
        if actual != want:
            raise ValueError(f"word {index}: {actual} != {want}")
    return {
        "page": record[406],
        "capacity": record[407],
        "source_s2": record[12],
        "spans": record[PREFIX:],
    }
