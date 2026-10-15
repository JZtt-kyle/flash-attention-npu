"""Shared helpers extracted from tests/mla/unit/mixed/test_flash_attn_npu_v3_mla_tnd_bsnd_metadata.py.

v4 mixed-layout CPU wire/buffer checks with an independent pinned-source oracle.
"""

import hashlib
import itertools
import json
from pathlib import Path
from tests.mla.support.components.bsnd_metadata import projection
from tests.mla.support.components.tnd_bsnd_schedule import (
    mixed_artifacts as mixed_artifacts,
    mixed_oracle as mixed_oracle,
    expected_spans,
    input_lengths,
    mixed_case,
)
from tests.mla.support.components.tnd_schedule import (
    check_tasks,
    evaluate,
    expected_spans as compact_spans,
    tnd_case,
)

PREFIX = 406

MAGIC = 0x4D4C4131

SENTINEL = 0x12345678


def wide_words(value):
    return [value & 0xFFFFFFFF, value >> 32]


def wire_spans(item):
    cq, q, kv = input_lengths(item)
    return [w for b in range(item[0]) for w in (cq[b], *wide_words(b * item[3]), q[b], kv[b])]


def metadata_cases(aic, ratio):
    result = []
    for b, heads, (qs, ks, capacity) in itertools.product(
        sorted({1, 2, aic, aic + 1, 2 * aic + 1}),
        (1, 128, 171),
        [
            ([3, 0, 1], [17, 0, 33], 64),
            ([0, 1, 3], [513, 0, 8193], 16384),
            ([257, 1, 0], [16384, 512, 0], 16384),
            ([2], [511], 512),
            ([1, 3, 7], [1, 513, 0], 8193),
            ([2, 0], [0, 513], 1024),
            ([3, 0, 1], [0], 513),
        ],
    ):
        q = [qs[i % len(qs)] for i in range(b)]
        k = [ks[i % len(ks)] for i in range(b)]
        if sum(q):
            result.append(
                mixed_case(q, k, heads=heads, sq=512, capacity=capacity, aic=aic, ratio=ratio)
            )
    return result


def check_records(rows, cases, oracle, directory, name, dtype):
    directory = Path(directory)
    mapped, native = evaluate(oracle, cases), evaluate(oracle, cases, native=True)
    (directory / f"{name}-inputs.json").write_text(json.dumps(cases) + "\n")
    (directory / f"{name}-outputs.json").write_text(
        json.dumps(
            dict(
                records=rows,
                mapped=mapped,
                native=native,
            )
        )
        + "\n"
    )
    counts = dict(cases=len(cases), native_non_fd=0, fa3_no_split_mapping=0, omitted_suffix=0)
    for row, item, want, source in zip(rows, cases, mapped, native):
        cq, q, kv = input_lengths(item)
        spans = expected_spans(item)
        assert row[:8] == [MAGIC, 4, 4, PREFIX + 5 * item[0], dtype, 0, 0, 0]
        assert len(row) == row[3]
        assert row[401:PREFIX] == [cq[-1], item[1], item[3], *wide_words(item[0] * item[3])]
        assert row[PREFIX:] == wire_spans(item)
        assert row[8:15] == [item[0], 1, item[2], spans[0], item[3], 512, 64]
        assert row[17:20] == [item[4], item[5], item[5] // item[4]]

        def wide(words):
            return [lo | hi << 32 for lo, hi in zip(words[::2], words[1::2])]

        workspace = wide(row[367:391])
        cursor = 0
        for offset, size in zip(
            workspace[4:8], [workspace[0] * 4, workspace[0] * 2, workspace[1] * 4, workspace[1] * 4]
        ):
            assert offset == cursor and offset % 32 == 0
            cursor += 2 * row[21] * size
        assert cursor == workspace[8] <= workspace[9]
        assert wide(row[391:401]) == [
            516096,
            174080 if row[19] == 2 else 178176,
            65536,
            65536,
            131072,
        ]
        got = projection(row) + spans
        assert got == want, (item, got, want)
        if source[5] == 0:
            assert got == source
            counts["native_non_fd"] += 1
        else:
            counts["fa3_no_split_mapping"] += 1
        compact = tnd_case(q, kv, heads=item[2])
        check_tasks(compact, [*got[:356], *compact_spans(compact)])
        counts["omitted_suffix"] += got[18 + got[3] - 1] < item[0]
    counts["records_sha256"] = hashlib.sha256(json.dumps(rows).encode()).hexdigest()
    (directory / f"{name}.json").write_text(json.dumps(counts, indent=2) + "\n")


EMPTY_CASES = [
    ([], [], 0),
    ([], [], 16384),
    ([0], [0], 64),
    ([0, 0], [1, 513], 1024),
    ([3, 0, 1], [0, 0, 0], 0),
]

FAILURES = [
    ({"has_q": False}, 3, 0),
    ({"has_k": False}, 3, 0),
    ({"qcount": 0}, 3, 0),
    ({"kcount": 0}, 3, 0),
    ({"kcount": 4}, 3, 0),
    ({"cq": [1, 3, 3, 4]}, 2, 1),
    ({"cq": [-1, 3, 3, 4]}, 2, 1),
    ({"cq": [0, 3, 2, 4]}, 2, 1),
    ({"cq": [0, 3, -(2**31), 4]}, 2, 1),
    ({"klens": [-1, 0, 33]}, 2, 1),
    ({"klens": [17, -(2**31), 33]}, 2, 1),
    ({"klens": [17, 0, 65]}, 2, 1),
    ({"klens": [17, 0, 2**31 - 1]}, 2, 1),
    ({"total_q": 3}, 2, 1),
    ({"total_q": 5}, 2, 1),
    ({"total_q": -1}, 2, 1),
    ({"sq": 2}, 2, 1),
    ({"capacity": 32}, 2, 1),
    ({"heads": 0}, 2, 1),
    ({"sq": -1}, 2, 1),
    ({"capacity": -1}, 2, 1),
    ({"capacity": 16385}, 2, 1),
    ({"total_q": 2**31}, 2, 2),
    ({"heads": 2**23}, 2, 2),
    ({"sq": 2**30 + 1}, 2, 2),
    ({"sq": 2**30, "heads": 4}, 2, 2),
    ({"capacity": 2**32}, 2, 2),
    ({"aic": 0, "aiv": 0}, 2, 3),
    ({"aic": 27}, 2, 3),
    ({"aiv": 21}, 2, 3),
    ({"block_dim": 21}, 2, 3),
    ({"libapi": 2**64 - 1}, 2, 5),
    ({"l1": 1}, 2, 4),
    ({"qlens": [], "klens": [], "has_q": False}, 3, 0),
    ({"qlens": [], "klens": [], "cq": [1]}, 2, 1),
    ({"qlens": [], "klens": [], "total_q": 1}, 2, 1),
    ({"qlens": [], "klens": [], "kcount": 1}, 3, 0),
    ({"qlens": [3, 0, 1], "klens": [0, 0, 0], "capacity": 0, "cq": [0, 3, 2, 4]}, 2, 1),
]
