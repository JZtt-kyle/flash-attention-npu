"""Shared helpers extracted from tests/mla/unit/tnd/test_flash_attn_npu_v3_mla_tnd_metadata.py.

v3 CPU wire checks and pinned-source differential, including full sanitizer runs.
"""

import hashlib
import itertools
import json
from pathlib import Path
from tests.mla.support.components.bsnd_metadata import projection
from tests.mla.support.components.tnd_schedule import (
    tnd_artifacts as tnd_artifacts,
    tnd_oracle as tnd_oracle,
    check_tasks,
    evaluate,
    expected_spans,
    prefixes,
    tnd_case,
)

PREFIX = 405

MAGIC = 0x4D4C4131

SENTINEL = 0x12345678


def metadata_cases(aic, ratio):
    result = []
    for b, heads, (qs, ks) in itertools.product(
        sorted({1, 2, aic, aic + 1, 2 * aic + 1}),
        (1, 128, 171),
        [
            ([3, 0, 1], [17, 0, 33]),
            ([0, 1, 3], [513, 0, 8193]),
            ([257, 1, 0], [16384, 512, 0]),
            ([2], [511]),
            ([1, 3, 7], [1, 513, 0]),
            ([2, 0], [0, 513]),
        ],
    ):
        q = [qs[i % len(qs)] for i in range(b)]
        k = [ks[i % len(ks)] for i in range(b)]
        if sum(q) and sum(k):
            result.append(tnd_case(q, k, heads=heads, sq=512, sk=16384, aic=aic, ratio=ratio))
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
        cq, ck = prefixes(item)
        spans = expected_spans(item)
        assert row[:8] == [MAGIC, 3, 3, PREFIX + 4 * item[0], dtype, 0, 0, 0]
        assert len(row) == row[3]
        assert row[401:PREFIX] == [cq[-1], ck[-1], item[1], item[3]]
        assert row[PREFIX:] == spans[2:]
        assert row[8:15] == [item[0], 1, item[2], *spans[:2], 512, 64]
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
        check_tasks(item, got)
        counts["omitted_suffix"] += got[18 + got[3] - 1] < item[0]
    counts["records_sha256"] = hashlib.sha256(json.dumps(rows).encode()).hexdigest()
    (directory / f"{name}.json").write_text(json.dumps(counts, indent=2) + "\n")


FAILURES = [
    ({"has_q": False}, 3, 0),
    ({"has_k": False}, 3, 0),
    ({"qcount": 0}, 3, 0),
    ({"kcount": 3}, 3, 0),
    ({"cq": [1, 3, 3, 4]}, 2, 1),
    ({"ck": [-1, 17, 17, 50]}, 2, 1),
    ({"cq": [0, 3, 2, 4]}, 2, 1),
    ({"ck": [0, 17, 16, 50]}, 2, 1),
    ({"cq": [0, 3, -(2**31), 4]}, 2, 1),
    ({"ck": [0, 17, 17, -1]}, 2, 1),
    ({"total_q": 5}, 2, 1),
    ({"total_k": 51}, 2, 1),
    ({"total_q": -1}, 2, 1),
    ({"total_k": -1}, 2, 1),
    ({"sq": 2}, 2, 1),
    ({"sk": 32}, 2, 1),
    ({"heads": 0}, 2, 1),
    ({"sq": -1}, 2, 1),
    ({"sk": 16385}, 2, 1),
    ({"total_q": 2**31}, 2, 2),
    ({"total_k": 2**31}, 2, 2),
    ({"heads": 2**23}, 2, 2),
    ({"sq": 2**30 + 1}, 2, 2),
    ({"sq": 2**30, "heads": 4}, 2, 2),
    ({"sk": 2**32}, 2, 2),
    ({"aic": 0, "aiv": 0}, 2, 3),
    ({"aic": 27}, 2, 3),
    ({"aiv": 21}, 2, 3),
    ({"block_dim": 21}, 2, 3),
    ({"libapi": 2**64 - 1}, 2, 5),
    ({"l1": 1}, 2, 4),
    ({"qlens": [], "klens": [], "has_q": False}, 3, 0),
    ({"qlens": [], "klens": [], "ck": [1]}, 2, 1),
    ({"qlens": [], "klens": [], "total_q": 1}, 2, 1),
    ({"qlens": [0, 0], "klens": [0, 0], "kcount": 2}, 3, 0),
]
