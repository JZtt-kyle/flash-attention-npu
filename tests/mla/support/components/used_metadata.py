"""Shared helpers extracted from tests/mla/unit/used/test_flash_attn_npu_v3_mla_used_metadata.py.

v5 allocation/used wire and buffers, checked against independent pinned CANN source.
"""

import itertools
import json
from pathlib import Path
from tests.mla.support.components.bsnd_metadata import projection
from tests.mla.support.components.tnd_bsnd_metadata import wide_words
from tests.mla.support.components.tnd_schedule import (
    check_tasks,
    tnd_case,
    expected_spans as compact_spans,
)
from tests.mla.support.components.used_schedule import (
    used_artifacts as used_artifacts,
    used_oracle as used_oracle,
    encode,
    execute,
    spans,
    used_case as used_case,
)

PREFIX = 410

MAGIC = 0x4D4C4131

SENTINEL = 0x12345678


def input_arrays(item):
    """Independent raw descriptors, including malformed inputs for negative tests."""
    layout = item["layout"]
    q, kv = item["q"], item["kv"]
    result = []
    for name, values, present, count_name in (
        (
            "q",
            item.get("cq", list(itertools.accumulate(q, initial=0)) if layout != 0 else []),
            layout != 0,
            "qcount",
        ),
        (
            "k",
            item.get("ck", list(itertools.accumulate(kv, initial=0)) if layout == 1 else []),
            layout == 1,
            "kcount",
        ),
        ("uq", item["uq"] or [], item["uq"] is not None, "uqcount"),
        ("uk", item["uk"] or [], item["uk"] is not None, "ukcount"),
    ):
        result.append((values, item.get("has_" + name, present), item.get(count_name, len(values))))
    return result


def allocation_words(item):
    layout = item["layout"]
    flags = int(item["uq"] is not None) | (int(item["uk"] is not None) << 1)
    return [
        int(layout != 0),
        *wide_words(item.get("total_q", sum(item["q"]))),
        item.get("sq", max(item["q"], default=0)),
        int(layout == 1),
        *wide_words(item.get("total_k", sum(item["kv"]))),
        item.get("sk", max(item["kv"], default=0)),
        flags,
    ]


def wire_spans(item):
    raw = spans(item)[4:]
    return [
        word
        for b in range(len(item["q"]))
        for word in (
            *wide_words(raw[6 * b]),
            *wide_words(raw[6 * b + 1]),
            *raw[6 * b + 2 : 6 * b + 6],
        )
    ]


def check_records(rows, items, oracle, directory, name, dtype):
    mapped, native = execute(oracle, items), execute(oracle, items, native=True)
    counts = dict(cases=len(items), native_non_fd=0, fa3_no_split_mapping=0)
    for row, item, want, original in zip(rows, items, mapped, native):
        values = encode(item)
        assert row[:8] == [MAGIC, 5, 5, PREFIX + len(item["q"]) * 8, dtype, 0, 0, 0]
        assert len(row) == row[3]
        assert row[401:PREFIX] == allocation_words(item)
        assert row[PREFIX:] == wire_spans(item)
        assert row[8:15] == [len(item["q"]), 1, values[2], max(item["q"]), max(item["kv"]), 512, 64]
        assert row[17:20] == [values[4], values[5], values[5] // values[4]]

        def wide(data):
            return [lo | hi << 32 for lo, hi in zip(data[::2], data[1::2])]

        workspace = wide(row[367:391])
        cursor = 0
        for offset, size in zip(
            workspace[4:8], (workspace[0] * 4, workspace[0] * 2, workspace[1] * 4, workspace[1] * 4)
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
        got = projection(row) + spans(item)
        assert got == want, (item, got, want)
        counts["native_non_fd" if original[5] == 0 else "fa3_no_split_mapping"] += 1
        if original[5] == 0:
            assert got == original
        compact = tnd_case(
            item["q"] if item["uq"] is None else item["uq"],
            item["kv"] if item["uk"] is None else item["uk"],
            heads=values[2],
        )
        check_tasks(compact, [*got[:356], *compact_spans(compact)])
    directory = Path(directory)
    (directory / f"{name}-inputs.json").write_text(json.dumps(items) + "\n")
    (directory / f"{name}-outputs.json").write_text(
        json.dumps(dict(records=rows, mapped=mapped, native=native)) + "\n"
    )
    (directory / f"{name}.json").write_text(json.dumps(counts, indent=2) + "\n")
