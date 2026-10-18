"""Mixed storage bounds and fixed-source ND address traces under ASan/UBSan."""

import itertools
import os
from pathlib import Path

import pytest

from tests.mla.support.mla_int8_layout_source import build_drivers, run_rows


@pytest.fixture(scope="module")
def drivers():
    return build_drivers(Path(os.environ["MLA_VALIDATION_RESULT"]) / "layout-sanitized")


def view(sizes, strides, offset, element_bytes):
    span = 0 if 0 in sizes else 1 + sum((n - 1) * s for n, s in zip(sizes, strides))
    return dict(
        sizes=list(sizes),
        strides=list(strides),
        offset=offset,
        storage=(offset + span) * element_bytes,
    )


def geometry(krow=64, vrow=512, table_stride=5):
    return dict(
        table=view((2, 3), (table_stride, 1), 3, 4),
        k=view((4, 128, 1, 64), (128 * krow + 19, krow, 64, 1), 7, 2),
        v=view((4, 128, 1, 512), (128 * vrow + 31, vrow, 512, 1), 13, 1),
    )


def encode(item, operation=0):
    return [
        operation,
        *itertools.chain.from_iterable(
            [*item[n]["sizes"], *item[n]["strides"], item[n]["offset"], item[n]["storage"]]
            for n in ("table", "k", "v")
        ),
    ]


BOUNDS = [
    ("ok", 0),
    ("v-byte-short", 3),
    ("k-byte-short", 3),
    ("table-byte-short", 3),
    ("negative-stride", 2),
    ("last-stride", 2),
    ("negative-offset", 3),
    ("page-size", 1),
    ("pool-mismatch", 1),
    ("heads", 1),
    ("dim", 1),
    ("pool-overflow", 1),
    ("row-overflow", 4),
    ("page-overflow", 4),
    ("offset-overflow", 4),
    ("table-overflow", 4),
    ("capacity-overflow", 4),
    ("int8-max-byte", 0),
    ("bf16-max-byte", 4),
    ("empty-pool", 0),
]


def storage_case(fault, expected):
    item = geometry()
    if fault.endswith("byte-short"):
        item[fault.split("-")[0]]["storage"] -= 1
    elif fault == "negative-stride":
        item["v"]["strides"][1] = -1
    elif fault == "last-stride":
        item["v"]["strides"][3] = 2
    elif fault == "negative-offset":
        item["k"]["offset"] = -1
    elif fault == "page-size":
        item["k"]["sizes"][1] = item["v"]["sizes"][1] = 64
    elif fault == "pool-mismatch":
        item["v"]["sizes"][0] += 1
    elif fault == "heads":
        item["k"]["sizes"][2] = item["v"]["sizes"][2] = 2
    elif fault == "dim":
        item["v"]["sizes"][3] = 64
    elif fault == "pool-overflow":
        item["k"]["sizes"][0] = item["v"]["sizes"][0] = 2**31 + 1
    elif fault in ("row-overflow", "page-overflow"):
        item["v"]["strides"][fault == "row-overflow"] = 2**63 - 1
    elif fault == "offset-overflow":
        item["v"]["offset"] = 2**63 - 1
    elif fault == "table-overflow":
        item["table"]["strides"][0] = 2**63 - 1
    elif fault == "capacity-overflow":
        item["table"] = view((0, 2**63 - 1), (0, 1), 0, 4)
    elif fault in ("int8-max-byte", "bf16-max-byte"):
        name = "v" if fault.startswith("int8") else "k"
        span = item[name]["storage"] // (1 if name == "v" else 2) - item[name]["offset"]
        item[name]["offset"] = 2**63 - 1 - span
        item[name]["storage"] = 2**63 - 1
    elif fault == "empty-pool":
        for name in ("k", "v"):
            item[name]["sizes"][0] = item[name]["storage"] = 0
    return item


def test_storage_bounds_atomic_failure(drivers):
    items = [storage_case(fault, expected) for fault, expected in BOUNDS]
    results = run_rows(drivers[0], [encode(item) for item in items])
    for (fault, expected), item, result in zip(BOUNDS, items, results):
        assert result[0] == expected, fault
        if expected:
            assert result[1:] == [17] * 15, fault
        else:
            assert result[1:7] == [2, 3, 384, item["k"]["sizes"][0], 128, 5], fault


def source_cases(krow, vrow, table_stride):
    item = geometry(krow, vrow, table_stride)
    table = [3, 1, 3, 0, 1, 2, 0, 2][: table_stride + 3]
    # All table entries are valid, including overlapped rows and repeated pages.
    layout_rows, source_rows, expected_segments, expected_traces = [], [], [], []
    for batch, start, count in [(0, 0, 1), (0, 127, 3), (1, 128, 128), (1, 255, 129)]:
        remaining, pos, segments = count, start, []
        while remaining:
            take = min(remaining, 128 - pos % 128)
            slot, row = batch * table_stride + pos // 128, pos % 128
            physical = table[slot]
            offsets = [
                physical * item[n]["strides"][0] + row * item[n]["strides"][1] for n in ("k", "v")
            ]
            segment = [0, slot, physical, row, take, *offsets]
            segments.append(segment)
            layout_rows.append(
                encode(item, 2) + [len(table), len(table), 1, *table, batch, 384, pos, remaining]
            )
            expected_segments.append(segment)
            remaining, pos = remaining - take, pos + take
        source_rows.append(
            [
                128,
                table_stride,
                *[item[n]["strides"][a] for n in ("k", "v") for a in (0, 1)],
                batch,
                start,
                count,
                32,
                256,
                32,
                256,
                len(table),
                *table,
            ]
        )
        qk, pv, done = [], [], 0
        for _, _, _, _, rows, ko, vo in segments:
            rope = [0, done * 16, 0, ko + 32, rows, 32, krow, 512]
            content = [1, done * 16, 1, vo + 256, rows, 256, vrow, 512]
            qk.extend([content, rope])
            pv.append(content)
            done += rows
        expected_traces.append(
            [
                len(segments),
                len(segments),
                3 * len(segments),
                *itertools.chain.from_iterable(qk + pv),
            ]
        )
    return layout_rows, source_rows, expected_segments, expected_traces


def test_pinned_source_segments_and_scalar_addresses(drivers):
    groups = [
        source_cases(*spec)
        for spec in [(64, 512, 5), (80, 544, 5), (0, 0, 0), (65536, 65536, 5), (64, 512, 1)]
    ]
    rows, source, expected, traces = [
        list(itertools.chain.from_iterable(group[i] for group in groups)) for i in range(4)
    ]
    assert run_rows(drivers[0], rows) == expected
    assert run_rows(drivers[1], source) == traces


PAGES = [
    ("padding", 0),
    ("empty", 0),
    ("negative-page", 3),
    ("large-page", 3),
    ("negative-length", 2),
    ("long-length", 2),
    ("missing-table", 1),
    ("short-table", 1),
    ("missing-length", 1),
]


def page_case(fault):
    item = geometry()
    table, lengths, available, has_table, has_lengths = (
        [3, -1, -1, -1, -1, 2, -1, -1],
        [1, 127],
        8,
        1,
        1,
    )
    if fault == "empty":
        table, lengths = [-1] * 8, [0, 0]
    elif fault == "negative-page":
        table[5] = -1
    elif fault == "large-page":
        table[0] = 4
    elif fault == "negative-length":
        lengths[1] = -1
    elif fault == "long-length":
        lengths[1] = 385
    elif fault == "missing-table":
        has_table = 0
    elif fault == "short-table":
        available = 7
    elif fault == "missing-length":
        has_lengths = 0
    row = encode(item, 1) + [8, available, has_table, *table, 2, 2, has_lengths, *lengths]
    return row


def test_effective_page_and_length_boundaries(drivers):
    assert run_rows(drivers[0], [page_case(fault) for fault, _ in PAGES]) == [
        [expected] for _, expected in PAGES
    ]
