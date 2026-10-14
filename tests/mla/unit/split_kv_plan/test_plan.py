"""Two-stage ownership, persistent partial buffers and original BSND addresses.

The plan is exercised here, not the historical split unit suite. Fixed-source
executables provide CANN ranges/workspace and FA3 slices/shapes; Python derives
visible keys, task coverage and memory addresses independently from raw cases.
"""

import contextlib
import itertools
import math
import os
from pathlib import Path

import pytest

from tests.mla.support.mla_cpu_driver import CpuDriver
from tests.mla.support.mla_split_kv_plan_build import compile_drivers


@pytest.fixture(scope="session")
def drivers(tmp_path_factory):
    directory = Path(
        os.environ.get("MLA_SCHEDULE_ARTIFACT_DIR") or tmp_path_factory.mktemp("split-kv-plan")
    )
    with contextlib.ExitStack() as stack:
        yield [stack.enter_context(CpuDriver(p)) for p in compile_drivers(directory)]


def config(b=2, q=9, k=1057, h=4, m=17, n=128, s=8, cores=3, block=2, ratio=2, lib=16777216):
    return [b, q, k, h, m, n, s, cores, block, ratio, lib]


def build(driver, c, lengths, window=(-1, -1, 0), bounds=None):
    return driver.evaluate(
        [
            [
                1,
                *c,
                *(bounds or c[1:3]),
                *window,
                len(lengths),
                *itertools.chain.from_iterable(lengths),
            ]
        ]
    )[0]


def norm(q, k, left, right, causal):
    left = -1 if left >= k - 1 else left
    right = -1 if right >= q - 1 else right
    if causal and not (q == 1 and left == right == -1):
        right = 0
    kind = 3 if left < 0 and right == 0 else 4 if left >= 0 or right >= 0 else 0
    return (
        kind,
        (2**31 - 1 if kind == 3 else k - 1 if left < 0 else left),
        (0 if kind == 3 else q - 1 if right < 0 else right),
    )


# Purposeful tuples include mask/geometry/ownership interactions; no implicit
# quota or cartesian product hides the expensive and boundary configurations.
CASES = [
    ("window-cross-wave", config(), [(9, 1057), (5, 3)], (530, 1, 0)),
    (
        "causal-original-q",
        config(q=17, h=8, m=129, n=512, s=2, cores=26, block=26),
        [(17, 1031), (9, 3)],
        (-1, -1, 1),
    ),
    (
        "leading-middle-trailing-zero-q",
        config(b=5, q=7, k=513, h=5, m=16, s=8, cores=7, block=3, ratio=1),
        [(0, 513), (7, 0), (0, 1), (2, 3), (0, 0)],
        (0, 0, 0),
    ),
    (
        "one-core-many-waves",
        config(b=1, q=3, k=513, cores=1, block=1, ratio=1),
        [(3, 513)],
        (-1, -1, 0),
    ),
    ("all-zero-used-kv", config(), [(9, 0), (4, 0)], (-1, -1, 0)),
    ("all-masked", config(q=33, k=3, m=16, n=32, s=2), [(33, 3), (17, 1)], (-1, -1, 1)),
    ("one-sided-causal", config(q=5, k=129, h=8, m=16, n=112), [(5, 129), (2, 7)], (7, -1, 1)),
    (
        "normalized-wide-window",
        config(q=1, k=16384, h=1, m=16, n=512, s=2),
        [(1, 16384), (1, 1)],
        (2**63 - 1, 2**63 - 1, 1),
    ),
    (
        "192-source-limit",
        config(b=1, q=2, k=1025, h=8, m=255, n=32, s=192),
        [(2, 1025)],
        (-1, -1, 0),
    ),
    (
        "193-explicit-extension",
        config(b=1, q=1, k=33, h=1, m=16, n=32, s=193),
        [(1, 33)],
        (0, 0, 0),
    ),
    (
        "255-repeat-boundary",
        config(b=1, q=1, k=16383, h=8, m=16, n=512, s=255),
        [(1, 16383)],
        (-1, -1, 0),
    ),
    (
        "256-split-longest",
        config(b=1, q=1, k=16384, h=8, m=16, n=32, s=256, cores=26, block=26),
        [(1, 16384)],
        (-1, -1, 0),
    ),
]


def check_layout(drivers, c):
    b, q, k, h, m, n, s, cores, block, ratio, lib = c
    got = drivers[0].evaluate([[0, *c]])[0]
    assert got["status"] == 0
    layout = got["layout"]
    empty = not (b and q and k)
    groups = 0 if empty else b * math.ceil(q * h / m)
    rows = groups * s * m
    assert layout["geometry"] == [groups, groups * s, rows, int(empty)]
    source = drivers[2].evaluate([[4, 0, b, q, b * q, h, s]])[0]
    assert source == [5, s, b, h, q, 512, 4, s, b, h, q]
    for key, shape in [("accum", [s, b, h, q, 512]), ("lse_accum", [s, b, h, q])]:
        tensor = layout[key]
        strides = [math.prod(max(1, v) for v in shape[i + 1 :]) for i in range(len(shape))]
        assert tensor == dict(
            rank=len(shape),
            shape=shape + [0] * (5 - len(shape)),
            strides=strides + [0] * (5 - len(shape)),
            elements=math.prod(shape),
            bytes=4 * math.prod(shape),
        )
    if empty:
        assert layout["buffers"] == [0] * 5 and layout["scratch"] == [0] * 12
        return layout
    source = drivers[2].evaluate([[3, m, n, q * h, cores, block, 0, 0]])[0]
    scratch = layout["scratch"]
    assert scratch[:4] + [scratch[-1]] == source
    mm1, mm2 = scratch[:2]
    offsets = [0, 2 * block * mm1 * 4, 2 * block * mm1 * 6, 2 * block * (mm1 * 6 + mm2 * 4)]
    kernel = 2 * block * (mm1 * 6 + mm2 * 8)
    assert scratch[4:9] == [*offsets, kernel]
    assert scratch[9] >= kernel
    out = ((scratch[9] + 31) // 32) * 32
    sums = out + rows * 512 * 4
    maximum = sums + rows * 8 * 4
    user = maximum + rows * 8 * 4
    assert layout["buffers"] == [out, sums, maximum, user, user + lib]
    assert all(x % 32 == 0 for x in layout["buffers"][:4])
    return layout


@pytest.mark.parametrize("name,c,lengths,window", CASES, ids=[c[0] for c in CASES])
def test_source_ranges_ownership_and_persistent_rows(drivers, name, c, lengths, window):
    b, qcap, kcap, h, m, n, s, _, cores, ratio, _ = c
    layout = check_layout(drivers, c)
    got = build(drivers[0], c, lengths, window)
    assert got["status"] == 0
    plan = got["plan"]
    prefix = [0, *itertools.accumulate(math.ceil(q * h / m) for q, _ in lengths)]
    assert plan == dict(groups=prefix[-1], tasks=prefix[-1] * s, prefix=prefix, layout=layout)
    mask, left, right = norm(qcap, kcap, *window)
    all_tasks = drivers[0].evaluate([[2, t] for t in range(plan["tasks"])])
    all_merges = drivers[0].evaluate([[3, g] for g in range(plan["groups"])])
    # Simulate the interleaved producer waves. No physical-core reuse may
    # overwrite another task's persistent slot before the merge phase.
    saved = {}
    for core in range(cores):
        for t in range(core, plan["tasks"], cores):
            task = all_tasks[t]
            assert task["status"] == 0 and task["task"][2] == t // cores
            assert task["task"][-1] == core
            for offset in task["offsets"]:
                assert offset not in saved
                saved[offset] = t
    visible_rows = set()
    for batch, (q, k) in enumerate(lengths):
        for mb in range(math.ceil(q * h / m)):
            g = prefix[batch] + mb
            begin, end = mb * m, min((mb + 1) * m, q * h)
            merge = all_merges[g]
            assert merge["status"] == 0
            assert merge["merge"] == [
                g,
                g // (cores * ratio),
                batch,
                mb,
                g % (cores * ratio),
                begin,
                end,
            ]
            spans = []
            for row in range(begin, end):
                query = row // h
                lo = max(0, query + k - q - left) if mask == 4 else 0
                hi = min(k, query + k - q + right + 1) if mask else k
                if lo < hi:
                    spans.append((lo, hi))
                assert (batch, row) not in visible_rows
                visible_rows.add((batch, row))
            lo = min((a for a, _ in spans), default=0)
            hi = max((z for _, z in spans), default=0)
            first, last = lo // n, math.ceil(hi / n)
            native = drivers[1].evaluate(
                [[0, mask, left, right, h, m, n, c[7], ratio, 1, q, k, mb]]
            )[0]
            assert native[2:4] == [first, last]
            cuts = drivers[2].evaluate([[1, first, last, s, i] for i in range(s)])
            covered = []
            for split in range(s):
                t = g * s + split
                actual = all_tasks[t]
                assert actual["task"] == [t, g, t // cores, batch, mb, split, t % cores]
                start, stop = cuts[split]
                start = min(start, stop)
                ka, kz = (max(lo, start * n), min(hi, stop * n)) if start < stop else (hi, hi)
                cost = sum(
                    6 * math.ceil((end - begin) / 16) + 10 * math.ceil(min(n, k - j * n) / 64)
                    for j in range(start, stop)
                )
                assert actual["fragment"] == [start, stop, ka, kz, cost]
                assert actual["whole"][:8] == [
                    begin,
                    end,
                    begin // h,
                    math.ceil(end / h),
                    lo,
                    hi,
                    first,
                    last,
                ]
                assert actual["whole"][9:] == native[4:7]
                covered.extend(range(start, stop))
                expected_offsets = [
                    base + t * m * width * 4 for base, width in zip(layout["buffers"], [512, 8, 8])
                ]
                assert actual["offsets"] == expected_offsets
                assert [saved[p] for p in expected_offsets] == [t] * 3
                if split == 0:
                    assert merge["offsets"] == expected_offsets
                # First/last row and a query/head boundary: these independently
                # distinguish BSND output, head-major accum and used-Q packing.
                probes = sorted({0, end - begin - 1, min(end - begin - 1, h - 1 - begin % h)})
                addresses = drivers[0].evaluate([[4, t, row] for row in probes])
                for row, address in zip(probes, addresses):
                    query, head = divmod(begin + row, h)
                    lse = ((batch * h + head) * qcap + query) * 4
                    alse = (((split * b + batch) * h + head) * qcap + query) * 4
                    assert address["status"] == 0
                    assert address["row"] == [
                        (batch * qcap * h + begin + row) * 512 * 2,
                        lse,
                        alse * 512,
                        alse,
                        *[p + row * width * 4 for p, width in zip(expected_offsets, [512, 8, 8])],
                    ]
            assert covered == list(range(first, last))
    assert visible_rows == {
        (batch, row) for batch, (q, _) in enumerate(lengths) for row in range(q * h)
    }


@pytest.mark.parametrize(
    "c,lengths",
    [
        (config(b=0), []),
        (config(q=0), [(0, 3), (0, 0)]),
        (config(k=0), [(9, 0), (3, 0)]),
        (config(), [(0, 1057), (0, 1)]),
    ],
    ids=["zero-batch", "zero-q-capacity", "zero-kv-capacity", "zero-q-used"],
)
def test_empty_capacity_and_empty_work_are_distinct(drivers, c, lengths):
    layout = check_layout(drivers, c)
    result = build(drivers[0], c, lengths)
    assert result["status"] == 0
    assert result["plan"] == dict(groups=0, tasks=0, prefix=[0] * (c[0] + 1), layout=layout)
    for item in drivers[0].evaluate([[2, 0], [3, 0], [4, 0, 0]]):
        assert item["status"] != 0 and item["unchanged"]
    if c[0]:
        bad = [*lengths]
        bad[-1] = (bad[-1][0], c[2] + 1)
        failure = build(drivers[0], c, bad)
        assert failure["status"] != 0 and failure["unchanged"]


INVALID = [
    (0, -1),
    (0, 2**32),
    (1, -1),
    (1, 2**30 + 1),
    (2, -1),
    (2, 16385),
    (3, 0),
    (3, (2**32 - 1) // 512 + 1),
    (4, 0),
    (4, 257),
    (5, 0),
    (5, 513),
    (6, -1),
    (6, 0),
    (6, 1),
    (6, 257),
    (6, 2**32 + 2),
    (7, 0),
    (7, 27),
    (8, 0),
    (8, 4),
    (9, 0),
    (9, 3),
    (10, 2**64 - 1),
]


@pytest.mark.parametrize("field,value", INVALID)
def test_invalid_config_is_atomic(drivers, field, value):
    c = config()
    assert build(drivers[0], c, [(9, 1057), (5, 3)])["status"] == 0
    before = drivers[0].evaluate([[2, 3], [4, 3, 0]])
    c[field] = value
    result = drivers[0].evaluate([[0, *c]])[0]
    assert result["status"] != 0 and result["unchanged"]
    failure = build(drivers[0], c, [(9, 1057), (5, 3)])
    assert failure["status"] != 0 and failure["unchanged"]
    assert drivers[0].evaluate([[2, 3], [4, 3, 0]]) == before


def test_lengths_window_query_errors_and_replacement(drivers):
    c = config()
    baseline = build(drivers[0], c, [(9, 1057), (5, 3)])
    before = drivers[0].evaluate([[2, 3], [4, 3, 0]])
    for lengths in [
        [(9, 1057)],
        [(9, 1057), (-1, 3)],
        [(9, 1057), (10, 0)],
        [(9, 1057), (1, -1)],
        [(9, 1057), (0, 1058)],
    ]:
        result = build(drivers[0], c, lengths)
        assert result["status"] != 0 and result["unchanged"]
    for bounds, window in [
        ((8, 1057), (-1, -1, 0)),
        ((9, 1056), (-1, -1, 0)),
        ((9, 1057), (-(2**63), 0, 0)),
    ]:
        result = build(drivers[0], c, [(9, 1057), (5, 3)], window, bounds)
        assert result["status"] != 0 and result["unchanged"]
    plan = baseline["plan"]
    for query in [
        [2, plan["tasks"]],
        [2, 2**64 - 1],
        [3, plan["groups"]],
        [3, 2**64 - 1],
        [4, 0, c[4]],
        [4, 0, 2**32 - 1],
        [4, plan["tasks"], 0],
    ]:
        result = drivers[0].evaluate([query])[0]
        assert result["status"] != 0 and result["unchanged"]
    assert drivers[0].evaluate([[2, 3], [4, 3, 0]]) == before
    replacement = build(drivers[0], c, [(0, 1), (1, 0)], (0, 0, 0))
    assert replacement["status"] == 0
    assert replacement["plan"]["prefix"] == [0, 0, 1]
    assert drivers[0].evaluate([[2, 0]])[0]["task"][3:6] == [1, 0, 0]


def test_u64_task_ids_and_last_row_without_enumeration(drivers):
    c = config(b=4, q=2**30, k=16384, h=1, m=1, s=256, cores=26, block=26)
    result = build(drivers[0], c, [(2**30, 16384)] * 4)
    assert result["status"] == 0
    tasks = result["plan"]["tasks"]
    assert tasks == 2**40
    last = drivers[0].evaluate([[2, tasks - 1], [4, tasks - 1, 0]])
    assert last[0]["task"] == [
        tasks - 1,
        2**32 - 1,
        (tasks - 1) // 26,
        3,
        2**30 - 1,
        255,
        (tasks - 1) % 26,
    ]
    assert last[1]["row"][:4] == [
        (2**32 - 1) * 1024,
        (2**32 - 1) * 4,
        (2**40 - 1) * 2048,
        (2**40 - 1) * 4,
    ]


def test_capacity_cost_boundary_and_byte_overflow(drivers):
    c = config(b=1, q=2**30, h=1, m=256, s=256, cores=26, block=26, lib=0)
    layout = check_layout(drivers, c)
    per_batch = c[1] * c[6] * (512 + 16) * 4
    maximum = (2**63 - 1 - layout["buffers"][0]) // per_batch
    c[0] = maximum
    largest = drivers[0].evaluate([[0, *c]])[0]
    assert largest["status"] == 0
    assert largest["layout"]["buffers"][-1] <= 2**63 - 1
    c[0] += 1
    too_large = drivers[0].evaluate([[0, *c]])[0]
    assert too_large["status"] == 5 and too_large["unchanged"]
    c = config()
    size = drivers[0].evaluate([[0, *c]])[0]["layout"]["buffers"][-2]
    c[-1] = 2**63 - 1 - size
    assert drivers[0].evaluate([[0, *c]])[0]["layout"]["buffers"][-1] == 2**63 - 1
    c[-1] += 1
    assert drivers[0].evaluate([[0, *c]])[0]["status"] == 5
    # Field/stride products must be rejected even when another axis is empty.
    for c in [
        config(b=0, q=2**30, h=8),
        config(b=2**32 - 1, q=2**30, h=1),
        config(b=2**32 - 1, q=0, h=(2**32 - 1) // 512, s=256),
    ]:
        result = drivers[0].evaluate([[0, *c]])[0]
        assert result["status"] != 0 and result["unchanged"]
