"""Native variable partitions, persistent FD slots and independent row coverage.

All oracle executables are compiled from pinned blobs for this module. Tile
ownership is enumerated independently from raw lengths and visible keys; no
historical unit test results or mathematical attention references are reused.
"""

import contextlib
import itertools
import math
import os
from pathlib import Path

import pytest

from tests.mla.support.mla_auto_split_kv_plan_build import compile_drivers
from tests.mla.support.mla_cpu_driver import CpuDriver


@pytest.fixture(scope="session")
def drivers(tmp_path_factory):
    directory = Path(
        os.environ.get("MLA_SCHEDULE_ARTIFACT_DIR") or tmp_path_factory.mktemp("auto-plan")
    )
    with contextlib.ExitStack() as stack:
        yield [stack.enter_context(CpuDriver(p)) for p in compile_drivers(directory)]


def config(b=3, q=17, k=8193, h=8, m=64, n=512, cores=7, ratio=2, lib=16777216):
    return [b, q, k, h, m, n, cores, ratio, lib]


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


def norm(c, window):
    _, q, k, *_ = c
    left, right, causal = window
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


def geometry(q, k, h, m, n, block, mask):
    kind, left, right = mask
    begin, end = block * m, min((block + 1) * m, q * h)
    spans = []
    for row in range(begin, end):
        query = row // h
        lo = max(0, query + k - q - left) if kind == 4 else 0
        hi = min(k, query + k - q + right + 1) if kind else k
        if lo < hi:
            spans.append((lo, hi))
    lo, hi = min((a for a, _ in spans), default=0), max((z for _, z in spans), default=0)
    return begin, end, lo, hi, lo // n, math.ceil(hi / n)


# Explicit tuples preserve interactions and independent boundaries, not a quota.
CASES = [
    ("representative-variable-fd", config(), [(17, 8193), (7, 4097), (2, 1)], (-1, -1, 0)),
    ("representative-causal-fd", config(cores=26), [(17, 8193), (7, 4097), (2, 1)], (-1, -1, 1)),
    ("representative-window-fd", config(m=17, n=112), [(17, 8193), (7, 4097), (2, 1)], (530, 1, 0)),
    ("one-core-no-fd", config(cores=1, ratio=1), [(17, 8193), (7, 4097), (2, 1)], (-1, -1, 0)),
    (
        "26-core-single-m",
        config(b=1, q=1, k=16384, h=128, m=256, cores=26),
        [(1, 16384)],
        (-1, -1, 0),
    ),
    (
        "fd-row-tail",
        config(b=1, q=3, k=16383, h=5, m=17, n=128, cores=26, ratio=1),
        [(3, 16383)],
        (-1, -1, 1),
    ),
    (
        "leading-middle-trailing-empty",
        config(b=5, q=7, k=513, h=5, m=16, n=128),
        [(0, 513), (7, 0), (0, 1), (2, 513), (7, 0)],
        (0, 0, 0),
    ),
    ("masked-prefix", config(q=33, k=3, m=16, n=3), [(33, 3), (17, 1), (1, 0)], (-1, -1, 1)),
    (
        "single-tile-no-fd",
        config(q=2, k=1, h=1, m=1, n=1, cores=26),
        [(2, 1), (1, 1), (2, 0)],
        (-1, -1, 0),
    ),
    (
        "one-sided-left",
        config(q=9, k=1057, m=31, n=32, cores=20, ratio=1),
        [(9, 1057), (5, 3), (0, 4)],
        (7, -1, 0),
    ),
    (
        "one-sided-right",
        config(q=9, k=1057, m=129, n=448, cores=2),
        [(9, 1057), (5, 3), (2, 0)],
        (-1, 1, 0),
    ),
    ("causal-with-window", config(q=5, k=129, m=16, n=112), [(5, 129), (2, 7), (1, 1)], (7, -1, 1)),
    (
        "wide-window-normalizes",
        config(b=1, q=1, k=16384, h=1, m=1),
        [(1, 16384)],
        (2**63 - 1, 2**63 - 1, 1),
    ),
    ("all-kv-used-zero", config(), [(17, 0), (4, 0), (0, 0)], (-1, -1, 0)),
    ("all-q-used-zero", config(), [(0, 8193), (0, 1), (0, 0)], (0, 0, 0)),
    ("zero-batch", config(b=0), [], (-1, -1, 0)),
    ("zero-q-capacity", config(q=0), [(0, 8193), (0, 1), (0, 0)], (-1, -1, 0)),
    ("zero-kv-capacity", config(k=0), [(17, 0), (4, 0), (0, 0)], (-1, -1, 1)),
]


def check_layout(drivers, c):
    b, q, k, h, m, n, cores, _, lib = c
    result = drivers[0].evaluate([[0, *c]])[0]
    assert result["status"] == 0
    layout = result["layout"]
    empty = not (b and q and k)
    groups = b * math.ceil(q * h / m) if not empty else 0
    slots = 2 * cores if cores > 1 and not empty else 0
    rows = slots * m
    assert layout["geometry"] == [groups, slots, rows, int(empty)]
    shapes = drivers[2].evaluate([[4, 0, b, q, b * q, h, cores]])[0]
    assert shapes == ([0, 0] if cores == 1 else [5, cores, b, h, q, 512, 4, cores, b, h, q])
    for key, shape in [("accum", [cores, b, h, q, 512]), ("lse_accum", [cores, b, h, q])]:
        if cores == 1:
            assert layout[key] == dict(rank=0, shape=[0] * 5, strides=[0] * 5, elements=0, bytes=0)
        else:
            assert layout[key] == dict(
                rank=len(shape),
                shape=shape + [0] * (5 - len(shape)),
                strides=[math.prod(max(1, v) for v in shape[i + 1 :]) for i in range(len(shape))]
                + [0] * (5 - len(shape)),
                elements=math.prod(shape),
                bytes=4 * math.prod(shape),
            )
    if empty:
        assert layout["buffers"] == [0] * 5 and layout["scratch"] == [0] * 12
        return layout
    scratch = layout["scratch"]
    assert (
        scratch[:4] + [scratch[-1]]
        == drivers[2].evaluate([[3, m, n, q * h, cores, cores, 0, 0]])[0]
    )
    mm1, mm2 = scratch[:2]
    assert scratch[4:9] == [
        0,
        8 * cores * mm1,
        12 * cores * mm1,
        cores * (12 * mm1 + 8 * mm2),
        cores * (12 * mm1 + 16 * mm2),
    ]
    assert scratch[9] >= scratch[8]
    out = (scratch[9] + 31) // 32 * 32
    sums, maximum = out + rows * 512 * 4, out + rows * (512 + 8) * 4
    user = maximum + rows * 8 * 4
    assert layout["buffers"] == [out, sums, maximum, user, user + lib]
    return layout


@pytest.mark.parametrize("name,c,lengths,window", CASES, ids=[c[0] for c in CASES])
def test_source_and_independent_ownership(drivers, name, c, lengths, window):
    b, qcap, kcap, h, m, n, cores, ratio, _ = c
    layout = check_layout(drivers, c)
    result = build(drivers[0], c, lengths, window)
    assert result["status"] == 0
    p = result["plan"]
    native = drivers[1].evaluate(
        [[1, *norm(c, window), h, m, n, cores, ratio, b, *itertools.chain.from_iterable(lengths)]]
    )[0]
    assert p["native"][:7] == native[:7]
    arrays = p["arrays"]
    offset = 7
    for index, a in enumerate(arrays):
        count = cores if index < 9 else cores * ratio
        assert a[:count] == native[offset : offset + count] and not any(a[count:])
        offset += count
    assert offset == len(native)
    empty = layout["geometry"][-1]
    prefix = [0, *itertools.accumulate(0 if empty else math.ceil(q * h / m) for q, _ in lengths)]
    assert p["prefix"] == prefix and p["groups"] == prefix[-1]
    all_geometry, tiles = {}, []
    mask = norm(c, window)
    for batch, (q, k) in enumerate(lengths):
        for mb in range(0 if empty else math.ceil(q * h / m)):
            g = geometry(q, k, h, m, n, mb, mask)
            all_geometry[batch, mb] = g
            tiles.extend((batch, mb, j) for j in range(g[4], g[5]))
    used, maximum, fd_count, max_splits, vectors, total, tile_count, _ = p["native"]
    owners = {}
    costs = [0] * used
    for tile in tiles:
        batch, mb, j = tile
        owner = next(i for i in range(used) if tile < tuple(a[i] for a in arrays[:3]))
        owners.setdefault((batch, mb), {}).setdefault(owner, []).append(j)
        rb, re, *_ = all_geometry[batch, mb]
        costs[owner] += 6 * math.ceil((re - rb) / 16) + 10 * math.ceil(
            min(n, lengths[batch][1] - j * n) / 64
        )
    assert (len(tiles), sum(costs), max(costs, default=0)) == (tile_count, total, maximum)
    fd_keys = sorted(key for key, parts in owners.items() if len(parts) > 1)
    assert list(zip(arrays[3][:fd_count], arrays[4][:fd_count])) == fd_keys
    assert max_splits == max((len(owners[key]) for key in fd_keys), default=0)
    fd_prefix = [0, *itertools.accumulate(len(owners[key]) for key in fd_keys)]
    assert p["slots"] == fd_prefix[-1] <= layout["geometry"][1]
    assert p["partial_prefix"] == fd_prefix + [0] * (27 - len(fd_prefix))
    # Reconstruct the source's first split index of every active AIC and the
    # two-FD-head maximum underlying our static persistent-slot bound.
    for core in range(used):
        core_keys = sorted(key for key, parts in owners.items() if core in parts)
        if core_keys:
            assert arrays[6][core] == list(owners[core_keys[0]]).index(core)
        assert sum(core in owners[key] for key in fd_keys) <= 2
    merge_keys = [
        (head, part)
        for head, key in enumerate(fd_keys)
        for part in range(math.ceil((all_geometry[key][1] - all_geometry[key][0]) / 8))
    ]
    vector_ends = [merge_keys.index((arrays[9][i], arrays[10][i])) + 1 for i in range(vectors)]
    assert not merge_keys or vector_ends[-1] == len(merge_keys)
    assert vector_ends == sorted(set(vector_ends))
    saved, merged = {}, set()
    observed_kinds = set()
    for (batch, mb), g in all_geometry.items():
        rb, re, lo, hi, first, last = g
        group = prefix[batch] + mb
        parts = list(owners.get((batch, mb), {}).items())
        count = len(parts) or 1
        for split in range(count):
            actual = drivers[0].evaluate([[2, group, split]])[0]
            assert actual["status"] == 0
            fd_head = fd_keys.index((batch, mb)) if len(parts) > 1 else 2**32 - 1
            slot = fd_prefix[fd_head] + split if len(parts) > 1 else 2**64 - 1
            kind = 1 if len(parts) > 1 else 0 if parts else 2
            observed_kinds.add(kind)
            owner = parts[split][0] if parts else group % cores
            assert actual["task"] == [group, slot, batch, mb, split, count, owner, fd_head, kind]
            start, stop = (parts[split][1][0], parts[split][1][-1] + 1) if parts else (last, last)
            ka, kz = (max(lo, start * n), min(hi, stop * n)) if parts else (hi, hi)
            cost = sum(
                6 * math.ceil((re - rb) / 16)
                + 10 * math.ceil(min(n, lengths[batch][1] - j * n) / 64)
                for j in range(start, stop)
            )
            assert actual["fragment"] == [start, stop, ka, kz, cost]
            assert actual["whole"][:8] == [rb, re, rb // h, math.ceil(re / h), lo, hi, first, last]
            offsets = (
                [base + slot * m * width * 4 for base, width in zip(layout["buffers"], [512, 8, 8])]
                if kind == 1
                else [0] * 3
            )
            assert actual["offsets"] == offsets
            if kind == 1:
                assert slot not in saved
                saved[slot] = (group, split)
            for row in sorted({0, re - rb - 1, min(re - rb - 1, h - 1 - rb % h)}):
                address = drivers[0].evaluate([[4, group, split, row]])[0]
                query, head = divmod(rb + row, h)
                lse = ((batch * h + head) * qcap + query) * 4
                accum = (((split * b + batch) * h + head) * qcap + query) * 4 if cores > 1 else 0
                partial = (
                    [x + row * width * 4 for x, width in zip(offsets, [512, 8, 8])]
                    if kind == 1
                    else [2**64 - 1] * 3
                )
                assert address["status"] == 0 and address["row"] == [
                    (batch * qcap * h + rb + row) * 1024,
                    lse,
                    accum * 512,
                    accum,
                    *partial,
                ]
        invalid = drivers[0].evaluate([[2, group, count], [4, group, 0, re - rb]])
        assert all(x["status"] != 0 and x["unchanged"] for x in invalid)
    for head, key in enumerate(fd_keys):
        batch, mb = key
        rb, re, *_ = all_geometry[key]
        count = len(owners[key])
        assert arrays[5][head] == count and arrays[7][head] == math.ceil((re - rb) / 8)
        assert arrays[8][head] == (re - rb - 1) % 8 + 1
        for part in range(math.ceil((re - rb) / 8)):
            owner = next(i for i in range(vectors) if (head, part) <= (arrays[9][i], arrays[10][i]))
            result = drivers[0].evaluate([[3, head, part]])[0]
            begin, end = rb + part * 8, min(re, rb + (part + 1) * 8)
            group = prefix[batch] + mb
            assert result["status"] == 0 and result["merge"] == [
                group,
                fd_prefix[head],
                batch,
                mb,
                head,
                part,
                owner,
                count,
                begin,
                end,
            ]
            assert result["offsets"] == [
                base + (fd_prefix[head] * m + part * 8) * width * 4
                for base, width in zip(layout["buffers"], [512, 8, 8])
            ]
            for row in range(begin, end):
                assert (group, row) not in merged
                merged.add((group, row))
            assert [saved[fd_prefix[head] + s] for s in range(count)] == [
                (group, s) for s in range(count)
            ]
    assert merged == {
        (prefix[batch] + mb, row)
        for batch, mb in fd_keys
        for row in range(all_geometry[batch, mb][0], all_geometry[batch, mb][1])
    }
    assert sorted(saved) == list(range(p["slots"]))
    for query in [[2, p["groups"], 0], [3, fd_count, 0], [3, 0, 2**32 - 1]]:
        error = drivers[0].evaluate([query])[0]
        assert error["status"] != 0 and error["unchanged"]
    if "representative" in name or name in ("26-core-single-m", "fd-row-tail"):
        assert fd_count > 0
    if name in ("one-core-no-fd", "single-tile-no-fd"):
        assert fd_count == 0 and 0 in observed_kinds


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
    (6, 0),
    (6, 27),
    (7, 0),
    (7, 3),
    (8, 2**64 - 1),
]


@pytest.mark.parametrize("field,value", INVALID)
def test_invalid_config_is_atomic(drivers, field, value):
    c = config()
    lengths = [(17, 8193), (7, 4097), (2, 1)]
    assert build(drivers[0], c, lengths)["status"] == 0
    before = drivers[0].evaluate([[2, 0, 0], [4, 0, 0, 0]])
    c[field] = value
    for got in [drivers[0].evaluate([[0, *c]])[0], build(drivers[0], c, lengths)]:
        assert got["status"] != 0 and got["unchanged"]
    assert drivers[0].evaluate([[2, 0, 0], [4, 0, 0, 0]]) == before


def test_lengths_window_errors_replacement_and_lifetime(drivers):
    c = config()
    lengths = [(17, 8193), (7, 4097), (2, 1)]
    assert build(drivers[0], c, lengths)["status"] == 0
    before = drivers[0].evaluate([[2, 0, 0], [4, 0, 0, 0]])
    for bad in [
        lengths[:-1],
        lengths + [(0, 0)],
        [(18, 1), *lengths[1:]],
        [(0, 8194), *lengths[1:]],
        [(-1, 0), *lengths[1:]],
        [(0, -1), *lengths[1:]],
    ]:
        result = build(drivers[0], c, bad)
        assert result["status"] != 0 and result["unchanged"]
    for bounds, window in [
        ((16, 8193), (-1, -1, 0)),
        ((17, 8192), (-1, -1, 0)),
        ((17, 8193), (-(2**63), 0, 0)),
    ]:
        result = build(drivers[0], c, lengths, window, bounds)
        assert result["status"] != 0 and result["unchanged"]
    for query in [[2, 2**64 - 1, 0], [2, 0, 2**32 - 1], [3, 2**32 - 1, 0], [4, 0, 0, 2**32 - 1]]:
        error = drivers[0].evaluate([query])[0]
        assert error["status"] != 0 and error["unchanged"]
    assert drivers[0].evaluate([[2, 0, 0], [4, 0, 0, 0]]) == before
    replacement = build(drivers[0], c, [(0, 1), (1, 0), (0, 0)], (0, 0, 0))
    assert replacement["status"] == 0 and replacement["plan"]["prefix"] == [0, 0, 1, 1]
    assert drivers[0].evaluate([[2, 0, 0]])[0]["task"][2:4] == [1, 0]


def test_capacity_cost_boundary_and_overflow(drivers):
    # Large logical plans store only batch/core prefixes. No tensor is allocated.
    c = config(b=5, q=2**30, k=1, h=1, m=1, cores=26, lib=0)
    layout = check_layout(drivers, c)
    result = build(drivers[0], c, [(2**30, 0)] * 5)
    assert result["status"] == 0 and result["plan"]["groups"] == 5 * 2**30
    last = 5 * 2**30 - 1
    t, r = drivers[0].evaluate([[2, last, 0], [4, last, 0, 0]])
    assert t["task"] == [last, 2**64 - 1, 4, 2**30 - 1, 0, 1, last % 26, 2**32 - 1, 2]
    assert r["row"][:4] == [last * 1024, last * 4, last * 2048, last * 4]
    # Source rectangular tile counter must reject before allocating/enumerating.
    bad = build(drivers[0], c, [(2**30, 1)] * 5)
    assert bad["status"] != 0 and bad["unchanged"]
    assert layout["geometry"][2] == 52
    c = config(lib=0)
    size = check_layout(drivers, c)["buffers"][-1]
    c[-1] = 2**63 - 1 - size
    assert drivers[0].evaluate([[0, *c]])[0]["layout"]["buffers"][-1] == 2**63 - 1
    c[-1] += 1
    error = drivers[0].evaluate([[0, *c]])[0]
    assert error["status"] != 0 and error["unchanged"]
    for bad in [
        config(b=0, q=2**30, h=8),
        config(b=2**32 - 1, q=2**30, h=1),
        config(b=2**32 - 1, q=0, h=(2**32 - 1) // 512),
        config(b=2**20, q=2**30, h=1),
    ]:
        error = drivers[0].evaluate([[0, *bad]])[0]
        assert error["status"] != 0 and error["unchanged"]


def test_default_plan_queries_are_atomic(drivers):
    # A separate fresh process has no previously built plan.
    with CpuDriver(
        drivers[0].binary, log_base=drivers[0].binary.with_name("default-plan")
    ) as fresh:
        for result in fresh.evaluate([[2, 0, 0], [3, 0, 0], [4, 0, 0, 0]]):
            assert result["status"] != 0 and result["unchanged"]
