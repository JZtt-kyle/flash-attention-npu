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

from tests.mla.support.mla_auto_split_tnd_plan_build import compile_drivers
from tests.mla.support.mla_cpu_driver import CpuDriver


@pytest.fixture(scope="session")
def drivers(tmp_path_factory):
    directory = Path(
        os.environ.get("MLA_SCHEDULE_ARTIFACT_DIR") or tmp_path_factory.mktemp("auto-tnd-plan")
    )
    with contextlib.ExitStack() as stack:
        yield [stack.enter_context(CpuDriver(p)) for p in compile_drivers(directory)]


def config(b=2, q=17, k=8193, h=8, m=64, n=512, cores=7, ratio=2, lib=16777216, tq=26, tk=8196):
    return [b, q, k, h, m, n, cores, ratio, lib, tq, tk]


def allocation_prefix(lengths):
    return [0, *itertools.accumulate(lengths)]


def build(driver, c, arrays=None, window=(-1, -1, 0), bounds=None, nulls=()):
    arrays = arrays if arrays is not None else [[0, 17, 26], [0, 8193, 8196], None, None]
    encoded = []
    for i, values in enumerate(arrays):
        encoded.extend(
            [int(values is not None and i not in nulls), len(values or []), *(values or [])]
        )
    return driver.evaluate([[1, *c, *(bounds or c[1:3]), *window, *encoded]])[0]


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
    (
        "representative-variable-fd-gaps",
        {},
        [(17, 8193), (9, 4097), (3, 1)],
        [13, 7, 2],
        [8191, 4095, 1],
        (-1, -1, 0),
    ),
    (
        "representative-causal-fd-gaps",
        dict(cores=26),
        [(17, 8193), (9, 4097), (3, 1)],
        [13, 7, 2],
        [8191, 4095, 1],
        (-1, -1, 1),
    ),
    (
        "representative-window-fd-gaps",
        dict(m=17, n=112),
        [(17, 8193), (9, 4097), (3, 1)],
        [13, 7, 2],
        [8191, 4095, 1],
        (530, 1, 0),
    ),
    (
        "one-core-no-fd",
        dict(cores=1, ratio=1),
        [(17, 8193), (7, 4097), (2, 1)],
        None,
        None,
        (-1, -1, 0),
    ),
    (
        "26-core-single-m",
        dict(q=1, k=16384, h=128, m=256, cores=26),
        [(1, 16384)],
        None,
        None,
        (-1, -1, 0),
    ),
    (
        "fd-row-tail",
        dict(q=3, k=16384, h=5, m=17, n=128, cores=26, ratio=1),
        [(3, 16384)],
        None,
        [16383],
        (-1, -1, 1),
    ),
    (
        "leading-middle-trailing-empty",
        dict(q=7, k=513, h=5, m=16, n=128),
        [(0, 513), (7, 0), (0, 1), (2, 513), (7, 0)],
        None,
        None,
        (0, 0, 0),
    ),
    (
        "masked-prefix",
        dict(q=33, k=3, m=16, n=3),
        [(33, 3), (17, 1), (1, 0)],
        None,
        None,
        (-1, -1, 1),
    ),
    (
        "single-tile-no-fd",
        dict(q=2, k=1, h=1, m=1, n=1, cores=26),
        [(2, 1), (1, 1), (2, 0)],
        None,
        None,
        (-1, -1, 0),
    ),
    (
        "one-sided-left",
        dict(q=9, k=1057, m=31, n=32, cores=20, ratio=1),
        [(9, 1057), (5, 3), (0, 4)],
        None,
        None,
        (7, -1, 0),
    ),
    (
        "one-sided-right",
        dict(q=9, k=1057, m=129, n=448, cores=2),
        [(9, 1057), (5, 3), (2, 0)],
        None,
        None,
        (-1, 1, 0),
    ),
    (
        "causal-with-window",
        dict(q=5, k=129, m=16, n=112),
        [(5, 129), (2, 7), (1, 1)],
        None,
        None,
        (7, -1, 1),
    ),
    (
        "wide-window-normalizes",
        dict(q=1, k=16384, h=1, m=1),
        [(1, 16384)],
        None,
        None,
        (2**63 - 1, 2**63 - 1, 1),
    ),
    ("all-kv-used-zero", {}, [(17, 8193), (9, 3)], [17, 4], [0, 0], (-1, -1, 0)),
    ("all-q-used-zero", {}, [(17, 8193), (9, 3)], [0, 0], None, (0, 0, 0)),
    (
        "direct-fd-empty-mix",
        dict(q=7, k=16384, h=8, m=16, n=128, cores=7),
        [(7, 16384), (3, 1), (2, 0)],
        [5, 2, 1],
        None,
        (-1, -1, 0),
    ),
    ("zero-batch", {}, [], None, None, (-1, -1, 0)),
    ("zero-q-total-positive-max", {}, [(0, 8193), (0, 1)], None, None, (-1, -1, 0)),
    ("zero-kv-total-positive-max", {}, [(17, 0), (4, 0)], None, None, (-1, -1, 1)),
    ("zero-q-max", dict(q=0), [(0, 8193)], None, None, (-1, -1, 0)),
    ("zero-kv-max", dict(k=0), [(17, 0)], None, None, (-1, -1, 1)),
]


def check_layout(drivers, c):
    b, q, k, h, m, n, cores, _, lib, tq, tk = c
    result = drivers[0].evaluate([[0, *c]])[0]
    assert result["status"] == 0
    layout = result["layout"]
    empty = not (tq and tk)
    active = min(b, tq)
    groups = (
        min(active * math.ceil(q * h / m), (tq * h + active * (m - 1)) // m) if not empty else 0
    )
    slots = 2 * cores if cores > 1 and not empty else 0
    rows = slots * m
    assert layout["geometry"] == [groups, slots, rows, int(empty)]
    shapes = drivers[2].evaluate([[4, 1, b, q, tq, h, cores]])[0]
    assert shapes == ([0, 0] if cores == 1 else [4, cores, h, tq, 512, 3, cores, h, tq])
    for key, shape in [("accum", [cores, h, tq, 512]), ("lse_accum", [cores, h, tq])]:
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


@pytest.mark.parametrize("name,overrides,alloc,qu,ku,window", CASES, ids=[c[0] for c in CASES])
def test_source_and_independent_ownership(drivers, name, overrides, alloc, qu, ku, window):
    cq, ck = allocation_prefix(x[0] for x in alloc), allocation_prefix(x[1] for x in alloc)
    c = config(b=len(alloc), tq=cq[-1], tk=ck[-1], **overrides)
    b, qcap, kcap, h, m, n, cores, ratio, _, tq, tk = c
    lengths = list(
        zip(
            qu if qu is not None else [x[0] for x in alloc],
            ku if ku is not None else [x[1] for x in alloc],
        )
    )
    layout = check_layout(drivers, c)
    result = build(drivers[0], c, [cq, ck, qu, ku], window)
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
            assert actual["sequence"] == [cq[batch], ck[batch], *alloc[batch], *lengths[batch]]
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
                token = cq[batch] + query
                lse = (head * tq + token) * 4
                accum = ((split * h + head) * tq + token) * 4 if cores > 1 else 0
                partial = (
                    [x + row * width * 4 for x, width in zip(offsets, [512, 8, 8])]
                    if kind == 1
                    else [2**64 - 1] * 3
                )
                assert address["status"] == 0 and address["row"] == [
                    (token * h + head) * 1024,
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
    if name == "direct-fd-empty-mix":
        assert observed_kinds == {0, 1, 2}
    if "representative" in name or name in ("26-core-single-m", "fd-row-tail"):
        assert fd_count > 0
    if name in ("one-core-no-fd", "single-tile-no-fd"):
        assert fd_count == 0 and 0 in observed_kinds


@pytest.mark.parametrize("axis", range(4), ids=["cu-q", "cu-kv", "used-q", "used-kv"])
def test_independent_prefix_used_counts_values_and_nulls(drivers, axis):
    c = config()
    arrays = [[0, 17, 26], [0, 8193, 8196], [4, 3], [7, 1]]
    assert build(drivers[0], c, arrays)["status"] == 0
    before = drivers[0].evaluate([[2, 0, 0], [4, 0, 0, 0]])
    bad_values = [[], [1], arrays[axis] + [0], None] if axis < 2 else [[], [1], arrays[axis] + [0]]
    if axis < 2:
        limit = c[axis + 1]
        end = arrays[axis][-1]
        bad_values += [
            [1, arrays[axis][1], end],
            [0, -1, end],
            [0, -(2**31), end],
            [0, end + 1, end],
            [0, arrays[axis][1], end - 1],
            [0, limit + 1, end],
        ]
    else:
        capacity = arrays[axis - 2][-1] - arrays[axis - 2][-2]
        bad_values += [[1, -1], [1, -(2**31)], [1, capacity + 1]]
    for values in bad_values:
        bad = list(arrays)
        bad[axis] = values
        got = build(drivers[0], c, bad)
        assert got["status"] != 0 and got["unchanged"]
    got = build(drivers[0], c, arrays, nulls=[axis])
    assert got["status"] != 0 and got["unchanged"]
    assert drivers[0].evaluate([[2, 0, 0], [4, 0, 0, 0]]) == before


def test_window_query_errors_and_replacement(drivers):
    c = config()
    baseline = build(drivers[0], c)["plan"]
    before = drivers[0].evaluate([[2, 0, 0], [4, 0, 0, 0]])
    for bounds, window in [
        ((16, 8193), (-1, -1, 0)),
        ((17, 8192), (-1, -1, 0)),
        ((17, 8193), (-(2**63), 0, 0)),
    ]:
        result = build(drivers[0], c, window=window, bounds=bounds)
        assert result["status"] != 0 and result["unchanged"]
    for query in [
        [2, baseline["groups"], 0],
        [2, 2**64 - 1, 0],
        [2, 0, 2**32 - 1],
        [3, 2**32 - 1, 0],
        [3, 0, 2**32 - 1],
        [4, 0, 0, c[4]],
        [4, 0, 0, 2**32 - 1],
        [4, baseline["groups"], 0, 0],
    ]:
        result = drivers[0].evaluate([query])[0]
        assert result["status"] != 0 and result["unchanged"]
    assert drivers[0].evaluate([[2, 0, 0], [4, 0, 0, 0]]) == before
    result = build(drivers[0], c, [[0, 17, 26], [0, 8193, 8196], [0, 1], [0, 0]])
    assert result["status"] == 0 and result["plan"]["prefix"] == [0, 0, 1]
    actual = drivers[0].evaluate([[2, 0, 0], [4, 0, 0, 0]])
    assert actual[0]["sequence"] == [17, 8193, 9, 3, 1, 0]
    assert actual[1]["row"][0] == 17 * 8 * 1024


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
    (9, -1),
    (9, 2**31),
    (10, -1),
    (10, 2**31),
    (9, 35),
    (10, 16387),
]


@pytest.mark.parametrize("field,value", INVALID)
def test_invalid_config_is_atomic(drivers, field, value):
    c = config()
    assert build(drivers[0], c)["status"] == 0
    before = drivers[0].evaluate([[2, 0, 0], [4, 0, 0, 0]])
    c[field] = value
    for result in [drivers[0].evaluate([[0, *c]])[0], build(drivers[0], c)]:
        assert result["status"] != 0 and result["unchanged"]
    assert drivers[0].evaluate([[2, 0, 0], [4, 0, 0, 0]]) == before


def test_static_bound_against_small_allocation_partitions(drivers):
    for b, q, h, m in itertools.product(range(1, 5), range(1, 5), [1, 3, 8], [1, 4, 7]):
        maxima = [0] * (b * q + 1)
        for alloc in itertools.product(range(q + 1), repeat=b):
            t = sum(alloc)
            maxima[t] = max(maxima[t], sum(math.ceil(v * h / m) for v in alloc))
        for t in range(b * q + 1):
            c = config(b=b, q=q, k=1, h=h, m=m, tq=t, tk=1)
            got = drivers[0].evaluate([[0, *c]])[0]
            assert got["status"] == 0
            assert maxima[t] <= got["layout"]["geometry"][0] <= b * math.ceil(q * h / m)
    # B*maxQ would overflow, but totalQ=2 is valid and tightly bounded.
    c = config(b=2**32 - 1, q=2**30, k=16384, h=1, m=256, tq=2, tk=1)
    assert check_layout(drivers, c)["geometry"][:3] == [2, 14, 3584]


def test_capacity_cost_boundary_u64_groups_and_overflow(drivers):
    # >uint32 M prefix, without enumerating tasks or allocating tensors.
    c = config(q=2**29, k=1, h=5, m=1, cores=26, lib=0, tq=2**30, tk=2)
    arrays = [[0, 2**29, 2**30], [0, 1, 2], None, [0, 0]]
    layout = check_layout(drivers, c)
    result = build(drivers[0], c, arrays)
    assert result["status"] == 0 and result["plan"]["groups"] == 5 * 2**30
    last = 5 * 2**30 - 1
    task, row = drivers[0].evaluate([[2, last, 0], [4, last, 0, 0]])
    assert task["task"] == [last, 2**64 - 1, 1, 5 * 2**29 - 1, 0, 1, last % 26, 2**32 - 1, 2]
    lse = (4 * 2**30 + (2**30 - 1)) * 4
    assert row["row"][:4] == [last * 1024, lse, lse * 512, lse]
    # Source uint32 rectangular tile counter overflow is rejected atomically.
    bad = build(drivers[0], c, [*arrays[:3], None])
    assert bad["status"] != 0 and bad["unchanged"]
    assert layout["geometry"][2] == 52
    # Largest int32 prefix terminal, sparse used rows after a huge gap.
    c = config(b=2, q=2**30, h=1, m=256, tq=2**31 - 1, tk=8196)
    result = build(drivers[0], c, [[0, 2**30, 2**31 - 1], [0, 8193, 8196], [1, 1], None])
    assert result["status"] == 0
    assert drivers[0].evaluate([[4, 1, 0, 0]])[0]["row"][0] == 2**30 * 1024
    # Largest KV prefix terminal is also a valid static compact allocation;
    # maxKV stays bounded per segment, independently of totalKV.
    c = config(b=2**17, k=16384, tq=1, tk=2**31 - 1)
    assert check_layout(drivers, c)["geometry"][0] == 1
    c = config(lib=0)
    size = check_layout(drivers, c)["buffers"][-1]
    c[8] = 2**63 - 1 - size
    assert drivers[0].evaluate([[0, *c]])[0]["layout"]["buffers"][-1] == 2**63 - 1
    c[8] += 1
    got = drivers[0].evaluate([[0, *c]])[0]
    assert got["status"] == 5 and got["unchanged"]
    # Independent exact accumulator byte boundary (FD storage is small).
    c = config(b=2**31 - 1, q=1, k=1, h=(2**32 - 1) // 512, m=256, cores=26, lib=0, tq=1, tk=1)
    limit = (2**63 - 1) // (26 * c[3] * 512 * 4)
    c[9] = limit
    assert drivers[0].evaluate([[0, *c]])[0]["status"] == 0
    c[9] += 1
    got = drivers[0].evaluate([[0, *c]])[0]
    assert got["status"] == 5 and got["unchanged"]
    for c in [
        config(q=2**30, h=8),
        config(b=0, q=2**30, h=8, tq=0, tk=0),
        config(b=0, tq=1, tk=0),
        config(b=0, tq=0, tk=1),
        config(tk=0, lib=2**63),
    ]:
        got = drivers[0].evaluate([[0, *c]])[0]
        assert got["status"] != 0 and got["unchanged"]


def test_empty_allocations_validate_prefixes_and_default_queries(drivers):
    with CpuDriver(
        drivers[0].binary, log_base=drivers[0].binary.with_name("default-plan")
    ) as fresh:
        for result in fresh.evaluate([[2, 0, 0], [3, 0, 0], [4, 0, 0, 0]]):
            assert result["status"] != 0 and result["unchanged"]
    for tq, tk, arrays in [
        (0, 8196, [[0, 0, 0], [0, 8193, 8196], None, None]),
        (26, 0, [[0, 17, 26], [0, 0, 0], None, None]),
    ]:
        c = config(tq=tq, tk=tk)
        assert build(drivers[0], c, arrays)["status"] == 0
        for axis in range(4):
            bad = list(arrays)
            bad[axis] = None if axis < 2 else [1]
            got = build(drivers[0], c, bad)
            assert got["status"] != 0 and got["unchanged"]
