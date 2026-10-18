"""Mixed compact-Q/rectangular-KV task bounds, allocation/used ownership and persistent addresses.

Compile the pinned CANN/FA3 CPU fragments; independently enumerate visible rows,
allocation tokens and slots. No historical unit suite or attention reference runs.
"""

import itertools
import math

import pytest


def config(
    b=2,
    q=17,
    k=1057,
    h=4,
    m=17,
    n=128,
    s=8,
    cores=3,
    block=2,
    ratio=2,
    lib=16777216,
    tq=14,
):
    return [b, q, k, h, m, n, s, cores, block, ratio, lib, tq]


def prefix(lengths):
    return [0, *itertools.accumulate(lengths)]


def build(driver, c, arrays=None, window=(-1, -1, 0), bounds=None, nulls=()):
    arrays = arrays if arrays is not None else [[0, 9, 14], None, None, [1057, 3]]
    encoded = []
    for i, values in enumerate(arrays):
        encoded.extend(
            [int(values is not None and i not in nulls), len(values or []), *(values or [])]
        )
    return driver.evaluate([[1, *c, *(bounds or c[1:3]), *window, *encoded]])[0]


def normalize(q, k, left, right, causal):
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


# Each tuple targets a distinct address, mask or ownership interaction.
CASES = [
    ("window-used-gaps", {}, [(9, 1057), (5, 3)], [4, 3], [1031, 1], (530, 1, 0)),
    (
        "causal-original-lengths",
        dict(h=8, m=129, n=512, s=2, cores=26, block=26),
        [(17, 1031), (9, 3)],
        [13, 9],
        None,
        (-1, -1, 1),
    ),
    (
        "no-mask-uneven-allocation",
        dict(m=16, s=3),
        [(1, 5), (12, 1025), (2, 19)],
        None,
        [2, 103, 0],
        (-1, -1, 0),
    ),
    (
        "zero-segments-and-used-gaps",
        dict(q=7, k=513, h=5, m=16, s=8, cores=7, block=3, ratio=1),
        [(0, 513), (7, 0), (0, 1), (2, 3), (4, 0)],
        [0, 7, 0, 1, 0],
        None,
        (0, 0, 0),
    ),
    ("one-core-many-waves", dict(cores=1, block=1, ratio=1), [(3, 513)], None, None, (-1, -1, 0)),
    ("all-zero-used-kv", {}, [(9, 1057), (5, 3)], None, [0, 0], (-1, -1, 0)),
    ("all-masked", dict(q=33, k=3, m=16, n=32, s=2), [(33, 3), (17, 1)], None, None, (-1, -1, 1)),
    (
        "one-sided-causal",
        dict(q=5, k=129, h=8, m=16, n=112),
        [(5, 129), (2, 7)],
        None,
        None,
        (7, -1, 1),
    ),
    (
        "normalized-wide-window",
        dict(q=1, k=16384, h=1, m=16, n=512, s=2),
        [(1, 16384), (1, 1)],
        None,
        None,
        (2**63 - 1, 2**63 - 1, 1),
    ),
    ("192-source-range", dict(h=8, m=255, n=32, s=192), [(2, 1025)], None, None, (-1, -1, 0)),
    ("193-explicit-extension", dict(h=1, m=16, n=32, s=193), [(1, 33)], None, None, (0, 0, 0)),
    (
        "255-repeat-boundary",
        dict(k=16383, h=8, m=16, n=512, s=255),
        [(1, 16383)],
        None,
        None,
        (-1, -1, 0),
    ),
    (
        "256-longest-kv",
        dict(k=16384, h=8, m=16, n=32, s=256, cores=26, block=26),
        [(1, 16384)],
        None,
        None,
        (-1, -1, 0),
    ),
]


def check_layout(drivers, c):
    b, q, k, h, m, n, s, cores, block, _, lib, tq = c
    got = drivers[0].evaluate([[0, *c]])[0]
    assert got["status"] == 0
    layout = got["layout"]
    empty = not (tq and b and k)
    # Bound validity itself is checked by exhaustive small allocations below.
    active = min(b, tq)
    groups = 0 if empty else min(active * math.ceil(q * h / m), (tq * h + active * (m - 1)) // m)
    rows = groups * s * m
    assert layout["geometry"] == [groups, groups * s, rows, int(empty)]
    source = drivers[2].evaluate([[4, 1, b, q, tq, h, s]])[0]
    assert source == [4, s, h, tq, 512, 3, s, h, tq]
    for key, shape in [("accum", [s, h, tq, 512]), ("lse_accum", [s, h, tq])]:
        strides = [math.prod(max(1, v) for v in shape[i + 1 :]) for i in range(len(shape))]
        assert layout[key] == dict(
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
    assert scratch[4:9] == [*offsets, kernel] and scratch[9] >= kernel
    out = ((scratch[9] + 31) // 32) * 32
    sums, maximum, user = out + rows * 2048, out + rows * 2080, out + rows * 2112
    assert layout["buffers"] == [out, sums, maximum, user, user + lib]
    assert all(x % 32 == 0 for x in layout["buffers"][:4])
    return layout


@pytest.mark.parametrize("name,options,alloc,uq,uk,window", CASES, ids=[v[0] for v in CASES])
def test_source_ranges_ownership_and_allocation_rows(drivers, name, options, alloc, uq, uk, window):
    cq = prefix(q for q, _ in alloc)
    c = config(b=len(alloc), tq=cq[-1], **options)
    b, qcap, kcap, h, m, n, s, _, cores, ratio, _, tq = c
    lengths = list(
        zip(
            uq if uq is not None else [q for q, _ in alloc],
            uk if uk is not None else [k for _, k in alloc],
        )
    )
    layout = check_layout(drivers, c)
    result = build(
        drivers[0], c, [cq, None, uq, uk if uk is not None else [k for _, k in alloc]], window
    )
    assert result["status"] == 0
    plan = result["plan"]
    groups = prefix(math.ceil(q * h / m) for q, _ in lengths)
    assert plan == dict(groups=groups[-1], tasks=groups[-1] * s, prefix=groups, layout=layout)
    assert groups[-1] <= layout["geometry"][0]
    mask, left, right = normalize(qcap, kcap, *window)
    tasks = drivers[0].evaluate([[2, t] for t in range(plan["tasks"])])
    merges = drivers[0].evaluate([[3, g] for g in range(plan["groups"])])
    saved = set()
    for core in range(cores):
        for t in range(core, plan["tasks"], cores):
            assert tasks[t]["task"][2] == t // cores and tasks[t]["task"][-1] == core
            for offset in tasks[t]["offsets"]:
                assert offset not in saved
                saved.add(offset)
    owned_rows = set()
    for batch, (q, k) in enumerate(lengths):
        for mb in range(math.ceil(q * h / m)):
            g = groups[batch] + mb
            begin, end = mb * m, min((mb + 1) * m, q * h)
            merge = merges[g]
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
                token_head = (cq[batch] + query, row % h)
                assert token_head not in owned_rows
                owned_rows.add(token_head)
            lo, hi = min((a for a, _ in spans), default=0), max((z for _, z in spans), default=0)
            first, last = lo // n, math.ceil(hi / n)
            native = drivers[1].evaluate(
                [[0, mask, left, right, h, m, n, c[7], ratio, 1, q, k, mb]]
            )[0]
            assert native[2:4] == [first, last]
            cuts = drivers[2].evaluate([[1, first, last, s, i] for i in range(s)])
            covered = []
            for split in range(s):
                t = g * s + split
                actual = tasks[t]
                assert actual["status"] == 0
                assert actual["task"] == [t, g, t // cores, batch, mb, split, t % cores]
                assert actual["sequence"] == [cq[batch], batch * kcap, alloc[batch][0], kcap, q, k]
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
                offsets = [
                    base + t * m * width * 4 for base, width in zip(layout["buffers"], [512, 8, 8])
                ]
                assert actual["offsets"] == offsets
                if split == 0:
                    assert merge["offsets"] == offsets
                probes = sorted({0, end - begin - 1, min(end - begin - 1, h - 1 - begin % h)})
                addresses = drivers[0].evaluate([[4, t, row] for row in probes])
                for row, address in zip(probes, addresses):
                    query, head = divmod(begin + row, h)
                    token = cq[batch] + query
                    alse = ((split * h + head) * tq + token) * 4
                    assert address["status"] == 0
                    assert address["row"] == [
                        (token * h + head) * 1024,
                        (head * tq + token) * 4,
                        alse * 512,
                        alse,
                        *[base + row * width * 4 for base, width in zip(offsets, [512, 8, 8])],
                    ]
                    assert address["row"][2] + 2048 <= layout["accum"]["bytes"]
                    assert address["row"][3] + 4 <= layout["lse_accum"]["bytes"]
                    for offset, limit, width in zip(
                        address["row"][4:], layout["buffers"][1:4], [512, 8, 8]
                    ):
                        assert offset + width * 4 <= limit
            assert covered == list(range(first, last))
    assert owned_rows == {
        (cq[batch] + i, head)
        for batch, (q, _) in enumerate(lengths)
        for i in range(q)
        for head in range(h)
    }


def test_static_bound_against_all_small_allocation_partitions(drivers):
    # Enumerate independent allocation partitions, including the counterexample
    # to ceil(totalQ*H/M): [1,1,1], H=1,M=4 needs THREE M groups, not one.
    for b, q, h, m in itertools.product(range(1, 5), range(1, 5), [1, 3, 8], [1, 4, 7]):
        maxima = [0] * (b * q + 1)
        for allocation in itertools.product(range(q + 1), repeat=b):
            t = sum(allocation)
            maxima[t] = max(maxima[t], sum(math.ceil(x * h / m) for x in allocation))
        cases = [config(b=b, q=q, k=1, h=h, m=m, tq=t) for t in range(b * q + 1)]
        for t, got in enumerate(drivers[0].evaluate([[0, *c] for c in cases])):
            assert got["status"] == 0
            bound = got["layout"]["geometry"][0]
            assert maxima[t] <= bound <= b * math.ceil(q * h / m)
    # Nominal B*maxQ does not overflow/reject a tiny valid compact allocation.
    c = config(b=2**32 - 1, q=2**30, k=16384, h=1, m=256, tq=2)
    assert check_layout(drivers, c)["geometry"][:3] == [2, 16, 4096]


@pytest.mark.parametrize(
    "c,arrays",
    [
        (config(b=0, tq=0), [[0], None, [], []]),
        (config(tq=0), [[0, 0, 0], None, None, None]),
        (config(q=0, tq=0), [[0, 0, 0], None, None, None]),
        (config(k=0), [[0, 9, 14], None, None, None]),
        (config(), [[0, 9, 14], None, [0, 0], None]),
    ],
    ids=[
        "empty-batch",
        "zero-q-total-positive-max",
        "zero-q-max",
        "zero-kv-max",
        "zero-q-used",
    ],
)
def test_empty_allocations_still_validate_lengths(drivers, c, arrays):
    layout = check_layout(drivers, c)
    result = build(drivers[0], c, arrays)
    assert result["status"] == 0
    assert result["plan"] == dict(groups=0, tasks=0, prefix=[0] * (c[0] + 1), layout=layout)
    for got in drivers[0].evaluate([[2, 0], [3, 0], [4, 0, 0]]):
        assert got["status"] != 0 and got["unchanged"]
    for axis in [0, 1]:
        bad = list(arrays)
        bad[axis] = None if axis == 0 else [0]
        got = build(drivers[0], c, bad)
        assert got["status"] != 0 and got["unchanged"]


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
    (6, 257),
    (6, 2**32 + 2),
    (7, 0),
    (7, 27),
    (8, 0),
    (8, 4),
    (9, 0),
    (9, 3),
    (10, 2**64 - 1),
    (11, -1),
    (11, 2**31),
    (11, 35),
]


@pytest.mark.parametrize("field,value", INVALID)
def test_invalid_config_is_atomic(drivers, field, value):
    c = config()
    assert build(drivers[0], c)["status"] == 0
    before = drivers[0].evaluate([[2, 3], [4, 3, 0]])
    c[field] = value
    for result in [drivers[0].evaluate([[0, *c]])[0], build(drivers[0], c)]:
        assert result["status"] != 0 and result["unchanged"]
    assert drivers[0].evaluate([[2, 3], [4, 3, 0]]) == before


@pytest.mark.parametrize("axis", [0, 2, 3], ids=["cu-q", "used-q", "used-kv"])
def test_independent_prefix_used_counts_values_and_nulls(drivers, axis):
    c = config()
    arrays = [[0, 9, 14], None, [4, 3], [7, 1]]
    assert build(drivers[0], c, arrays)["status"] == 0
    before = drivers[0].evaluate([[2, 3], [4, 3, 0]])
    bad_values = [[], [1], arrays[axis] + [0]]
    if axis == 0:
        bad_values += [
            None,
            [1, 9, 14],
            [0, -1, 14],
            [0, -(2**31), 14],
            [0, 15, 14],
            [0, 9, 13],
            [0, 18, 14],
        ]
    else:
        capacity = 5 if axis == 2 else c[2]
        bad_values += [[1, -1], [1, -(2**31)], [1, capacity + 1]]
    for values in bad_values:
        bad = list(arrays)
        bad[axis] = values
        got = build(drivers[0], c, bad)
        assert got["status"] != 0 and got["unchanged"]
    got = build(drivers[0], c, arrays, nulls=[axis])
    assert got["status"] != 0 and got["unchanged"]
    assert drivers[0].evaluate([[2, 3], [4, 3, 0]]) == before


def test_kv_prefix_is_forbidden_and_absent_used_means_full_capacity(drivers):
    c = config()
    for prefix in [[], [0], [0, 1057, 2114]]:
        got = build(drivers[0], c, [[0, 9, 14], prefix, None, None])
        assert got["status"] != 0 and got["unchanged"]
    assert build(drivers[0], c, [[0, 9, 14], None, None, None])["status"] == 0
    assert drivers[0].evaluate([[2, 3]])[0]["sequence"] == [0, 0, 9, 1057, 9, 1057]


def test_window_query_errors_and_replacement(drivers):
    c = config()
    baseline = build(drivers[0], c)["plan"]
    before = drivers[0].evaluate([[2, 3], [4, 3, 0]])
    for bounds, window in [
        ((16, 1057), (-1, -1, 0)),
        ((17, 1056), (-1, -1, 0)),
        ((17, 1057), (-(2**63), 0, 0)),
    ]:
        result = build(drivers[0], c, window=window, bounds=bounds)
        assert result["status"] != 0 and result["unchanged"]
    for query in [
        [2, baseline["tasks"]],
        [2, 2**64 - 1],
        [3, baseline["groups"]],
        [3, 2**64 - 1],
        [4, 0, c[4]],
        [4, 0, 2**32 - 1],
        [4, baseline["tasks"], 0],
    ]:
        result = drivers[0].evaluate([query])[0]
        assert result["status"] != 0 and result["unchanged"]
    assert drivers[0].evaluate([[2, 3], [4, 3, 0]]) == before
    result = build(drivers[0], c, [[0, 9, 14], None, [0, 1], [0, 0]])
    assert result["status"] == 0 and result["plan"]["prefix"] == [0, 0, 1]
    actual = drivers[0].evaluate([[2, 0], [4, 0, 0]])
    assert actual[0]["sequence"] == [9, 1057, 5, 1057, 1, 0]
    assert actual[1]["row"][0] == 9 * 4 * 1024


def test_u64_task_ids_and_last_row_without_enumeration(drivers):
    c = config(q=2**29, k=16384, h=4, m=1, s=256, cores=26, block=26, tq=2**30)
    result = build(drivers[0], c, [[0, 2**29, 2**30], None, None, None])
    assert result["status"] == 0
    tasks = result["plan"]["tasks"]
    assert tasks == 2**40 and result["plan"]["groups"] == 2**32
    last = drivers[0].evaluate([[2, tasks - 1], [4, tasks - 1, 0]])
    assert last[0]["task"] == [
        tasks - 1,
        2**32 - 1,
        (tasks - 1) // 26,
        1,
        2**31 - 1,
        255,
        (tasks - 1) % 26,
    ]
    assert last[1]["row"][:4] == [
        (2**32 - 1) * 1024,
        (2**32 - 1) * 4,
        (2**40 - 1) * 2048,
        (2**40 - 1) * 4,
    ]
    # Int32 prefix terminal itself: huge logical allocation, only boundary queries.
    c = config(b=2, q=2**30, h=1, m=256, s=2, tq=2**31 - 1)
    result = build(drivers[0], c, [[0, 2**30, 2**31 - 1], None, [1, 1], None])
    assert result["status"] == 0
    assert drivers[0].evaluate([[4, 2, 0]])[0]["row"][0] == 2**30 * 1024


def test_capacity_cost_boundary_and_byte_overflow(drivers):
    c = config(lib=0)
    size = check_layout(drivers, c)["buffers"][-1]
    c[10] = 2**63 - 1 - size
    assert drivers[0].evaluate([[0, *c]])[0]["layout"]["buffers"][-1] == 2**63 - 1
    c[10] += 1
    result = drivers[0].evaluate([[0, *c]])[0]
    assert result["status"] == 5 and result["unchanged"]
    # Find the exact static byte boundary without allocating a tensor or B spans.
    c = config(b=2**31 - 1, q=1, k=1, h=(2**32 - 1) // 512, m=256, s=256, tq=1, lib=0)
    low, high = 1, 2**31 - 1
    while low < high:
        middle = (low + high + 1) // 2
        c[11] = middle
        if drivers[0].evaluate([[0, *c]])[0]["status"] == 0:
            low = middle
        else:
            high = middle - 1
    c[11] = low
    accepted = drivers[0].evaluate([[0, *c]])[0]
    assert accepted["status"] == 0 and accepted["layout"]["buffers"][-1] <= 2**63 - 1
    c[11] += 1
    rejected = drivers[0].evaluate([[0, *c]])[0]
    assert rejected["status"] == 5 and rejected["unchanged"]
    for c in [
        config(q=2**30, h=8),
        config(b=0, q=2**30, h=8, tq=0),
        config(b=0, tq=1),
        config(k=0, lib=2**63),
    ]:
        result = drivers[0].evaluate([[0, *c]])[0]
        assert result["status"] != 0 and result["unchanged"]


def test_representative_wide_kv_origin_and_sparse_q(drivers):
    # First physical KV token past uint32; only two Q tokens, no large tensors.
    b, k = 262146, 16384
    cq = [0] * (b - 1) + [1, 2]
    c = config(b=b, q=1, k=k, h=1, m=16, s=2, tq=2)
    layout = check_layout(drivers, c)
    assert layout["geometry"][:2] == [2, 4]
    got = build(drivers[0], c, [cq, None, None, None])
    assert got["status"] == 0 and got["plan"]["groups"] == 2
    task, row = drivers[0].evaluate([[2, 3], [4, 3, 0]])
    assert task["sequence"] == [1, (b - 1) * k, 1, k, 1, k]
    assert task["sequence"][1] > 2**32
    assert row["row"][:4] == [1024, 4, 3 * 2048, 12]
