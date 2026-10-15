"""Native PA geometry/AIC/FD/Vector from fixed CANN; FA3 accumulator shapes.

Independent row/token/page enumeration checks the new composition, not just
its unmasked predecessor. No physical cache or tensor allocation is needed.
"""

import itertools
import math

import pytest

from tests.mla.unit.paged_bsnd.test_flash_attn_npu_v3_mla_paged_schedule import (
    encode as paged_encode,
    paged_case,
)


def config(
    b=2,
    q=17,
    k=1120,
    p=112,
    h=8,
    ratio=2,
    cores=26,
    lib=16777216,
    total=None,
    layout=0,
    block=-1,
    memory=0,
):
    return [
        b,
        q,
        k,
        p,
        h,
        ratio,
        cores,
        lib,
        b * q if total is None else total,
        layout,
        block,
        memory,
    ]


def arrays(c, uq=None, kv=None, cu=None, counts=None, absent=()):
    b, q, k = c[:3]
    values = [cu, uq, [k] * b if kv is None else kv]
    result = []
    for i, v in enumerate(values):
        result.extend(
            [
                int(v is not None and i not in absent),
                len(v or []) if counts is None else counts[i],
                len(v or []),
                *(v or []),
            ]
        )
    return result


def build(
    driver,
    c=None,
    *,
    uq=None,
    kv=None,
    cu=None,
    window=(-1, -1, 0),
    bounds=None,
    counts=None,
    absent=(),
):
    c = c or config()
    return driver.evaluate(
        [[1, *c, *(bounds or [c[1], c[2]]), *window, *arrays(c, uq, kv, cu, counts, absent)]]
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


def check_budget(drivers, c):
    b, q, k, page, h, ratio, cores, lib = c[:8]
    got = drivers[0].evaluate([[0, *c]])[0]
    assert got["status"] == 0, got
    layout = got["layout"]
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
    source = drivers[3].evaluate([[4, 0, b, q, b * q, h, cores]])[0]
    assert source == ([0, 0] if cores == 1 else [5, cores, b, h, q, 512, 4, cores, b, h, q])
    if not b or not q or not k:
        assert layout["bounds"] == [0] * 6
        return layout
    m = math.ceil(min(q * h, 256) / 16) * 16
    base = 512 // page * page if page < 512 and 512 % page else 512
    n = math.ceil(min(k, base) / 32) * 32
    scratch = 2 * cores * (6 * m * n + 8 * m * 512 + 8 * 256)
    slots = 2 * cores if cores > 1 else 0
    user = scratch + slots * 256 * 2112
    assert layout["bounds"] == [
        b * math.ceil(q * h / 128),
        slots,
        slots * 256,
        scratch,
        user,
        user + lib,
    ]
    return layout


# Named interactions: both masks and PA N geometry are exercised, not inherited
# from unmasked/non-PA tests. All page/mask pairs include a core-count boundary.
PAIRS = [
    (p, w, (1, 7, 26)[(pi + mi) % 3])
    for pi, p in enumerate((16, 112, 512, 528, 1024))
    for mi, w in enumerate(((-1, -1, 0), (-1, -1, 1), (530, 1, 0)))
]
CORNERS = [
    (
        "direct-fd-empty-mix",
        dict(b=3, q=65, h=8, k=16384, p=512),
        [65, 3, 65],
        [16384, 1, 0],
        (-1, -1, 0),
    ),
    ("leading-masked-m", dict(q=257, h=1), [257, 1], [1, 0], (-1, -1, 1)),
    ("divisor-128", dict(q=208, h=8), None, [1025, 1025], (-1, -1, 0)),
    ("divisor-249", dict(q=249, h=13), None, [1025, 1025], (19, 1, 0)),
    ("fd-row-tail", dict(b=1, q=1, h=257, k=16384, p=1024), None, [16384], (-1, -1, 0)),
    ("one-core-no-fd", dict(cores=1), [17, 3], [1057, 1], (-1, -1, 1)),
    ("zero-b", dict(b=0), [], [], (-1, -1, 0)),
    ("zero-q", dict(q=0), [0, 0], [1120, 1], (-1, -1, 1)),
    ("zero-k", dict(k=0), [17, 3], [0, 0], (1, 0, 0)),
    ("zero-used-q", {}, [0, 0], [1120, 1], (0, 0, 0)),
    ("zero-used-k", {}, [17, 3], [0, 0], (-1, -1, 1)),
    ("no-used-q", {}, None, [1120, 1], (-1, -1, 0)),
    ("nonzero-window-start", dict(q=416, h=8), None, [1025, 1025], (19, 1, 0)),
]


def check_source(drivers, c, qs, ks, window, name=""):
    b, qcap, kcap, page, h, ratio, cores = c[:7]
    budget = check_budget(drivers, c)
    result = build(drivers[0], c, uq=qs, kv=ks, window=window)
    assert result["status"] == 0, result
    m, n, source_s2, _, _ = result["tiling"]
    if b and qcap and max(ks, default=0):
        oracle = drivers[4].evaluate(
            [
                paged_encode(
                    paged_case(
                        q=[qcap] * b,
                        lengths=ks,
                        uq=qs,
                        page=page,
                        capacity=kcap,
                        heads=h,
                        aic=cores,
                        block_dim=cores,
                    )
                )
            ]
        )[0]
        assert [m, n, source_s2] == [oracle[1], oracle[2], oracle[-1]]
        assert source_s2 == math.ceil(max(ks) / page) * page
    m, n = result["tiling"][3:]
    lengths = list(zip([qcap] * b if qs is None else qs, ks))
    layout = result["plan"]["layout"]
    assert layout["buffers"][3] <= budget["bounds"][4]
    assert layout["geometry"][2] <= budget["bounds"][2]
    if b and qcap and kcap:
        assert (
            layout["scratch"][:4] + [layout["scratch"][-1]]
            == drivers[3].evaluate([[3, m, n, qcap * h, cores, cores, 0, 0]])[0]
        )
    p = result["plan"]
    native = drivers[2].evaluate(
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
            for tile in range(start, stop):
                begin, end = tile * n, min((tile + 1) * n, lengths[batch][1])
                pos = begin
                pieces = []
                while pos < end:
                    finish = min(end, (pos // page + 1) * page)
                    pieces.extend(range(pos, finish))
                    pos = finish
                assert pieces == list(range(begin, end))
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
                token = batch * qcap + query
                lse = ((batch * h + head) * qcap + query) * 4
                accum = (((split * b + batch) * h + head) * qcap + query) * 4 if cores > 1 else 0
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
    if name == "fd-row-tail":
        assert fd_count > 0
    if name in ("one-core-no-fd", "single-tile-no-fd"):
        assert fd_count == 0 and 0 in observed_kinds


@pytest.mark.parametrize("page,window,cores", PAIRS)
def test_source_ranges_rows_and_pages(drivers, page, window, cores):
    check_source(
        drivers,
        config(p=page, k=math.ceil(1057 / page) * page, cores=cores),
        [13, 3],
        [1057, 1],
        window,
    )


@pytest.mark.parametrize("name,options,qs,ks,window", CORNERS, ids=[x[0] for x in CORNERS])
def test_source_corners(drivers, name, options, qs, ks, window):
    check_source(drivers, config(**options), qs, ks, window, name)


def test_representative(drivers):
    check_source(drivers, config(p=528, k=16368), [13, 3], [16367, 1], (530, 1, 0))


def test_capacity_cost_boundary(drivers):
    c = config(b=1025, q=2**30, k=16, p=16, h=1)
    got = build(drivers[0], c, kv=[0] * c[0])
    assert got["status"] == 0 and got["plan"]["groups"] > 2**32
    g = got["plan"]["groups"] - 1
    task, row = drivers[0].evaluate([[2, g, 0], [4, g, 0, 255]])
    assert task["task"][:5] == [g, 2**64 - 1, 1024, 2**22 - 1, 0]
    assert task["task"][-1] == 2 and row["row"][0] > 2**32


def test_accumulator_overflow_separate_from_fd_workspace(drivers):
    q = 2**30
    limit = (2**63 - 1) // (q * 26 * 512 * 4)
    for k in (0, 16):
        a, b = drivers[0].evaluate(
            [
                [0, *config(b=limit, q=q, k=k, p=16, h=1)],
                [0, *config(b=limit + 1, q=q, k=k, p=16, h=1)],
            ]
        )
        assert a["status"] == 0 and b["status"] and b["unchanged"]
        assert a["layout"]["bounds"][2] == (52 * 256 if k else 0)


@pytest.mark.parametrize(
    "update",
    [
        {"b": -1},
        {"b": 2**32},
        {"q": -1},
        {"q": 2**30 + 1},
        {"q": 2**30, "h": 4},
        {"h": 0},
        {"h": 2**23},
        {"k": -1},
        {"k": 16385},
        {"k": 111},
        *({"p": p} for p in (0, 15, 17, 1040, 2**63 - 1)),
        *({"ratio": r} for r in (0, 1, 3)),
        {"cores": 0},
        {"cores": 27},
        {"block": 1},
        {"total": 1},
        {"total": -1},
        {"layout": 1},
        {"memory": 1},
        {"lib": 2**64 - 1},
    ],
)
def test_invalid_static_budget_is_atomic(drivers, update):
    got = drivers[0].evaluate([[0, *config(**update)]])[0]
    assert got["status"] and got["unchanged"]


@pytest.mark.parametrize(
    "options",
    [
        {"uq": [-1, 0]},
        {"uq": [18, 0]},
        {"kv": [-1, 0]},
        {"kv": [0, 1121]},
        {"cu": [0, 17, 34]},
        {"counts": [0, 0, 1]},
        {"absent": [2]},
        {"bounds": [16, 1120]},
        {"bounds": [17, 1119]},
        {"window": [-(2**63), -1, 0]},
    ],
)
def test_invalid_lengths_preserve_owned_plan(drivers, options):
    assert build(drivers[0])["status"] == 0
    got = build(drivers[0], **options)
    assert got["status"] and got["unchanged"]


def test_exact_total_workspace_overflow(drivers):
    c = config()
    user = check_budget(drivers, c)["bounds"][4]
    c[7] = 2**63 - 1 - user
    got = drivers[0].evaluate([[0, *c]])[0]
    assert got["status"] == 0 and got["layout"]["bounds"][-1] == 2**63 - 1
    c[7] += 1
    got = drivers[0].evaluate([[0, *c]])[0]
    assert got["status"] and got["unchanged"]
