"""PA geometry from fixed CANN; explicit cuts/accumulators from fixed FA3.

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
    s=8,
    cores=26,
    lib=16777216,
    total=None,
    layout=1,
    block=-1,
    memory=0,
):
    return [b, q, k, p, h, s, cores, lib, b * q if total is None else total, layout, block, memory]


def allocations(c):
    remaining = c[8]
    result = []
    for _ in range(c[0]):
        value = min(max(remaining, 0), max(c[1], 0))
        result.append(value)
        remaining -= value
    return result


def arrays(c, uq=None, kv=None, cu=None, counts=None, absent=()):
    b, q, k = c[:3]
    if cu is None:
        cu = [0, *itertools.accumulate(allocations(c))]
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


def check_budget(driver, c):
    b, q, k, p, h, s, cores, lib = c[:8]
    total = c[8]
    got = driver.evaluate([[0, *c]])[0]
    assert got["status"] == 0, got
    budget = got["layout"]
    for key, shape in [("accum", [s, h, total, 512]), ("lse_accum", [s, h, total])]:
        assert budget[key]["shape"] == shape + [0] * (5 - len(shape))
        assert budget[key]["strides"] == [
            math.prod(max(v, 1) for v in shape[i + 1 :]) for i in range(len(shape))
        ] + [0] * (5 - len(shape))
        assert budget[key]["bytes"] == math.prod(shape) * 4
    if not b or not total or not k:
        assert budget["bounds"] == [0] * 6
        return budget
    m = math.ceil(min(q * h, 256) / 16) * 16
    base = 512 // p * p if p < 512 and 512 % p else 512
    n = math.ceil(min(k, base) / 32) * 32
    scratch = math.ceil(2 * cores * (6 * m * n + 8 * m * 512 + 8 * 256) / 32) * 32
    active = min(b, total)
    groups = min(active * math.ceil(q * h / 128), (total * h + active * 127) // 128)
    rows = (total * h + active * 255) * s
    user = scratch + rows * 2112
    assert budget["bounds"] == [groups, groups * s, rows, scratch, user, user + lib]
    return budget


# Every page/mask/split pair appears, with both dtypes covered by metadata tests.
PAIRS = [
    (p, mask, (2, 8, 256)[(pi + mi) % 3])
    for pi, p in enumerate((16, 112, 512, 528, 1024))
    for mi, mask in enumerate(((-1, -1, 0), (-1, -1, 1), (19, 1, 0)))
]


@pytest.mark.parametrize("page,window,splits", PAIRS)
def test_source_ranges_rows_and_pages(drivers, page, window, splits):
    c = config(p=page, k=math.ceil(1057 / page) * page, s=splits, total=22)
    b, qcap, kcap, _, h, s, cores = c[:7]
    qs, ks = [13, 3], [1057, 1]
    budget = check_budget(drivers[0], c)
    got = build(drivers[0], c, uq=qs, kv=ks, window=window)
    assert got["status"] == 0, got
    plan = got["plan"]
    oracle = drivers[4].evaluate(
        [
            paged_encode(
                paged_case(
                    layout=2,
                    q=allocations(c),
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
    m, n, source_s2, _, _ = got["tiling"]
    assert [m, n, source_s2] == [oracle[1], oracle[2], oracle[-1]]
    assert source_s2 == math.ceil(max(ks) / page) * page
    prefixes = [0, *itertools.accumulate(math.ceil(q * h / m) for q in qs)]
    assert plan["prefix"] == prefixes
    assert plan["groups"] == prefixes[-1] and plan["tasks"] == prefixes[-1] * s
    layout = plan["layout"]
    assert layout["buffers"][3] <= budget["bounds"][4]
    native_ws = drivers[3].evaluate([[3, m, n, qcap * h, cores, cores, 0, 0]])[0]
    assert layout["scratch"][:4] + [layout["scratch"][-1]] == native_ws
    shape = drivers[3].evaluate([[4, 1, b, qcap, c[8], h, s]])[0]
    assert shape == [4, s, h, c[8], 512, 3, s, h, c[8]]
    tasks = drivers[0].evaluate([[2, t] for t in range(plan["tasks"])])
    all_slots = set()
    for bi, (q, k) in enumerate(zip(qs, ks)):
        for mb in range(math.ceil(q * h / m)):
            g = prefixes[bi] + mb
            a, z = mb * m, min((mb + 1) * m, q * h)
            left, right, causal = window
            kind = 3 if causal else 4 if left >= 0 or right >= 0 else 0
            keys = {
                j
                for row in range(a, z)
                for j in range(k)
                if (not kind or j <= row // h + k - q + (right if kind == 4 else 0))
                and (kind != 4 or j >= row // h + k - q - left)
            }
            lo, hi = (min(keys), max(keys) + 1) if keys else (0, 0)
            first, stop = lo // n, math.ceil(hi / n)
            source = drivers[2].evaluate(
                [
                    [
                        0,
                        kind,
                        2**31 - 1 if kind == 3 else left,
                        0 if kind == 3 else right,
                        h,
                        m,
                        n,
                        cores,
                        2,
                        1,
                        q,
                        k,
                        mb,
                    ]
                ]
            )[0]
            assert source[2:4] == [first, stop]
            cuts = drivers[3].evaluate([[1, first, stop, s, i] for i in range(s)])
            merge = drivers[0].evaluate([[3, g]])[0]
            assert merge["merge"] == [g, g // (cores * 2), bi, mb, g % (cores * 2), a, z]
            covered = []
            for split, cut in enumerate(cuts):
                t = g * s + split
                task = tasks[t]
                assert task["task"] == [t, g, t // cores, bi, mb, split, t % cores]
                begin, end = min(cut[0], stop), cut[1]
                assert task["fragment"][:2] == [begin, end]
                covered.extend(range(begin, end))
                # Independently split every logical tile into physical page segments.
                for tile in range(begin, end):
                    tokens = list(range(tile * n, min((tile + 1) * n, k)))
                    segments = []
                    pos = tile * n
                    while pos < min((tile + 1) * n, k):
                        finish = min((tile + 1) * n, k, (pos // page + 1) * page)
                        segments.extend(range(pos, finish))
                        pos = finish
                    assert segments == tokens
                for offset in task["offsets"]:
                    assert offset not in all_slots
                    all_slots.add(offset)
                for row in sorted({0, z - a - 1}):
                    out = drivers[0].evaluate([[4, t, row]])[0]
                    query, head = divmod(a + row, h)
                    query += sum(allocations(c)[:bi])
                    index = (split * h + head) * c[8] + query
                    assert out["row"] == [
                        (query * h + head) * 1024,
                        (head * c[8] + query) * 4,
                        index * 2048,
                        index * 4,
                        task["offsets"][0] + row * 2048,
                        task["offsets"][1] + row * 32,
                        task["offsets"][2] + row * 32,
                    ]
            assert covered == list(range(first, stop))


@pytest.mark.parametrize(
    "spec,uq,kv,window",
    [
        ({"b": 0}, [], [], (-1, -1, 0)),
        ({"q": 0}, [], [1120, 0], (-1, -1, 1)),
        ({"k": 0}, [0, 17], [0, 0], (1, 0, 0)),
        ({}, [0, 0], [1120, 1], (0, 0, 0)),
        ({}, [17, 3], [0, 0], (-1, -1, 1)),
        ({"q": 257, "h": 1, "s": 256}, [257, 1], [16, 0], (-1, -1, 1)),
        ({"b": 1, "q": 1, "h": 257, "p": 1024, "k": 16384}, [1], [16384], (530, 0, 0)),
        ({"q": 416, "h": 8}, None, [1025, 1025], (-1, -1, 0)),
    ],
)
def test_empty_and_geometry_corners(drivers, spec, uq, kv, window):
    c = config(**spec)
    if c[1] == 0:
        uq = [0] * c[0]
    got = build(drivers[0], c, uq=uq, kv=kv, window=window)
    assert got["status"] == 0, got
    check_budget(drivers[0], c)
    plan = got["plan"]
    m = got["tiling"][3]
    groups = (
        0
        if not c[0] * c[8] * c[2]
        else sum(math.ceil(q * c[4] / m) for q in (uq or allocations(c)))
    )
    assert plan["groups"] == groups
    for cmd in ([2, plan["tasks"]], [3, groups], [4, plan["tasks"], 0], [2, 2**64 - 1]):
        failed = drivers[0].evaluate([cmd])[0]
        assert failed["status"] and failed["unchanged"]


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
        *({"s": s} for s in (0, 1, 257, 2**63 - 1)),
        {"cores": 0},
        {"cores": 27},
        {"block": 1},
        {"total": 35},
        {"total": 2**31},
        {"total": -1},
        {"layout": 0},
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
        {"cu": [0, 18, 34]},
        {"absent": [0]},
        {"cu": [1, 17, 34]},
        {"cu": [0, 17, 33]},
        {"cu": [0, 20, 19]},
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


def test_representative(drivers):
    test_source_ranges_rows_and_pages(drivers, 528, (19, 1, 0), 8)


def test_capacity_cost_boundary(drivers):
    # Large metadata prefix and >uint32 task IDs, without allocating workspace.
    c = config(b=32768, q=65535, k=16, p=16, h=1024, s=256)
    got = build(drivers[0], c, kv=[0] * c[0])
    assert got["status"] == 0 and got["plan"]["tasks"] > 2**32
    t = got["plan"]["tasks"] - 1
    task = drivers[0].evaluate([[2, t]])[0]
    row = drivers[0].evaluate([[4, t, 255]])[0]
    assert task["task"][3] == c[0] - 1 and row["row"][0] > 2**32
    assert got["plan"]["layout"]["buffers"][3] <= got["budget"]["bounds"][4]


def test_accumulator_overflow_separate_from_empty_workspace(drivers):
    # Accumulator FP32 bytes overflow while final dtype rows still fit.
    # Keep each sample's row count valid while total FP32 accumulator overflows.
    b, q, s = 65536, 32767, 256
    total = b * q
    limit = (2**63 - 1) // (total * s * 512 * 4)
    for k in (0, 16):
        ok = drivers[0].evaluate([[0, *config(b=b, q=q, k=k, p=16, h=limit, s=s)]])[0]
        bad = drivers[0].evaluate([[0, *config(b=b, q=q, k=k, p=16, h=limit + 1, s=s)]])[0]
        if k == 0:
            assert ok["status"] == 0
        assert bad["status"] and bad["unchanged"]


@pytest.mark.parametrize("q,h", [(208, 8), (249, 13)])
def test_source_divisor_m_and_static_bound(drivers, q, h):
    c = config(q=q, h=h)
    got = build(drivers[0], c, kv=[1025, 1025])
    source = drivers[4].evaluate(
        [
            paged_encode(
                paged_case(
                    layout=2,
                    q=[q, q],
                    lengths=[1025, 1025],
                    page=112,
                    capacity=1120,
                    heads=h,
                    aic=26,
                    block_dim=26,
                )
            )
        ]
    )[0]
    assert got["status"] == 0 and got["tiling"][:2] == source[1:3]
    assert got["tiling"][0] == q * h // 13 < 256
    assert got["plan"]["layout"]["geometry"][2] <= got["budget"]["bounds"][2]
    assert got["plan"]["layout"]["buffers"][3] <= got["budget"]["bounds"][4]


def test_exact_total_workspace_overflow(drivers):
    c = config()
    user = check_budget(drivers[0], c)["bounds"][4]
    c[7] = 2**63 - 1 - user
    got = drivers[0].evaluate([[0, *c]])[0]
    assert got["status"] == 0 and got["layout"]["bounds"][-1] == 2**63 - 1
    c[7] += 1
    got = drivers[0].evaluate([[0, *c]])[0]
    assert got["status"] and got["unchanged"]


@pytest.mark.parametrize("heads", [1, 129, 257])
def test_small_partition_enumeration_bounds_all_pa_m_values(drivers, heads):
    # Enumerate independent allocation partitions and every possible source M;
    # totals stay compact even when maxQ is a much looser caller bound.
    for caps in itertools.product(range(4), repeat=3):
        total = sum(caps)
        c = config(b=3, q=10, total=total, h=heads, cores=2, s=2)
        budget = check_budget(drivers[0], c)
        for m in range(128, 257):
            groups = sum(math.ceil(q * heads / m) for q in caps)
            assert groups <= budget["bounds"][0]
            assert groups * 2 * m <= budget["bounds"][2]
        got = build(drivers[0], c, cu=[0, *itertools.accumulate(caps)], kv=[1057] * 3)
        assert got["status"] == 0, (caps, got)
        if total:
            source = drivers[4].evaluate(
                [
                    paged_encode(
                        paged_case(
                            layout=2,
                            q=caps,
                            lengths=[1057] * 3,
                            page=112,
                            capacity=1120,
                            heads=heads,
                            aic=2,
                            block_dim=2,
                        )
                    )
                ]
            )[0]
            assert got["tiling"][:2] == source[1:3]
        assert got["plan"]["layout"]["buffers"][3] <= budget["bounds"][4]


def test_compact_budget_ignores_rectangular_maximum(drivers):
    c = config(b=100000, q=2**30, total=1, h=1, k=16, p=16)
    budget = check_budget(drivers[0], c)
    assert budget["bounds"][0] == 1
    assert budget["accum"]["bytes"] == c[5] * 512 * 4
    got = build(drivers[0], c, kv=[0] * c[0])
    assert got["status"] == 0 and got["plan"]["groups"] == 1
