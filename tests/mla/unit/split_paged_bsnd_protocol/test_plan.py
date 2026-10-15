"""PA geometry from fixed CANN; explicit cuts/accumulators from fixed FA3.

Independent row/token/page enumeration checks the new composition, not just
its unmasked predecessor. No physical cache or tensor allocation is needed.
"""

import itertools
import math

import pytest

from tests.mla.support.components.paged_schedule import encode as paged_encode, paged_case


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
    layout=0,
    block=-1,
    memory=0,
):
    return [b, q, k, p, h, s, cores, lib, b * q if total is None else total, layout, block, memory]


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


def check_budget(driver, c):
    b, q, k, p, h, s, cores, lib = c[:8]
    got = driver.evaluate([[0, *c]])[0]
    assert got["status"] == 0, got
    budget = got["layout"]
    for key, shape in [("accum", [s, b, h, q, 512]), ("lse_accum", [s, b, h, q])]:
        assert budget[key]["shape"] == shape + [0] * (5 - len(shape))
        assert budget[key]["strides"] == [
            math.prod(max(v, 1) for v in shape[i + 1 :]) for i in range(len(shape))
        ] + [0] * (5 - len(shape))
        assert budget[key]["bytes"] == math.prod(shape) * 4
    if not b or not q or not k:
        assert budget["bounds"] == [0] * 6
        return budget
    m = math.ceil(min(q * h, 256) / 16) * 16
    base = 512 // p * p if p < 512 and 512 % p else 512
    n = math.ceil(min(k, base) / 32) * 32
    scratch = math.ceil(2 * cores * (6 * m * n + 8 * m * 512 + 8 * 256) / 32) * 32
    groups = b * math.ceil(q * h / 128)
    rows = b * math.ceil(q * h / 256) * s * 256
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
    c = config(p=page, k=math.ceil(1057 / page) * page, s=splits)
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
    shape = drivers[3].evaluate([[4, 0, b, qcap, b * qcap, h, s]])[0]
    assert shape == [5, s, b, h, qcap, 512, 4, s, b, h, qcap]
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
                    index = ((split * b + bi) * h + head) * qcap + query
                    assert out["row"] == [
                        ((bi * qcap + query) * h + head) * 1024,
                        ((bi * h + head) * qcap + query) * 4,
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
        0 if not c[0] * c[1] * c[2] else sum(math.ceil(q * c[4] / m) for q in (uq or [c[1]] * c[0]))
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


def test_representative(drivers):
    test_source_ranges_rows_and_pages(drivers, 528, (19, 1, 0), 8)


def test_capacity_cost_boundary(drivers):
    # Large metadata prefix and >uint32 task IDs, without allocating workspace.
    c = config(b=65538, q=65536, k=16, p=16, h=1, s=256)
    got = build(drivers[0], c, kv=[0] * c[0])
    assert got["status"] == 0 and got["plan"]["tasks"] > 2**32
    t = got["plan"]["tasks"] - 1
    task = drivers[0].evaluate([[2, t]])[0]
    row = drivers[0].evaluate([[4, t, 255]])[0]
    assert task["task"][3] == c[0] - 1 and row["row"][0] > 2**32
    assert got["plan"]["layout"]["buffers"][3] <= got["budget"]["bounds"][4]


def test_accumulator_overflow_separate_from_empty_workspace(drivers):
    # Accumulator FP32 bytes overflow while final dtype rows still fit.
    q = 2**30
    h = 1
    s = 256
    limit = (2**63 - 1) // (q * h * s * 512 * 4)
    for k in (0, 16):
        ok = drivers[0].evaluate([[0, *config(b=limit, q=q, k=k, p=16, h=h, s=s)]])[0]
        bad = drivers[0].evaluate([[0, *config(b=limit + 1, q=q, k=k, p=16, h=h, s=s)]])[0]
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
