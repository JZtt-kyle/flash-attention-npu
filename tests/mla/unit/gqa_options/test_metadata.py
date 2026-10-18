"""Production codec: fixed-source partitions, independent geometry and corruption."""

import contextlib
import itertools
import os
from pathlib import Path
import pytest
from tests.mla.support.mla_cpu_driver import CpuDriver
from tests.mla.support.mla_gqa_options_build import compile_drivers


@pytest.fixture(scope="module")
def drivers(tmp_path_factory):
    directory = Path(
        os.environ.get("MLA_SCHEDULE_ARTIFACT_DIR") or tmp_path_factory.mktemp("gqa-options")
    )
    with contextlib.ExitStack() as stack:
        yield tuple(stack.enter_context(CpuDriver(p)) for p in compile_drivers(directory))


def case(layout=0, dim=512, pack=0, splits=0, mask="window", dtype=1, **updates):
    q, k = [5, 0, 9], [1025, 0, 513]
    page = 16 if layout >= 3 else 0
    sk = 1040 if page else 1025
    tnd = layout in (1, 2, 4)
    cq = [0, *itertools.accumulate(q)] if tnd else []
    ck = [0, *itertools.accumulate(k)] if layout == 1 else []
    left, right, causal = {"none": (-1, -1, 0), "causal": (-1, -1, 1), "window": (2, 1, 0)}[mask]
    c = dict(
        batch=3,
        sq=9,
        sk=sk,
        heads=8,
        kvheads=1 if dim == 512 else 2,
        splits=splits,
        dtype=dtype,
        cores=26,
        margin=2,
        pack=pack,
        layout=layout,
        totalq=sum(q) if tnd else 0,
        totalk=sum(k) if layout == 1 else 0,
        page=page,
        flags=3,
        dim=dim,
        cap=0,
        left=left,
        right=right,
        causal=causal,
        fault=-1,
        value=0,
        cq=cq,
        ck=ck,
        uq=[3, 0, 7],
        uk=[1023, 0, 511],
    )
    c.update(updates)
    return c


def build(driver, c):
    keys = (
        "batch",
        "sq",
        "sk",
        "heads",
        "kvheads",
        "splits",
        "dtype",
        "cores",
        "margin",
        "pack",
        "layout",
        "totalq",
        "totalk",
        "page",
        "flags",
        "dim",
        "cap",
        "left",
        "right",
        "causal",
        "fault",
        "value",
    )
    arrays = [c[k] for k in ("cq", "ck", "uq", "uk")]
    return driver.evaluate(
        [[1, *(c[k] for k in keys), *(v for a in arrays for v in (len(a), *a))]]
    )[0]


# Both dtypes per distinct protocol boundary, without a Cartesian switch expansion.
CASES = [
    (
        layout,
        dim,
        (layout + (dim == 128)) % 2,
        [0, 1, 3][(layout + (dim == 128)) % 3],
        ["none", "causal", "window"][(layout + 2 * (dim == 128)) % 3],
    )
    for layout in range(5)
    for dim in (512, 128)
]


@pytest.mark.parametrize("dtype", [1, 2])
@pytest.mark.parametrize("layout,dim,pack,splits,mask", CASES)
def test_source_and_geometry(drivers, dtype, layout, dim, pack, splits, mask):
    c = case(layout, dim, pack, splits, mask, dtype)
    r = build(drivers[0], c)
    assert r["status"] == 0, r
    record = r["record"]
    assert record[:6] == [0x4D4C4131, 47, 47, len(record), dtype, 0]
    groups = c["kvheads"] if pack else c["heads"]
    g = c["heads"] // groups
    m, n, cores = record[25:28]
    sparse, left, right = {
        "none": (0, c["sk"] - 1, c["sq"] - 1),
        "causal": (3, 2**31 - 1, 0),
        "window": (4, 2, 1),
    }[mask]
    virtual = [v for pair in zip(c["uq"], c["uk"]) for _ in range(groups) for v in pair]
    native = drivers[1].evaluate(
        [[1, sparse, left, right, g, m, n, cores, 2, c["batch"] * groups, *virtual]]
    )[0]
    tail = record[32 + 8 * c["batch"] :]
    assert [tail[3], tail[82], tail[83], tail[84]] == [native[0], native[2], native[3], native[4]]
    pos = 7
    for start in (4, 30, 56, 86, 112, 138, 164, 190, 216, 242, 294):
        count = 2 * cores if start >= 242 else cores
        assert tail[start : start + count] == native[pos : pos + count]
        pos += count
    seen, group = set(), 0
    for b, (q, k) in enumerate(zip(c["uq"], c["uk"])):
        for hgroup in range(groups):
            for row in range(0, q * g, m):
                task = drivers[0].evaluate([[2, group, 0]])[0]
                assert task[:3] == [1, b * groups + hgroup, row // m]
                assert task[4:6] == [row, min(row + m, q * g)]
                visible = {
                    j
                    for flat in range(row, min(row + m, q * g))
                    for j in range(k)
                    if mask == "none"
                    or (
                        j <= flat // g + k - q
                        if mask == "causal"
                        else flat // g + k - q - 2 <= j <= flat // g + k - q + 1
                    )
                }
                lo, hi = (min(visible) // n, (max(visible) + n) // n) if visible else (0, 0)
                cursor = lo
                for split in range(task[3]):
                    part = drivers[0].evaluate([[2, group, split]])[0]
                    assert part[6] == cursor and part[7] >= cursor
                    cursor = part[7]
                assert cursor == hi
                for flat in range(task[4], task[5]):
                    coord = (b, hgroup * g + flat % g, flat // g)
                    assert coord not in seen
                    seen.add(coord)
                assert drivers[0].evaluate([[2, group, task[3]]])[0][0] == 0
                group += 1
    assert group == r["groups"]
    assert seen == {
        (b, h, q)
        for b, count in enumerate(c["uq"])
        for h in range(c["heads"])
        for q in range(count)
    }


@pytest.mark.parametrize(
    "word", list(range(32)) + [32, 34, 36, 38, 39, 56, 59, 60, 86, 112, 138, 142, 402, 404]
)
def test_record_corruption(drivers, word):
    c = case(layout=1)
    r = build(drivers[0], c)
    assert r["status"] == 0
    c.update(fault=word, value=r["record"][word] ^ 0x80)
    assert build(drivers[0], c)["status"] != 0


@pytest.mark.parametrize("fault", range(-16, -1))
def test_failures_are_atomic(drivers, fault):
    c = case(layout=2 if fault == -13 else 1, fault=fault)
    r = build(drivers[0], c)
    assert r["status"] == 0 if fault == -13 else r["status"] != 0


@pytest.mark.parametrize(
    "changes",
    [
        dict(margin=26),
        dict(margin=2**32 - 1),
        dict(cores=0),
        dict(cores=27),
        dict(pack=2),
        dict(kvheads=3),
        dict(dim=64),
        dict(layout=5),
        dict(page=16),
        dict(splits=257),
        dict(dtype=3),
        dict(cap=2),
        dict(totalq=2**31),
        dict(batch=2**32 - 1),
        dict(heads=2**32 - 1),
        dict(sk=16385),
        dict(sq=2**30 + 1),
        dict(flags=4),
    ],
)
def test_bounds(drivers, changes):
    r = build(drivers[0], case(**changes))
    assert not r["layout"] or r["status"] != 0


@pytest.mark.parametrize("layout", range(5))
@pytest.mark.parametrize("dtype", [1, 2])
def test_empty_capacity_and_one_core(drivers, layout, dtype):
    c = case(
        layout=layout,
        dtype=dtype,
        margin=25,
        sk=0,
        uk=[0, 0, 0],
        totalk=0,
        ck=[0, 0, 0, 0] if layout == 1 else [],
    )
    assert build(drivers[0], c)["status"] == 0
    c.update(sq=0, totalq=0, uq=[0, 0, 0], cq=[0, 0, 0, 0] if layout in (1, 2, 4) else [])
    assert build(drivers[0], c)["status"] == 0


@pytest.mark.parametrize("dim", [128, 512])
@pytest.mark.parametrize("splits", [0, 1, 3, 19])
@pytest.mark.parametrize("empty", [False, True])
def test_runnable_slots_preserve_partials_and_distribute_heads(drivers, dim, splits, empty):
    c = case(
        dim=dim,
        batch=2,
        sq=3,
        sk=4097,
        cores=20,
        margin=1,
        splits=splits,
        left=256,
        right=2,
        uq=[0, 0] if empty else [3, 1],
        uk=[4097, 1057],
    )
    r = build(drivers[0], c)
    assert r["status"] == 0
    launch = drivers[0].evaluate([[3]])[0]
    expected = set()
    for group in range(r["groups"]):
        for split in range(launch["capacity"]):
            if drivers[0].evaluate([[2, group, split]])[0][0]:
                expected.add((group, split))
    actual = launch["tasks"]
    assert {(group, split) for _, group, split, _ in actual} == expected
    assert len(actual) == len(expected)
    m = r["record"][25]
    assert len({row for _, _, _, row in actual}) == len(actual)
    assert all(row == (g * launch["capacity"] + s) * m for _, g, s, row in actual)
    if empty:
        assert not actual
    elif splits == 0:
        # This regression formerly sent all 16 unsplit head groups to core 0.
        assert launch["slots"] == 1 < launch["capacity"]
        assert len(actual) == 16
        assert len({core for core, *_ in actual}) == 16


@pytest.mark.parametrize("cores,kv,parts", [(20, 4097, 9), (17, 16384, 16)])
def test_auto_split_slots_retain_all_source_fragments(drivers, cores, kv, parts):
    c = case(
        batch=1,
        sq=1,
        sk=kv,
        heads=1,
        kvheads=1,
        cores=cores,
        margin=1,
        mask="none",
        uq=[1],
        uk=[kv],
    )
    r = build(drivers[0], c)
    assert r["status"] == 0
    launch = drivers[0].evaluate([[3]])[0]
    assert launch["slots"] == parts <= launch["capacity"]
    assert [(core, group, split) for core, group, split, _ in launch["tasks"]] == [
        (split, 0, split) for split in range(parts)
    ]
    fragments = drivers[0].evaluate([[2, 0, split] for split in range(parts)])
    assert fragments[0][6] == 0
    assert fragments[-1][7] * r["record"][26] >= kv
    assert all(a[7] == b[6] for a, b in zip(fragments, fragments[1:]))
