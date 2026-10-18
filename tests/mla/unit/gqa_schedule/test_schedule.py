"""Independent physical-row enumeration and pinned source scheduling contracts."""

import contextlib
import itertools
import os
from pathlib import Path
import pytest

from tests.mla.support.mla_cpu_driver import CpuDriver
from tests.mla.support.mla_gqa_schedule_build import compile_drivers


@pytest.fixture(scope="module")
def drivers(tmp_path_factory):
    directory = Path(os.environ.get("MLA_SCHEDULE_ARTIFACT_DIR") or tmp_path_factory.mktemp("gqa"))
    paths = compile_drivers(directory)
    with contextlib.ExitStack() as stack:
        yield tuple(stack.enter_context(CpuDriver(path)) for path in paths)


def request(
    *,
    layout="bsnd",
    dim=512,
    pack=0,
    splits=0,
    mask="none",
    margin=2,
    cores=26,
    heads=8,
    kvheads=None,
    q=(5, 0, 9),
    k=(11, 0, 65),
    uq=(3, 0, 7),
    uk=(9, 0, 61),
    m=32,
    n=16,
    mutate=0,
    **updates,
):
    kvheads = (1 if dim == 512 else 2) if kvheads is None else kvheads
    tnd = layout != "bsnd" and layout != "paged_bsnd"
    ktnd = layout == "tnd"
    page = 16 if layout.startswith("paged") else 0
    maxq, maxk = max(q, default=0), max(k, default=0)
    if page:
        maxk = (maxk + page - 1) // page * page
    q = q if tnd else (maxq,) * len(q)
    k = k if ktnd else (maxk,) * len(k)
    cq, ck = [0, *itertools.accumulate(q)], [0, *itertools.accumulate(k)]
    left, right, causal = {"none": (-1, -1, 0), "causal": (-1, -1, 1), "window": (2, 1, 0)}[mask]
    c = dict(
        batch=len(q),
        heads=heads,
        kvheads=kvheads,
        dim=dim,
        qlayout=int(tnd),
        klayout=int(ktnd),
        totalq=sum(q),
        totalk=sum(k),
        maxq=maxq,
        maxk=maxk,
        m=m,
        n=n,
        splits=splits,
        page=page,
        pack=pack,
        margin=margin,
        cores=cores,
        left=left,
        right=right,
        causal=causal,
        mutate=mutate,
        cq=cq if tnd else [],
        ck=ck if ktnd else [],
        uq=list(uq) if uq is not None else [],
        uk=list(uk) if uk is not None else [],
    )
    c.update(updates)
    return c


def wire(c):
    keys = (
        "batch",
        "heads",
        "kvheads",
        "dim",
        "qlayout",
        "klayout",
        "totalq",
        "totalk",
        "maxq",
        "maxk",
        "m",
        "n",
        "splits",
        "page",
        "pack",
        "margin",
        "cores",
        "left",
        "right",
        "causal",
        "mutate",
    )
    arrays = [c[name] for name in ("cq", "ck", "uq", "uk")]
    flags = c.get("flags", sum(bool(values) << i for i, values in enumerate(arrays)))
    return [1, *(c[key] for key in keys), flags, *(x for a in arrays for x in (len(a), *a))]


def build(driver, c):
    return driver.evaluate([wire(c)])[0]


def lengths(c):
    q = c["uq"] or (
        [b - a for a, b in zip(c["cq"], c["cq"][1:])] if c["qlayout"] else [c["maxq"]] * c["batch"]
    )
    k = c["uk"] or (
        [b - a for a, b in zip(c["ck"], c["ck"][1:])] if c["klayout"] else [c["maxk"]] * c["batch"]
    )
    return q, k


# Each tuple has a distinct head/layout/mask/split boundary. Exhaustive head
# and row enumeration is inside each case; CPU geometry is dtype-independent.
CASES = [
    ("bsnd", 512, -1, 0, "none"),
    ("bsnd", 128, 0, 3, "window"),
    ("tnd", 512, 0, 0, "causal"),
    ("tnd", 128, 1, 1, "window"),
    ("mixed", 512, 1, 3, "causal"),
    ("mixed", 128, 0, 0, "none"),
    ("paged_bsnd", 512, 0, 1, "window"),
    ("paged_bsnd", 128, -1, 0, "causal"),
    ("paged_tnd", 512, 1, 1, "none"),
    ("paged_tnd", 128, 0, 3, "causal"),
]


@pytest.mark.parametrize("layout,dim,pack,splits,mask", CASES)
def test_physical_rows_and_fragments(drivers, layout, dim, pack, splits, mask):
    driver = drivers[0]
    c = request(layout=layout, dim=dim, pack=pack, splits=splits, mask=mask)
    result = build(driver, c)
    assert result["status"] == 0, result
    packed = pack != 0
    hg, g = (c["kvheads"], c["heads"] // c["kvheads"]) if packed else (c["heads"], 1)
    assert result["geometry"] == [int(packed), hg, g, c["cores"] - c["margin"]]
    assert result["capacity"] == (26 if splits == 0 else splits)
    qlens, klens = lengths(c)
    expected_rows, seen = set(), set()
    group = 0
    prefix = [0]
    for b, (q, k) in enumerate(zip(qlens, klens)):
        qstart = c["cq"][b] if c["qlayout"] else b * c["maxq"]
        kstart = c["ck"][b] if c["klayout"] else b * c["maxk"]
        expected_rows.update((b, h, i) for h in range(c["heads"]) for i in range(q))
        for hgroup in range(hg):
            for start in range(0, q * g, c["m"]):
                stop = min(start + c["m"], q * g)
                base = driver.evaluate([[2, group, 0, 0]])[0]
                assert base["task_status"] == 0
                count = base["task"][5]
                assert count == splits if splits else 1 <= count <= c["cores"] - c["margin"]
                visible = {
                    j
                    for flat in range(start, stop)
                    for j in range(k)
                    if mask == "none"
                    or (
                        j <= flat // g + k - q
                        if mask == "causal"
                        else flat // g + k - q - 2 <= j <= flat // g + k - q + 1
                    )
                }
                lo = min(visible) // c["n"] if visible else 0
                hi = (max(visible) + c["n"]) // c["n"] if visible else 0
                cursor = lo
                for split in range(count):
                    values = driver.evaluate([[2, group, split, r] for r in range(stop - start)])
                    t = values[0]["task"]
                    kh = hgroup if packed else hgroup // (c["heads"] // c["kvheads"])
                    assert t[:6] == [b, hgroup, kh, start // c["m"], split, count]
                    assert 0 <= t[6] < c["cores"] - c["margin"]
                    assert t[7:9] == [start, stop]
                    assert t[9] == cursor and t[10] >= cursor
                    if splits:
                        width = (hi - lo + splits - 1) // splits
                        assert t[9:11] == [
                            min(lo + split * width, hi),
                            min(lo + (split + 1) * width, hi),
                        ]
                        assert t[6] == (group * splits + split) % (c["cores"] - c["margin"])
                    cursor = t[10]
                    for r, value in enumerate(values):
                        assert value["row_status"] == 0
                        flat = start + r
                        h, i = hgroup * g + flat % g, flat // g
                        token = qstart + i
                        lse = (
                            h * c["totalq"] + token
                            if c["qlayout"]
                            else (b * c["heads"] + h) * c["maxq"] + i
                        )
                        accum = split * c["totalq"] * c["heads"] + lse
                        assert value["row"] == [
                            b,
                            h,
                            kh,
                            i,
                            token,
                            kstart,
                            (token * c["heads"] + h) * dim * 2,
                            lse * 4,
                            accum * dim * 4 if splits != 1 else 2**64 - 1,
                            accum * 4 if splits != 1 else 2**64 - 1,
                        ]
                        if split == 0:
                            assert (b, h, i) not in seen
                            seen.add((b, h, i))
                assert cursor == hi
                invalid = driver.evaluate([[2, group, count, 0], [2, group, 0, stop - start]])
                assert invalid[0]["task_status"] != 0 and invalid[0]["task"][0] == 123
                assert invalid[1]["row_status"] != 0 and invalid[1]["row"][6] == 456
                group += 1
            prefix.append(group)
    assert seen == expected_rows and result["prefix"] == prefix
    assert group == result["groups"]


@pytest.mark.parametrize(
    "pack,mask,m,n,margin",
    [
        (0, "none", 32, 16, 0),
        (1, "causal", 64, 32, 1),
        (-1, "window", 128, 512, 2),
        (0, "window", 512, 8192, 25),
        (1, "none", 256, 2048, 13),
        (0, "causal", 512, 33, 0),
    ],
)
def test_pinned_native_endpoints_and_fd(drivers, pack, mask, m, n, margin):
    c = request(
        layout="tnd",
        dim=128,
        pack=pack,
        mask=mask,
        m=m,
        n=n,
        margin=margin,
        q=(35, 0, 129),
        k=(16384, 0, 8017),
        uq=(33, 0, 127),
        uk=(16381, 0, 8001),
    )
    result = build(drivers[0], c)
    assert result["status"] == 0, result
    _, hg, g, cores = result["geometry"]
    q, k = lengths(c)
    sparse, left, right = {
        "none": (0, c["maxk"] - 1, c["maxq"] - 1),
        "causal": (3, 2**31 - 1, 0),
        "window": (4, 2, 1),
    }[mask]
    virtual = [value for pair in zip(q, k) for _ in range(hg) for value in pair]
    source = drivers[1].evaluate(
        [[1, sparse, left, right, g, m, n, cores, 2, len(q) * hg, *virtual]]
    )[0]
    assert result["native"] == source


@pytest.mark.parametrize(
    "pack,physical,margin,h,hk",
    [
        (-1, 26, 0, 8, 2),
        (0, 26, 1, 8, 2),
        (1, 26, 25, 8, 2),
        (1, 1, 0, 8, 1),
        (-1, 26, 13, 2, 2),
        (0, 26, 0, 2, 2),
    ],
)
def test_fixed_fa3_options_and_resource_mapping(drivers, pack, physical, margin, h, hk):
    result = drivers[0].evaluate([[0, h, hk, pack, margin, physical, 2 * physical, -1]])[0]
    source = drivers[2].evaluate([[pack, physical, margin, int(h != hk)]])[0]
    assert result[0] == 0
    assert result[3:5] == [int(bool(source[0]) and h != hk), source[1]]
    remaining = physical - margin
    assert result[5] == 2 * remaining
    assert result[6:32] == list(range(1, remaining + 1)) + [0] * (26 - remaining)
    assert result[32:] == [16777216, 196608]


@pytest.mark.parametrize(
    "values",
    [
        (0, 1, 0, 0, 26, 52, -1),
        (8, 0, 0, 0, 26, 52, -1),
        (8, 3, 0, 0, 26, 52, -1),
        (8, 1, -2, 0, 26, 52, -1),
        (8, 1, 2, 0, 26, 52, -1),
        (2**63 - 1, 1, 1, 0, 26, 52, -1),
        (8, 1, 1, -1, 26, 52, -1),
        (8, 1, 1, 26, 26, 52, -1),
        (8, 1, 1, 2**63 - 1, 26, 52, -1),
        (8, 1, 1, 0, 0, 0, -1),
        (8, 1, 1, 0, 27, 54, -1),
        (8, 1, 1, 0, 26, 26, -1),
        (8, 1, 1, 0, 26, 52, 0),
        (8, 1, 1, 0, 26, 52, 25),
    ],
)
def test_resource_rejections_are_atomic(drivers, values):
    result = drivers[0].evaluate([[0, *values]])[0]
    assert result[0] != 0 and result[1] == 123


@pytest.mark.parametrize(
    "updates",
    [
        dict(dim=64),
        dict(kvheads=2),
        dict(m=0),
        dict(m=513),
        dict(m=33),
        dict(n=0),
        dict(n=8193),
        dict(splits=-1),
        dict(splits=257),
        dict(page=-16),
        dict(page=15),
        dict(page=1040),
        dict(page=16),
        dict(totalq=-1),
        dict(totalk=-1),
        dict(maxq=2**31),
        dict(maxk=16385),
        dict(batch=2**32),
        dict(batch=2**31, heads=8, pack=0),
        dict(uq=[1]),
        dict(uk=[1]),
        dict(uq=[-1, 0, 7]),
        dict(uk=[9, 0, 99]),
        dict(uq=[99, 0, 7]),
        dict(flags=0),
        dict(cq=[0, 9, 18, 27]),
        dict(qlayout=2),
        dict(klayout=2),
    ],
)
def test_static_and_length_failure_preserves_plan(drivers, updates):
    c = request()
    good = build(drivers[0], c)
    bad = build(drivers[0], dict(c, **updates))
    assert bad["status"] != 0
    assert {k: v for k, v in bad.items() if k != "status"} == {
        k: v for k, v in good.items() if k != "status"
    }


@pytest.mark.parametrize(
    "updates",
    [
        dict(cq=[0, 5, 4, 14]),
        dict(cq=[1, 5, 5, 14]),
        dict(cq=[0, 5, 5, 13]),
        dict(page=16),
        dict(totalq=2**31),
    ],
)
def test_tnd_allocation_rejections(drivers, updates):
    assert build(drivers[0], request(layout="tnd", **updates))["status"] != 0


def test_snapshot_and_large_offsets(drivers):
    c = request(
        layout="mixed",
        dim=128,
        splits=3,
        pack=0,
        mutate=1,
        q=(400_000_000, 0, 9),
        k=(65, 65, 65),
        uq=(0, 0, 1),
        uk=(1, 0, 61),
    )
    result = build(drivers[0], c)
    assert result["status"] == 0
    row = drivers[0].evaluate([[2, result["groups"] - 1, 2, 0]])[0]
    assert row["row_status"] == 0 and row["row"][4] == 400_000_000
    assert row["row"][6] > 2**32 and row["row"][8] > 2**32


@pytest.mark.parametrize("pack", [-1, 0, 1])
def test_empty_and_single_head(drivers, pack):
    for layout in ("bsnd", "tnd", "mixed", "paged_bsnd", "paged_tnd"):
        for q, k in (((), ()), ((0, 0), (0, 0)), ((5, 9), (0, 0))):
            c = request(
                layout=layout,
                pack=pack,
                dim=128,
                heads=1,
                kvheads=1,
                q=q,
                k=k,
                uq=None,
                uk=None,
                cores=1,
                margin=0,
            )
            r = build(drivers[0], c)
            assert r["status"] == 0 and r["geometry"] == [0, 1, 1, 1]
            assert r["capacity"] == 26
            if r["groups"]:
                t = drivers[0].evaluate([[2, 0, 0, 0]])[0]
                assert t["task_status"] == 0 and t["task"][9:] == [0, 0]


def test_explicit_split_upper_bound(drivers):
    c = request(splits=256, pack=0, q=(3,), k=(17,), uq=(3,), uk=(17,))
    assert build(drivers[0], c)["status"] == 0
    for split in (0, 1, 2, 255):
        t = drivers[0].evaluate([[2, 0, split, 2]])[0]
        assert t["task_status"] == t["row_status"] == 0
        assert t["task"][5] == 256
        assert t["row"][8] == (split * 24 + 2) * 512 * 4


def test_source_ceildiv_and_total_tile_overflow(drivers):
    # Each shape passes ordinary used/allocation bounds. The stricter source
    # M=512 ceil-div and total N-tile arithmetic must reject before SplitCore.
    cases = [
        request(
            heads=4, pack=1, m=512, q=(1073741724,), uq=(1073741724,), k=(0,), uk=(0,), splits=1
        ),
        request(
            heads=1,
            pack=0,
            m=32,
            n=1,
            q=(1073741824,),
            uq=(1073741824,),
            k=(16384,),
            uk=(16384,),
            splits=1,
        ),
    ]
    for c in cases:
        assert build(drivers[0], c)["status"] == 2
