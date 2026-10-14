"""v20 TND native partition protocol: pinned source, independent geometry and corruption.

No historical unit evidence is reused. Attention references are unnecessary for
this CPU wire protocol; the production numerical path is not opened here.
"""

import contextlib
import itertools
import math
import os
from pathlib import Path

import pytest

from tests.mla.support.mla_auto_split_tnd_metadata_build import compile_drivers
from tests.mla.support.mla_cpu_driver import CpuDriver


@pytest.fixture(scope="module")
def drivers(tmp_path_factory):
    directory = Path(
        os.environ.get("MLA_SCHEDULE_ARTIFACT_DIR") or tmp_path_factory.mktemp("auto-metadata")
    )
    with contextlib.ExitStack() as stack:
        yield [stack.enter_context(CpuDriver(p)) for p in compile_drivers(directory)]


def request(
    b=3,
    q=35,
    k=8193,
    h=8,
    cores=26,
    ratio=2,
    dtype=1,
    uq=1,
    uk=1,
    left=-1,
    right=-1,
    causal=0,
    mode=0,
    qs=None,
    ks=None,
    cq=None,
    ck=None,
    tq=None,
    tk=None,
):
    cq = cq if cq is not None else [i * q for i in range(b + 1)]
    ck = ck if ck is not None else [i * k for i in range(b + 1)]
    return [
        1,
        b,
        q,
        k,
        h,
        cores,
        ratio,
        dtype,
        uq,
        uk,
        left,
        right,
        causal,
        mode,
        cq[-1] if tq is None else tq,
        ck[-1] if tk is None else tk,
        *cq,
        *ck,
        *(qs if qs is not None else [cq[i + 1] - cq[i] for i in range(b)]),
        *(ks if ks is not None else [ck[i + 1] - ck[i] for i in range(b)]),
    ]


CASES = [
    (
        "representative-variable-fd",
        dict(cq=[0, 35, 44, 47], ck=[0, 8193, 12290, 12291], qs=[31, 7, 2], ks=[8191, 4095, 1]),
    ),
    (
        "representative-causal-fd",
        dict(
            cq=[0, 35, 44, 47],
            ck=[0, 8193, 12290, 12291],
            qs=[31, 7, 2],
            ks=[8191, 4095, 1],
            causal=1,
        ),
    ),
    (
        "representative-window-fd",
        dict(
            left=530,
            right=1,
            cq=[0, 35, 54, 57],
            ck=[0, 8193, 12290, 12291],
            qs=[31, 17, 2],
            ks=[8191, 4095, 1],
        ),
    ),
    ("one-core-no-fd", dict(cores=1, qs=[35, 7, 2], ks=[8193, 4097, 1])),
    ("single-m-max-kv", dict(b=1, q=1, h=128, k=16384)),
    ("fd-row-tail", dict(b=1, q=3, h=5, k=16383, causal=1)),
    ("zero-query-spans", dict(qs=[0, 7, 0], ks=[8193, 4097, 0])),
    ("empty-kv-suffix", dict(ks=[8193, 0, 0])),
    ("masked-prefix", dict(q=513, k=3, h=1, causal=1)),
    ("single-tile-no-fd", dict(q=2, k=1, h=1)),
    ("one-sided-left", dict(q=9, k=1057, h=5, cores=7, left=7)),
    ("one-sided-right", dict(q=9, k=1057, cores=2, right=1)),
    ("causal-window", dict(q=5, k=129, left=7, causal=1)),
    ("wide-window", dict(b=1, q=1, k=16384, h=1, left=2**63 - 1, right=2**63 - 1, causal=1)),
    ("all-kv-used-zero", dict(ks=[0, 0, 0])),
    ("all-q-used-zero", dict(qs=[0, 0, 0], left=0, right=0)),
    ("zero-batch", dict(b=0)),
    ("zero-q-capacity", dict(q=0)),
    ("zero-kv-capacity", dict(k=0, causal=1)),
    (
        "direct-fd-empty-mix",
        dict(
            q=7, k=16384, h=8, cores=7, cq=[0, 7, 10, 12], ck=[0, 16384, 16385, 16385], qs=[5, 2, 1]
        ),
    ),
    ("zero-allocation-spans", dict(cq=[0, 0, 7, 7], ck=[0, 8193, 8193, 8196])),
    ("zero-q-total-positive-max", dict(cq=[0, 0, 0, 0])),
    ("zero-kv-total-positive-max", dict(ck=[0, 0, 0, 0], causal=1)),
    ("loose-maxima", dict(cq=[0, 5, 7, 8], ck=[0, 7, 9, 10], uq=0, uk=0, left=3, right=1)),
    ("implicit-lengths", dict(uq=0, uk=0, cq=[0, 17, 26, 29], ck=[0, 8193, 8196, 8197])),
    ("implicit-q", dict(uq=0)),
    ("implicit-kv", dict(uk=0)),
]


def visible(q, k, h, m, n, mb, mask):
    kind, left, right = mask
    start, stop = mb * m, min((mb + 1) * m, q * h)
    spans = []
    for row in range(start, stop):
        center = row // h + k - q
        lo = max(0, center - left) if kind == 2 else 0
        hi = min(k, center + (0 if kind == 1 else right) + 1) if kind else k
        if lo < hi:
            spans.append((lo, hi))
    lo = min((x for x, _ in spans), default=0)
    hi = max((y for _, y in spans), default=0)
    return start, stop, lo, hi, lo // n, math.ceil(hi / n)


def check_wire_and_tasks(drivers, req, result):
    assert result["built"] and result["generated"] == result["read"] == 0, result
    _, b, q, k, h, cores, ratio, dtype, uq, uk, *_ = req
    record = result["record"]
    assert len(record) == 758 + 10 * b and record[1:5] == [20, 20, len(record), dtype]
    base = 410 + 8 * b
    kind, left, right = record[base : base + 3]
    left, right = left if left < 2**31 else left - 2**32, right if right < 2**31 else right - 2**32
    tq, tk = req[14:16]
    cq, ck = req[16 : 17 + b], req[17 + b : 18 + 2 * b]
    qs, ks = req[18 + 2 * b : 18 + 3 * b], req[18 + 3 * b :]
    empty = not (tq and tk)
    m, n = (256, 512) if empty else record[15:17]
    lengths = [
        (qs[i] if uq else cq[i + 1] - cq[i], ks[i] if uk else ck[i + 1] - ck[i]) for i in range(b)
    ]
    shapes = drivers[2].evaluate([[4, 1, b, q, tq, h, cores]])[0]
    assert shapes == ([0, 0] if cores == 1 else [4, cores, h, tq, 512, 3, cores, h, tq])
    native = drivers[1].evaluate(
        [
            [
                1,
                3 if kind == 1 else 4 if kind == 2 else 0,
                2**31 - 1 if kind == 1 else left,
                0 if kind == 1 else right,
                h,
                m,
                n,
                cores,
                ratio,
                b,
                *itertools.chain.from_iterable(lengths),
            ]
        ]
    )[0]
    used, _, fd_count, max_split, vectors = native[:5]
    assert record[base + 3] == used
    assert record[base + 82 : base + 86] == [fd_count, max_split, vectors, 8]
    arrays = []
    for offset, size in [
        (4, 26),
        (30, 26),
        (56, 26),
        (86, 26),
        (112, 26),
        (138, 26),
        (164, 26),
        (190, 26),
        (216, 26),
        (242, 52),
        (294, 52),
    ]:
        arrays.append(record[base + offset : base + offset + size])
    pos = 7
    for i, array in enumerate(arrays):
        size = cores if i < 9 else cores * ratio
        assert array[:size] == native[pos : pos + size] and not any(array[size:])
        pos += size
    assert pos == len(native)
    prefix = [0, *itertools.accumulate(0 if empty else math.ceil(ql * h / m) for ql, _ in lengths)]
    assert [
        record[base + 346 + 2 * i] + (record[base + 347 + 2 * i] << 32) for i in range(b + 1)
    ] == prefix
    ends = list(zip(*(a[:used] for a in arrays[:3])))
    geometry, parts = {}, {}
    for batch, (ql, kl) in enumerate(lengths):
        for mb in range(0 if empty else math.ceil(ql * h / m)):
            g = visible(ql, kl, h, m, n, mb, (kind, left, right))
            geometry[batch, mb] = g
            fragments = {}
            for tile in range(g[4], g[5]):
                owner = next(i for i, end in enumerate(ends) if (batch, mb, tile) < end)
                fragments.setdefault(owner, []).append(tile)
            parts[batch, mb] = list(fragments.items())
    fd = sorted(key for key, owned in parts.items() if len(owned) > 1)
    assert list(zip(arrays[3][:fd_count], arrays[4][:fd_count])) == fd
    slots = [0, *itertools.accumulate(len(parts[key]) for key in fd)]
    partial_rows = (2 * cores * m if cores > 1 else 0) if not empty else 0
    if empty:
        buffers = [0] * 4
    else:
        # The wire's v5 schedule selects scratch from actual allocation maxima.
        # maxQ is a loose host bound; unlike BSND it need not equal any segment.
        allocated_rows = max(cq[i + 1] - cq[i] for i in range(b)) * h
        source_ws = drivers[2].evaluate([[3, m, n, allocated_rows, cores, cores, 0, 0]])[0]
        reserve = source_ws[-1]
        out = (reserve + 31) // 32 * 32
        buffers = [
            out,
            out + partial_rows * 512 * 4,
            out + partial_rows * 520 * 4,
            out + partial_rows * 528 * 4,
        ]
    assert result["summary"] == [
        prefix[-1],
        used,
        fd_count,
        max_split,
        vectors,
        slots[-1],
        *buffers,
    ]
    saved = set()
    for (batch, mb), g in geometry.items():
        rb, re, lo, hi, _, stop = g
        owned = parts[batch, mb]
        count = len(owned) or 1
        group = prefix[batch] + mb
        queries = drivers[0].evaluate([[2, group, i] for i in range(count)])
        for split, actual in enumerate(queries):
            assert actual["ok"]
            task_kind = 1 if count > 1 else 0 if owned else 2
            head = fd.index((batch, mb)) if task_kind == 1 else 2**32 - 1
            slot = slots[head] + split if task_kind == 1 else 2**64 - 1
            owner = owned[split][0] if owned else group % cores
            assert actual["sequence"] == [
                cq[batch],
                ck[batch],
                cq[batch + 1] - cq[batch],
                ck[batch + 1] - ck[batch],
                *lengths[batch],
            ]
            assert actual["task"] == [group, slot, batch, mb, split, count, owner, head, task_kind]
            first, last = (owned[split][1][0], owned[split][1][-1] + 1) if owned else (stop, stop)
            kb, ke = (max(lo, first * n), min(hi, last * n)) if owned else (hi, hi)
            assert actual["range"] == [rb, re, first, last, kb, ke]
            offsets = (
                [x + slot * m * width * 4 for x, width in zip(buffers, (512, 8, 8))]
                if task_kind == 1
                else [0] * 3
            )
            assert actual["offsets"] == offsets
            if task_kind == 1:
                assert slot not in saved
                saved.add(slot)
            for row in sorted({0, re - rb - 1, min(re - rb - 1, h - 1 - rb % h)}):
                r = drivers[0].evaluate([[4, group, split, row]])[0]
                query, qhead = divmod(rb + row, h)
                query += cq[batch]
                lse = (qhead * tq + query) * 4
                accum = (split * h * tq + qhead * tq + query) * 4 if cores > 1 else 0
                raw = (
                    [x + row * width * 4 for x, width in zip(offsets, (512, 8, 8))]
                    if task_kind == 1
                    else [2**64 - 1] * 3
                )
                assert r == dict(
                    ok=1, row=[(query * h + qhead) * 1024, lse, accum * 512, accum, *raw]
                )
    assert saved == set(range(slots[-1]))
    merge_keys = [
        (head, part)
        for head, key in enumerate(fd)
        for part in range(math.ceil((geometry[key][1] - geometry[key][0]) / 8))
    ]
    vector_ends = list(zip(arrays[9][:vectors], arrays[10][:vectors]))
    for head, part in merge_keys:
        batch, mb = fd[head]
        rb, re, *_ = geometry[batch, mb]
        row = rb + part * 8
        owner = next(i for i, end in enumerate(vector_ends) if (head, part) <= end)
        r = drivers[0].evaluate([[3, head, part]])[0]
        assert r == dict(
            ok=1,
            merge=[
                prefix[batch] + mb,
                slots[head],
                batch,
                mb,
                head,
                part,
                owner,
                len(parts[batch, mb]),
                row,
                min(row + 8, re),
            ],
            offsets=[
                x + (slots[head] * m + part * 8) * width * 4
                for x, width in zip(buffers, (512, 8, 8))
            ],
        )


@pytest.mark.parametrize("name,case", CASES, ids=[name for name, _ in CASES])
@pytest.mark.parametrize("dtype", [1, 2])
def test_source_and_independent_geometry(drivers, name, case, dtype):
    req = request(**case, dtype=dtype)
    check_wire_and_tasks(drivers, req, drivers[0].evaluate([req])[0])


@pytest.mark.parametrize(
    "change",
    [
        dict(qs=[-1, 0, 0]),
        dict(qs=[36, 0, 0]),
        dict(ks=[-1, 0, 0]),
        dict(ks=[8194, 0, 0]),
        dict(dtype=3),
        dict(b=0, q=2**40),
        dict(tq=2**31),
        dict(tk=2**31),
        dict(cq=[1, 35, 70, 105]),
        dict(cq=[0, 35, 34, 50]),
        dict(ck=[1, 8193, 16386, 24579]),
        dict(ck=[0, 8193, 8192, 10000]),
        dict(tq=104),
        dict(tk=24578),
        dict(cq=[0, 36, 70, 105]),
        dict(ck=[0, 8194, 16386, 24579]),
        dict(q=65536, h=65536, qs=[0] * 3),
        dict(cores=0),
        dict(cores=27),
        dict(ratio=1),
        dict(left=-(2**63)),
        dict(mode=2),
        dict(mode=3),
        dict(mode=4),
        dict(mode=5),
        dict(mode=6),
    ],
)
def test_failed_producer_clears_payload(drivers, change):
    r = drivers[0].evaluate([request(**change)])[0]
    assert r["built"] and r["generated"] and r["read"] and r["unchanged"]
    assert not any(r["record"][8:])


def test_overlap_rejected_before_write(drivers):
    r = drivers[0].evaluate([request(mode=1)])[0]
    assert not r["built"] and r["intact"] and r["unchanged"]
    assert set(r["record"]) == {0xDEADBEEF}


@pytest.mark.parametrize("field", ["header", "base", "window", "aic", "fd", "vector", "prefix"])
def test_corruption_rejected_before_output(drivers, field):
    r = drivers[0].evaluate([request()])[0]
    assert r["read"] == 0
    base = 434
    ranges = {
        "header": range(8),
        "base": [8, 9, 15, 16, 18, 19, *range(398, base)],
        "window": range(base, base + 3),
        "aic": range(base + 3, base + 82),
        "fd": range(base + 82, base + 242),
        "vector": range(base + 242, base + 346),
        "prefix": range(base + 346, len(r["record"])),
    }
    # Check every wire slot in each owned field family, including inactive tails.
    for word in ranges[field]:
        a = drivers[0].evaluate([[5, word, 0xFEDCBA98]])[0]
        assert a["read"] and a["unchanged"], (word, a)


@pytest.mark.parametrize(
    "mode,value",
    [
        (0, 1),
        (1, 1),
        (2, 3),
        (3, 1),
        (4, 2),
        (5, 1),
        (6, 2**32 - 1),
        (7, 2),
        (8, 104),
        (9, 24578),
        (10, 34),
        (11, 3),
        (12, 3),
        (13, 0),
        (14, 0),
        (15, 1),
        (16, 1),
        (17, 0),
        (18, 0),
        (19, 0),
        (20, 2),
        (21, 1),
        (13, 2**64 - 4),
        (14, 2**64 - 4),
        (5, 2**64 - 4),
        (21, 2**64 - 4),
    ],
)
def test_workspace_stale_lengths_and_context(drivers, mode, value):
    r = drivers[0].evaluate([request()])[0]
    assert r["read"] == 0
    a = drivers[0].evaluate([[6, mode, value]])[0]
    assert a["read"] and a["unchanged"]


def test_in_range_partition_and_reduction_corruption(drivers):
    r = drivers[0].evaluate([request()])[0]
    assert r["read"] == 0
    a, base = r["record"], 434
    used, heads, vectors = a[base + 3], a[base + 82], a[base + 84]
    assert heads and vectors and used > 1
    changes = [
        (base + 82, 0),  # omit all FD owners despite interior AIC boundaries
        (base + 83, 1),  # wrong maximum, still a legal platform-sized integer
        (base + 138, 1),  # one partial cannot describe a cross-core M
        (base + 164, 1),  # core zero cannot start at the second fragment
        (base + 190, a[base + 190] - 1),  # lose eight reduction rows
        (base + 216, a[base + 216] % 8 + 1),  # incorrect FD tail row count
        (base + 294 + vectors - 1, a[base + 294 + vectors - 1] - 1),
        (base + 4 + used - 1, a[base + 4 + used - 1] - 1),  # omit final sample
        (base + 347, 1),  # a forged high word must not make a valid M prefix
    ]
    assert all(value >= 0 for _, value in changes)
    for index, value in changes:
        result = drivers[0].evaluate([[5, index, value]])[0]
        assert result["read"] and result["unchanged"], (index, value, result)


def test_exact_budget_and_query_boundaries(drivers):
    r = drivers[0].evaluate([request()])[0]
    assert r["read"] == 0
    budget = r["summary"][-1]
    a = drivers[0].evaluate([[6, 0, budget - 1]])[0]
    assert a["read"] and a["unchanged"]
    assert drivers[0].evaluate([[6, 0, budget]])[0]["read"] == 0
    # A successful revalidation enables queries; rejected indices must not write.
    for cmd in (
        [2, r["summary"][0], 0],
        [2, 0, 2**32 - 1],
        [3, 26, 0],
        [3, 0, 2**32 - 1],
        [4, 0, 0, 2**32 - 1],
    ):
        a = drivers[0].evaluate([cmd])[0]
        field = {2: "task", 3: "merge", 4: "row"}[cmd[0]]
        assert not a["ok"] and a[field][0] == 12345


def test_word_count_boundary(drivers):
    maximum = (2**32 - 1 - 758) // 10
    for b in (-1, 0, maximum, maximum + 1, 2**63 - 1):
        r = drivers[0].evaluate([[0, b]])[0]
        ok = 0 <= b <= maximum
        assert r == dict(ok=int(ok), words=758 + 10 * b if ok else 12345)


def test_capacity_cost_boundary(drivers):
    # Over uint32 M groups with compact totalQ in int32; never enumerate groups.
    b, q, h = 300, 512, (2**32 - 1) // 512
    req = request(b=b, q=q, k=1, h=h, ks=[0] * b)
    r = drivers[0].evaluate([req])[0]
    assert r["built"] and r["generated"] == r["read"] == 0
    groups = r["summary"][0]
    assert groups > 2**32 and r["summary"][2:6] == [0, 0, 0, 0]
    m = r["record"][15]
    per_batch = math.ceil(q * h / m)
    for group in (0, 2**32 - 1, 2**32, groups - 1):
        t, row = drivers[0].evaluate([[2, group, 0], [4, group, 0, 0]])
        assert t["ok"] and t["task"][0] == group and t["task"][-1] == 2
        assert t["task"][6] == group % 26
        batch, mb = divmod(group, per_batch)
        token, head = divmod(mb * m, h)
        token += batch * q
        lse = (head * b * q + token) * 4
        assert row["row"][:4] == [(token * h + head) * 1024, lse, lse * 512, lse]
    word = 410 + 8 * b + 347 + 2 * b
    a = drivers[0].evaluate([[5, word, 0]])[0]
    assert a["read"] and a["unchanged"]


def test_accumulator_capacity_cost_boundary(drivers):
    # Independent exact int64 byte boundary, no attention/accumulator allocation.
    q, h, cores = 512, (2**32 - 1) // 512, 26
    limit = (2**63 - 1) // (cores * h * 512 * 4)
    for total in (limit, limit + 1):
        b = math.ceil(total / q)
        cq = [min(i * q, total) for i in range(b + 1)]
        r = drivers[0].evaluate(
            [request(b=b, q=q, k=1, h=h, cq=cq, ks=[0] * b, mode=7 if total > limit else 0)]
        )[0]
        assert r["built"]
        if total == limit:
            assert r["generated"] == r["read"] == 0
        else:
            # Forge from v5 to bypass the producer's allocation guard.
            assert r["generated"] and r["record"][5] == 0
            assert r["read"] == 9 and r["unchanged"]


def test_int32_total_and_unused_allocation_gap(drivers):
    req = request(b=2, q=2**30, k=1, h=1, cq=[0, 2**30, 2**31 - 1], qs=[0, 1], ks=[0, 0])
    r = drivers[0].evaluate([req])[0]
    assert r["read"] == 0 and r["summary"][0] == 1
    t, row = drivers[0].evaluate([[2, 0, 0], [4, 0, 0, 0]])
    assert t["sequence"] == [2**30, 1, 2**30 - 1, 1, 1, 0]
    assert row["row"][:4] == [2**30 * 1024, 2**30 * 4, 2**30 * 2048, 2**30 * 4]


@pytest.mark.parametrize("empty_axis", ["q", "k"])
def test_global_empty_still_checks_all_inputs(drivers, empty_axis):
    req = request(**({"cq": [0] * 4} if empty_axis == "q" else {"ck": [0] * 4}))
    r = drivers[0].evaluate([req])[0]
    assert r["read"] == 0 and r["summary"][0] == 0 and r["summary"][-4:] == [0] * 4
    for mode, value in [(8, 1), (9, 1), (11, 3), (12, 3), (13, 0), (14, 0), (19, 0)]:
        got = drivers[0].evaluate([[6, mode, value]])[0]
        assert got["read"] and got["unchanged"]


@pytest.mark.parametrize("mask", [(0, -1, -1), (1, -1, -1), (0, 530, 1)])
def test_shared_bsnd_tnd_partition_and_distinct_identity(drivers, mask):
    causal, left, right = mask
    req = request(qs=[35, 7, 2], ks=[8193, 4097, 1], causal=causal, left=left, right=right)
    tnd = drivers[0].evaluate([req])[0]
    b = req[1]
    bsnd = drivers[3].evaluate([req[:14] + req[18 + 2 * b :]])[0]
    assert tnd["read"] == bsnd["read"] == 0
    base = 410 + 8 * b
    assert tnd["record"][base:] == bsnd["record"][base:]
    assert tnd["summary"] == bsnd["summary"]
    for group in range(tnd["summary"][0]):
        first = drivers[0].evaluate([[2, group, 0]])[0]
        for split in range(first["task"][5]):
            got = drivers[0].evaluate([[2, group, split]])[0]
            old = drivers[3].evaluate([[2, group, split]])[0]
            assert {key: value for key, value in got.items() if key != "sequence"} == old
            # Existing BSND address semantics remain independently checked.
            row = drivers[3].evaluate([[4, group, split, 0]])[0]
            batch, rb = old["task"][2], old["range"][0]
            query, head = divmod(rb, req[4])
            lse = ((batch * req[4] + head) * req[2] + query) * 4
            accum = split * b * req[4] * req[2] * 4 + lse
            assert row["row"][:4] == [
                (batch * req[2] * req[4] + rb) * 1024,
                lse,
                accum * 512,
                accum,
            ]
    for version in (18, 19, 20):
        got = drivers[0].evaluate([[8, version]])[0]
        assert got["read"] and got["unchanged"]
    for version in (18, 19):
        got = drivers[0].evaluate([[5, 1, version]])[0]
        assert got["read"] and got["unchanged"]
    for word in (1, 2):
        got = drivers[3].evaluate([[5, word, 20]])[0]
        assert got["read"] and got["unchanged"]
