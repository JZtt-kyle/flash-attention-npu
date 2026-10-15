"""v28 allocation/used safety and fixed-source TND M/N and native partitions."""

import contextlib
import itertools
import os
from pathlib import Path
import pytest
from tests.mla.support.mla_cpu_driver import CpuDriver
from tests.mla.support.mla_prefill_bsnd_build import compile_drivers


@pytest.fixture(scope="module")
def drivers(tmp_path_factory):
    root = Path(
        os.environ.get("MLA_SCHEDULE_ARTIFACT_DIR") or tmp_path_factory.mktemp("prefill-tnd")
    )
    with contextlib.ExitStack() as stack:
        yield [
            stack.enter_context(CpuDriver(p, args=() if i == 1 else ("--tnd",)))
            for i, p in enumerate(compile_drivers(root))
        ]


def request(
    q=(7, 0, 17),
    k=(1057, 31, 0),
    used_q=None,
    used_k=None,
    mask=0,
    splits=0,
    heads=8,
    kv_heads=2,
    dtype=1,
    fault=-1,
    value=0,
    overrides=None,
):
    cq = [0, *itertools.accumulate(q)]
    ck = [0, *itertools.accumulate(k)]
    maxq, maxk = max(q, default=0), max(k, default=0)
    flags = int(used_q is not None) + 2 * int(used_k is not None)
    left, right = (maxk - 1, maxq - 1) if mask == 0 else (maxk - 1, 0) if mask == 1 else (7, 2)
    header = [
        len(q),
        maxq,
        maxk,
        heads,
        kv_heads,
        splits,
        20,
        dtype,
        flags,
        mask,
        left,
        right,
        fault,
        value,
        sum(q),
        sum(k),
    ]
    for i, v in (overrides or {}).items():
        header[i] = v
    return (
        header
        + list(q if used_q is None else used_q)
        + list(k if used_k is None else used_k)
        + cq
        + ck
    )


@pytest.mark.parametrize(
    "q,k",
    [
        ((7, 0, 17), (1057, 31, 0)),
        ((1, 1), (16384, 513)),
        ((129, 3), (517, 1025)),
        ((), ()),
        ((0, 0), (31, 3)),
        ((7, 2), (0, 0)),
    ],
)
@pytest.mark.parametrize("mask,splits,dtype", [(0, 1, 1), (1, 2, 2), (2, 0, 1), (0, 256, 2)])
def test_source_partitions(drivers, q, k, mask, splits, dtype):
    r = drivers[0].evaluate([request(q, k, mask=mask, splits=splits, dtype=dtype)])[0]
    assert r["built"] and r["status"] == 0, r
    record = r["record"]
    assert record[1:5] == [28, 28, len(record), dtype]
    assert record[20:24] == [sum(q), sum(k), 0, 0]
    if not sum(q) * sum(k):
        return
    m, n = record[13:15]
    assert [m, n] == drivers[2].evaluate([[max(q), max(k), 4, mask]])[0]
    assert m == 512
    kind, left, right = record[16:19]
    native = drivers[1].evaluate(
        [
            [
                1,
                3 if kind == 1 else 4 if kind == 2 else 0,
                2**31 - 1 if kind == 1 else left,
                0 if kind == 1 else right,
                4,
                m,
                n,
                20,
                2,
                len(q) * 2,
                *itertools.chain.from_iterable(pair for pair in zip(q, k) for _ in range(2)),
            ]
        ]
    )[0]
    tail = 24 + 8 * len(q)
    assert record[tail + 3] == native[0]
    assert record[tail + 82 : tail + 86] == [*native[2:5], 8]
    pos = 7
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
        count = 20 if size == 26 else 40
        assert record[tail + offset : tail + offset + count] == native[pos : pos + count]
        assert not any(record[tail + offset + count : tail + offset + size])
        pos += count


@pytest.mark.parametrize("flags", [0, 1, 2, 3])
@pytest.mark.parametrize("dtype", [1, 2])
def test_allocation_not_used_prefix(drivers, flags, dtype):
    q, k = (7, 0, 17), (1057, 31, 0)
    uq, uk = ((3, 0, 1) if flags & 1 else None), ((513, 0, 0) if flags & 2 else None)
    r = drivers[0].evaluate([request(q, k, used_q=uq, used_k=uk, dtype=dtype)])[0]
    assert r["status"] == 0
    for b, (qs, ks) in enumerate(zip([0, 7, 7], [0, 1057, 1088])):
        assert r["record"][24 + b * 8 : 32 + b * 8] == [
            qs,
            0,
            ks,
            0,
            q[b],
            k[b],
            (uq or q)[b],
            (uk or k)[b],
        ]


@pytest.mark.parametrize(
    "word", [*range(24), 24, 25, 26, 27, 28, 29, 30, 31, 40, 51, 130, 134, 160, 394, 400]
)
def test_corruption_atomic(drivers, word):
    assert drivers[0].evaluate([request(fault=word, value=0xFFFFFFFF)])[0]["status"] != 0


@pytest.mark.parametrize("core", [0, 1, 19, 20, 25])
@pytest.mark.parametrize("dtype", [1, 2])
def test_fd_start_ordinal_rejected(drivers, core, dtype):
    shape = dict(q=(1, 1), k=(16384, 513), dtype=dtype)
    valid = drivers[0].evaluate([request(**shape)])[0]
    assert valid["status"] == 0
    word = 24 + 8 * 2 + 164 + core
    invalid = request(**shape, fault=word, value=valid["record"][word] + 1)
    assert drivers[0].evaluate([invalid])[0]["status"] != 0


@pytest.mark.parametrize("fault", [-2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12])
def test_addresses(drivers, fault):
    r = drivers[0].evaluate([request(used_q=(7, 0, 17), used_k=(1057, 31, 0), fault=fault)])[0]
    assert not r["built"] or r["status"] != 0


@pytest.mark.parametrize(
    "axis,index,value",
    [
        (0, 0, 1),
        (0, 1, -1),
        (0, 2, 6),
        (0, 3, 23),
        (0, 1, 25),
        (1, 0, 1),
        (1, 1, -1),
        (1, 2, 1056),
        (1, 3, 1087),
        (1, 1, 1089),
    ],
)
def test_bad_prefix(drivers, axis, index, value):
    req = request()
    req[16 + 6 + axis * 4 + index] = value
    r = drivers[0].evaluate([req])[0]
    assert r["built"] and r["record"][5] and not any(r["record"][6:])


@pytest.mark.parametrize(
    "q,k",
    [
        ((-1, 0, 1), (513, 0, 0)),
        ((8, 0, 1), (1, 0, 0)),
        ((1, 1, 1), (1, 0, 0)),
        ((1, 0, 1), (1058, 0, 0)),
    ],
)
def test_bad_used(drivers, q, k):
    assert drivers[0].evaluate([request(used_q=q, used_k=k)])[0]["status"] != 0


@pytest.mark.parametrize(
    "override", [{3: 3}, {5: 257}, {6: 27}, {14: 2**31}, {15: 2**31}, {1: 2**30}, {2: 16385}]
)
def test_bounds(drivers, override):
    assert not drivers[0].evaluate([request(overrides=override)])[0]["layout"]
