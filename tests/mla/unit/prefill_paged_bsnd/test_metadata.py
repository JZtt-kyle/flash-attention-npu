"""BSND PA prefill protocol, native partition differential and independent corruption."""

import contextlib
import itertools
import os
from pathlib import Path
import pytest
from tests.mla.support.mla_cpu_driver import CpuDriver
from tests.mla.support.mla_prefill_bsnd_build import compile_drivers, compile_paged_layout


@pytest.fixture(scope="module")
def drivers(tmp_path_factory):
    directory = Path(
        os.environ.get("MLA_SCHEDULE_ARTIFACT_DIR") or tmp_path_factory.mktemp("prefill-paged-bsnd")
    )
    with contextlib.ExitStack() as stack:
        binaries = compile_drivers(directory)
        binaries.append(compile_paged_layout(directory))
        yield [
            stack.enter_context(CpuDriver(p, args=("--paged",) if i in (0, 2) else ()))
            for i, p in enumerate(binaries)
        ]


def request(
    shape=(2, 17, 1120, 8, 2),
    splits=0,
    mask=0,
    flags=3,
    fault=-1,
    value=0,
    cores=20,
    dtype=1,
    qlens=None,
    klens=None,
    page=16,
):
    b, q, k, h, hk = shape
    left, right = (k - 1, q - 1) if mask == 0 else (k - 1, 0) if mask == 1 else (7, 2)
    return [
        b,
        q,
        k,
        h,
        hk,
        splits,
        cores,
        dtype,
        flags,
        mask,
        left,
        right,
        fault,
        value,
        page,
        *(qlens if qlens is not None else [q] * b),
        *(klens if klens is not None else [k] * b),
    ]


@pytest.mark.parametrize(
    "shape",
    [
        (2, 17, 1120, 8, 2),
        (1, 1, 8208, 4, 1),
        (1, 7, 8208, 8, 1),
        (1, 16, 16384, 12, 1),
        (1, 65, 16, 8, 8),
        (0, 17, 32, 8, 2),
        (2, 0, 32, 8, 2),
        (2, 7, 0, 8, 2),
    ],
)
@pytest.mark.parametrize("mask,splits,dtype", [(0, 1, 1), (1, 2, 2), (2, 0, 1), (0, 256, 2)])
def test_source_partitions(drivers, shape, mask, splits, dtype):
    req = request(shape, splits, mask, dtype=dtype)
    result = drivers[0].evaluate([req])[0]
    assert result["built"] and result["status"] == 0, result
    record = result["record"]
    b, q, k, h, hk = shape
    assert record[1:5] == [30, 30, len(record), dtype]
    if not b * q * k:
        return
    m, n = record[13:15]
    assert [m, n] == drivers[2].evaluate([[q, k, h // hk, mask, 16]])[0]
    kind, left, right = record[16:19]
    native = drivers[1].evaluate(
        [
            [
                1,
                3 if kind == 1 else 4 if kind == 2 else 0,
                2**31 - 1 if kind == 1 else left,
                0 if kind == 1 else right,
                h // hk,
                m,
                n,
                20,
                2,
                b * hk,
                *itertools.chain.from_iterable([(q, k)] * (b * hk)),
            ]
        ]
    )[0]
    base = 24 + 2 * b
    assert record[base + 3] == native[0]
    assert record[base + 82 : base + 86] == [*native[2:5], 8]
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
        assert record[base + offset : base + offset + count] == native[pos : pos + count]
        assert not any(record[base + offset + count : base + offset + size])
        pos += count


@pytest.mark.parametrize(
    "fault",
    [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        31,
        110,
        374,
        383,
    ],
)
def test_corruption_atomic(drivers, fault):
    result = drivers[0].evaluate([request(fault=fault, value=0xFFFFFFFF)])[0]
    assert result["status"] != 0


@pytest.mark.parametrize(
    "q,k", [([0, 17], [1120, 0]), ([0, 0], [0, 0]), ([18, 17], [1, 1]), ([17, 17], [-1, 1120])]
)
def test_lengths(drivers, q, k):
    r = drivers[0].evaluate([request(qlens=q, klens=k)])[0]
    assert (r["status"] == 0) == (all(0 <= x <= 17 for x in q) and all(0 <= x <= 1120 for x in k))


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(shape=(1, 17, 32, 3, 2)),
        dict(splits=257),
        dict(cores=27),
        dict(shape=(1, 2**30, 32, 8, 2)),
        dict(shape=(1, 17, 16400, 8, 2)),
    ],
)
def test_budget(drivers, kwargs):
    assert not drivers[0].evaluate([request(**kwargs)])[0]["layout"]


@pytest.mark.parametrize("fault", [-2, -3, -4, -5, -6, -7, -8])
def test_address_bounds_and_atomicity(drivers, fault):
    result = drivers[0].evaluate([request(fault=fault)])[0]
    assert result["status"] != 0
    assert bool(result["built"]) == (fault in (-6, -7))
    if result["built"]:
        assert result["record"][5] != 0 and not any(result["record"][6:])


@pytest.mark.parametrize(
    "g,sk,page",
    [(4, 512, 16), (5, 560, 112), (9, 528, 528), (10, 2048, 1024), (4, 4480, 112), (4, 9216, 1024)],
)
@pytest.mark.parametrize("mask", [0, 1, 2])
def test_generic_paged_tiling(drivers, g, sk, page, mask):
    result = drivers[0].evaluate([request(shape=(1, 16, sk, g, 1), flags=0, page=page, mask=mask)])[
        0
    ]
    assert result["built"] and result["status"] == 0
    assert result["record"][13:15] == drivers[2].evaluate([[16, sk, g, mask, page]])[0]


@pytest.mark.parametrize(
    "page,capacity", [(1, 1120), (15, 1120), (17, 1120), (1040, 1040), (112, 1121)]
)
def test_invalid_paged_geometry(drivers, page, capacity):
    assert not drivers[0].evaluate([request(shape=(1, 17, capacity, 8, 2), page=page)])[0]["layout"]


def test_invalid_dtype_zeroes_payload(drivers):
    result = drivers[0].evaluate([request(dtype=99)])[0]
    assert result["built"] and result["record"][5] != 0
    assert not any(result["record"][6:])


@pytest.mark.parametrize("heads,page", [(1, 16), (2, 112), (3, 528), (8, 1024)])
def test_paged_independent_addresses(drivers, heads, page):
    result = drivers[3].evaluate([[0, heads, page]])[0]
    assert result["layout"] == result["status"] == 0
    expected = []
    for start in [0, page - 1, 2 * page + 1]:
        physical = [2, 0, 2][start // page]
        row = start % page
        expected.append(
            [
                physical,
                row,
                min(page - row, 3 * page - 1 - start),
                physical * (page * (heads * 208 + 32) + 96)
                + row * (heads * 208 + 32)
                + (heads - 1) * 208,
                physical * (page * (heads * 160 + 64) + 128)
                + row * (heads * 160 + 64)
                + (heads - 1) * 160,
            ]
        )
    assert result["segments"] == expected


@pytest.mark.parametrize("fault", range(1, 25))
def test_paged_descriptor_boundaries(drivers, fault):
    result = drivers[3].evaluate([[fault, 3, 112]])[0]
    assert (result.get("status", -1) == 0) == (fault in (12, 13, 14))
