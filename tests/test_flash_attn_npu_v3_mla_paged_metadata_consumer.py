"""Shared v6 CPU/device consumer: current lengths, full cache spans, effective pages."""

import ctypes as ct
import itertools
import json
import os
from pathlib import Path
import subprocess

import pytest

from tests.mla_metadata_build import MLA, ROOT, sha256
from tests.test_flash_attn_npu_v3_mla_metadata_consumer import Memory
from tests.test_flash_attn_npu_v3_mla_bsnd_metadata_consumer import (
    expected_schedule as dense_schedule,
)
from tests.test_flash_attn_npu_v3_mla_paged_metadata import (
    paged_builder as paged_builder,
    paged_artifacts as paged_artifacts,
    paged_oracles as paged_oracles,
    check_source,
    run_builder,
    encode,
    wide,
    paged_case,
)

SENTINEL = 0x12345678
PREFIX = 408
ARRAY_NAMES = ("cu_q", "used_q", "kv_lengths")


class Query(ct.Structure):
    _fields_ = [(name, ct.c_uint32) for name in ("layout", "low", "high", "maximum")]


class Context(ct.Structure):
    _fields_ = (
        [(name, ct.c_uint32) for name in ("dtype", "batch", "heads", "flags")]
        + [("query", Query)]
        + [(name, ct.c_uint32) for name in ("page", "capacity", "aic", "aiv", "block_dim")]
        + [("workspace", ct.c_uint64), ("libapi", ct.c_uint64), ("memory", Memory)]
    )


class Array(ct.Structure):
    _fields_ = [("address", ct.c_uint64), ("count", ct.c_uint64)]


class Inputs(ct.Structure):
    _fields_ = [(name, Array) for name in ARRAY_NAMES]


class TensorLayout(ct.Structure):
    _fields_ = [
        (name, ct.c_uint64) for name in ("batch_stride", "row_stride", "head_stride", "span")
    ]


class CacheLayout(ct.Structure):
    _fields_ = [
        (name, ct.c_uint64)
        for name in ("batch", "width", "capacity", "pool", "page", "row_stride", "span")
    ] + [("k", TensorLayout), ("v", TensorLayout)]


class Cache(ct.Structure):
    _fields_ = [("layout", CacheLayout)] + [(name, Array) for name in ("table", "k", "v")]


class TableView(ct.Structure):
    _fields_ = [
        ("sizes", ct.c_int64 * 2),
        ("strides", ct.c_int64 * 2),
        ("offset", ct.c_int64),
        ("storage", ct.c_uint64),
    ]


class TensorView(ct.Structure):
    _fields_ = [
        ("sizes", ct.c_int64 * 4),
        ("strides", ct.c_int64 * 4),
        ("offset", ct.c_int64),
        ("storage", ct.c_uint64),
    ]


@pytest.fixture(scope="module")
def paged_consumer(tmp_path_factory):
    directory = Path(
        os.environ.get("MLA_PAGED_CONSUMER_CPU_DIR", tmp_path_factory.mktemp("mla-paged-consumer"))
    )
    directory.mkdir(parents=True, exist_ok=True)
    sources = [ROOT / "tests/mla_paged_metadata_consumer_driver.cpp"] + [
        MLA / name
        for name in (
            "mla_paged_metadata.cpp",
            "mla_tnd_schedule.cpp",
            "mla_metadata.cpp",
            "mla_schedule.cpp",
            "detail/cann_split_core.cpp",
            "mla_paged_layout.cpp",
            "mla_bsnd_layout.cpp",
        )
    ]
    common = ["g++", "-std=c++17", "-Wall", "-Wextra", "-Werror", f"-I{MLA}", *map(str, sources)]
    library = directory / "consumer.so"
    commands = [[*common, "-O2", "-shared", "-fPIC", "-o", str(library)]]
    if os.environ.get("MLA_CONSUMER_SANITIZE") == "1":
        binary = directory / "consumer-sanitize"
        commands += [
            [
                *common,
                "-O1",
                "-g",
                "-DMLA_CONSUMER_SANITIZE",
                "-fsanitize=address,undefined",
                "-fno-sanitize-recover=all",
                "-fno-omit-frame-pointer",
                "-o",
                str(binary),
            ],
            [str(binary)],
        ]
    (directory / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    with (directory / "build.log").open("w") as log:
        for command in commands:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
    sources += list(MLA.glob("*.h")) + [MLA / "detail/cann_split_core.h", Path(__file__)]
    (directory / "build-manifest.json").write_text(
        json.dumps(
            dict(
                library=str(library.resolve()),
                library_sha256=sha256(library),
                sources={str(p.relative_to(ROOT)): sha256(p) for p in sources},
                sanitizer=os.environ.get("MLA_CONSUMER_SANITIZE") == "1",
            ),
            indent=2,
        )
        + "\n"
    )
    lib = ct.CDLL(str(library.resolve()))
    lib.consume_paged_metadata.argtypes = [
        ct.c_void_p,
        ct.c_uint32,
        ct.POINTER(Context),
        ct.POINTER(Inputs),
        ct.POINTER(Cache),
        ct.c_void_p,
        ct.POINTER(ct.c_uint64),
    ]
    lib.consume_paged_metadata.restype = ct.c_uint32
    lib.build_paged_cache.argtypes = [
        ct.POINTER(TableView),
        ct.POINTER(TensorView),
        ct.POINTER(TensorView),
        ct.POINTER(CacheLayout),
    ]
    lib.build_paged_cache.restype = ct.c_uint32
    return lib, directory


def context_for(row, item, dtype):
    v = encode(item)
    # Only budget/block count comes from the independently generated good record;
    # mutations keep this invocation fixed. Geometry/lengths come from the call.
    return Context(
        dtype,
        v[0],
        v[1],
        int(item["uq"] is not None),
        Query(v[2], *wide(v[3]), v[4]),
        v[5],
        v[6],
        v[7],
        v[8],
        row[21],
        row[385] | row[386] << 32,
        v[14] if row[15] else 0,
        Memory(*v[9:14]),
    )


def input_arrays(item):
    values, offset, arrays = encode(item), 16, []
    for _ in ARRAY_NAMES:
        present, count, size = values[offset : offset + 3]
        arrays.append((values[offset + 3 : offset + 3 + size], present, count))
        offset += 3 + size
    assert offset == len(values)
    return arrays


def expected_schedule(row):
    return dense_schedule(row) + row[401:408]


def cache_spec(item, *, pool=7, row_stride=None, k_strides=None, v_strides=None):
    batch, page, capacity = len(item["q"]), item["page"], item["sk"]
    width = capacity // page
    stride = width + 3 if row_stride is None else row_stride
    span = (batch - 1) * stride + width if batch and width else 0

    def tensor(dim, strides):
        bs, rs, hs = strides if strides is not None else (page * dim + 19, dim, dim)
        return TensorLayout(bs, rs, hs, (pool - 1) * bs + (page - 1) * rs + dim if pool else 0)

    layout = CacheLayout(
        batch,
        width,
        capacity,
        pool,
        page,
        stride,
        span,
        tensor(64, k_strides),
        tensor(512, v_strides),
    )
    # Non-effective entries and row padding stay poison; all effective pages
    # are deliberately permuted and reused, also for overlapping table rows.
    values = [-1] * span
    for b, length in enumerate(item["uk"]):
        for j in range((length + page - 1) // page):
            values[b * stride + j] = (4, 1, 4, 0)[j % 4] % max(pool, 1)
    return layout, values


def change(obj, modifications):
    for path, value in (modifications or {}).items():
        fields = path.split(".")
        target = obj
        for name in fields[:-1]:
            target = getattr(target, name)
        setattr(target, fields[-1], value)


def decode(
    consumer,
    row,
    context,
    item,
    *,
    words=None,
    address=None,
    descriptors=None,
    cache_changes=None,
    spec=None,
    table_values=None,
):
    storage = (ct.c_uint32 * (len(row) + 2))(SENTINEL, *row, SENTINEL)
    output = (ct.c_uint32 * 98)(*([SENTINEL] * 98))
    reads = (ct.c_uint64 * 5)()
    inputs, keep = Inputs(), []
    for name, (values, present, count) in zip(ARRAY_NAMES, input_arrays(item)):
        array = (ct.c_uint32 * (len(values) + 2))(SENTINEL, *values, SENTINEL)
        setattr(inputs, name, Array(ct.addressof(array) + 4 if present else 0, count))
        keep.append((array, list(array)))
    change(inputs, descriptors)
    layout, values = cache_spec(
        dict(item, uk=[0] * len(item["q"])) if table_values is not None else item, **(spec or {})
    )
    values = values if table_values is None else table_values
    table = (ct.c_int32 * (len(values) + 2))(SENTINEL, *values, SENTINEL)
    cache = Cache(
        layout,
        Array(ct.addressof(table) + 4, len(values)),
        Array(4096, layout.k.span),
        Array(8192, layout.v.span),
    )
    change(cache, cache_changes)
    status = consumer[0].consume_paged_metadata(
        ct.addressof(storage) + 4 if address is None else address,
        len(row) if words is None else words,
        ct.byref(context),
        ct.byref(inputs),
        ct.byref(cache),
        ct.addressof(output) + 4,
        reads,
    )
    assert list(storage) == [SENTINEL, *row, SENTINEL]
    assert all(list(array) == before for array, before in keep)
    assert list(table) == [SENTINEL, *values, SENTINEL]
    assert output[0] == output[-1] == SENTINEL
    if status:
        assert list(output) == [SENTINEL] * 98
    return status, list(output)[1:-1], list(reads)


def matrix(page, layout, **kwargs):
    capacity = 16384 // page * page
    q = [7, 7, 7] if layout == 0 else [7, 0, 3]
    return [
        paged_case(layout, q, [length, 0, min(17, length)], uq, page, capacity, **kwargs)
        for length, uq in itertools.product(
            sorted({1, page - 1, page + 1, 513, capacity}), (None, [3, 0, 1])
        )
    ]


@pytest.mark.parametrize(
    "page,layout,dtype", itertools.product(range(16, 1025, 16), (0, 2), (1, 2))
)
def test_source_records(paged_consumer, paged_builder, paged_oracles, page, layout, dtype):
    items = matrix(page, layout)
    rows = check_source(
        paged_builder, paged_oracles, items, dtype, f"consumer-{page}-{layout}-{dtype}"
    )
    for item, row in zip(items, rows):
        status, output, _ = decode(paged_consumer, row, context_for(row, item, dtype), item)
        assert status == 0, (item, status)
        assert output == expected_schedule(row)


# Exact error categories for damaged header, geometry, lengths and resources.
CORRUPTIONS = [
    (0, 0, 2),
    (3, 407, 2),
    (7, 1, 2),
    (4, 3, 4),
    (5, 1, 3),
    (6, 1, 3),
    *[(i, v, 2) for i in (1, 2) for v in (1, 2, 3, 4, 5, 7)],
    *[(i, 0xFFFFFFFF, 5) for i in (*range(8, 15), *range(401, 408))],
    *[(i, 0xFFFFFFFF, 6) for i in range(17, 23)],
    *[(i, 0xFFFFFFFF, 7) for i in (15, 16, 103, 104, 105, 106, 107, 366)],
    *[(i, 0xFFFFFFFF, 8) for i in (23, 48, 49, 74, 75, 100)],
    *[(i, 0xFFFFFFFF, 9) for i in range(367, 391)],
    *[(i, 0xFFFFFFFF, 10) for i in range(391, 401)],
    *[
        (408 + 5 * b + i, 0xFFFFFFFF, 12 if i < 3 else 13 if i == 3 else 11)
        for b in range(2)
        for i in range(5)
    ],
]


@pytest.mark.parametrize("word,value,status", CORRUPTIONS)
def test_corruption(paged_consumer, paged_builder, word, value, status):
    item = paged_case(2, [1, 1], [257, 257], [1, 1], 112, 448, heads=8)
    row = run_builder(paged_builder, [item])[0]
    c = context_for(row, item, 1)
    assert row[word] != value
    row[word] = value
    got, _, reads = decode(paged_consumer, row, c, item)
    assert got == status
    if word < 8:
        assert reads[0] <= 8 and reads[1:] == [0] * 4


@pytest.mark.parametrize("layout", (0, 2))
def test_all_current_lengths(paged_consumer, paged_builder, layout):
    item = paged_case(layout, [7] * 42, [17, 0, 513] * 14, [3, 0, 1] * 14, 112, 16352)
    row = run_builder(paged_builder, [item])[0]
    c = context_for(row, item, 1)
    for axis, (values, present, _) in enumerate(input_arrays(item)):
        if not present:
            continue
        for index in range(len(values)):
            for bad in (-1, 2**31 - 1, values[index] + 1):
                altered = values.copy()
                altered[index] = bad
                current = dict(item, **{("cq", "uq", "uk")[axis]: altered})
                # Keep table independent of invalid lengths.
                result = decode(paged_consumer, row, c, current, table_values=cache_spec(item)[1])
                assert result[0] == (12, 13, 11)[axis]
    changed = dict(item, uq=[0] * 42, uk=[0] * 42)
    updated = run_builder(paged_builder, [changed])[0]
    assert decode(paged_consumer, updated, context_for(updated, changed, 1), changed)[0] == 0


def empty_cases(layout):
    return [
        paged_case(layout, q, kv, uq, page=112, capacity=capacity)
        for q, kv, uq, capacity in (
            ([], [], None, 0),
            ([], [], [], 448),
            ([0, 0], [257, 1], None, 448),
            ([7, 7], [0, 0], None, 0),
            ([7, 7], [0, 0], [3, 0], 448),
            ([7, 7], [257, 1], [0, 0], 448),
        )
    ]


@pytest.mark.parametrize("layout,dtype", itertools.product((0, 2), (1, 2)))
def test_empty_views_and_effective_pages(paged_consumer, paged_builder, layout, dtype):
    for item in empty_cases(layout):
        row = run_builder(paged_builder, [item], dtype)[0]
        c = context_for(row, item, dtype)
        for pool in (0, 7) if not any(item["uk"]) else (7,):
            result = decode(paged_consumer, row, c, item, spec=dict(pool=pool))
            assert result[0] == 0
            assert result[1] == expected_schedule(row)
        if any(item["uk"]):
            assert decode(paged_consumer, row, c, item, spec=dict(pool=0))[0] == 15
            values = cache_spec(item)[1]
            for b, length in enumerate(item["uk"]):
                for j in range((length + c.page - 1) // c.page):
                    index = b * (c.capacity // c.page + 3) + j
                    altered = values.copy()
                    altered[index] = -1
                    assert decode(paged_consumer, row, c, item, table_values=altered)[0] == 15


CACHE_FAULTS = [
    ("layout.batch", 0, 5),
    ("layout.width", 0, 5),
    ("layout.capacity", 0, 5),
    ("layout.page", 16, 5),
    ("layout.pool", 2**31 + 1, 5),
    ("layout.row_stride", 2**63 - 1, 14),
    ("layout.span", 0, 14),
    ("table.address", 0, 14),
    ("table.address", 3, 14),
    ("table.address", 2**64 - 4, 14),
    ("table.count", 0, 14),
    ("table.count", 2**64 - 1, 14),
    *[(f"{x}.address", v, 1) for x in ("k", "v") for v in (0, 1, 2**64 - 2)],
    *[(f"{x}.count", v, 1) for x in ("k", "v") for v in (0, 2**64 - 1)],
    *[
        (f"layout.{x}.{field}", v, 1)
        for x in ("k", "v")
        for field, v in (
            ("span", 0),
            ("batch_stride", 2**63 - 1),
            ("row_stride", 2**63 - 1),
            ("head_stride", 2**63),
        )
    ],
]


@pytest.mark.parametrize("path,value,status", CACHE_FAULTS)
def test_cache_descriptor_rejected_before_read(paged_consumer, paged_builder, path, value, status):
    item = paged_case(2, [1, 1], [257, 257], [1, 1], 112, 448, heads=8)
    row = run_builder(paged_builder, [item])[0]
    result = decode(
        paged_consumer, row, context_for(row, item, 1), item, cache_changes={path: value}
    )
    assert result[0] == status and result[2][1:] == [0] * 4


@pytest.mark.parametrize("layout", (0, 2))
def test_bad_descriptors_and_size(paged_consumer, paged_builder, layout):
    item = paged_case(layout, [7, 7], [257, 1], [3, 0])
    row = run_builder(paged_builder, [item])[0]
    c = context_for(row, item, 1)
    for words in (0, 8, 407, len(row) - 1, len(row) + 1, 2**32 - 1):
        result = decode(paged_consumer, row, c, item, words=words)
        assert result[0] == 1 and result[2] == [0] * 5
    for address in (0, 1, 2**64 - 4):
        assert decode(paged_consumer, row, c, item, address=address)[0] == 1
    for axis, name in enumerate(ARRAY_NAMES):
        for field, value in (("address", 1), ("address", 2**64 - 4), ("count", 2**64 - 1)):
            result = decode(paged_consumer, row, c, item, descriptors={name + "." + field: value})
            assert result[0] == (12, 13, 11)[axis] and result[2][1:] == [0] * 4


@pytest.mark.parametrize("layout", (0, 2))
def test_reuse_strides_and_signed_pages(paged_consumer, paged_builder, layout):
    item = paged_case(layout, [7] * 3, [257, 1, 0], [3, 0, 1], page=112, capacity=448)
    row = run_builder(paged_builder, [item])[0]
    c = context_for(row, item, 1)
    for pool, stride, ks, vs in itertools.product(
        (1, 7),
        (0, 1, 4, 7),
        ((0, 0, 2**63 - 1), (9000, 128, 1)),
        ((0, 0, 1), (70000, 600, 2**63 - 1)),
    ):
        spec = dict(pool=pool, row_stride=stride, k_strides=ks, v_strides=vs)
        assert decode(paged_consumer, row, c, item, spec=spec)[0] == 0
    good = cache_spec(item)[1]
    for b, length in enumerate(item["uk"]):
        for j in range((length + 111) // 112):
            index = b * 7 + j
            for bad in (-1, -(2**31), 7, 2**31 - 1):
                table = good.copy()
                table[index] = bad
                assert decode(paged_consumer, row, c, item, table_values=table)[0] == 15
    # Arithmetic-only wide descriptors: page IDs may reach INT32_MAX; no cache
    # data is read, and no >uint32 device allocation is claimed by this test.
    spec = dict(pool=2**31, k_strides=(65536, 0, 0), v_strides=(1, 0, 0))
    values = cache_spec(item, **spec)[1]
    values[0] = 2**31 - 1
    assert decode(paged_consumer, row, c, item, spec=spec, table_values=values)[0] == 0


CONTEXT_FAULTS = [
    ("dtype", 3, 4),
    ("batch", 1, 1),
    ("batch", 2**32 - 1, 1),
    ("heads", 0, 5),
    ("heads", 2**32 - 1, 5),
    ("flags", 2, 5),
    ("query.layout", 2, 5),
    ("query.low", 0, 5),
    ("query.high", 1, 5),
    ("query.maximum", 2**32 - 1, 5),
    ("page", 0, 5),
    ("page", 15, 5),
    ("page", 1040, 5),
    ("capacity", 449, 5),
    ("capacity", 16464, 5),
    ("aic", 0, 6),
    ("aiv", 20, 6),
    ("block_dim", 0, 6),
    ("workspace", 0, 9),
    ("libapi", 2**64 - 1, 9),
    *[("memory." + name, 0, 10) for name in ("l1", "ub", "l0a", "l0b", "l0c")],
]


@pytest.mark.parametrize("path,value,status", CONTEXT_FAULTS)
def test_independent_context(paged_consumer, paged_builder, path, value, status):
    item = paged_case(2, [1, 1], [257, 257], [1, 1], 112, 448, heads=8)
    row = run_builder(paged_builder, [item])[0]
    c = context_for(row, item, 1)
    change(c, {path: value})
    assert decode(paged_consumer, row, c, item)[0] == status


def test_wide_query_workspace_and_tile_overflow(paged_consumer, paged_builder):
    batch = 65538
    item = paged_case(0, [65536] * batch, [0] * batch, [0] * batch, 1024, 16384, heads=1)
    row = run_builder(paged_builder, [item])[0]
    c = context_for(row, item, 1)
    assert row[403] == row[-4] == 1
    assert decode(paged_consumer, row, c, item, spec=dict(row_stride=0, pool=0))[0] == 0
    row[-4] = 0
    assert decode(paged_consumer, row, c, item, spec=dict(row_stride=0, pool=0))[0] == 12
    item = paged_case(2, page=528)
    first = run_builder(paged_builder, [item])[0]
    item["libapi"] = 2**64 - 1 - (first[385] | first[386] << 32)
    row = run_builder(paged_builder, [item])[0]
    assert row[389:391] == [2**32 - 1] * 2
    assert decode(paged_consumer, row, context_for(row, item, 1), item)[0] == 0
    item = paged_case(0, [65536] * 9, [16384] * 9, [0] * 9, 1024, 16384, heads=65535)
    row = run_builder(paged_builder, [item])[0]
    c = context_for(row, item, 1)
    assert decode(paged_consumer, row, c, item)[0] == 0
    item["uq"] = item["q"]
    for b in range(9):
        row[PREFIX + 5 * b + 3] = 65536
    assert decode(paged_consumer, row, c, item)[0] == 8


def test_allocation_bounds_and_inactive_fields(paged_consumer, paged_builder):
    items = [
        paged_case(2, [2**30, 2**30 - 1], [16, 0], [0, 0], 16, 32, heads=1),
        paged_case(2, [3, 0, 1], [16, 0, 1], [2, 0, 1], 16, 32, heads=1, sq=2**30),
    ]
    for item in items:
        row = run_builder(paged_builder, [item])[0]
        c = context_for(row, item, 1)
        assert decode(paged_consumer, row, c, item)[0] == 0
    item = paged_case(2, [7, 0, 3], [257, 0, 17], [3, 0, 1])
    row = run_builder(paged_builder, [item])[0]
    c = context_for(row, item, 1)
    for word in [
        *range(107, 367),
        *[23 + 26 * axis + i for axis in range(3) for i in range(row[20], 26)],
    ]:
        bad = row.copy()
        bad[word] = 1
        assert decode(paged_consumer, bad, c, item)[0] == (7 if word >= 107 else 8)
    # maxCost is diagnostic: valid addressing need not authenticate optimality.
    row[101:103] = [2**32 - 1] * 2
    assert decode(paged_consumer, row, c, item)[0] == 0


@pytest.mark.parametrize("page", (16, 112, 128, 256, 528, 1024))
def test_host_layout_matches_invocation(paged_consumer, paged_builder, page):
    for batch, pool, stride in ((3, 7, 6), (3, 1, 0), (3, 0, 4), (0, 7, 0)):
        item = paged_case(0, [7] * batch, [0] * batch, page=page, capacity=4 * page)
        spec = dict(
            pool=pool,
            row_stride=stride,
            k_strides=(page * 131, 128, 1),
            v_strides=(page * 601, 600, 1),
        )
        layout, _ = cache_spec(item, **spec)
        table = TableView(
            (ct.c_int64 * 2)(batch, 4), (ct.c_int64 * 2)(stride, 1), 3, (3 + layout.span) * 4
        )
        views = [
            TensorView(
                (ct.c_int64 * 4)(pool, page, 1, dim),
                (ct.c_int64 * 4)(x.batch_stride, x.row_stride, x.head_stride, 1),
                offset,
                (offset + x.span) * 2,
            )
            for x, dim, offset in ((layout.k, 64, 7), (layout.v, 512, 13))
        ]
        actual = CacheLayout()
        assert (
            paged_consumer[0].build_paged_cache(
                ct.byref(table), ct.byref(views[0]), ct.byref(views[1]), ct.byref(actual)
            )
            == 0
        )
        assert bytes(actual) == bytes(layout)
        row = run_builder(paged_builder, [item])[0]
        assert decode(paged_consumer, row, context_for(row, item, 1), item, spec=spec)[0] == 0
