"""Shared helpers extracted from tests/mla/unit/paged_bsnd/test_flash_attn_npu_v3_mla_paged_metadata_consumer.py.

Shared v6 CPU/device consumer: current lengths, full cache spans, effective pages.
"""

import ctypes as ct
from tests.mla.support.components.metadata_consumer import Memory
from tests.mla.support.components.bsnd_metadata_consumer import expected_schedule as dense_schedule
from tests.mla.support.components.paged_metadata import encode, wide, paged_case as paged_case

SENTINEL = 0x12345678

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
