"""Shared helpers extracted from tests/mla/unit/used/test_flash_attn_npu_v3_mla_used_metadata_consumer.py.

v5 allocation/used decoder, shared by CPU sanitizers and release device probes.
"""

import ctypes as ct
from tests.mla.support.components.metadata_consumer import Memory
from tests.mla.support.components.bsnd_metadata_consumer import expected_schedule as bsnd_schedule
from tests.mla.support.components.used_metadata import (
    used_artifacts as used_artifacts,
    used_oracle as used_oracle,
    SENTINEL as SENTINEL,
    check_records as check_records,
    allocation_words,
    used_case as used_case,
    encode,
)


class Allocation(ct.Structure):
    _fields_ = [(name, ct.c_uint32) for name in ("layout", "low", "high", "maximum")]


class Context(ct.Structure):
    _fields_ = (
        [(name, ct.c_uint32) for name in ("dtype", "batch", "heads", "flags")]
        + [
            ("query", Allocation),
            ("kv", Allocation),
        ]
        + [(name, ct.c_uint32) for name in ("aic", "aiv", "block_dim")]
        + [
            ("workspace", ct.c_uint64),
            ("libapi", ct.c_uint64),
            ("memory", Memory),
        ]
    )


class Array(ct.Structure):
    _fields_ = [("address", ct.c_uint64), ("count", ct.c_uint64)]


ARRAY_NAMES = ("cu_q", "cu_k", "used_q", "used_k")


class Inputs(ct.Structure):
    _fields_ = [(name, Array) for name in ARRAY_NAMES]


def context_for(row, item, dtype):
    values = encode(item)
    alloc = allocation_words(item)
    return Context(
        dtype,
        len(item["q"]),
        values[2],
        alloc[8],
        Allocation(*alloc[:4]),
        Allocation(*alloc[4:8]),
        values[4],
        values[5],
        row[21],
        row[385] | row[386] << 32,
        values[11] if row[15] else 0,
        Memory(*values[6:11]),
    )


def expected_schedule(row):
    return bsnd_schedule(row) + row[401:410]
