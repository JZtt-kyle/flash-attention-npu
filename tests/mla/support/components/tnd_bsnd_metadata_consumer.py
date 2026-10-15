"""Shared helpers extracted from tests/mla/unit/mixed/test_flash_attn_npu_v3_mla_tnd_bsnd_metadata_consumer.py.

v4 shared decoder: source schedules, physical KV spans and rejection boundaries.
"""

import ctypes as ct
from tests.mla.support.components.metadata_consumer import Memory
from tests.mla.support.components.bsnd_metadata_consumer import expected_schedule as bsnd_schedule
from tests.mla.support.components.tnd_bsnd_metadata import (
    SENTINEL as SENTINEL,
    input_lengths,
    mixed_case as mixed_case,
)


class Context(ct.Structure):
    _fields_ = [
        (n, ct.c_uint32)
        for n in (
            "dtype",
            "batch",
            "total_q",
            "heads",
            "max_q",
            "capacity",
            "aic",
            "aiv",
            "block_dim",
        )
    ] + [("workspace", ct.c_uint64), ("libapi", ct.c_uint64), ("memory", Memory)]


def prefixes(values):
    q, _, k = input_lengths(values)
    return q, k


def context_for(row, values, dtype):
    cq, _ = prefixes(values)
    return Context(
        dtype,
        values[0],
        cq[-1],
        values[2],
        values[1],
        values[3],
        values[4],
        values[5],
        row[21],
        row[385] | row[386] << 32,
        values[11] if cq[-1] and values[3] else 0,
        Memory(*values[6:11]),
    )


def expected_schedule(row):
    return bsnd_schedule(row) + [row[401], *row[404:406]]
