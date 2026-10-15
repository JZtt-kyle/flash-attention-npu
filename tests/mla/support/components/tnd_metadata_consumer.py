"""Shared helpers extracted from tests/mla/unit/tnd/test_flash_attn_npu_v3_mla_tnd_metadata_consumer.py.

v3 shared decoder: source schedules, compact spans and rejection boundaries.
"""

import ctypes as ct
from tests.mla.support.components.metadata_consumer import Memory
from tests.mla.support.components.bsnd_metadata_consumer import expected_schedule as bsnd_schedule
from tests.mla.support.components.tnd_metadata import (
    SENTINEL as SENTINEL,
    prefixes,
    tnd_case as tnd_case,
)


class Context(ct.Structure):
    _fields_ = [
        (n, ct.c_uint32)
        for n in (
            "dtype",
            "batch",
            "total_q",
            "total_k",
            "heads",
            "max_q",
            "max_k",
            "aic",
            "aiv",
            "block_dim",
        )
    ] + [("workspace", ct.c_uint64), ("libapi", ct.c_uint64), ("memory", Memory)]


def context_for(row, values, dtype):
    cq, ck = prefixes(values)
    return Context(
        dtype,
        values[0],
        cq[-1],
        ck[-1],
        values[2],
        values[1],
        values[3],
        values[4],
        values[5],
        row[21],
        row[385] | row[386] << 32,
        values[11] if cq[-1] and ck[-1] else 0,
        Memory(*values[6:11]),
    )


def expected_schedule(row):
    return bsnd_schedule(row) + row[401:403]
