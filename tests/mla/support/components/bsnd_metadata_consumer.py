"""Shared helpers extracted from tests/mla/unit/bsnd/test_flash_attn_npu_v3_mla_bsnd_metadata_consumer.py.

Shared CPU/device v2 decoder: source records, stale lengths and unsafe holes.
"""

from tests.mla.support.components.bsnd_metadata import arguments
from tests.mla.support.components.metadata_consumer import Context


def context_for(row, values, dtype):
    # Independent trusted generation supplies workspace/blockDim. Mutations
    # never change this caller context, and shape/platform come from the call.
    p = arguments(values, dtype).platform
    return Context(
        dtype,
        *values[:4],
        p.aic,
        p.aiv,
        row[21],
        row[385] | row[386] << 32,
        0 if not all(values[i] for i in (0, 1, 3)) else p.libapi,
        p.memory,
    )


def expected_schedule(row):
    used = row[20]
    return [row[i] for i in (8, 11, 10, 12, 15, 16, 20, 21, 19, 367, 369)] + [
        row[23 + axis * 26 + core] if core < used else 0 for axis in range(3) for core in range(26)
    ]
