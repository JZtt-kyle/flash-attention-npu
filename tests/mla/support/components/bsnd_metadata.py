"""Shared helpers extracted from tests/mla/unit/bsnd/test_flash_attn_npu_v3_mla_bsnd_metadata.py.

CPU wire/entry checks for BSND length snapshots, independently source checked.
"""

import ctypes as ct
from tests.mla.support.components.metadata_consumer import Args, Memory, Platform, Shape

PREFIX = 401

MAGIC = 0x4D4C4131

SENTINEL = 0x12345678


def arguments(values, dtype=1):
    return Args(
        0,
        PREFIX + values[0],
        dtype,
        Shape(*values[:4]),
        Platform(
            values[4],
            values[5],
            Memory(*values[6:11]),
            values[11],
            (ct.c_uint32 * 26)(*[values[12] or c for c in range(1, 27)]),
        ),
    )


def projection(row):
    # Fixed wire offsets from design 6.2, independent of C++ struct casts.
    def wide(values):
        return [lo | hi << 32 for lo, hi in zip(values[::2], values[1::2])]

    s = row[8:PREFIX]
    w = wide(s[359:383])
    return (
        [0, s[7], s[8], s[12], wide(s[93:95])[0], *s[95:99], *w[:4], *w[9:12], s[13], s[14]]
        + s[15:93]
        + s[99:359]
    )
