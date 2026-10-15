"""Shared helpers extracted from tests/mla/unit/dense/test_flash_attn_npu_v3_mla_metadata_consumer.py.

CPU checks of the device decoder, including truncated and corrupt records.
"""

import ctypes as ct

WORDS = 401

SENTINEL = 0x12345678


class Memory(ct.Structure):
    _fields_ = [(n, ct.c_uint64) for n in ("l1", "ub", "l0a", "l0b", "l0c")]


class Context(ct.Structure):
    _fields_ = [
        (n, ct.c_uint32) for n in ("dtype", "batch", "sq", "heads", "sk", "aic", "aiv", "block_dim")
    ] + [("workspace", ct.c_uint64), ("libapi", ct.c_uint64), ("memory", Memory)]


class Shape(ct.Structure):
    _fields_ = [(n, ct.c_int64) for n in ("batch", "sq", "heads", "sk")]


class Platform(ct.Structure):
    _fields_ = [
        ("aic", ct.c_uint32),
        ("aiv", ct.c_uint32),
        ("memory", Memory),
        ("libapi", ct.c_uint64),
        ("block_dims", ct.c_uint32 * 26),
    ]


class Args(ct.Structure):
    _fields_ = [
        ("output", ct.c_uint64),
        ("words", ct.c_uint32),
        ("dtype", ct.c_uint32),
        ("shape", Shape),
        ("platform", Platform),
        ("cache_seqlens", ct.c_uint64),
        ("cache_seqlens_count", ct.c_uint64),
    ]
