"""Shared helpers extracted from tests/mla/unit/used/test_flash_attn_npu_v3_mla_used_metadata_aicpu.py.

910B v5 independent generation; no used attention consumer or public entry.
"""

import ctypes as ct
import os
from pathlib import Path
import pytest
import torch

import torch_npu  # noqa: F401 -- register the NPU backend before torch.npu helpers
from tests.mla.support.mla_metadata_build import build_probe
from tests.mla.support.components.metadata import Memory, Platform, load_probe
from tests.mla.support.components.tnd_bsnd_metadata_aicpu import storage_for as storage_for
from tests.mla.support.components.used_metadata import PREFIX, SENTINEL, encode, input_arrays


class Allocation(ct.Structure):
    _fields_ = [("layout", ct.c_int32), ("total", ct.c_int64), ("maximum", ct.c_int64)]


class UsedShape(ct.Structure):
    _fields_ = [
        ("batch", ct.c_int64),
        ("heads", ct.c_int64),
        ("query", Allocation),
        ("kv", Allocation),
    ]


class Array(ct.Structure):
    _fields_ = [("address", ct.c_uint64), ("count", ct.c_uint64)]


ARRAY_NAMES = ("cu_q", "cu_k", "used_q", "used_k")


class UsedArgs(ct.Structure):
    _fields_ = [
        ("output", ct.c_uint64),
        ("words", ct.c_uint32),
        ("dtype", ct.c_uint32),
        ("shape", UsedShape),
        ("platform", Platform),
    ] + [(name, Array) for name in ARRAY_NAMES]


def arguments(item, dtype):
    values = encode(item)
    return UsedArgs(
        0,
        PREFIX + values[0] * 8,
        dtype,
        UsedShape(
            values[0],
            values[2],
            Allocation(int(item["layout"] != 0), item.get("total_q", sum(item["q"])), values[1]),
            Allocation(int(item["layout"] == 1), item.get("total_k", sum(item["kv"])), values[3]),
        ),
        Platform(
            values[4],
            values[5],
            Memory(*values[6:11]),
            values[11],
            (ct.c_uint32 * 26)(*[values[12] or c for c in range(1, 27)]),
        ),
        *[Array(0, count) for _, _, count in input_arrays(item)],
    )


@pytest.fixture(scope="module")
def used_probe(tmp_path_factory):
    directory = Path(
        os.environ.get("MLA_USED_METADATA_DEVICE_DIR", tmp_path_factory.mktemp("mla-used-aicpu"))
    )
    with load_probe(build_probe(directory, used=True), directory) as probe:
        lib = probe["lib"]
        lib.used_metadata_args_size.restype = ct.c_uint32
        assert lib.used_metadata_args_size() == ct.sizeof(UsedArgs)
        lib.launch_used_metadata.argtypes = [ct.c_void_p] * 4 + [ct.POINTER(UsedArgs)]
        lib.launch_used_metadata.restype = ct.c_int
        lib.used_metadata_host.argtypes = [ct.POINTER(UsedArgs), ct.c_void_p]
        lib.used_metadata_host.restype = ct.c_uint32
        yield probe


def host_record(probe, item, dtype):
    args = arguments(item, dtype)
    keep = []
    for name, (values, present, _) in zip(ARRAY_NAMES, input_arrays(item)):
        data = (ct.c_int32 * (len(values) + 2))(SENTINEL, *values, SENTINEL)
        getattr(args, name).address = ct.addressof(data) + 4 if present else 0
        keep.append((data, [SENTINEL, *values, SENTINEL]))
    output = (ct.c_uint32 * args.words)()
    assert probe["lib"].used_metadata_host(ct.byref(args), output) == 0
    assert all(list(data) == before for data, before in keep)
    return list(output)


def device_inputs(item):
    return [
        torch.tensor([SENTINEL, *values, SENTINEL], dtype=torch.int32, device="npu")[1:-1]
        if present
        else None
        for values, present, _ in input_arrays(item)
    ]


def enqueue(probe, args, storage, inputs, worker=0, *, descriptors=None):
    output = storage[1:-1]
    assert output.is_contiguous() and output.numel() == args.words
    args.output = output.data_ptr()
    for name, tensor in zip(ARRAY_NAMES, inputs):
        # An explicit B=0 view still has a presence flag. Tensor.data_ptr()
        # may be zero for empty views; the guarded backing allocation is live.
        getattr(args, name).address = (
            0
            if tensor is None
            else (
                tensor.untyped_storage().data_ptr()
                + tensor.storage_offset() * tensor.element_size()
            )
        )
    for name, (address, count) in (descriptors or {}).items():
        setattr(args, name, Array(address, count))
    stream = probe["workers"][worker]
    ready, done = probe["events"][worker * 2 : worker * 2 + 2]
    assert (
        probe["lib"].launch_used_metadata(
            torch.npu.current_stream().npu_stream, stream.npu_stream, ready, done, ct.byref(args)
        )
        == 0
    )
    for tensor in (output, *inputs):
        if tensor is not None:
            tensor.record_stream(stream)
