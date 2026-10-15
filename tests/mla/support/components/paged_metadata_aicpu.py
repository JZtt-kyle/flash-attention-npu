"""Shared helpers extracted from tests/mla/unit/paged_bsnd/test_flash_attn_npu_v3_mla_paged_metadata_aicpu.py.

910B standalone v6 producer: CPU/pinned PA source, bounds and stream lifetime.

The diagnostic wrapper calls the exact entry and captures its return code;
it does not claim device consumer, Trap, Copy or attention acceptance.
"""

import ctypes as ct
import json
import os
from pathlib import Path
import pytest
import torch

import torch_npu  # noqa: F401 -- register the NPU backend before torch.npu helpers
from tests.mla.support.mla_metadata_build import ROOT, build_probe, sha256
from tests.mla.support.components.metadata import Memory, Platform, load_probe, SENTINEL
from tests.mla.support.components.paged_metadata import PREFIX, encode
from tests.mla.support.components.tnd_bsnd_metadata_aicpu import storage_for as storage_for
from tests.mla.support.components.used_metadata_aicpu import Allocation, Array


class PagedShape(ct.Structure):
    _fields_ = [
        ("batch", ct.c_int64),
        ("heads", ct.c_int64),
        ("query", Allocation),
        ("capacity", ct.c_int64),
        ("page", ct.c_int64),
    ]


ARRAY_NAMES = ("cu_q", "used_q", "kv_lengths")


class PagedArgs(ct.Structure):
    _fields_ = [
        ("output", ct.c_uint64),
        ("words", ct.c_uint32),
        ("dtype", ct.c_uint32),
        ("shape", PagedShape),
        ("platform", Platform),
    ] + [(name, Array) for name in ARRAY_NAMES]


class ProbeArgs(ct.Structure):
    _fields_ = [("input", PagedArgs), ("status", ct.c_uint64), ("null_args", ct.c_uint32)]


def input_arrays(item):
    values, offset, arrays = encode(item), 16, []
    for _ in ARRAY_NAMES:
        present, count, size = values[offset : offset + 3]
        arrays.append((values[offset + 3 : offset + 3 + size], present, count))
        offset += 3 + size
    assert offset == len(values)
    return arrays


def arguments(item, dtype):
    v = encode(item)
    return PagedArgs(
        0,
        PREFIX + 5 * v[0],
        dtype,
        PagedShape(v[0], v[1], Allocation(*v[2:5]), v[6], v[5]),
        Platform(
            v[7],
            v[8],
            Memory(*v[9:14]),
            v[14],
            (ct.c_uint32 * 26)(*[v[15] or c for c in range(1, 27)]),
        ),
        *[Array(0, count) for _, _, count in input_arrays(item)],
    )


@pytest.fixture(scope="module")
def paged_probe(tmp_path_factory):
    target = os.environ.get("MLA_PAGED_METADATA_DEVICE_DIR")
    directory = Path(target) if target else tmp_path_factory.mktemp("mla-paged-aicpu")
    library = os.environ.get("MLA_PAGED_METADATA_LIBRARY")
    if library:
        manifest = json.loads((Path(library).parent / "build-manifest.json").read_text())
        assert sha256(library) == manifest["library_sha256"]
        for path, digest in manifest["sources"].items():
            assert sha256(ROOT / path) == digest, path
    else:
        library = build_probe(directory, paged=True)
    with load_probe(library, directory) as probe:
        lib = probe["lib"]
        for name, typ in (
            ("paged_metadata_args_size", PagedArgs),
            ("paged_metadata_probe_args_size", ProbeArgs),
        ):
            getattr(lib, name).restype = ct.c_uint32
            assert getattr(lib, name)() == ct.sizeof(typ)
        lib.launch_paged_metadata.argtypes = [ct.c_void_p] * 4 + [
            ct.POINTER(PagedArgs),
            ct.c_uint64,
            ct.c_bool,
        ]
        lib.launch_paged_metadata.restype = ct.c_int
        lib.paged_metadata_host.argtypes = [ct.POINTER(PagedArgs), ct.c_void_p]
        lib.paged_metadata_host.restype = ct.c_uint32
        yield probe


def host_record(probe, item, dtype, *, causal=False):
    args, keep = arguments(item, dtype), []
    for name, (values, present, _) in zip(ARRAY_NAMES, input_arrays(item)):
        data = (ct.c_int32 * (len(values) + 2))(SENTINEL, *values, SENTINEL)
        getattr(args, name).address = ct.addressof(data) + 4 if present else 0
        keep.append((data, [SENTINEL, *values, SENTINEL]))
    output = (ct.c_uint32 * args.words)()
    function = (
        probe["lib"].causal_paged_metadata_host if causal else probe["lib"].paged_metadata_host
    )
    function.argtypes = [ct.POINTER(PagedArgs), ct.c_void_p]
    function.restype = ct.c_uint32
    assert function(ct.byref(args), output) == 0
    assert all(list(data) == before for data, before in keep)
    return list(output)


def device_inputs(item):
    return [
        torch.tensor([SENTINEL, *values, SENTINEL], dtype=torch.int32, device="npu")[1:-1]
        if present
        else None
        for values, present, _ in input_arrays(item)
    ]


def enqueue(
    probe,
    args,
    storage,
    inputs,
    worker=0,
    *,
    descriptors=None,
    status=None,
    null_args=False,
    output=None,
):
    # Keep explicit zero-count presence using the live guarded allocation.
    args.output = storage[1:-1].data_ptr() if output is None else output
    for name, tensor in zip(ARRAY_NAMES, inputs):
        getattr(args, name).address = (
            0
            if tensor is None
            else tensor.untyped_storage().data_ptr() + 4 * tensor.storage_offset()
        )
    for name, value in (descriptors or {}).items():
        setattr(args, name, Array(*value))
    stream = probe["workers"][worker]
    ready, done = probe["events"][2 * worker : 2 * worker + 2]
    assert (
        probe["lib"].launch_paged_metadata(
            torch.npu.current_stream().npu_stream,
            stream.npu_stream,
            ready,
            done,
            ct.byref(args),
            0 if status is None else status.data_ptr(),
            null_args,
        )
        == 0
    )
    for tensor in (storage, *inputs, status):
        if tensor is not None:
            tensor.record_stream(stream)
