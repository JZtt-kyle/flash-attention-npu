"""Shared helpers extracted from tests/mla/unit/tnd/test_flash_attn_npu_v3_mla_tnd_metadata_aicpu.py.

910B v3 AICPU producer; no TND attention/device consumer is enabled.
"""

import ctypes as ct
import json
import os
from pathlib import Path
import pytest
import torch

import torch_npu  # noqa: F401 -- register the NPU backend before torch.npu helpers
from tests.mla.support.mla_metadata_build import ROOT, build_probe, sha256
from tests.mla.support.mla_metadata_error_probe import forbid_host_readback as forbid_host_readback
from tests.mla.support.components.metadata import (
    Args as Args,
    Memory,
    Platform,
    Shape as Shape,
    load_probe,
)
from tests.mla.support.components.tnd_metadata import (
    tnd_artifacts as tnd_artifacts,
    tnd_oracle as tnd_oracle,
    FAILURES as FAILURES,
    PREFIX,
    SENTINEL,
    check_records as check_records,
    metadata_cases as metadata_cases,
    prefixes,
    tnd_case as tnd_case,
)


class TndShape(ct.Structure):
    _fields_ = [
        (name, ct.c_int64) for name in ("batch", "total_q", "total_k", "heads", "max_q", "max_k")
    ]


class TndArgs(ct.Structure):
    _fields_ = [
        ("output", ct.c_uint64),
        ("words", ct.c_uint32),
        ("dtype", ct.c_uint32),
        ("shape", TndShape),
        ("platform", Platform),
        ("cu_q", ct.c_uint64),
        ("qcount", ct.c_uint64),
        ("cu_k", ct.c_uint64),
        ("kcount", ct.c_uint64),
    ]


def arguments(values, dtype):
    tail = 16 + values[15]
    return TndArgs(
        0,
        PREFIX + values[0] * 4,
        dtype,
        TndShape(values[0], values[tail], values[tail + 1], values[2], values[1], values[3]),
        Platform(
            values[4],
            values[5],
            Memory(*values[6:11]),
            values[11],
            (ct.c_uint32 * 26)(*[values[12] or c for c in range(1, 27)]),
        ),
        0,
        values[tail + 3],
        0,
        values[14],
    )


@pytest.fixture(scope="module")
def tnd_probe(tmp_path_factory):
    directory = Path(
        os.environ.get("MLA_TND_METADATA_DEVICE_DIR", tmp_path_factory.mktemp("mla-tnd-aicpu"))
    )
    library = os.environ.get("MLA_TND_METADATA_LIBRARY")
    if library:
        manifest = json.loads((Path(library).parent / "build-manifest.json").read_text())
        assert sha256(library) == manifest["library_sha256"]
        for path, digest in manifest["sources"].items():
            assert sha256(ROOT / path) == digest, path
    else:
        library = build_probe(directory, tnd=True)
    with load_probe(library, directory) as probe:
        lib = probe["lib"]
        lib.tnd_metadata_args_size.restype = ct.c_uint32
        assert lib.tnd_metadata_args_size() == ct.sizeof(TndArgs)
        lib.launch_tnd_metadata.argtypes = [ct.c_void_p] * 4 + [ct.POINTER(TndArgs)]
        lib.launch_tnd_metadata.restype = ct.c_int
        lib.tnd_metadata_host.argtypes = [ct.POINTER(TndArgs), ct.c_void_p]
        lib.tnd_metadata_host.restype = ct.c_uint32
        yield probe


def host_record(probe, values, dtype):
    args = arguments(values, dtype)
    cq, ck = prefixes(values)
    q = (ct.c_int32 * (len(cq) + 2))(SENTINEL, *cq, SENTINEL)
    k = (ct.c_int32 * (len(ck) + 2))(SENTINEL, *ck, SENTINEL)
    args.cu_q = ct.addressof(q) + 4 if values[18 + values[15]] else 0
    args.cu_k = ct.addressof(k) + 4 if values[13] else 0
    output = (ct.c_uint32 * args.words)()
    assert probe["lib"].tnd_metadata_host(ct.byref(args), output) == 0
    assert list(q) == [SENTINEL, *cq, SENTINEL] and list(k) == [SENTINEL, *ck, SENTINEL]
    return list(output)


def device_inputs(values):
    def tensor(prefix, present):
        if not present:
            return None
        return torch.tensor([SENTINEL, *prefix, SENTINEL], dtype=torch.int32, device="npu")[1:-1]

    cq, ck = prefixes(values)
    return tensor(cq, values[18 + values[15]]), tensor(ck, values[13])


def enqueue(probe, args, storage, q, k, worker=0):
    output = storage[1:-1]
    assert output.is_contiguous() and output.numel() == args.words
    args.output = output.data_ptr()
    args.cu_q = 0 if q is None else q.data_ptr()
    args.cu_k = 0 if k is None else k.data_ptr()
    stream = probe["workers"][worker]
    ready, done = probe["events"][worker * 2 : worker * 2 + 2]
    assert (
        probe["lib"].launch_tnd_metadata(
            torch.npu.current_stream().npu_stream, stream.npu_stream, ready, done, ct.byref(args)
        )
        == 0
    )
    for tensor in (output, q, k):
        if tensor is not None:
            tensor.record_stream(stream)


def storage_for(args):
    return torch.full((args.words + 2,), SENTINEL, dtype=torch.int32, device="npu")


def check_storage(storage, expected):
    words = [w & 0xFFFFFFFF for w in storage.cpu().tolist()]
    assert words == [SENTINEL, *expected, SENTINEL]
    return words[1:-1]
