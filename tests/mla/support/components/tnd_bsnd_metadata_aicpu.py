"""Shared helpers extracted from tests/mla/unit/mixed/test_flash_attn_npu_v3_mla_tnd_bsnd_metadata_aicpu.py.

910B v4 mixed-layout AICPU generation, independently of attention.
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
from tests.mla.support.components.metadata import Memory, Platform, load_probe
from tests.mla.support.components.bsnd_metadata import Args as BsndArgs
from tests.mla.support.components.bsnd_schedule import length_case as length_case
from tests.mla.support.components.tnd_bsnd_metadata import (
    mixed_artifacts as mixed_artifacts,
    mixed_oracle as mixed_oracle,
    FAILURES as FAILURES,
    PREFIX,
    SENTINEL,
    check_records as check_records,
    metadata_cases as metadata_cases,
    EMPTY_CASES as EMPTY_CASES,
    input_lengths,
    mixed_case as mixed_case,
    wide_words as wide_words,
    wire_spans as wire_spans,
)


class MixedShape(ct.Structure):
    _fields_ = [(name, ct.c_int64) for name in ("batch", "total_q", "heads", "max_q", "capacity")]


class MixedArgs(ct.Structure):
    _fields_ = [
        ("output", ct.c_uint64),
        ("words", ct.c_uint32),
        ("dtype", ct.c_uint32),
        ("shape", MixedShape),
        ("platform", Platform),
        ("cu_q", ct.c_uint64),
        ("qcount", ct.c_uint64),
        ("kv_lengths", ct.c_uint64),
        ("kcount", ct.c_uint64),
    ]


def arguments(values, dtype):
    tail = 16 + values[15]
    return MixedArgs(
        0,
        PREFIX + values[0] * 5,
        dtype,
        MixedShape(values[0], values[tail], values[2], values[1], values[3]),
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
def mixed_probe(tmp_path_factory):
    directory = Path(
        os.environ.get(
            "MLA_TND_BSND_METADATA_DEVICE_DIR", tmp_path_factory.mktemp("mla-mixed-aicpu")
        )
    )
    library = os.environ.get("MLA_TND_BSND_METADATA_LIBRARY")
    if library:
        manifest = json.loads((Path(library).parent / "build-manifest.json").read_text())
        assert sha256(library) == manifest["library_sha256"]
        for path, digest in manifest["sources"].items():
            assert sha256(ROOT / path) == digest, path
    else:
        library = build_probe(directory, mixed=True, bsnd=True)
    with load_probe(library, directory) as probe:
        lib = probe["lib"]
        lib.mixed_metadata_args_size.restype = ct.c_uint32
        assert lib.mixed_metadata_args_size() == ct.sizeof(MixedArgs)
        lib.launch_mixed_metadata.argtypes = [ct.c_void_p] * 4 + [ct.POINTER(MixedArgs)]
        lib.launch_mixed_metadata.restype = ct.c_int
        lib.mixed_metadata_host.argtypes = [ct.POINTER(MixedArgs), ct.c_void_p]
        lib.mixed_metadata_host.restype = ct.c_uint32
        lib.launch_bsnd_metadata.argtypes = [ct.c_void_p] * 4 + [ct.POINTER(BsndArgs)]
        lib.launch_bsnd_metadata.restype = ct.c_int
        yield probe


def host_record(probe, values, dtype):
    args = arguments(values, dtype)
    cq, _, ck = input_lengths(values)
    q = (ct.c_int32 * (len(cq) + 2))(SENTINEL, *cq, SENTINEL)
    k = (ct.c_int32 * (len(ck) + 2))(SENTINEL, *ck, SENTINEL)
    args.cu_q = ct.addressof(q) + 4 if values[18 + values[15]] else 0
    args.kv_lengths = ct.addressof(k) + 4 if values[13] else 0
    output = (ct.c_uint32 * args.words)()
    assert probe["lib"].mixed_metadata_host(ct.byref(args), output) == 0
    assert list(q) == [SENTINEL, *cq, SENTINEL] and list(k) == [SENTINEL, *ck, SENTINEL]
    return list(output)


def device_inputs(values):
    def tensor(prefix, present):
        if not present:
            return None
        return torch.tensor([SENTINEL, *prefix, SENTINEL], dtype=torch.int32, device="npu")[1:-1]

    cq, _, ck = input_lengths(values)
    return tensor(cq, values[18 + values[15]]), tensor(ck, values[13])


def enqueue(probe, args, storage, q, k, worker=0, *, descriptors=None):
    output = storage[1:-1]
    assert output.is_contiguous() and output.numel() == args.words
    args.output = output.data_ptr()
    args.cu_q = 0 if q is None else q.data_ptr()
    args.kv_lengths = 0 if k is None else k.data_ptr()
    for name, value in (descriptors or {}).items():
        setattr(args, name, value)
    stream = probe["workers"][worker]
    ready, done = probe["events"][worker * 2 : worker * 2 + 2]
    assert (
        probe["lib"].launch_mixed_metadata(
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
