"""Shared helpers extracted from tests/mla/unit/dense/test_flash_attn_npu_v3_mla_metadata.py.

AICPU dense schedule component: pinned source, host and device must agree.

Only this test bridge exposes the component; public MLA metadata stays rejected.
Synthetic platform inputs exercise the shared CPU algorithm on AICPU and do not
constitute attention-kernel acceptance on other chips or C1V1 hardware.
"""

import ctypes as ct
from contextlib import contextmanager
import json
import os
from pathlib import Path
import pytest
import torch

import torch_npu  # noqa: F401 -- register the NPU backend before torch.npu helpers
from tests.mla.support.mla_metadata_build import build_probe

WORDS = 401

SENTINEL = 0x12345678


class Shape(ct.Structure):
    _fields_ = [(name, ct.c_int64) for name in ("batch", "sq", "heads", "sk")]


class Memory(ct.Structure):
    _fields_ = [(name, ct.c_uint64) for name in ("l1", "ub", "l0a", "l0b", "l0c")]


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


@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    directory = Path(
        os.environ.get("MLA_METADATA_ARTIFACT_DIR", tmp_path_factory.mktemp("mla-metadata"))
    )
    with load_probe(build_probe(directory), directory) as result:
        yield result


@contextmanager
def load_probe(library, directory):
    """Own test-only AICPU streams/events, also for standalone diagnostics."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.npu.set_device(0)
    assert "910B" in torch.npu.get_device_name(0)
    lib = ct.CDLL(str(library))
    lib.metadata_args_size.restype = ct.c_uint32
    lib.metadata_words.restype = ct.c_uint32
    assert lib.metadata_args_size() == ct.sizeof(Args)
    assert lib.metadata_words() == WORDS
    lib.metadata_platform.argtypes = [ct.POINTER(Platform)]
    lib.metadata_platform.restype = ct.c_int
    lib.metadata_host.argtypes = [ct.POINTER(Args), ct.c_void_p]
    lib.metadata_host.restype = None
    lib.launch_metadata.argtypes = [ct.c_void_p] * 4 + [ct.POINTER(Args)]
    lib.launch_metadata.restype = ct.c_int
    p = Platform()
    assert lib.metadata_platform(ct.byref(p)) == 0
    acl = ct.CDLL(str(Path(os.environ["ASCEND_HOME_PATH"]) / "lib64/libascendcl.so"))
    acl.aclrtCreateEvent.argtypes = [ct.POINTER(ct.c_void_p)]
    acl.aclrtDestroyEvent.argtypes = [ct.c_void_p]
    events = [ct.c_void_p() for _ in range(4)]
    for event in events:
        assert acl.aclrtCreateEvent(ct.byref(event)) == 0
    result = dict(
        lib=lib,
        directory=directory,
        platform=p,
        events=events,
        workers=[torch.npu.Stream(), torch.npu.Stream()],
    )
    (directory / "platform.json").write_text(
        json.dumps(
            dict(
                aic=p.aic,
                aiv=p.aiv,
                memory=[getattr(p.memory, n) for n, _ in Memory._fields_],
                libapi=p.libapi,
                block_dims=list(p.block_dims),
            ),
            indent=2,
        )
        + "\n"
    )
    try:
        yield result
    finally:
        torch.npu.synchronize()
        for event in events:
            assert acl.aclrtDestroyEvent(event) == 0


def enqueue(probe, args, storage, worker=0, cache_seqlens=None):
    output = storage[1:-1]
    assert output.is_contiguous() and output.numel() == WORDS
    args.output = output.data_ptr()
    if cache_seqlens is not None:
        assert cache_seqlens.dtype == torch.int32 and cache_seqlens.is_contiguous()
        assert cache_seqlens.device == output.device and cache_seqlens.ndim == 1
        args.cache_seqlens = cache_seqlens.data_ptr()
        args.cache_seqlens_count = cache_seqlens.numel()
    producer = torch.npu.current_stream()
    worker_stream = probe["workers"][worker]
    ready, done = probe["events"][2 * worker : 2 * worker + 2]
    assert (
        probe["lib"].launch_metadata(
            producer.npu_stream, worker_stream.npu_stream, ready, done, ct.byref(args)
        )
        == 0
    )
    # The fixture also retains storage until synchronization; explicitly record
    # the auxiliary stream, as the production allocator integration will do.
    output.record_stream(worker_stream)
    if cache_seqlens is not None:
        cache_seqlens.record_stream(worker_stream)
    # Independent host schedule for full capacity only; never read device
    # lengths here. Rejection tests separately assert their expected header.
    host = (ct.c_uint32 * WORDS)()
    probe["lib"].metadata_host(ct.byref(args), host)
    return list(host)
