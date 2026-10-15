"""Shared helpers extracted from tests/mla/unit/dense/test_flash_attn_npu_v3_mla_metadata_dense.py.

AICPU -> immutable metadata -> dense device consumer, without public dispatch.
"""

import ctypes as ct
import torch

import torch_npu  # noqa: F401 -- register the NPU backend before torch.npu helpers
from tests.mla.support.components.dense import check_case, launch, load_probes, make_case
from tests.mla.support.components.metadata import probe as probe
from tests.mla.support.components.metadata import Args, Shape, enqueue
from tests.mla.support.components.metadata_consumer import SENTINEL, WORDS, Context, Memory


def configure(libraries, directory, dtype):
    probes = load_probes(libraries, directory, dtype)
    for name in ("metadata", "metadata_checked"):
        lib = probes[0][name]
        lib.launch_dense_metadata.argtypes = [ct.c_void_p] * 10 + [
            ct.c_uint32,
            ct.c_void_p,
            ct.POINTER(Context),
            ct.c_float,
        ]
        lib.launch_dense_metadata.restype = ct.c_int
    return probes


def make_metadata_case(probes, probe, shape, *, scale=None, worker=0):
    case = make_case(probes, *shape, scale=scale, offset=1)
    d = case["details"]
    dtype = 1 if probes[0]["dtype"] == torch.float16 else 2
    case["context"] = Context(
        dtype, *shape, d[2], d[3], case["schedule"].block_dim, d[0], d[9], Memory(*d[4:9])
    )
    storage = torch.full((WORDS + 2,), SENTINEL, dtype=torch.int32, device="npu")
    case["metadata_storage"] = storage
    case["metadata"] = storage[1:-1]  # Odd int32 offset, only 4-byte aligned.
    case["status_storage"] = torch.full(
        ((3 * case["context"].block_dim + 2) * 16,), SENTINEL, dtype=torch.int32, device="npu"
    )
    case["status"] = case["status_storage"][16:-16]
    case["cache_seqlens"] = torch.full((shape[0],), shape[3], dtype=torch.int32, device="npu")
    args = Args(0, WORDS, dtype, Shape(*shape), probe["platform"])
    # No D2H or synchronization between the AICPU producer and the consumer.
    case["metadata_expected"] = enqueue(
        probe, args, storage, worker, cache_seqlens=case["cache_seqlens"]
    )
    return case


def launch_consumer(probes, case, *, words=WORDS, address=None, context=None, checked=True):
    tensors = [case["device"][n] for n in ("qv", "q", "v", "k")]
    tensors += [case["desc"], *case["outputs"]["migrated"], case["workspace"]["migrated"]]
    result = probes[0]["metadata_checked" if checked else "metadata"].launch_dense_metadata(
        torch.npu.current_stream().npu_stream,
        *(t.data_ptr() for t in tensors),
        case["metadata"].data_ptr() if address is None else address,
        words,
        None if checked else case["status"].data_ptr(),
        ct.byref(case["context"] if context is None else context),
        case["scale"],
    )
    assert result == 0, result


def assert_status(case, expected):
    words = case["status_storage"].cpu()
    expected_words = torch.full_like(words, SENTINEL)
    expected_words[16:-16:16] = expected
    assert torch.equal(words, expected_words), words.tolist()


def check_metadata_case(probes, case, repeats=3):
    def dispatch(lib, current, name):
        if name == "migrated":
            launch_consumer(probes, current)
        else:
            launch(lib, current, name)

    record = check_case(probes, case, repeats=repeats, launch_fn=dispatch)
    actual = case["metadata_storage"].cpu().tolist()
    assert actual[0] == actual[-1] == SENTINEL
    assert [x & 0xFFFFFFFF for x in actual[1:-1]] == case["metadata_expected"]
    return record
