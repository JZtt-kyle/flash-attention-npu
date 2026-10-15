"""Shared helpers extracted from tests/mla/unit/bsnd/test_flash_attn_npu_v3_mla_bsnd_metadata_aicpu.py.

910B generation of real BSND length records; no attention consumer yet.
"""

import ctypes as ct
import torch

import torch_npu  # noqa: F401 -- register the NPU backend before torch.npu helpers
from tests.mla.support.components.bsnd_metadata import MAGIC, SENTINEL, arguments


def host_record(probe, values, dtype):
    args = arguments(values, dtype)
    lengths = (ct.c_int32 * max(1, len(values[16:])))(*values[16:])
    args.cache_seqlens = ct.addressof(lengths) if values[13] else 0
    args.cache_seqlens_count = values[14]
    result = (ct.c_uint32 * args.words)()
    assert probe["lib"].bsnd_metadata_host(ct.byref(args), result) == 0
    assert list(result)[:5] == [MAGIC, 2, 2, args.words, dtype]
    return list(result)


def enqueue(probe, args, storage, lengths=None, worker=0, *, dense=False):
    output = storage[1:-1]
    assert output.numel() == args.words and output.is_contiguous()
    args.output = output.data_ptr()
    if lengths is not None:
        assert lengths.is_contiguous() and lengths.ndim == 1 and lengths.dtype == torch.int32
        assert lengths.device == output.device
        args.cache_seqlens = lengths.data_ptr()
        args.cache_seqlens_count = lengths.numel()
    stream = probe["workers"][worker]
    ready, done = probe["events"][2 * worker : 2 * worker + 2]
    launch = probe["lib"].launch_metadata if dense else probe["lib"].launch_bsnd_metadata
    assert (
        launch(
            torch.npu.current_stream().npu_stream, stream.npu_stream, ready, done, ct.byref(args)
        )
        == 0
    )
    output.record_stream(stream)
    if lengths is not None:
        lengths.record_stream(stream)


def device_lengths(values):
    if not values[13]:
        return None
    storage = torch.full((values[0] + 2,), SENTINEL, dtype=torch.int32, device="npu")
    storage[1:-1].copy_(torch.tensor(values[16:], dtype=torch.int32, device="npu"))
    return storage[1:-1]
