"""Shared helpers extracted from tests/mla/unit/bsnd/test_flash_attn_npu_v3_mla_bsnd_metadata_device.py.

910B v2 validation only: no short-length attention or public dispatch.
"""

import ctypes as ct
import torch

import torch_npu  # noqa: F401 -- register the NPU backend before torch.npu helpers
from tests.mla.support.mla_metadata_build import ROOT, sha256
from tests.mla.support.mla_metadata_error_probe import forbid_host_readback
from tests.mla.support.components.bsnd_metadata import SENTINEL, arguments
from tests.mla.support.components.bsnd_metadata_aicpu import device_lengths, enqueue, host_record
from tests.mla.support.components.bsnd_metadata_consumer import context_for, expected_schedule
from tests.mla.support.components.bsnd_schedule import length_case
from tests.mla.support.components.metadata_consumer import Context


def load_consumer(manifest, dtype):
    assert manifest["ndebug"]
    for name, digest in manifest["sources"].items():
        assert sha256(ROOT / name) == digest, name
    row = manifest["libraries"]["fp16" if dtype == 1 else "bf16"]
    assert sha256(row["path"]) == row["sha256"]
    lib = ct.CDLL(row["path"])
    lib.launch_bsnd_consumer.argtypes = [
        ct.c_void_p,
        ct.c_void_p,
        ct.c_uint32,
        ct.POINTER(Context),
        ct.c_void_p,
        ct.c_uint64,
        *([ct.c_void_p] * 4),
        ct.c_uint32,
        ct.c_bool,
    ]
    lib.launch_bsnd_consumer.restype = ct.c_int
    return lib


def make_case(consumer, producer, lengths, *, shape=None, aic=None, worker=0, block_dim=0):
    shape = shape or (len(lengths), 3, 171, 16384)
    p = producer["platform"]
    values = length_case(
        lengths,
        batch=shape[0],
        sq=shape[1],
        heads=shape[2],
        sk=shape[3],
        aic=aic or p.aic,
        block_dim=block_dim,
        ratio=2,
        libapi=p.libapi,
        **{name: getattr(p.memory, name) for name in ("l1", "ub", "l0a", "l0b", "l0c")},
    )
    row = host_record(producer, values, consumer["dtype"])
    assert row[5:8] == [0, 0, 0]
    c = context_for(row, values, consumer["dtype"])
    storage = torch.full((len(row) + 2,), SENTINEL, dtype=torch.int32, device="npu")
    current = device_lengths(values)
    args = arguments(values, consumer["dtype"])
    case = dict(
        context=c,
        args=args,
        metadata_storage=storage,
        metadata=storage[1:-1],
        lengths=current,
        expected=row,
    )
    # Each core owns complete cache lines; independently flushed scalar writers
    # must never overwrite another core's diagnostic or output values.
    case["buffers"] = [
        torch.full((3 * c.block_dim * stride + 32,), SENTINEL, dtype=torch.int32, device="npu")
        for stride in (16, 96, 16, 16)
    ]
    with forbid_host_readback():
        enqueue(producer, args, storage, current, worker)
    return case


def launch(
    consumer,
    case,
    *,
    checked=True,
    address=None,
    words=None,
    length_address=None,
    count=None,
    context=None,
    blocks=None,
):
    c = context if context is not None else case["context"]
    lengths = case["lengths"]
    stream = torch.npu.current_stream()
    tensors = [case["metadata"], *case["buffers"]]
    if lengths is not None:
        tensors.append(lengths)
    for tensor in tensors:
        tensor.record_stream(stream)
    assert (
        consumer["lib"].launch_bsnd_consumer(
            stream.npu_stream,
            case["metadata"].data_ptr() if address is None else address,
            case["metadata"].numel() if words is None else words,
            ct.byref(c),
            (lengths.data_ptr() if lengths is not None else 0)
            if length_address is None
            else length_address,
            (lengths.numel() if lengths is not None else 0) if count is None else count,
            *(tensor.data_ptr() + 64 for tensor in case["buffers"]),
            c.block_dim if blocks is None else blocks,
            checked,
        )
        == 0
    )


def check(case, status=0, *, record=None):
    blocks = case["context"].block_dim
    for i, buffer in enumerate(case["buffers"]):
        actual = buffer.cpu()
        expected = torch.full_like(actual, SENTINEL)
        if i == 0:
            expected[16:-16:16] = status
        elif status == 0:
            if i == 1:
                decoded = torch.tensor(expected_schedule(case["expected"]), dtype=torch.int32)
                for core in range(3 * blocks):
                    expected[16 + core * 96 : 16 + core * 96 + 89] = decoded
            else:
                expected[16:-16:16] = 1
        assert torch.equal(actual, expected), (i, status, actual.tolist())
    actual_record = [word & 0xFFFFFFFF for word in case["metadata_storage"].cpu().tolist()]
    assert actual_record == [SENTINEL, *(case["expected"] if record is None else record), SENTINEL]
