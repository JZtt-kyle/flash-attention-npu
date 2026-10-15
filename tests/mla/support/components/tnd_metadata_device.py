"""Shared helpers extracted from tests/mla/unit/tnd/test_flash_attn_npu_v3_mla_tnd_metadata_device.py.

910B release validation of compact TND records, independently of attention.
"""

import ctypes as ct
import torch

import torch_npu  # noqa: F401 -- register the NPU backend before torch.npu helpers
from tests.mla.support.mla_metadata_build import ROOT, sha256
from tests.mla.support.mla_metadata_error_probe import forbid_host_readback
from tests.mla.support.components.tnd_metadata_aicpu import (
    arguments,
    device_inputs,
    enqueue,
    host_record,
    storage_for,
)
from tests.mla.support.components.tnd_metadata_consumer import (
    Context,
    SENTINEL,
    context_for,
    expected_schedule,
    tnd_case,
)


def load_consumer(manifest, dtype):
    assert manifest["ndebug"]
    for name, digest in manifest["sources"].items():
        assert sha256(ROOT / name) == digest, name
    row = manifest["libraries"]["fp16" if dtype == 1 else "bf16"]
    assert sha256(row["path"]) == row["sha256"]
    lib = ct.CDLL(row["path"])
    lib.launch_tnd_consumer.argtypes = [
        ct.c_void_p,
        ct.c_void_p,
        ct.c_uint32,
        ct.POINTER(Context),
        ct.c_void_p,
        ct.c_uint64,
        ct.c_void_p,
        ct.c_uint64,
        *([ct.c_void_p] * 4),
        ct.c_uint32,
        ct.c_bool,
    ]
    lib.launch_tnd_consumer.restype = ct.c_int
    return lib


def make_case(consumer, producer, q=(3, 0, 1), k=(17, 0, 33), *, values=None, worker=0, **shape):
    if values is None:
        p = producer["platform"]
        values = tnd_case(
            q,
            k,
            **dict(
                aic=p.aic,
                ratio=2,
                libapi=p.libapi,
                **{name: getattr(p.memory, name) for name in ("l1", "ub", "l0a", "l0b", "l0c")},
                **shape,
            ),
        )
    row = host_record(producer, values, consumer["dtype"])
    assert row[5:8] == [0, 0, 0]
    c = context_for(row, values, consumer["dtype"])
    args = arguments(values, consumer["dtype"])
    storage = storage_for(args)
    query, kv = device_inputs(values)
    case = dict(
        context=c,
        args=args,
        values=values,
        metadata_storage=storage,
        metadata=storage[1:-1],
        query=query,
        kv=kv,
        expected=row,
    )
    # Whole cache lines per core avoid scalar cache flushes overwriting peers.
    case["buffers"] = [
        torch.full((3 * c.block_dim * stride + 32,), SENTINEL, dtype=torch.int32, device="npu")
        for stride in (16, 96, 16, 16)
    ]
    with forbid_host_readback():
        enqueue(producer, args, storage, query, kv, worker)
    return case


def launch(
    consumer,
    case,
    *,
    checked=True,
    address=None,
    words=None,
    qaddress=None,
    kaddress=None,
    qcount=None,
    kcount=None,
    context=None,
    blocks=None,
):
    c = context if context is not None else case["context"]
    stream = torch.npu.current_stream()
    for tensor in (case["metadata"], case["query"], case["kv"], *case["buffers"]):
        tensor.record_stream(stream)
    assert (
        consumer["lib"].launch_tnd_consumer(
            stream.npu_stream,
            case["metadata"].data_ptr() if address is None else address,
            case["metadata"].numel() if words is None else words,
            ct.byref(c),
            case["query"].data_ptr() if qaddress is None else qaddress,
            case["query"].numel() if qcount is None else qcount,
            case["kv"].data_ptr() if kaddress is None else kaddress,
            case["kv"].numel() if kcount is None else kcount,
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
                    expected[16 + core * 96 : 16 + core * 96 + 91] = decoded
            else:
                expected[16:-16:16] = 1
        assert torch.equal(actual, expected), (i, status, actual.tolist())
    actual_record = [w & 0xFFFFFFFF for w in case["metadata_storage"].cpu().tolist()]
    assert actual_record == [SENTINEL, *(case["expected"] if record is None else record), SENTINEL]
