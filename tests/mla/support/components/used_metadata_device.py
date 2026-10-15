"""Shared helpers extracted from tests/mla/unit/used/test_flash_attn_npu_v3_mla_used_metadata_device.py.

910B release v5 validation only: exact schedules, rejection, Trap and lifetime.
"""

import ctypes as ct
import torch

import torch_npu  # noqa: F401 -- register the NPU backend before torch.npu helpers
from tests.mla.support.mla_metadata_build import ROOT, sha256
from tests.mla.support.mla_metadata_error_probe import forbid_host_readback
from tests.mla.support.components.used_metadata_aicpu import (
    arguments,
    device_inputs,
    enqueue,
    host_record,
    storage_for,
)
from tests.mla.support.components.used_metadata_consumer import (
    Context,
    Inputs,
    Array,
    ARRAY_NAMES,
    SENTINEL,
    context_for,
    expected_schedule,
    used_case,
)


def load_consumer(manifest, dtype):
    assert manifest["ndebug"]
    for name, digest in manifest["sources"].items():
        assert sha256(ROOT / name) == digest, name
    row = manifest["libraries"]["fp16" if dtype == 1 else "bf16"]
    assert sha256(row["path"]) == row["sha256"]
    lib = ct.CDLL(row["path"])
    lib.launch_used_consumer.argtypes = [
        ct.c_void_p,
        ct.c_void_p,
        ct.c_uint32,
        ct.POINTER(Context),
        ct.POINTER(Inputs),
        *([ct.c_void_p] * 4),
        ct.c_uint32,
        ct.c_bool,
    ]
    lib.launch_used_consumer.restype = ct.c_int
    return lib


def make_case(consumer, producer, item=None, *, worker=0):
    item = dict(item or used_case(uq=[3, 1], uk=[17, 0]))
    p = producer["platform"]
    item.update(
        aic=p.aic,
        ratio=2,
        libapi=p.libapi,
        **{name: getattr(p.memory, name) for name in ("l1", "ub", "l0a", "l0b", "l0c")},
    )
    row = host_record(producer, item, consumer["dtype"])
    assert row[5:8] == [0, 0, 0]
    c = context_for(row, item, consumer["dtype"])
    args = arguments(item, consumer["dtype"])
    storage, inputs = storage_for(args), device_inputs(item)
    case = dict(
        context=c,
        args=args,
        item=item,
        metadata_storage=storage,
        metadata=storage[1:-1],
        inputs=inputs,
        expected=row,
    )
    case["buffers"] = [
        torch.full((3 * c.block_dim * stride + 32,), SENTINEL, dtype=torch.int32, device="npu")
        for stride in (16, 112, 16, 16)
    ]
    with forbid_host_readback():
        enqueue(producer, args, storage, inputs, worker)
    return case


def input_address(tensor):
    return (
        0 if tensor is None else tensor.untyped_storage().data_ptr() + tensor.storage_offset() * 4
    )


def launch(
    consumer,
    case,
    *,
    checked=True,
    address=None,
    words=None,
    descriptors=None,
    context=None,
    blocks=None,
):
    c = context if context is not None else case["context"]
    stream = torch.npu.current_stream()
    inputs = Inputs()
    for name, tensor in zip(ARRAY_NAMES, case["inputs"]):
        setattr(inputs, name, Array(input_address(tensor), 0 if tensor is None else tensor.numel()))
    for name, changes in (descriptors or {}).items():
        for field, value in changes.items():
            setattr(getattr(inputs, name), field, value)
    for tensor in (case["metadata"], *case["inputs"], *case["buffers"]):
        if tensor is not None:
            tensor.record_stream(stream)
    assert (
        consumer["lib"].launch_used_consumer(
            stream.npu_stream,
            case["metadata"].data_ptr() if address is None else address,
            case["metadata"].numel() if words is None else words,
            ct.byref(c),
            ct.byref(inputs),
            *(t.data_ptr() + 64 for t in case["buffers"]),
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
                decoded = torch.tensor(
                    [w if w < 2**31 else w - 2**32 for w in expected_schedule(case["expected"])],
                    dtype=torch.int32,
                )
                for core in range(3 * blocks):
                    expected[16 + core * 112 : 16 + core * 112 + 98] = decoded
            else:
                expected[16:-16:16] = 1
        assert torch.equal(actual, expected), (i, status)
    actual = [w & 0xFFFFFFFF for w in case["metadata_storage"].cpu().tolist()]
    assert actual == [SENTINEL, *(case["expected"] if record is None else record), SENTINEL]


def fault_case(consumer, producer, fault):
    q, k = [7, 0, 3, 7, 0], [513, 7, 33, 17, 16384]
    uq, uk = [3, 0, 1, 0, 0], [513, 7, 1, 0, 16384]
    if fault == "empty_stale":
        q, uq = [0] * 5, [0] * 5
    if fault == "empty_batch":
        q, k, uq, uk = [], [], [], []
    case = make_case(consumer, producer, used_case(1, q, k, uq, uk, heads=171))
    mutations = dict(
        v1=(1, 1, 2),
        v2=(2, 2, 2),
        v3=(1, 3, 2),
        v4=(2, 4, 2),
        layout=(401, 0, 5),
        flags=(409, 0, 5),
        span=(410, -1, 12),
        wide_start=(411, 1, 12),
        allocation=(414, 1, 12),
        workspace=(386, -1, 9),
    )
    status, options = 12, {}
    if fault in mutations:
        word, value, status = mutations[fault]
        case["metadata"][word] = value
        case["expected"][word] = value & 0xFFFFFFFF
    if fault.startswith("stale_") or fault in ("empty_stale", "empty_batch"):
        axis = {"stale_q": 0, "stale_k": 1, "stale_uq": 2, "stale_uk": 3}.get(fault, 0)
        case["inputs"][axis][0] = -1
        status = 12 if axis < 2 else 13
    if fault in ("suffix", "variable_m"):
        case["metadata"][20] = 1
        case["expected"][20] = 1
        case["metadata"][23:101] = 0
        case["expected"][23:101] = [0] * 78
        for word, value in (
            (23, 1 if fault == "suffix" else 2),
            (49, 0 if fault == "suffix" else 1),
        ):
            case["metadata"][word] = value
            case["expected"][word] = value
        status = 8
    if fault == "generation":
        case["inputs"][2][0] = -1
        enqueue(producer, case["args"], case["metadata_storage"], case["inputs"])
        case["expected"][5:7] = [2, 1]
        case["expected"][8:] = [0] * (len(case["expected"]) - 8)
        status = 3
    if fault == "words":
        options, status = dict(words=401), 1
    if fault in ("missing_q", "count_uk", "align_uq"):
        descriptor = {
            "missing_q": ("cu_q", {"address": 0}),
            "count_uk": ("used_k", {"count": 1}),
            "align_uq": ("used_q", {"address": 1}),
        }[fault]
        options = dict(descriptors={descriptor[0]: descriptor[1]})
        status = 12 if fault == "missing_q" else 13
    return case, options, status
