"""Shared helpers extracted from tests/mla/unit/paged_bsnd/test_flash_attn_npu_v3_mla_paged_metadata_device.py.

910B release v6 PA validation only: exact schedules, rejection, Trap and lifetime.
"""

import ctypes as ct
import torch

import torch_npu  # noqa: F401 -- register the NPU backend before torch.npu helpers
from tests.mla.support.mla_metadata_build import ROOT, sha256
from tests.mla.support.mla_metadata_error_probe import forbid_host_readback
from tests.mla.support.components.paged_metadata_aicpu import (
    arguments,
    device_inputs,
    enqueue,
    host_record,
    storage_for,
)
from tests.mla.support.components.paged_metadata_consumer import (
    Context,
    Inputs,
    Array,
    Cache,
    ARRAY_NAMES,
    SENTINEL,
    context_for,
    expected_schedule,
    cache_spec,
    change,
    paged_case,
)


def load_consumer(manifest, dtype):
    assert manifest["ndebug"]
    for name, digest in manifest["sources"].items():
        assert sha256(ROOT / name) == digest, name
    row = manifest["libraries"]["fp16" if dtype == 1 else "bf16"]
    assert sha256(row["path"]) == row["sha256"]
    lib = ct.CDLL(row["path"])
    lib.launch_paged_consumer.argtypes = [
        ct.c_void_p,
        ct.c_void_p,
        ct.c_uint32,
        ct.POINTER(Context),
        ct.POINTER(Inputs),
        ct.POINTER(Cache),
        *([ct.c_void_p] * 4),
        ct.c_uint32,
        ct.c_bool,
    ]
    lib.launch_paged_consumer.restype = ct.c_int
    return lib


def make_case(consumer, producer, item=None, *, worker=0):
    item = dict(item or paged_case(uq=[3, 1]))
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
    case.update(make_cache(item, consumer["dtype"]))
    case["buffers"] = [
        torch.full((3 * c.block_dim * stride + 32,), SENTINEL, dtype=torch.int32, device="npu")
        for stride in (16, 112, 16, 16)
    ]
    with forbid_host_readback():
        enqueue(producer, args, storage, inputs, worker)
    return case


def make_cache(item, dtype, **spec):
    layout, values = cache_spec(item, **spec)
    table_storage = torch.tensor([SENTINEL, *values, SENTINEL], dtype=torch.int32, device="npu")
    table = table_storage[1:-1]
    tensors = [table]
    for x, dim, offset in ((layout.k, 64, 7), (layout.v, 512, 13)):
        data = torch.empty(
            x.span + offset + 1, dtype=torch.float16 if dtype == 1 else torch.bfloat16, device="npu"
        )
        tensors.append(
            data.as_strided(
                (layout.pool, layout.page, 1, dim),
                (x.batch_stride, x.row_stride, x.head_stride, 1),
                offset,
            )
        )
    cache = Cache(
        layout,
        *(
            Array(
                t.untyped_storage().data_ptr() + t.storage_offset() * t.element_size(),
                (t.untyped_storage().nbytes() // t.element_size()) - t.storage_offset(),
            )
            for t in tensors
        ),
    )
    return dict(
        cache=cache, cache_tensors=tensors, table_storage=table_storage, table_values=values
    )


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
    cache_changes=None,
):
    c = context if context is not None else case["context"]
    stream = torch.npu.current_stream()
    inputs = Inputs()
    for name, tensor in zip(ARRAY_NAMES, case["inputs"]):
        setattr(inputs, name, Array(input_address(tensor), 0 if tensor is None else tensor.numel()))
    change(inputs, descriptors)
    cache = Cache.from_buffer_copy(case["cache"])
    change(cache, cache_changes)
    for tensor in (case["metadata"], *case["inputs"], *case["buffers"], *case["cache_tensors"]):
        if tensor is not None:
            tensor.record_stream(stream)
    assert (
        consumer["lib"].launch_paged_consumer(
            stream.npu_stream,
            case["metadata"].data_ptr() if address is None else address,
            case["metadata"].numel() if words is None else words,
            ct.byref(c),
            ct.byref(inputs),
            ct.byref(cache),
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
                    expected[16 + core * 112 : 16 + core * 112 + 96] = decoded
            else:
                expected[16:-16:16] = 1
        assert torch.equal(actual, expected), (i, status)
    actual = [w & 0xFFFFFFFF for w in case["metadata_storage"].cpu().tolist()]
    assert actual == [SENTINEL, *(case["expected"] if record is None else record), SENTINEL]


def fault_case(consumer, producer, fault):
    q, uq, kv = [7, 0, 3], [3, 0, 1], [257, 1, 0]
    if fault == "qzero_page":
        q, uq = [0] * 3, [0] * 3
    if fault == "usedzero_page":
        uq = [0] * 3
    if fault == "empty_table_stale":
        kv = [0] * 3
    if fault == "empty_batch":
        q, uq, kv = [], [], []
    item = paged_case(2, q, kv, uq, 112, 0 if fault == "empty_table_stale" else 448)
    case = make_case(consumer, producer, item)
    options, status = {}, 12
    changes = {
        "page": (406, 16, 5),
        "capacity": (407, 112, 5),
        "source_s2": (12, 448, 5),
        "span": (410, 1, 12),
        "wide_start": (409, 1, 12),
        "workspace": (386, -1, 9),
        "local_memory": (391, 0, 10),
    }
    if fault in changes or fault.startswith("v"):
        word, value, status = changes[fault] if fault in changes else (1, int(fault[1:]), 2)
        case["metadata"][word] = value
        case["expected"][word] = value & 0xFFFFFFFF
    if fault in ("stale_q", "stale_uq", "stale_kv", "empty_batch", "empty_table_stale"):
        axis = {
            "stale_q": 0,
            "stale_uq": 1,
            "stale_kv": 2,
            "empty_batch": 0,
            "empty_table_stale": 2,
        }[fault]
        case["inputs"][axis][0] = -1
        status = (12, 13, 11)[axis]
    if fault == "words":
        options, status = dict(words=401), 1
    if fault == "missing_q":
        options, status = dict(descriptors={"cu_q.address": 0}), 12
    if fault == "align_used":
        options, status = dict(descriptors={"used_q.address": 1}), 13
    if fault == "table_descriptor":
        options, status = dict(cache_changes={"table.count": 0}), 14
    if fault == "cache_span":
        options, status = dict(cache_changes={"layout.k.span": 0}), 1
    if fault in ("page_negative", "page_oob", "qzero_page", "usedzero_page"):
        case["cache_tensors"][0][0] = 7 if fault == "page_oob" else -1
        status = 15
    if fault == "empty_pool":
        case.update(make_cache(item, consumer["dtype"], pool=0))
        status = 15
    if fault == "suffix":
        case["metadata"][20] = 1
        case["expected"][20] = 1
        case["metadata"][23:101] = 0
        case["expected"][23:101] = [0] * 78
        case["metadata"][23] = 0
        case["metadata"][49] = 1
        case["expected"][23] = 0
        case["expected"][49] = 1
        status = 8
    if fault == "generation":
        case["inputs"][1][0] = -1
        enqueue(producer, case["args"], case["metadata_storage"], case["inputs"])
        case["expected"][5:7] = [2, 1]
        case["expected"][8:] = [0] * (len(case["expected"]) - 8)
        status = 3
    return case, options, status
