"""910B v5 independent generation; no used attention consumer or public entry."""

import ctypes as ct
import json
import os
from pathlib import Path
import statistics
import time

import pytest
import torch
import torch_npu  # noqa: F401

from tests.mla.support.mla_metadata_build import build_probe
from tests.mla.support.mla_metadata_error_probe import forbid_host_readback
from tests.mla.unit.dense.test_flash_attn_npu_v3_mla_metadata import Memory, Platform, load_probe
from tests.mla.unit.mixed.test_flash_attn_npu_v3_mla_tnd_bsnd_metadata_aicpu import (
    storage_for,
    check_storage,
)
from tests.mla.unit.used.test_flash_attn_npu_v3_mla_used_metadata import (
    used_artifacts as used_artifacts,
    used_oracle as used_oracle,
    PREFIX,
    SENTINEL,
    FAILURES,
    EMPTY_CASES,
    allocation_words,
    check_records,
    encode,
    input_arrays,
    metadata_cases,
    used_case,
    wide_case,
    wire_spans,
)


class Allocation(ct.Structure):
    _fields_ = [("layout", ct.c_int32), ("total", ct.c_int64), ("maximum", ct.c_int64)]


class UsedShape(ct.Structure):
    _fields_ = [
        ("batch", ct.c_int64),
        ("heads", ct.c_int64),
        ("query", Allocation),
        ("kv", Allocation),
    ]


class Array(ct.Structure):
    _fields_ = [("address", ct.c_uint64), ("count", ct.c_uint64)]


ARRAY_NAMES = ("cu_q", "cu_k", "used_q", "used_k")


class UsedArgs(ct.Structure):
    _fields_ = [
        ("output", ct.c_uint64),
        ("words", ct.c_uint32),
        ("dtype", ct.c_uint32),
        ("shape", UsedShape),
        ("platform", Platform),
    ] + [(name, Array) for name in ARRAY_NAMES]


def arguments(item, dtype):
    values = encode(item)
    return UsedArgs(
        0,
        PREFIX + values[0] * 8,
        dtype,
        UsedShape(
            values[0],
            values[2],
            Allocation(int(item["layout"] != 0), item.get("total_q", sum(item["q"])), values[1]),
            Allocation(int(item["layout"] == 1), item.get("total_k", sum(item["kv"])), values[3]),
        ),
        Platform(
            values[4],
            values[5],
            Memory(*values[6:11]),
            values[11],
            (ct.c_uint32 * 26)(*[values[12] or c for c in range(1, 27)]),
        ),
        *[Array(0, count) for _, _, count in input_arrays(item)],
    )


@pytest.fixture(scope="module")
def used_probe(tmp_path_factory):
    directory = Path(
        os.environ.get("MLA_USED_METADATA_DEVICE_DIR", tmp_path_factory.mktemp("mla-used-aicpu"))
    )
    with load_probe(build_probe(directory, used=True), directory) as probe:
        lib = probe["lib"]
        lib.used_metadata_args_size.restype = ct.c_uint32
        assert lib.used_metadata_args_size() == ct.sizeof(UsedArgs)
        lib.launch_used_metadata.argtypes = [ct.c_void_p] * 4 + [ct.POINTER(UsedArgs)]
        lib.launch_used_metadata.restype = ct.c_int
        lib.used_metadata_host.argtypes = [ct.POINTER(UsedArgs), ct.c_void_p]
        lib.used_metadata_host.restype = ct.c_uint32
        yield probe


def host_record(probe, item, dtype):
    args = arguments(item, dtype)
    keep = []
    for name, (values, present, _) in zip(ARRAY_NAMES, input_arrays(item)):
        data = (ct.c_int32 * (len(values) + 2))(SENTINEL, *values, SENTINEL)
        getattr(args, name).address = ct.addressof(data) + 4 if present else 0
        keep.append((data, [SENTINEL, *values, SENTINEL]))
    output = (ct.c_uint32 * args.words)()
    assert probe["lib"].used_metadata_host(ct.byref(args), output) == 0
    assert all(list(data) == before for data, before in keep)
    return list(output)


def device_inputs(item):
    return [
        torch.tensor([SENTINEL, *values, SENTINEL], dtype=torch.int32, device="npu")[1:-1]
        if present
        else None
        for values, present, _ in input_arrays(item)
    ]


def enqueue(probe, args, storage, inputs, worker=0, *, descriptors=None):
    output = storage[1:-1]
    assert output.is_contiguous() and output.numel() == args.words
    args.output = output.data_ptr()
    for name, tensor in zip(ARRAY_NAMES, inputs):
        # An explicit B=0 view still has a presence flag. Tensor.data_ptr()
        # may be zero for empty views; the guarded backing allocation is live.
        getattr(args, name).address = (
            0
            if tensor is None
            else (
                tensor.untyped_storage().data_ptr()
                + tensor.storage_offset() * tensor.element_size()
            )
        )
    for name, (address, count) in (descriptors or {}).items():
        setattr(args, name, Array(address, count))
    stream = probe["workers"][worker]
    ready, done = probe["events"][worker * 2 : worker * 2 + 2]
    assert (
        probe["lib"].launch_used_metadata(
            torch.npu.current_stream().npu_stream, stream.npu_stream, ready, done, ct.byref(args)
        )
        == 0
    )
    for tensor in (output, *inputs):
        if tensor is not None:
            tensor.record_stream(stream)


@pytest.mark.parametrize("dtype", (1, 2))
@pytest.mark.parametrize("layout", range(3))
@pytest.mark.parametrize("aic,ratio", ((None, 2), (1, 1), (15, 2), (26, 1)))
def test_aicpu_source_records(used_probe, used_oracle, dtype, layout, aic, ratio):
    items = metadata_cases(layout, aic or used_probe["platform"].aic, ratio)
    if aic is None:
        platform = used_probe["platform"]
        assert list(platform.block_dims)[: platform.aic] == list(range(1, platform.aic + 1))
        for item in items:
            item.update(
                {name: getattr(platform.memory, name) for name in ("l1", "ub", "l0a", "l0b", "l0c")}
            )
            item["libapi"] = platform.libapi
    pending, expected = [], []
    for i, item in enumerate(items):
        args = arguments(item, dtype)
        inputs, storage = device_inputs(item), storage_for(args)
        with forbid_host_readback():
            enqueue(used_probe, args, storage, inputs, i % 2)
        pending.append(storage)
        expected.append(host_record(used_probe, item, dtype))
    rows = [check_storage(s, e) for s, e in zip(pending, expected)]
    check_records(
        rows,
        items,
        used_oracle,
        used_probe["directory"],
        f"device-{dtype}-{layout}-{aic}-{ratio}",
        dtype,
    )


@pytest.mark.parametrize("dtype", (1, 2))
def test_failures_empty_and_reuse(used_probe, dtype):
    reports = []
    for item in [entry[0] for entry in FAILURES] + EMPTY_CASES:
        args = arguments(item, dtype)
        inputs, storage = device_inputs(item), storage_for(args)
        with forbid_host_readback():
            enqueue(used_probe, args, storage, inputs)
        row = check_storage(storage, host_record(used_probe, item, dtype))
        if row[5]:
            assert not any(row[8:])
        reports.append(dict(case=item, record=row))
        good = used_case(item["layout"], [7] * args.shape.batch, [33] * args.shape.batch)
        args, inputs = arguments(good, dtype), device_inputs(good)
        with forbid_host_readback():
            enqueue(used_probe, args, storage, inputs)
        check_storage(storage, host_record(used_probe, good, dtype))
    (used_probe["directory"] / f"failures-{dtype}.json").write_text(json.dumps(reports) + "\n")


@pytest.mark.parametrize("dtype", (1, 2))
def test_four_input_descriptors(used_probe, dtype):
    item = used_case(1, uq=[3, 1], uk=[17, 0])
    args, inputs = arguments(item, dtype), device_inputs(item)
    storage = storage_for(args)
    expected = host_record(used_probe, item, dtype)
    failure = expected[:5] + [3, 0, 0] + [0] * (args.words - 8)
    for name, tensor in zip(ARRAY_NAMES, inputs):
        count = getattr(args, name).count
        for address, size in (
            (0, count),
            (tensor.data_ptr() + 1, count),
            (tensor.data_ptr() + 2, count),
            (tensor.data_ptr(), count - 1),
            (2**64 - 4, count),
            (2**64 - 4, 2**64 - 1),
        ):
            with forbid_host_readback():
                enqueue(used_probe, args, storage, inputs, descriptors={name: (address, size)})
            check_storage(storage, failure)
            args = arguments(item, dtype)
            with forbid_host_readback():
                enqueue(used_probe, args, storage, inputs)
            check_storage(storage, expected)
    for tensor, (values, _, _) in zip(inputs, input_arrays(item)):
        assert tensor._base.cpu().tolist() == [SENTINEL, *values, SENTINEL]


@pytest.mark.parametrize("dtype", (1, 2))
def test_each_array_update_and_owned_snapshot(used_probe, dtype):
    b = used_probe["platform"].aic + 1
    item = used_case(1, [7] * b, [33] * b, [3] * b, [17] * b, sq=17, sk=64)
    args, inputs = arguments(item, dtype), device_inputs(item)
    storage = storage_for(args)
    for i in range(b):
        if i > 0:
            item["q"][i - 1] -= 1
            item["q"][i] += 1
            item["kv"][i - 1] -= 1
            item["kv"][i] += 1
            inputs[0][i] -= 1
            inputs[1][i] -= 1
        item["uq"][i] = (0, 1, 3)[i % 3]
        item["uk"][i] = (0, 1, 17)[i % 3]
        inputs[2][i], inputs[3][i] = item["uq"][i], item["uk"][i]
        with forbid_host_readback():
            enqueue(used_probe, args, storage, inputs)
        row = check_storage(storage, host_record(used_probe, item, dtype))
        assert row[PREFIX:] == wire_spans(item)
    frozen = storage.clone()
    for tensor in inputs:
        tensor.zero_()
    del inputs
    assert torch.equal(frozen.cpu(), storage.cpu())


def test_alias_cross_stream_allocator_lifetime(used_probe):
    pending, consumer = [], torch.npu.Stream()
    for worker, stream in enumerate((torch.npu.Stream(), torch.npu.Stream())):
        for i in range(24):
            with torch.npu.stream(stream):
                item = used_case(1, [3, 3], [3, 3], [0, 3], [0, 3])
                args, inputs = arguments(item, worker + 1), device_inputs(item)
                if i % 2 == 0:
                    inputs = [inputs[0], inputs[0], inputs[0][:-1], inputs[0][:-1]]
                storage = storage_for(args)
                with forbid_host_readback():
                    enqueue(used_probe, args, storage, inputs, worker)
                    ready = torch.npu.Event()
                    ready.record(stream)
            with torch.npu.stream(consumer), forbid_host_readback():
                consumer.wait_event(ready)
                storage.record_stream(consumer)
                copied = storage.clone()
            pending.append((copied, host_record(used_probe, item, worker + 1)))
            del args, inputs, storage
            with torch.npu.stream(stream):
                torch.empty((32768,), dtype=torch.int32, device="npu").fill_(SENTINEL)
    for storage, expected in pending:
        check_storage(storage, expected)


@pytest.mark.parametrize("layout", (0, 2))
def test_wide_capacity_without_attention_allocation(used_probe, layout):
    item = wide_case(layout)
    args, inputs = arguments(item, 1), device_inputs(item)
    storage = storage_for(args)
    with forbid_host_readback():
        enqueue(used_probe, args, storage, inputs)
    row = check_storage(storage, host_record(used_probe, item, 1))
    assert row[5:8] == [0, 0, 0]
    assert row[401:PREFIX] == allocation_words(item) and row[PREFIX:] == wire_spans(item)
    assert row[-5] == 1 and (layout != 0 or row[-7] > 0)
    (used_probe["directory"] / f"wide-{layout}.json").write_text(
        json.dumps(
            dict(
                layout=layout,
                batch=args.shape.batch,
                totals=[sum(item["q"]), sum(item["kv"])],
                last_span=row[-8:],
                words=len(row),
                attention_allocated=False,
                cpu_device_equal=True,
            )
        )
        + "\n"
    )


@pytest.mark.parametrize("dtype", (1, 2))
def test_generation_timing(used_probe, dtype):
    items = [
        used_case(uq=[3, 1], uk=[17, 0]),
        used_case(1, [3, 0, 1], [17, 0, 33], [2, 0, 1], [7, 0, 17]),
        used_case(2, [3, 0, 1], [16384] * 3, [2, 0, 1], [17, 0, 8193]),
        used_case(1, [2] * 10, [511] * 10),
        used_case(1, [2] * 10, [511] * 10, uk=[511] * 10),
        used_case(0, [257] * 41, [16384] * 41, [0] * 41, [0] * 41),
        used_case(2, [3, 0, 1], [0] * 3),
    ]
    reports = []
    for item in items:
        args, inputs = arguments(item, dtype), device_inputs(item)
        storage, samples = storage_for(args), []
        for i in range(25):
            torch.npu.synchronize()
            start = time.perf_counter_ns()
            enqueue(used_probe, args, storage, inputs)
            torch.npu.synchronize()
            if i >= 5:
                samples.append((time.perf_counter_ns() - start) / 1000)
        check_storage(storage, host_record(used_probe, item, dtype))
        reports.append(dict(case=item, wall_us=samples, median_us=statistics.median(samples)))
    (used_probe["directory"] / f"timing-{dtype}.json").write_text(
        json.dumps(
            dict(
                dtype=dtype,
                records=reports,
                cpu_affinity=sorted(os.sched_getaffinity(0)),
                scope="Independent AICPU generation including ctypes/events/per-call sync; no attention or public E2E",
            ),
            indent=2,
        )
        + "\n"
    )
