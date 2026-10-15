"""910B v4 mixed-layout AICPU generation, independently of attention."""

import json
import os
import statistics
import time

import pytest
import torch
import torch_npu  # noqa: F401

from tests.mla.support.mla_metadata_error_probe import forbid_host_readback
from tests.mla.support.components.bsnd_metadata import arguments as bsnd_arguments
from tests.mla.support.components.bsnd_metadata_aicpu import enqueue as bsnd_enqueue
from tests.mla.support.components.bsnd_schedule import length_case
from tests.mla.support.components.tnd_bsnd_metadata import (
    mixed_artifacts as mixed_artifacts,
    mixed_oracle as mixed_oracle,
    FAILURES,
    PREFIX,
    SENTINEL,
    check_records,
    metadata_cases,
    EMPTY_CASES,
    input_lengths,
    mixed_case,
    wide_words,
    wire_spans,
)

from tests.mla.support.components.tnd_bsnd_metadata_aicpu import (
    MixedArgs as MixedArgs,
    MixedShape as MixedShape,
    arguments as arguments,
    check_storage as check_storage,
    device_inputs as device_inputs,
    enqueue as enqueue,
    host_record as host_record,
    mixed_probe as mixed_probe,
    storage_for as storage_for,
)


@pytest.mark.parametrize("dtype", (1, 2))
@pytest.mark.parametrize("aic,ratio", [(None, 2), (1, 1), (15, 2), (26, 1)])
def test_aicpu_source_records(mixed_probe, mixed_oracle, dtype, aic, ratio):
    cases = metadata_cases(aic or mixed_probe["platform"].aic, ratio)
    if aic is None:
        p = mixed_probe["platform"]
        assert list(p.block_dims)[: p.aic] == list(range(1, p.aic + 1))
        for values in cases:
            values[6:11] = [getattr(p.memory, key) for key in ("l1", "ub", "l0a", "l0b", "l0c")]
            values[11] = p.libapi
    expected = [host_record(mixed_probe, c, dtype) for c in cases]
    pending = []
    for i, values in enumerate(cases):
        args = arguments(values, dtype)
        q, k = device_inputs(values)
        storage = storage_for(args)
        with forbid_host_readback():
            enqueue(mixed_probe, args, storage, q, k, i % 2)
        pending.append(storage)
    rows = [check_storage(s, e) for s, e in zip(pending, expected)]
    check_records(
        rows, cases, mixed_oracle, mixed_probe["directory"], f"device-{dtype}-{aic}-{ratio}", dtype
    )


@pytest.mark.parametrize("dtype", (1, 2))
def test_failures_empty_and_buffer_reuse(mixed_probe, dtype):
    cases = [mixed_case(**kwargs) for kwargs, _, _ in FAILURES]
    cases += [
        mixed_case(q, k, sq=17, capacity=capacity, has_k=bool(k)) for q, k, capacity in EMPTY_CASES
    ]
    reports = []
    for values in cases:
        args = arguments(values, dtype)
        q, k = device_inputs(values)
        storage = storage_for(args)
        with forbid_host_readback():
            enqueue(mixed_probe, args, storage, q, k)
        row = check_storage(storage, host_record(mixed_probe, values, dtype))
        reports.append(dict(case=values, record=row))
        if row[5]:
            assert not any(row[8:])
        # Rewrite the failed buffer with valid prefixes of the same batch.
        good = mixed_case([3] * values[0], [513] * values[0], capacity=1024)
        args = arguments(good, dtype)
        q, k = device_inputs(good)
        with forbid_host_readback():
            enqueue(mixed_probe, args, storage, q, k)
        check_storage(storage, host_record(mixed_probe, good, dtype))
    (mixed_probe["directory"] / f"failures-empty-{dtype}.json").write_text(
        json.dumps(reports) + "\n"
    )


@pytest.mark.parametrize("dtype", (1, 2))
def test_input_buffer_descriptors(mixed_probe, dtype):
    values = mixed_case()
    args = arguments(values, dtype)
    q, k = device_inputs(values)
    storage = storage_for(args)
    expected = host_record(mixed_probe, values, dtype)
    failure = expected[:5] + [3, 0, 0] + [0] * (args.words - 8)
    for address_name, count_name, tensor, count in (
        ("cu_q", "qcount", q, 4),
        ("kv_lengths", "kcount", k, 3),
    ):
        for address, size in (
            (0, count),
            (tensor.data_ptr() + 1, count),
            (tensor.data_ptr() + 2, count),
            (tensor.data_ptr(), count - 1),
            (2**64 - 4, count),
            (2**64 - 4, 2**64 - 1),
        ):
            with forbid_host_readback():
                enqueue(
                    mixed_probe,
                    args,
                    storage,
                    q,
                    k,
                    descriptors={address_name: address, count_name: size},
                )
            check_storage(storage, failure)
            args = arguments(values, dtype)
            with forbid_host_readback():
                enqueue(mixed_probe, args, storage, q, k)
            check_storage(storage, expected)
    assert q._base.cpu().tolist() == [SENTINEL, 0, 3, 3, 4, SENTINEL]
    assert k._base.cpu().tolist() == [SENTINEL, 17, 0, 33, SENTINEL]


@pytest.mark.parametrize("dtype", (1, 2))
def test_each_input_update_and_snapshot_ownership(mixed_probe, dtype):
    b = 2 * mixed_probe["platform"].aic + 1
    values = mixed_case([3] * b, [513] * b, sq=512, capacity=16384)
    cq, _, kv = input_lengths(values)
    q, k = device_inputs(values)
    args = arguments(values, dtype)
    storage = storage_for(args)
    for i in range(b):
        if i > 0:
            cq[i] -= i % 2 + 1
            q[i] = cq[i]
        kv[i] = (0, 1, 16384)[i % 3]
        k[i] = kv[i]
        values = mixed_case([3] * b, kv, cq=cq, sq=512, capacity=16384)
        with forbid_host_readback():
            enqueue(mixed_probe, args, storage, q, k)
        row = check_storage(storage, host_record(mixed_probe, values, dtype))
        assert row[PREFIX + 1 :: 5] == [i * 16384 for i in range(b)]
    assert q._base.cpu().tolist() == [SENTINEL, *cq, SENTINEL]
    assert k._base.cpu().tolist() == [SENTINEL, *kv, SENTINEL]
    frozen = storage.clone()
    q.zero_()
    k.zero_()
    del q, k
    assert torch.equal(frozen.cpu(), storage.cpu())


def test_alias_cross_stream_and_allocator_lifetime(mixed_probe):
    pending = []
    consumer = torch.npu.Stream()
    for worker, stream in enumerate([torch.npu.Stream(), torch.npu.Stream()]):
        for i in range(24):
            with torch.npu.stream(stream):
                klens = [17, 0, 33] if i % 2 else [0, 3, 3]
                values = mixed_case([3, 0, 1], klens)
                args = arguments(values, worker + 1)
                q, k = device_inputs(values)
                if i % 2 == 0:
                    k = q[:-1]  # Required lengths differ; overlapping read-only views are legal.
                storage = storage_for(args)
                with forbid_host_readback():
                    enqueue(mixed_probe, args, storage, q, k, worker)
                    ready = torch.npu.Event()
                    ready.record(stream)
            with torch.npu.stream(consumer), forbid_host_readback():
                consumer.wait_event(ready)
                storage.record_stream(consumer)
                copied = storage.clone()
            pending.append((copied, host_record(mixed_probe, values, worker + 1)))
            del args, q, k, storage
            with torch.npu.stream(stream):
                torch.empty((32768,), dtype=torch.int32, device="npu").fill_(SENTINEL)
    for storage, expected in pending:
        check_storage(storage, expected)


@pytest.mark.parametrize("dtype", (1, 2))
def test_64bit_physical_capacity_without_cache_allocation(mixed_probe, dtype):
    b = 2**18 + 2
    values = mixed_case([0] * b, [0] * b, capacity=16384)
    args = arguments(values, dtype)
    q, k = device_inputs(values)
    storage = storage_for(args)
    with forbid_host_readback():
        enqueue(mixed_probe, args, storage, q, k)
    row = check_storage(storage, host_record(mixed_probe, values, dtype))
    assert row[5:8] == [0, 0, 0]
    assert row[401:PREFIX] == [0, 0, 16384, *wide_words(b * 16384)]
    assert row[PREFIX:] == wire_spans(values) and row[-4:-2] == [16384, 1]
    (mixed_probe["directory"] / f"wide-{dtype}.json").write_text(
        json.dumps(
            dict(
                batch=b,
                total_kv_capacity=b * 16384,
                last_kv_start=(b - 1) * 16384,
                record_words=len(row),
                physical_cache_allocated=False,
                cpu_device_equal=True,
            )
        )
        + "\n"
    )


@pytest.mark.parametrize("dtype", (1, 2))
def test_generation_diagnostics(mixed_probe, dtype):
    reports = []
    for qlens, klens, heads, capacity in [
        ([1, 1], [257, 257], 8, 512),
        ([3, 0, 1], [17, 0, 33], 128, 64),
        ([171] * 21, [8193] * 21, 3, 16384),
        ([3, 0, 1] * 14, [16384, 0, 513] * 14, 171, 16384),
        ([3, 0, 1], [0, 0, 0], 128, 513),
        ([3, 0, 1], [0, 0, 0], 128, 0),
    ]:
        values = mixed_case(qlens, klens, heads=heads, capacity=capacity)
        args = arguments(values, dtype)
        q, k = device_inputs(values)
        storage = storage_for(args)
        samples = []
        for iteration in range(25):
            torch.npu.synchronize()
            start = time.perf_counter_ns()
            enqueue(mixed_probe, args, storage, q, k)
            torch.npu.synchronize()
            if iteration >= 5:
                samples.append((time.perf_counter_ns() - start) / 1000)
        check_storage(storage, host_record(mixed_probe, values, dtype))
        reports.append(dict(case=values, wall_us=samples, median_us=statistics.median(samples)))
    (mixed_probe["directory"] / f"timing-{dtype}.json").write_text(
        json.dumps(
            dict(
                dtype=dtype,
                cpu_affinity=sorted(os.sched_getaffinity(0)),
                records=reports,
                scope="AICPU generation with ctypes/events/per-call sync; no attention or public E2E",
            ),
            indent=2,
        )
        + "\n"
    )


@pytest.mark.parametrize("dtype", (1, 2))
def test_generation_bsnd_comparison(mixed_probe, dtype):
    reports = []
    for b, sq, heads, sk, used in [
        (2, 1, 8, 512, 257),
        (10, 2, 128, 512, 511),
        (21, 171, 3, 16384, 8193),
    ]:
        values = mixed_case([sq] * b, [used] * b, heads=heads, capacity=sk)
        args = arguments(values, dtype)
        q, k = device_inputs(values)
        mixed_storage = storage_for(args)
        bsnd = bsnd_arguments(length_case([used] * b, batch=b, sq=sq, heads=heads, sk=sk), dtype)
        bsnd_storage = storage_for(bsnd)
        launchers = {
            "v2-bsnd": lambda: bsnd_enqueue(mixed_probe, bsnd, bsnd_storage, k),
            "v4-mixed": lambda: enqueue(mixed_probe, args, mixed_storage, q, k),
        }
        samples = {name: [] for name in launchers}
        for iteration in range(25):
            order = list(launchers) if iteration % 2 == 0 else list(launchers)[::-1]
            for name in order:
                torch.npu.synchronize()
                start = time.perf_counter_ns()
                launchers[name]()
                torch.npu.synchronize()
                if iteration >= 5:
                    samples[name].append((time.perf_counter_ns() - start) / 1000)
        mixed_row = check_storage(mixed_storage, host_record(mixed_probe, values, dtype))
        bsnd_row = [v & 0xFFFFFFFF for v in bsnd_storage.cpu().tolist()]
        assert bsnd_row[0] == bsnd_row[-1] == SENTINEL
        assert bsnd_row[1:9] == [0x4D4C4131, 2, 2, 401 + b, dtype, 0, 0, 0]
        assert bsnd_row[9:402] == mixed_row[8:401] and bsnd_row[402:-1] == [used] * b
        medians = {name: statistics.median(v) for name, v in samples.items()}
        reports.append(
            dict(
                shape=[b, sq, heads, sk],
                used=used,
                wall_us=samples,
                median_us=medians,
                ratio_v4_v2=medians["v4-mixed"] / medians["v2-bsnd"],
            )
        )
    (mixed_probe["directory"] / f"comparison-{dtype}.json").write_text(
        json.dumps(
            dict(
                dtype=dtype,
                records=reports,
                cpu_affinity=sorted(os.sched_getaffinity(0)),
                scope="Alternating AICPU generation with identical schedules; no attention or public E2E",
            ),
            indent=2,
        )
        + "\n"
    )
