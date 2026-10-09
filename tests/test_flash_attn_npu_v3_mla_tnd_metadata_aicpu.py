"""910B v3 AICPU producer; no TND attention/device consumer is enabled."""

import ctypes as ct
import json
import os
from pathlib import Path
import statistics
import time

import pytest
import torch
import torch_npu  # noqa: F401

from tests.mla_metadata_build import ROOT, build_probe, sha256
from tests.mla_metadata_error_probe import forbid_host_readback
from tests.test_flash_attn_npu_v3_mla_metadata import Args, Memory, Platform, Shape, load_probe
from tests.test_flash_attn_npu_v3_mla_tnd_metadata import (
    tnd_artifacts as tnd_artifacts,
    tnd_oracle as tnd_oracle,
    FAILURES,
    PREFIX,
    SENTINEL,
    check_records,
    metadata_cases,
    prefixes,
    tnd_case,
)


class TndShape(ct.Structure):
    _fields_ = [
        (name, ct.c_int64) for name in ("batch", "total_q", "total_k", "heads", "max_q", "max_k")
    ]


class TndArgs(ct.Structure):
    _fields_ = [
        ("output", ct.c_uint64),
        ("words", ct.c_uint32),
        ("dtype", ct.c_uint32),
        ("shape", TndShape),
        ("platform", Platform),
        ("cu_q", ct.c_uint64),
        ("qcount", ct.c_uint64),
        ("cu_k", ct.c_uint64),
        ("kcount", ct.c_uint64),
    ]


def arguments(values, dtype):
    tail = 16 + values[15]
    return TndArgs(
        0,
        PREFIX + values[0] * 4,
        dtype,
        TndShape(values[0], values[tail], values[tail + 1], values[2], values[1], values[3]),
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
def tnd_probe(tmp_path_factory):
    directory = Path(
        os.environ.get("MLA_TND_METADATA_DEVICE_DIR", tmp_path_factory.mktemp("mla-tnd-aicpu"))
    )
    library = os.environ.get("MLA_TND_METADATA_LIBRARY")
    if library:
        manifest = json.loads((Path(library).parent / "build-manifest.json").read_text())
        assert sha256(library) == manifest["library_sha256"]
        for path, digest in manifest["sources"].items():
            assert sha256(ROOT / path) == digest, path
    else:
        library = build_probe(directory, tnd=True)
    with load_probe(library, directory) as probe:
        lib = probe["lib"]
        lib.tnd_metadata_args_size.restype = ct.c_uint32
        assert lib.tnd_metadata_args_size() == ct.sizeof(TndArgs)
        lib.launch_tnd_metadata.argtypes = [ct.c_void_p] * 4 + [ct.POINTER(TndArgs)]
        lib.launch_tnd_metadata.restype = ct.c_int
        lib.tnd_metadata_host.argtypes = [ct.POINTER(TndArgs), ct.c_void_p]
        lib.tnd_metadata_host.restype = ct.c_uint32
        yield probe


def host_record(probe, values, dtype):
    args = arguments(values, dtype)
    cq, ck = prefixes(values)
    q = (ct.c_int32 * (len(cq) + 2))(SENTINEL, *cq, SENTINEL)
    k = (ct.c_int32 * (len(ck) + 2))(SENTINEL, *ck, SENTINEL)
    args.cu_q = ct.addressof(q) + 4 if values[18 + values[15]] else 0
    args.cu_k = ct.addressof(k) + 4 if values[13] else 0
    output = (ct.c_uint32 * args.words)()
    assert probe["lib"].tnd_metadata_host(ct.byref(args), output) == 0
    assert list(q) == [SENTINEL, *cq, SENTINEL] and list(k) == [SENTINEL, *ck, SENTINEL]
    return list(output)


def device_inputs(values):
    def tensor(prefix, present):
        if not present:
            return None
        return torch.tensor([SENTINEL, *prefix, SENTINEL], dtype=torch.int32, device="npu")[1:-1]

    cq, ck = prefixes(values)
    return tensor(cq, values[18 + values[15]]), tensor(ck, values[13])


def enqueue(probe, args, storage, q, k, worker=0):
    output = storage[1:-1]
    assert output.is_contiguous() and output.numel() == args.words
    args.output = output.data_ptr()
    args.cu_q = 0 if q is None else q.data_ptr()
    args.cu_k = 0 if k is None else k.data_ptr()
    stream = probe["workers"][worker]
    ready, done = probe["events"][worker * 2 : worker * 2 + 2]
    assert (
        probe["lib"].launch_tnd_metadata(
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


@pytest.mark.parametrize("dtype", (1, 2))
@pytest.mark.parametrize("aic,ratio", [(None, 2), (1, 1), (15, 2), (26, 1)])
def test_aicpu_source_records(tnd_probe, tnd_oracle, dtype, aic, ratio):
    cases = metadata_cases(aic or tnd_probe["platform"].aic, ratio)
    if aic is None:
        p = tnd_probe["platform"]
        assert list(p.block_dims)[: p.aic] == list(range(1, p.aic + 1))
        for values in cases:
            values[6:11] = [getattr(p.memory, key) for key in ("l1", "ub", "l0a", "l0b", "l0c")]
            values[11] = p.libapi
    expected = [host_record(tnd_probe, c, dtype) for c in cases]
    pending = []
    for i, values in enumerate(cases):
        args = arguments(values, dtype)
        q, k = device_inputs(values)
        storage = storage_for(args)
        with forbid_host_readback():
            enqueue(tnd_probe, args, storage, q, k, i % 2)
        pending.append(storage)
    rows = [check_storage(s, e) for s, e in zip(pending, expected)]
    check_records(
        rows, cases, tnd_oracle, tnd_probe["directory"], f"device-{dtype}-{aic}-{ratio}", dtype
    )


@pytest.mark.parametrize("dtype", (1, 2))
def test_failures_empty_and_buffer_reuse(tnd_probe, dtype):
    cases = [tnd_case(**kwargs) for kwargs, _, _ in FAILURES]
    cases += [
        tnd_case(q, k, sq=17, sk=16384)
        for q, k in [([], []), ([0], [0]), ([0, 0], [0, 513]), ([3, 0, 1], [0, 0, 0])]
    ]
    reports = []
    for values in cases:
        args = arguments(values, dtype)
        q, k = device_inputs(values)
        storage = storage_for(args)
        with forbid_host_readback():
            enqueue(tnd_probe, args, storage, q, k)
        row = check_storage(storage, host_record(tnd_probe, values, dtype))
        reports.append(dict(case=values, record=row))
        if row[5]:
            assert not any(row[8:])
        # Rewrite the failed buffer with valid prefixes of the same batch.
        good = tnd_case([3] * values[0], [513] * values[0])
        args = arguments(good, dtype)
        q, k = device_inputs(good)
        with forbid_host_readback():
            enqueue(tnd_probe, args, storage, q, k)
        check_storage(storage, host_record(tnd_probe, good, dtype))
    (tnd_probe["directory"] / f"failures-empty-{dtype}.json").write_text(json.dumps(reports) + "\n")


@pytest.mark.parametrize("dtype", (1, 2))
def test_each_prefix_update_and_snapshot_ownership(tnd_probe, dtype):
    b = 2 * tnd_probe["platform"].aic + 1
    values = tnd_case([3] * b, [513] * b, sq=512, sk=16384)
    cq, ck = prefixes(values)
    q, k = device_inputs(values)
    args = arguments(values, dtype)
    storage = storage_for(args)
    for i in range(1, b):
        cq[i] -= i % 2 + 1
        ck[i] -= (i % 2 + 1) * 100
        q[i], k[i] = cq[i], ck[i]
        values = tnd_case([3] * b, [513] * b, cq=cq, ck=ck, sq=512, sk=16384)
        with forbid_host_readback():
            enqueue(tnd_probe, args, storage, q, k)
        check_storage(storage, host_record(tnd_probe, values, dtype))
    assert q._base.cpu().tolist() == [SENTINEL, *cq, SENTINEL]
    assert k._base.cpu().tolist() == [SENTINEL, *ck, SENTINEL]
    frozen = storage.clone()
    q.zero_()
    k.zero_()
    del q, k
    assert torch.equal(frozen.cpu(), storage.cpu())


def test_alias_and_stream_allocator_lifetime(tnd_probe):
    pending = []
    for worker, stream in enumerate([torch.npu.Stream(), torch.npu.Stream()]):
        with torch.npu.stream(stream):
            for i in range(24):
                qlens, klens = ([3, 0, 1], [17, 0, 33]) if i % 2 else ([3, 0, 1], [3, 0, 1])
                values = tnd_case(qlens, klens)
                args = arguments(values, worker + 1)
                q, k = device_inputs(values)
                if i % 2 == 0:
                    k = q
                storage = storage_for(args)
                with forbid_host_readback():
                    enqueue(tnd_probe, args, storage, q, k, worker)
                    copied = storage.clone()
                pending.append((copied, host_record(tnd_probe, values, worker + 1)))
                del args, q, k, storage
                torch.empty((32768,), dtype=torch.int32, device="npu").fill_(SENTINEL)
    for storage, expected in pending:
        check_storage(storage, expected)


@pytest.mark.parametrize("dtype", (1, 2))
def test_generation_diagnostics(tnd_probe, dtype):
    reports = []
    for qlens, klens, heads in [
        ([1, 1], [257, 257], 8),
        ([3, 0, 1], [17, 0, 33], 128),
        ([171] * 21, [8193] * 21, 3),
        ([3, 0, 1] * 14, [16384, 0, 513] * 14, 171),
        ([0, 0], [0, 0], 128),
    ]:
        values = tnd_case(qlens, klens, heads=heads)
        args = arguments(values, dtype)
        q, k = device_inputs(values)
        storage = storage_for(args)
        samples = []
        for iteration in range(25):
            torch.npu.synchronize()
            start = time.perf_counter_ns()
            enqueue(tnd_probe, args, storage, q, k)
            torch.npu.synchronize()
            if iteration >= 5:
                samples.append((time.perf_counter_ns() - start) / 1000)
        check_storage(storage, host_record(tnd_probe, values, dtype))
        reports.append(dict(case=values, wall_us=samples, median_us=statistics.median(samples)))
    (tnd_probe["directory"] / f"timing-{dtype}.json").write_text(
        json.dumps(
            dict(
                dtype=dtype,
                cpu_affinity=sorted(os.sched_getaffinity(0)),
                records=reports,
                scope="AICPU generation including ctypes, events and per-call sync; no attention/public E2E claim",
            ),
            indent=2,
        )
        + "\n"
    )


@pytest.mark.parametrize("dtype", (1, 2))
def test_generation_dense_comparison(tnd_probe, dtype):
    # Uniform cases with identical source schedules isolate protocol/input
    # handling overhead. This comparison is not attention or public E2E.
    reports = []
    for b, sq, heads, sk in [(2, 1, 8, 257), (1, 7, 16, 65), (21, 171, 3, 8193)]:
        values = tnd_case([sq] * b, [sk] * b, heads=heads)
        args = arguments(values, dtype)
        q, k = device_inputs(values)
        tnd_storage = storage_for(args)
        dense = Args(0, 401, dtype, Shape(b, sq, heads, sk), args.platform)
        dense_storage = storage_for(dense)
        dense.output = dense_storage[1:-1].data_ptr()
        stream = tnd_probe["workers"][0]
        ready, done = tnd_probe["events"][:2]

        def launch_dense():
            assert (
                tnd_probe["lib"].launch_metadata(
                    torch.npu.current_stream().npu_stream,
                    stream.npu_stream,
                    ready,
                    done,
                    ct.byref(dense),
                )
                == 0
            )
            dense_storage.record_stream(stream)

        launchers = {
            "v1-dense": launch_dense,
            "v3-tnd": lambda: enqueue(tnd_probe, args, tnd_storage, q, k),
        }
        samples = {key: [] for key in launchers}
        for iteration in range(25):
            order = list(launchers) if iteration % 2 == 0 else list(launchers)[::-1]
            for name in order:
                torch.npu.synchronize()
                start = time.perf_counter_ns()
                launchers[name]()
                torch.npu.synchronize()
                if iteration >= 5:
                    samples[name].append((time.perf_counter_ns() - start) / 1000)
        tnd_row = check_storage(tnd_storage, host_record(tnd_probe, values, dtype))
        dense_row = [v & 0xFFFFFFFF for v in dense_storage.cpu().tolist()]
        assert dense_row[0] == dense_row[-1] == SENTINEL
        assert dense_row[1:9] == [0x4D4C4131, 1, 1, 401, dtype, 0, 0, 0]
        assert dense_row[9:-1] == tnd_row[8:401]
        medians = {name: statistics.median(v) for name, v in samples.items()}
        reports.append(
            dict(
                shape=[b, sq, heads, sk],
                wall_us=samples,
                median_us=medians,
                ratio_v3_v1=medians["v3-tnd"] / medians["v1-dense"],
            )
        )
    (tnd_probe["directory"] / f"comparison-{dtype}.json").write_text(
        json.dumps(
            dict(
                dtype=dtype,
                records=reports,
                cpu_affinity=sorted(os.sched_getaffinity(0)),
                scope="Alternating ctypes AICPU generation with identical schedules; no attention/public E2E",
            ),
            indent=2,
        )
        + "\n"
    )
