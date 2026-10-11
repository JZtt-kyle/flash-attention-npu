"""910B standalone v6 producer: CPU/pinned PA source, bounds and stream lifetime.

The diagnostic wrapper calls the exact entry and captures its return code;
it does not claim device consumer, Trap, Copy or attention acceptance.
"""

import ctypes as ct
import itertools
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
from tests.test_flash_attn_npu_v3_mla_metadata import Memory, Platform, load_probe, SENTINEL
from tests.test_flash_attn_npu_v3_mla_paged_metadata import (
    PREFIX,
    decode_record,
    encode,
)
from tests.test_flash_attn_npu_v3_mla_paged_schedule import (
    paged_artifacts as paged_artifacts,
    paged_oracles as paged_oracles,
    execute,
    paged_case,
)
from tests.test_flash_attn_npu_v3_mla_tnd_bsnd_metadata_aicpu import storage_for, check_storage
from tests.test_flash_attn_npu_v3_mla_used_metadata_aicpu import Allocation, Array


class PagedShape(ct.Structure):
    _fields_ = [
        ("batch", ct.c_int64),
        ("heads", ct.c_int64),
        ("query", Allocation),
        ("capacity", ct.c_int64),
        ("page", ct.c_int64),
    ]


ARRAY_NAMES = ("cu_q", "used_q", "kv_lengths")


class PagedArgs(ct.Structure):
    _fields_ = [
        ("output", ct.c_uint64),
        ("words", ct.c_uint32),
        ("dtype", ct.c_uint32),
        ("shape", PagedShape),
        ("platform", Platform),
    ] + [(name, Array) for name in ARRAY_NAMES]


class ProbeArgs(ct.Structure):
    _fields_ = [("input", PagedArgs), ("status", ct.c_uint64), ("null_args", ct.c_uint32)]


def input_arrays(item):
    values, offset, arrays = encode(item), 16, []
    for _ in ARRAY_NAMES:
        present, count, size = values[offset : offset + 3]
        arrays.append((values[offset + 3 : offset + 3 + size], present, count))
        offset += 3 + size
    assert offset == len(values)
    return arrays


def arguments(item, dtype):
    v = encode(item)
    return PagedArgs(
        0,
        PREFIX + 5 * v[0],
        dtype,
        PagedShape(v[0], v[1], Allocation(*v[2:5]), v[6], v[5]),
        Platform(
            v[7],
            v[8],
            Memory(*v[9:14]),
            v[14],
            (ct.c_uint32 * 26)(*[v[15] or c for c in range(1, 27)]),
        ),
        *[Array(0, count) for _, _, count in input_arrays(item)],
    )


@pytest.fixture(scope="module")
def paged_probe(tmp_path_factory):
    target = os.environ.get("MLA_PAGED_METADATA_DEVICE_DIR")
    directory = Path(target) if target else tmp_path_factory.mktemp("mla-paged-aicpu")
    library = os.environ.get("MLA_PAGED_METADATA_LIBRARY")
    if library:
        manifest = json.loads((Path(library).parent / "build-manifest.json").read_text())
        assert sha256(library) == manifest["library_sha256"]
        for path, digest in manifest["sources"].items():
            assert sha256(ROOT / path) == digest, path
    else:
        library = build_probe(directory, paged=True)
    with load_probe(library, directory) as probe:
        lib = probe["lib"]
        for name, typ in (
            ("paged_metadata_args_size", PagedArgs),
            ("paged_metadata_probe_args_size", ProbeArgs),
        ):
            getattr(lib, name).restype = ct.c_uint32
            assert getattr(lib, name)() == ct.sizeof(typ)
        lib.launch_paged_metadata.argtypes = [ct.c_void_p] * 4 + [
            ct.POINTER(PagedArgs),
            ct.c_uint64,
            ct.c_bool,
        ]
        lib.launch_paged_metadata.restype = ct.c_int
        lib.paged_metadata_host.argtypes = [ct.POINTER(PagedArgs), ct.c_void_p]
        lib.paged_metadata_host.restype = ct.c_uint32
        yield probe


def host_record(probe, item, dtype):
    args, keep = arguments(item, dtype), []
    for name, (values, present, _) in zip(ARRAY_NAMES, input_arrays(item)):
        data = (ct.c_int32 * (len(values) + 2))(SENTINEL, *values, SENTINEL)
        getattr(args, name).address = ct.addressof(data) + 4 if present else 0
        keep.append((data, [SENTINEL, *values, SENTINEL]))
    output = (ct.c_uint32 * args.words)()
    assert probe["lib"].paged_metadata_host(ct.byref(args), output) == 0
    assert all(list(data) == before for data, before in keep)
    return list(output)


def device_inputs(item):
    return [
        torch.tensor([SENTINEL, *values, SENTINEL], dtype=torch.int32, device="npu")[1:-1]
        if present
        else None
        for values, present, _ in input_arrays(item)
    ]


def enqueue(
    probe,
    args,
    storage,
    inputs,
    worker=0,
    *,
    descriptors=None,
    status=None,
    null_args=False,
    output=None,
):
    # Keep explicit zero-count presence using the live guarded allocation.
    args.output = storage[1:-1].data_ptr() if output is None else output
    for name, tensor in zip(ARRAY_NAMES, inputs):
        getattr(args, name).address = (
            0
            if tensor is None
            else tensor.untyped_storage().data_ptr() + 4 * tensor.storage_offset()
        )
    for name, value in (descriptors or {}).items():
        setattr(args, name, Array(*value))
    stream = probe["workers"][worker]
    ready, done = probe["events"][2 * worker : 2 * worker + 2]
    assert (
        probe["lib"].launch_paged_metadata(
            torch.npu.current_stream().npu_stream,
            stream.npu_stream,
            ready,
            done,
            ct.byref(args),
            0 if status is None else status.data_ptr(),
            null_args,
        )
        == 0
    )
    for tensor in (storage, *inputs, status):
        if tensor is not None:
            tensor.record_stream(stream)


def source_records(probe, oracles, items, dtype, name):
    source = execute(oracles[0], items)
    native = execute(oracles[0], items, native=True)
    rows, pending = [], []
    for i, item in enumerate(items):
        args, inputs = arguments(item, dtype), device_inputs(item)
        storage = storage_for(args)
        # AICPU must derive TND totals even if host geometry has a poison total.
        if item["layout"] == 2:
            args.shape.query.total = -(2**63)
        with forbid_host_readback():
            enqueue(probe, args, storage, inputs, i % 2)
        pending.append((storage, host_record(probe, item, dtype)))
    for (storage, expected), item, wanted in zip(pending, items, source):
        row = check_storage(storage, expected)
        decode_record(row, item, dtype, wanted)
        rows.append(row)
    non_fd = sum(row[5] == 0 for row in native)
    assert all(wanted == original for wanted, original in zip(source, native) if original[5] == 0)
    (probe["directory"] / (name + ".json")).write_text(
        json.dumps(
            dict(
                cases=len(items),
                native_non_fd=non_fd,
                fa3_no_split_mapping=len(items) - non_fd,
                inputs=items,
                records=rows,
                source=source,
                native=native,
            )
        )
        + "\n"
    )
    return rows


@pytest.mark.parametrize(
    "page,layout,dtype", itertools.product(range(16, 1025, 16), (0, 2), (1, 2))
)
def test_all_page_sizes(paged_probe, paged_oracles, page, layout, dtype):
    capacity = 16384 // page * page
    q = [7, 7, 7] if layout == 0 else [7, 0, 3]
    p = paged_probe["platform"]
    items = [
        paged_case(
            layout,
            q,
            [length, 0, min(17, length)],
            uq,
            page,
            capacity,
            aic=p.aic,
            ratio=2,
            libapi=p.libapi,
            **{name: getattr(p.memory, name) for name in ("l1", "ub", "l0a", "l0b", "l0c")},
        )
        for length, uq in itertools.product(
            sorted({1, page - 1, page, page + 1, 449, 513, capacity}), (None, [3, 0, 1])
        )
    ]
    source_records(paged_probe, paged_oracles, items, dtype, f"pages-{page}-{layout}-{dtype}")


@pytest.mark.parametrize("aic,ratio,dtype", itertools.product(range(1, 27), (1, 2), (1, 2)))
def test_synthetic_platforms(paged_probe, paged_oracles, aic, ratio, dtype):
    items = [
        paged_case(layout, [2] * b, [1025] * b, uq, page, aic=aic, ratio=ratio)
        for page, layout, b in itertools.product((112, 528), (0, 2), sorted({1, aic, aic + 1}))
        for uq in (None, [1] * b, [0] * b)
    ]
    source_records(paged_probe, paged_oracles, items, dtype, f"cores-{aic}-{ratio}-{dtype}")


@pytest.mark.parametrize("layout,dtype", itertools.product((0, 2), (1, 2)))
def test_empty_allocation_and_zero_lengths(paged_probe, layout, dtype):
    rows = []
    for q, kv, capacity, sq in (
        ([], [], 224, 7),
        ([0, 0], [17, 0], 224, 0),
        ([7, 7], [0, 0], 224, 7),
        ([7, 7], [0, 0], 0, 7),
    ):
        for uq, present in itertools.product(
            (None, [0] * len(q), q), (False, True) if not q else (True,)
        ):
            item = paged_case(layout, q, kv, uq, capacity=capacity, sq=sq, has_uk=present)
            args, inputs = arguments(item, dtype), device_inputs(item)
            storage = storage_for(args)
            with forbid_host_readback():
                enqueue(paged_probe, args, storage, inputs)
            row = check_storage(storage, host_record(paged_probe, item, dtype))
            decode_record(row, item, dtype, None)
            rows.append(dict(case=item, record=row))
    (paged_probe["directory"] / f"empty-{layout}-{dtype}.json").write_text(json.dumps(rows) + "\n")


@pytest.mark.parametrize("dtype", (1, 2))
def test_layout_allocation_and_zero_count_alias(paged_probe, dtype):
    bsnd = paged_case(0, capacity=224)
    cases = [dict(bsnd, total_q=total) for total in (-1, 13, 2**63 - 1)]
    cases += [dict(bsnd, has_q=True, cq=[0, 7, 14]), dict(bsnd, layout=1)]
    for item in cases:
        args, inputs = arguments(item, dtype), device_inputs(item)
        storage = storage_for(args)
        with forbid_host_readback():
            enqueue(paged_probe, args, storage, inputs)
        row = check_storage(storage, host_record(paged_probe, item, dtype))
        assert row[5] != 0 and not any(row[8:])
    # At B=0, zero-span aliases retain presence without reading/writing lengths.
    for layout in (0, 2):
        item = paged_case(layout, [], [], [], capacity=224, sq=7)
        args, inputs = arguments(item, dtype), device_inputs(item)
        storage = storage_for(args)
        address = storage[1:-1].data_ptr()
        with forbid_host_readback():
            enqueue(
                paged_probe,
                args,
                storage,
                inputs,
                descriptors={name: (address, 0) for name in ("used_q", "kv_lengths")},
            )
        row = check_storage(storage, host_record(paged_probe, item, dtype))
        decode_record(row, item, dtype, None)
        assert row[405] == 1


FAILURES = [
    *[dict(page=p) for p in (-1, 0, 15, 17, 1025, 2**63 - 1)],
    dict(sk=223),
    dict(sk=16464),
    dict(sq=-1),
    dict(sq=2**32),
    dict(heads=0),
    dict(heads=2**23),
    dict(cq=[1, 7, 14]),
    dict(cq=[0, 8, 7]),
    dict(cq=[0, -(2**31), 14]),
    dict(cq=[0, 7, -1]),
    dict(uq=[-1, 0]),
    dict(uq=[8, 0]),
    dict(uk=[0, -(2**31)]),
    dict(uk=[0, 225]),
    dict(has_q=False),
    dict(qcount=0),
    dict(has_uk=False),
    dict(ukcount=0),
    dict(uqcount=1),
    dict(aic=0, aiv=0),
    dict(aic=27),
    dict(ratio=3),
    dict(block_dim=21),
    dict(libapi=2**64 - 1),
    dict(l1=516095),
    dict(q=[0, 0], sq=0, uk=[0, -1]),
    dict(uq=[0, 0], uk=[0, 225]),
]


@pytest.mark.parametrize("dtype", (1, 2))
def test_failure_clears_payload_and_reuses_buffer(paged_probe, dtype):
    reports = []
    good = paged_case(2, capacity=224)
    storage = storage_for(arguments(good, dtype))
    for updates in FAILURES:
        item = dict(good, **updates)
        # CPU comparison supplies the same actual prefix total, not a host guess.
        if "cq" in item:
            item["total_q"] = item["cq"][-1]
        args, inputs = arguments(item, dtype), device_inputs(item)
        with forbid_host_readback():
            enqueue(paged_probe, args, storage, inputs)
        row = check_storage(storage, host_record(paged_probe, item, dtype))
        assert row[5] != 0 and not any(row[8:])
        reports.append(dict(case=item, record=row))
        with forbid_host_readback():
            enqueue(paged_probe, arguments(good, dtype), storage, device_inputs(good))
        check_storage(storage, host_record(paged_probe, good, dtype))
    (paged_probe["directory"] / f"failures-{dtype}.json").write_text(json.dumps(reports) + "\n")


@pytest.mark.parametrize("dtype", (1, 2))
def test_descriptors_before_prefix_read(paged_probe, dtype):
    item = paged_case(2, uq=[3, 0])
    args, inputs = arguments(item, dtype), device_inputs(item)
    storage = storage_for(args)
    expected = host_record(paged_probe, item, dtype)
    failure = expected[:5] + [3, 0, 0] + [0] * (args.words - 8)
    checked = 0
    for name, tensor in zip(ARRAY_NAMES, inputs):
        count = getattr(args, name).count
        for address, size in (
            (0, count),
            (tensor.data_ptr() + 1, count),
            (tensor.data_ptr() + 2, count),
            (tensor.data_ptr(), count - 1),
            (tensor.data_ptr(), count + 1),
            (2**64 - 4, count),
            (2**64 - 4, 2**64 - 1),
        ):
            with forbid_host_readback():
                enqueue(paged_probe, args, storage, inputs, descriptors={name: (address, size)})
            check_storage(storage, failure)
            args = arguments(item, dtype)
            checked += 1
    for tensor, (values, _, _) in zip(inputs, input_arrays(item)):
        assert tensor._base.cpu().tolist() == [SENTINEL, *values, SENTINEL]
    # Invalid dtype must not read even aligned, non-overlapping bogus inputs.
    for bad_dtype in (0, 3, 2**32 - 1):
        args = arguments(item, bad_dtype)
        descriptors = {name: (4, getattr(args, name).count) for name in ARRAY_NAMES}
        with forbid_host_readback():
            enqueue(paged_probe, args, storage, inputs, descriptors=descriptors)
        check_storage(storage, expected[:4] + [bad_dtype, 1, 0, 0] + [0] * (args.words - 8))
    (paged_probe["directory"] / f"descriptors-{dtype}.json").write_text(
        json.dumps(dict(invalid_descriptors=checked, invalid_dtypes=3)) + "\n"
    )


@pytest.mark.parametrize("dtype", (1, 2))
def test_unsafe_output_no_writes_on_device(paged_probe, dtype):
    item = paged_case(2, uq=[3, 0])
    args, inputs = arguments(item, dtype), device_inputs(item)
    storage, status = storage_for(args), torch.full((1,), SENTINEL, dtype=torch.int32, device="npu")
    address, words = storage[1:-1].data_ptr(), args.words
    cases = [
        (dict(descriptors={name: (address + offset, count)}), {}, dtype)
        for name, offset, count in itertools.product(
            ARRAY_NAMES, (-4, 0, 1, 4 * (words - 1)), (3, 2**64 - 1)
        )
    ]
    cases += [(dict(output=value), {}, dtype) for value in (0, address + 1, address + 2, 2**64 - 4)]
    cases += [({}, dict(words=value), dtype) for value in (0, 401, words - 1, words + 1, 2**32 - 1)]
    cases += [(dict(null_args=True), {}, dtype)]
    cases += [(dict(descriptors={"cu_q": (address, 3)}), {}, 3)]
    for options, mutations, code in cases:
        args = arguments(item, code)
        for key, value in mutations.items():
            setattr(args, key, value)
        with forbid_host_readback():
            enqueue(paged_probe, args, storage, inputs, status=status, **options)
        assert status.cpu().tolist() == [1]
        assert storage.cpu().tolist() == [SENTINEL] * (words + 2)
    for tensor, (values, _, _) in zip(inputs, input_arrays(item)):
        assert tensor._base.cpu().tolist() == [SENTINEL, *values, SENTINEL]
    (paged_probe["directory"] / f"unsafe-output-{dtype}.json").write_text(
        json.dumps(
            dict(
                cases=len(cases),
                returncode=1,
                output_unchanged=True,
                scope="AICPU diagnostic wrapper around exact entry; not a consumer Trap",
            )
        )
        + "\n"
    )


@pytest.mark.parametrize("dtype", (1, 2))
def test_device_updates_prefix_tail_used_and_kv(paged_probe, paged_oracles, dtype):
    item = paged_case(2, [7, 7, 0], [17, 0, 113], [3, 1, 0], capacity=224, sq=17)
    args, inputs = arguments(item, dtype), device_inputs(item)
    storage, pending, items = storage_for(args), [], []
    for total in (14, 15, 3, 0, 19):
        item = dict(
            item, q=[total // 2, total - total // 2, 0], uq=[0, min(1, total), 0], uk=[17, 0, total]
        )
        for tensor, (values, _, _) in zip(inputs, input_arrays(item)):
            tensor.copy_(torch.tensor(values, dtype=torch.int32, device="npu"))
        with forbid_host_readback():
            enqueue(paged_probe, args, storage, inputs)
            pending.append(storage.clone())
        items.append(item)
    for snapshot, item in zip(pending, items):
        row = check_storage(snapshot, host_record(paged_probe, item, dtype))
        source = None if not sum(item["q"]) else execute(paged_oracles[0], [item])[0]
        decode_record(row, item, dtype, source)
    frozen = storage.clone()
    for tensor in inputs:
        tensor.zero_()
    del inputs
    assert torch.equal(storage.cpu(), frozen.cpu())


def test_readonly_alias_cross_stream_allocator_lifetime(paged_probe):
    pending, consumer = [], torch.npu.Stream()
    for worker, stream in enumerate((torch.npu.Stream(), torch.npu.Stream())):
        for i in range(32):
            with torch.npu.stream(stream):
                item = paged_case(2, [3, 3], [0, 3], [0, 3])
                args, inputs = arguments(item, worker + 1), device_inputs(item)
                if i % 2 == 0:
                    inputs = [inputs[0], inputs[0][:-1], inputs[0][:-1]]
                storage = storage_for(args)
                with forbid_host_readback():
                    enqueue(paged_probe, args, storage, inputs, worker)
                    ready = torch.npu.Event()
                    ready.record(stream)
            with torch.npu.stream(consumer), forbid_host_readback():
                consumer.wait_event(ready)
                storage.record_stream(consumer)
                copied = storage.clone()
            pending.append((copied, host_record(paged_probe, item, worker + 1)))
            del args, inputs, storage
            with torch.npu.stream(stream):
                torch.empty((32768,), dtype=torch.int32, device="npu").fill_(SENTINEL)
    for storage, expected in pending:
        check_storage(storage, expected)


@pytest.mark.parametrize("dtype", (1, 2))
def test_wide_query_and_workspace(paged_probe, dtype):
    batch = 65538
    item = paged_case(
        0, [65536] * batch, [0] * batch, [0] * batch, page=1024, capacity=16384, heads=1
    )
    args, inputs = arguments(item, dtype), device_inputs(item)
    storage = storage_for(args)
    with forbid_host_readback():
        enqueue(paged_probe, args, storage, inputs)
    row = check_storage(storage, host_record(paged_probe, item, dtype))
    decode_record(row, item, dtype, None)
    assert row[403] == row[-4] == 1
    item = paged_case(2, page=528)
    row = host_record(paged_probe, item, dtype)
    budget = row[385] | row[386] << 32
    item["libapi"] = 2**64 - 1 - budget
    args, inputs = arguments(item, dtype), device_inputs(item)
    storage = storage_for(args)
    with forbid_host_readback():
        enqueue(paged_probe, args, storage, inputs)
    row = check_storage(storage, host_record(paged_probe, item, dtype))
    assert row[389:391] == [2**32 - 1] * 2


@pytest.mark.parametrize("dtype", (1, 2))
def test_generation_timing(paged_probe, dtype):
    reports = []
    for page, layout in itertools.product((16, 112, 128, 256, 528, 1024), (0, 2)):
        item = paged_case(layout, [2] * 10, [1025] * 10, page=page, aic=paged_probe["platform"].aic)
        args, inputs = arguments(item, dtype), device_inputs(item)
        storage, samples = storage_for(args), []
        for i in range(25):
            torch.npu.synchronize()
            start = time.perf_counter_ns()
            enqueue(paged_probe, args, storage, inputs)
            torch.npu.synchronize()
            if i >= 5:
                samples.append((time.perf_counter_ns() - start) / 1000)
        check_storage(storage, host_record(paged_probe, item, dtype))
        reports.append(dict(case=item, wall_us=samples, median_us=statistics.median(samples)))
    (paged_probe["directory"] / f"timing-{dtype}.json").write_text(
        json.dumps(
            dict(
                dtype=dtype,
                records=reports,
                cpu_affinity=sorted(os.sched_getaffinity(0)),
                scope="Standalone AICPU generation including ctypes/events/sync; no attention or public E2E",
            ),
            indent=2,
        )
        + "\n"
    )
