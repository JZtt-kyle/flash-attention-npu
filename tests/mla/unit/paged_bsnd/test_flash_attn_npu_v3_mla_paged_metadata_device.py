"""910B release v6 PA validation only: exact schedules, rejection, Trap and lifetime."""

import ctypes as ct
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest
import torch
import torch_npu  # noqa: F401

from tests.mla.support.mla_bsnd_consumer_build import build_consumers
from tests.mla.support.mla_metadata_build import ROOT, sha256
from tests.mla.support.mla_metadata_error_probe import forbid_host_readback
from tests.mla.unit.paged_bsnd.test_flash_attn_npu_v3_mla_paged_metadata_aicpu import (
    paged_probe as paged_probe,
    arguments,
    device_inputs,
    enqueue,
    host_record,
    storage_for,
)
from tests.mla.unit.paged_bsnd.test_flash_attn_npu_v3_mla_paged_metadata_consumer import (
    paged_artifacts as paged_artifacts,
    paged_oracles as paged_oracles,
    Context,
    Inputs,
    Array,
    Cache,
    ARRAY_NAMES,
    CORRUPTIONS,
    CACHE_FAULTS,
    SENTINEL,
    context_for,
    expected_schedule,
    matrix,
    empty_cases,
    cache_spec,
    change,
    paged_case,
)
from tests.mla.unit.paged_bsnd.test_flash_attn_npu_v3_mla_paged_metadata import decode_record
from tests.mla.unit.paged_bsnd.test_flash_attn_npu_v3_mla_paged_schedule import execute
from tests.mla.unit.used.test_flash_attn_npu_v3_mla_used_metadata import (
    used_builder as used_builder,
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


@pytest.fixture(scope="module")
def paged_consumer_build(tmp_path_factory):
    directory = Path(
        os.environ.get("MLA_PAGED_CONSUMER_DEVICE_DIR", tmp_path_factory.mktemp("mla-paged-device"))
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = os.environ.get("MLA_PAGED_CONSUMER_MANIFEST")
    manifest = (
        json.loads(Path(path).read_text()) if path else build_consumers(directory, paged=True)
    )
    (directory / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return directory, manifest


@pytest.fixture(scope="module", params=(1, 2))
def paged_device(paged_consumer_build, request):
    directory, manifest = paged_consumer_build
    return dict(
        lib=load_consumer(manifest, request.param), dtype=request.param, directory=directory
    )


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


def reset(case):
    for buffer in case["buffers"]:
        buffer.fill_(SENTINEL)


@pytest.mark.parametrize(
    "page,layout", [(p, layout) for p in range(16, 1025, 16) for layout in (0, 2)]
)
def test_aicpu_source_and_all_pages(paged_device, paged_probe, paged_oracles, page, layout):
    items, rows = [], []
    for item in matrix(page, layout):
        case = make_case(paged_device, paged_probe, item)
        with forbid_host_readback():
            launch(paged_device, case)
        check(case)
        items.append(case["item"])
        rows.append(case["expected"])
    source = execute(paged_oracles[0], items)
    native = execute(paged_oracles[0], items, native=True)
    for row, item, wanted, original in zip(rows, items, source, native):
        decode_record(row, item, paged_device["dtype"], wanted)
        if original[5] == 0:
            assert wanted == original
    report = dict(
        inputs=items,
        records=rows,
        source=source,
        native=native,
        cases=len(items),
        native_non_fd=sum(r[5] == 0 for r in native),
    )
    (paged_device["directory"] / f"source-{paged_device['dtype']}-{page}-{layout}.json").write_text(
        json.dumps(report) + "\n"
    )


def test_corrupt_record_and_descriptors_do_not_write(paged_device, paged_probe):
    case = make_case(
        paged_device, paged_probe, paged_case(2, [1, 1], [257, 257], [1, 1], 112, 448, heads=8)
    )
    good = case["metadata"].clone()
    for word, value, status in CORRUPTIONS:
        case["metadata"].copy_(good)
        case["metadata"][word] = value if value < 2**31 else value - 2**32
        altered = case["expected"].copy()
        altered[word] = value
        with forbid_host_readback():
            launch(paged_device, case, checked=False)
        check(case, status, record=altered)
    case["metadata"].copy_(good)
    for path, value, status in CACHE_FAULTS:
        launch(paged_device, case, checked=False, cache_changes={path: value})
        check(case, status)
    for axis, name in enumerate(ARRAY_NAMES):
        for field, value in (
            ("address", 0),
            ("address", 1),
            ("address", 2**64 - 4),
            ("count", 0),
            ("count", 2**64 - 1),
        ):
            launch(paged_device, case, checked=False, descriptors={name + "." + field: value})
            check(case, (12, 13, 11)[axis])
    for words in (0, 8, 407, len(case["expected"]) - 1, len(case["expected"]) + 1, 2**32 - 1):
        launch(paged_device, case, checked=False, words=words)
        check(case, 1)
    for address in (0, 1, 2**64 - 4):
        launch(paged_device, case, checked=False, address=address)
        check(case, 1)


@pytest.mark.parametrize("layout", (0, 2))
def test_empty_pool_table_and_zero_query(paged_device, paged_probe, layout):
    for item in empty_cases(layout):
        case = make_case(paged_device, paged_probe, item)
        pools = (0, 7) if not any(item["uk"]) else (7,)
        for pool in pools:
            case.update(make_cache(item, paged_device["dtype"], pool=pool))
            with forbid_host_readback():
                launch(paged_device, case)
            check(case)
            if pool == 0:
                reset(case)
                launch(
                    paged_device,
                    case,
                    cache_changes={"k.address": 0, "k.count": 0, "v.address": 0, "v.count": 0},
                )
                check(case)
        if not case["cache"].layout.span:
            launch(paged_device, case, cache_changes={"table.address": 0, "table.count": 0})
            check(case)
        if any(item["uk"]):
            reset(case)
            values = case["table_values"]
            for b, length in enumerate(item["uk"]):
                for j in range((length + item["page"] - 1) // item["page"]):
                    offset = b * case["cache"].layout.row_stride + j
                    for bad in (-1, -(2**31), 7, 2**31 - 1):
                        case["cache_tensors"][0][offset] = bad
                        launch(paged_device, case, checked=False)
                        check(case, 15)
                    case["cache_tensors"][0][offset] = values[offset]
            case.update(make_cache(item, paged_device["dtype"], pool=0))
            launch(paged_device, case, checked=False)
            check(case, 15)


@pytest.mark.parametrize("layout", (0, 2))
def test_page_remap_stride_reuse(paged_device, paged_probe, layout):
    item = paged_case(layout, [7] * 3, [257, 1, 0], [3, 0, 1], 112, 448)
    case = make_case(paged_device, paged_probe, item)
    for pool, stride, ks, vs in (
        (7, 7, (9000, 128, 1), (70000, 600, 1)),
        (1, 0, (0, 0, 1), (0, 0, 1)),
        (7, 1, (1, 0, 1), (60000, 513, 1)),
        (3, 4, (8000, 64, 1), (0, 0, 1)),
    ):
        case.update(
            make_cache(
                item,
                paged_device["dtype"],
                pool=pool,
                row_stride=stride,
                k_strides=ks,
                v_strides=vs,
            )
        )
        with forbid_host_readback():
            launch(paged_device, case)
        check(case)
        # Change effective IDs with the same immutable metadata: legal remaps
        # must be read on each new invocation, not bound to old pool addresses.
        for b, length in enumerate(item["uk"]):
            for j in range((length + 111) // 112):
                case["cache_tensors"][0][b * stride + j] = pool - 1
        reset(case)
        launch(paged_device, case)
        check(case)
        reset(case)
        case["cache_tensors"][0][0] = pool
        launch(paged_device, case, checked=False)
        check(case, 15)


def test_current_arrays_update_and_regenerate(paged_device, paged_probe):
    item = paged_case(2, [7, 0, 3], [257, 1, 0], [3, 0, 1], 112, 448)
    case = make_case(paged_device, paged_probe, item)
    for axis, tensor in enumerate(case["inputs"]):
        good = tensor.clone()
        for b in range(tensor.numel()):
            tensor[b] = -1
            with forbid_host_readback():
                launch(paged_device, case, checked=False)
            check(case, (12, 13, 11)[axis])
            tensor.copy_(good)
    changed = dict(item, q=[3, 0, 7], uq=[1, 0, 3], uk=[1, 0, 257])
    case["inputs"] = device_inputs(changed)
    case["args"] = arguments(changed, paged_device["dtype"])
    # Same total/max Q and P/C; source platform remains that of this invocation.
    changed.update(
        {
            key: case["item"][key]
            for key in ("aic", "ratio", "libapi", "l1", "ub", "l0a", "l0b", "l0c")
        }
    )
    case["args"] = arguments(changed, paged_device["dtype"])
    case["expected"] = host_record(paged_probe, changed, paged_device["dtype"])
    case["context"] = context_for(case["expected"], changed, paged_device["dtype"])
    # Context block count can change; create appropriately sized diagnostics.
    case["buffers"] = [
        torch.full(
            (3 * case["context"].block_dim * s + 32,), SENTINEL, dtype=torch.int32, device="npu"
        )
        for s in (16, 112, 16, 16)
    ]
    case.update(make_cache(changed, paged_device["dtype"]))
    with forbid_host_readback():
        enqueue(paged_probe, case["args"], case["metadata_storage"], case["inputs"])
        launch(paged_device, case)
    check(case)


def test_cross_stream_lifetime(paged_device, paged_probe):
    producer, consumer = torch.npu.Stream(), torch.npu.Stream()
    pending = []
    for i in range(12):
        with torch.npu.stream(producer):
            item = paged_case(2, [3, 0, 3], [3, 0, 3], [3, 0, 3], 112, 448)
            case = make_case(paged_device, paged_probe, item, worker=i % 2)
            # Legal alias of current Q-used and KV arrays, and cloned metadata.
            case["inputs"][2] = case["inputs"][1]
            copy = case["metadata_storage"].clone()
            case["metadata_storage"], case["metadata"] = copy, copy[1:-1]
            ready = torch.npu.Event()
            ready.record(producer)
        with torch.npu.stream(consumer):
            consumer.wait_event(ready)
            with forbid_host_readback():
                launch(paged_device, case)
                copies = [t.clone() for t in case["buffers"]]
                meta_copy = case["metadata_storage"].clone()
            pending.append(
                dict(
                    context=case["context"],
                    expected=case["expected"],
                    buffers=copies,
                    metadata_storage=meta_copy,
                )
            )
            del case, copy
            torch.empty(32768, device="npu", dtype=torch.int32).fill_(SENTINEL)
    for case in pending:
        check(case)


FAULTS = (
    "v1",
    "v2",
    "v3",
    "v4",
    "v5",
    "page",
    "capacity",
    "source_s2",
    "span",
    "wide_start",
    "words",
    "stale_q",
    "stale_uq",
    "stale_kv",
    "missing_q",
    "align_used",
    "table_descriptor",
    "cache_span",
    "page_negative",
    "page_oob",
    "qzero_page",
    "usedzero_page",
    "empty_pool",
    "empty_table_stale",
    "empty_batch",
    "suffix",
    "workspace",
    "local_memory",
    "generation",
)


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


@pytest.mark.parametrize("fault", FAULTS)
def test_release_trap(paged_device, paged_probe, fault):
    directory = paged_device["directory"] / f"error-{paged_device['dtype']}-{fault}"
    directory.mkdir(exist_ok=True)
    torch.npu.synchronize()
    sync = {"stale_uq": "event", "stale_kv": "stream"}.get(fault, "device")
    command = [
        sys.executable,
        "-m",
        "tests.mla.support.mla_paged_metadata_error_probe",
        str(paged_device["directory"] / "build-manifest.json"),
        str(paged_probe["directory"] / "build-manifest.json"),
        str(paged_device["dtype"]),
        fault,
        sync,
        str(directory),
    ]
    (directory / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    with (directory / "process.log").open("w") as log:
        with subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        ) as child:
            try:
                returncode = child.wait(timeout=90)
            finally:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
    assert returncode == 0, (directory / "process.log").read_text()[-8000:]
    report = json.loads((directory / "runtime-error.json").read_text())
    assert report["stage"] in ("launch", sync)
    assert "trap error" in report["error"] and any(
        code in report["error"] for code in ("507015", "507016", "507035", "507036")
    )
    assert report["elapsed_seconds"] < 15 and "910B" in report["device"]


@pytest.fixture(scope="module")
def v5_comparison(tmp_path_factory):
    directory = Path(
        os.environ.get("MLA_PAGED_COMPARISON_DIR", tmp_path_factory.mktemp("mla-v5-comparison"))
    )
    return build_consumers(directory / "consumer", used=True)


def make_v5_case(builder, dtype, item):
    # Comparison measures consumers only. Prepare the v5 record with the exact
    # CPU serializer before timing; no second AICPU module shares this process.
    from tests.mla.unit.used.test_flash_attn_npu_v3_mla_used_metadata import build
    from tests.mla.unit.used.test_flash_attn_npu_v3_mla_used_metadata_consumer import (
        context_for as v5_context,
    )
    from tests.mla.unit.used.test_flash_attn_npu_v3_mla_used_metadata_aicpu import (
        device_inputs as v5_inputs,
    )

    row = build(builder, [item], dtype)[0]
    assert row[5:8] == [0, 0, 0]
    context = v5_context(row, item, dtype)
    storage = torch.tensor(
        [SENTINEL, *[w if w < 2**31 else w - 2**32 for w in row], SENTINEL],
        dtype=torch.int32,
        device="npu",
    )
    return dict(
        context=context,
        expected=row,
        metadata_storage=storage,
        metadata=storage[1:-1],
        inputs=v5_inputs(item),
        buffers=[
            torch.full(
                (3 * context.block_dim * stride + 32,), SENTINEL, dtype=torch.int32, device="npu"
            )
            for stride in (16, 112, 16, 16)
        ],
    )


def test_validation_cost_vs_v5(paged_device, paged_probe, v5_comparison, used_builder):
    import statistics
    import time
    from tests.mla.unit.used.test_flash_attn_npu_v3_mla_used_metadata_device import (
        load_consumer as load_v5,
        launch as launch_v5,
        check as check_v5,
    )
    from tests.mla.unit.used.test_flash_attn_npu_v3_mla_used_metadata import used_case

    manifest = v5_comparison
    v5 = dict(lib=load_v5(manifest, paged_device["dtype"]), dtype=paged_device["dtype"])
    reports = []
    # Fixed before sampling: all representative P, BSND/TND, short/long KV,
    # single/multiple Q, and zero-length tails. Five warmups, 40 paired samples.
    for page in (16, 112, 128, 256, 528, 1024):
        for layout in (0, 2):
            q = [1, 1] if layout == 0 else [7, 0, 3] * 14
            capacity = page if layout == 0 else 16384 // page * page
            kv = [capacity, 0] if layout == 0 else [capacity, 0, min(513, capacity)] * 14
            uq = None if layout == 0 else [3, 0, 1] * 14
            item = paged_case(layout, q, kv, uq, page, capacity)
            new = make_case(paged_device, paged_probe, item)
            old_item = used_case(
                layout,
                q,
                [capacity] * len(q),
                uq,
                kv,
                **{
                    key: new["item"][key]
                    for key in ("aic", "ratio", "libapi", "l1", "ub", "l0a", "l0b", "l0c")
                },
            )
            old = make_v5_case(used_builder, paged_device["dtype"], old_item)
            samples = {"v5": [], "v6": []}
            device_us = {"v5": [], "v6": []}
            begin, end = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
            for i in range(45):
                for name in ("v5", "v6") if i % 2 else ("v6", "v5"):
                    torch.npu.synchronize()
                    start = time.perf_counter_ns()
                    begin.record()
                    if name == "v5":
                        launch_v5(v5, old)
                    else:
                        launch(paged_device, new)
                    end.record()
                    end.synchronize()
                    elapsed = (time.perf_counter_ns() - start) / 1000
                    if i >= 5:
                        samples[name].append(elapsed)
                        device_us[name].append(begin.elapsed_time(end) * 1000)
            check_v5(old)
            check(new)
            reports.append(
                dict(
                    case=item,
                    wall_us=samples,
                    device_event_us=device_us,
                    median_us={n: statistics.median(v) for n, v in samples.items()},
                    median_event_us={n: statistics.median(v) for n, v in device_us.items()},
                )
            )
    (paged_device["directory"] / f"comparison-{paged_device['dtype']}.json").write_text(
        json.dumps(
            dict(
                records=reports,
                cpu_affinity=sorted(os.sched_getaffinity(0)),
                scope="v6 PA vs v5 nonpaged consumer, matching logical inputs; v5 CPU record upload and v6 AICPU generation excluded; diagnostic writes/ctypes/stream/events/sync included; device event interval is not isolated kernel time; no attention or public E2E; fixed CANN has no matching validating consumer",
            ),
            indent=2,
        )
        + "\n"
    )


def test_context_compile_dtype_and_blockdim(paged_device, paged_probe):
    from tests.mla.unit.paged_bsnd.test_flash_attn_npu_v3_mla_paged_metadata_consumer import (
        CONTEXT_FAULTS,
    )

    case = make_case(
        paged_device, paged_probe, paged_case(2, [1, 1], [257, 257], [1, 1], 112, 448, heads=8)
    )
    for path, value, status in CONTEXT_FAULTS:
        c = Context.from_buffer_copy(case["context"])
        change(c, {path: value})
        launch(paged_device, case, checked=False, context=c, blocks=case["context"].block_dim)
        check(case, status)
    c = Context.from_buffer_copy(case["context"])
    c.dtype = 3 - c.dtype
    launch(paged_device, case, checked=False, context=c)
    check(case, 4)


def test_wide_query(paged_device, paged_probe):
    batch = 65538
    item = paged_case(0, [65536] * batch, [0] * batch, [0] * batch, 1024, 16384, heads=1)
    case = make_case(paged_device, paged_probe, item)
    case.update(make_cache(item, paged_device["dtype"], row_stride=0, pool=0))
    launch(paged_device, case)
    check(case)
    assert case["expected"][403] == case["expected"][-4] == 1
    case["metadata"][-4] = 0
    case["expected"][-4] = 0
    reset(case)
    launch(paged_device, case, checked=False)
    check(case, 12)
