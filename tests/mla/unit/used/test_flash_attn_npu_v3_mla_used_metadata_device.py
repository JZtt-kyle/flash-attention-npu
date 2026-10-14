"""910B release v5 validation only: exact schedules, rejection, Trap and lifetime."""

import ctypes as ct
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

import pytest
import torch
import torch_npu  # noqa: F401

from tests.mla.support.mla_bsnd_consumer_build import build_consumers
from tests.mla.support.mla_metadata_build import ROOT, sha256
from tests.mla.support.mla_metadata_error_probe import forbid_host_readback
from tests.mla.unit.used.test_flash_attn_npu_v3_mla_used_metadata_aicpu import (
    used_probe as used_probe,
    arguments,
    device_inputs,
    enqueue,
    host_record,
    storage_for,
)
from tests.mla.unit.used.test_flash_attn_npu_v3_mla_used_metadata_consumer import (
    used_artifacts as used_artifacts,
    used_oracle as used_oracle,
    Context,
    Inputs,
    Array,
    ARRAY_NAMES,
    USED_CORRUPTIONS,
    SENTINEL,
    EMPTY_CASES,
    check_records,
    context_for,
    expected_schedule,
    metadata_cases,
    used_case,
    wide_case,
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


@pytest.fixture(scope="module")
def used_consumer_build(tmp_path_factory):
    directory = Path(
        os.environ.get("MLA_USED_CONSUMER_DEVICE_DIR", tmp_path_factory.mktemp("mla-used-device"))
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = os.environ.get("MLA_USED_CONSUMER_MANIFEST")
    manifest = json.loads(Path(path).read_text()) if path else build_consumers(directory, used=True)
    (directory / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return directory, manifest


@pytest.fixture(scope="module", params=(1, 2))
def used_device(used_consumer_build, request):
    directory, manifest = used_consumer_build
    return dict(
        lib=load_consumer(manifest, request.param), dtype=request.param, directory=directory
    )


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


def reset(case):
    for buffer in case["buffers"]:
        buffer.fill_(SENTINEL)


@pytest.mark.parametrize("layout", range(3))
def test_aicpu_source_differential(used_device, used_probe, used_oracle, layout):
    items = metadata_cases(layout, used_probe["platform"].aic, 2)
    rows, actual_items = [], []
    for item in items:
        case = make_case(used_device, used_probe, item)
        with forbid_host_readback():
            launch(used_device, case)
        check(case)
        rows.append(case["expected"])
        actual_items.append(case["item"])
    check_records(
        rows,
        actual_items,
        used_oracle,
        used_device["directory"],
        f"source-{used_device['dtype']}-{layout}",
        used_device["dtype"],
    )


def test_u01_t01_presence_empty_reuse(used_device, used_probe):
    items = EMPTY_CASES + [
        used_case(uq=[3, 1], uk=[17, 0]),
        used_case(1, [3, 0, 1], [17, 0, 33], [2, 0, 1], [7, 0, 17]),
        used_case(2, [3, 0, 1], [64] * 3, [2, 0, 1], [17, 0, 33]),
    ]
    for item in items:
        case = make_case(used_device, used_probe, item)
        for checked in (False, True, True):
            with forbid_host_readback():
                launch(used_device, case, checked=checked)
            check(case)
        if item["uq"] is not None:
            reset(case)
            launch(
                used_device, case, checked=False, descriptors={"used_q": {"address": 0, "count": 0}}
            )
            check(case, 13)


def test_corrupt_records_and_descriptors_do_not_write(used_device, used_probe):
    case = make_case(
        used_device, used_probe, used_case(1, [1, 1], [257, 257], [1, 1], [257, 257], heads=8)
    )
    good = case["metadata"].clone()
    for word, value, status in USED_CORRUPTIONS:
        case["metadata"].copy_(good)
        case["metadata"][word] = value if value < 2**31 else value - 2**32
        altered = case["expected"].copy()
        altered[word] = value
        with forbid_host_readback():
            launch(used_device, case, checked=False)
        check(case, status, record=altered)
    case["metadata"].copy_(good)
    for name in ARRAY_NAMES:
        for fields in (
            {"address": 0},
            {"address": 1},
            {"address": 2**64 - 4},
            {"count": 0},
            {"count": 2**64 - 1},
        ):
            launch(used_device, case, checked=False, descriptors={name: fields})
            check(case, 12 if name.startswith("cu_") else 13)
    for options, status in [
        ({"words": 401}, 1),
        ({"address": 0}, 1),
        ({"address": 1}, 1),
        ({"address": 2**64 - 4}, 1),
    ]:
        launch(used_device, case, checked=False, **options)
        check(case, status)
    c = Context.from_buffer_copy(case["context"])
    c.dtype = 3 - used_device["dtype"]
    launch(used_device, case, checked=False, context=c)
    check(case, 4)
    c = Context.from_buffer_copy(case["context"])
    c.block_dim += 1
    launch(used_device, case, checked=False, context=c, blocks=case["context"].block_dim)
    check(case, 6)


def test_each_current_array_update_and_regenerate(used_device, used_probe):
    item = used_case(1, [7] * 3, [513] * 3, [3, 0, 1], [0, 17, 513])
    case = make_case(used_device, used_probe, item)
    for axis in range(4):
        for b in range(case["inputs"][axis].numel()):
            changed = case["inputs"][axis].clone()
            case["inputs"][axis][b] = -1
            with forbid_host_readback():
                launch(used_device, case, checked=False)
            check(case, 12 if axis < 2 else 13)
            case["inputs"][axis].copy_(changed)
    # Change all four arrays without changing total allocation or used presence.
    changed = dict(
        case["item"], q=[6, 1, 14], kv=[512, 1, 1026], sq=14, sk=1026, uq=[1, 0, 0], uk=[0, 1, 0]
    )
    args = arguments(changed, used_device["dtype"])
    arrays = device_inputs(changed)
    for old, new in zip(case["inputs"], arrays):
        old.copy_(new)
    with forbid_host_readback():
        launch(used_device, case, checked=False)
    check(case, 12)
    case["args"] = args
    case["expected"] = host_record(used_probe, changed, used_device["dtype"])
    case["context"] = context_for(case["expected"], changed, used_device["dtype"])
    # Allocate diagnostic buffers for the new block count.
    case["buffers"] = [
        torch.full(
            (3 * case["context"].block_dim * stride + 32,),
            SENTINEL,
            dtype=torch.int32,
            device="npu",
        )
        for stride in (16, 112, 16, 16)
    ]
    with forbid_host_readback():
        enqueue(used_probe, args, case["metadata_storage"], case["inputs"])
        launch(used_device, case)
    check(case)


def test_cross_stream_alias_clone_allocator_lifetime(used_device, used_probe):
    producer, consumer = torch.npu.Stream(), torch.npu.Stream()
    pending = []
    for i in range(16):
        with torch.npu.stream(producer):
            case = make_case(
                used_device,
                used_probe,
                used_case(1, [1, 3, 4], [1, 3, 4], [1, 3, 4], [1, 3, 4]),
                worker=i % 2,
            )
            backing = torch.tensor(
                [SENTINEL, 0, 1, 3, 4, 8, SENTINEL], device="npu", dtype=torch.int32
            )
            # Prefix [0,1,4,8] is a different guarded allocation; both axes alias.
            prefix = case["inputs"][0].clone()
            used = backing[2:5]
            case["inputs"] = [prefix, prefix, used, used]
            copied = case["metadata_storage"].clone()
            case["metadata_storage"], case["metadata"] = copied, copied[1:-1]
            ready = torch.npu.Event()
            ready.record(producer)
        with torch.npu.stream(consumer):
            consumer.wait_event(ready)
            with forbid_host_readback():
                launch(used_device, case)
                copies = [t.clone() for t in case["buffers"]]
                metadata_copy = case["metadata_storage"].clone()
            pending.append(
                dict(
                    context=case["context"],
                    expected=case["expected"],
                    buffers=copies,
                    metadata_storage=metadata_copy,
                )
            )
            del case, copied, prefix, used, backing
            torch.empty((32768,), device="npu", dtype=torch.int32).fill_(SENTINEL)
    for case in pending:
        check(case)


@pytest.mark.parametrize("layout", (0, 2))
def test_wide_physical_offsets(used_device, used_probe, layout):
    case = make_case(used_device, used_probe, wide_case(layout))
    with forbid_host_readback():
        launch(used_device, case)
    check(case)
    word = -7 if layout == 0 else -5
    assert case["expected"][word] > 0
    case["metadata"][word] = 0
    case["expected"][word] = 0
    reset(case)
    launch(used_device, case, checked=False)
    check(case, 12)


FAULTS = (
    "v1",
    "v2",
    "v3",
    "v4",
    "layout",
    "flags",
    "span",
    "wide_start",
    "allocation",
    "words",
    "stale_q",
    "stale_k",
    "stale_uq",
    "stale_uk",
    "missing_q",
    "count_uk",
    "align_uq",
    "suffix",
    "variable_m",
    "workspace",
    "generation",
    "empty_stale",
    "empty_batch",
)


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


@pytest.mark.parametrize("fault", FAULTS)
def test_runtime_trap(used_device, used_probe, fault):
    directory = used_device["directory"] / f"error-{used_device['dtype']}-{fault}"
    directory.mkdir()
    torch.npu.synchronize()
    sync = {"stale_uq": "event", "stale_uk": "stream"}.get(fault, "device")
    command = [
        sys.executable,
        "-m",
        "tests.mla.support.mla_used_metadata_error_probe",
        str(used_device["directory"] / "build-manifest.json"),
        str(used_probe["directory"] / "build-manifest.json"),
        str(used_device["dtype"]),
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
    assert returncode == 0, (directory / "process.log").read_text()
    report = json.loads((directory / "runtime-error.json").read_text())
    assert report["stage"] in ("launch", sync)
    assert any(code in report["error"] for code in ("507015", "507016", "507035", "507036"))
    assert "trap error" in report["error"] and report["elapsed_seconds"] < 15
    assert "910B" in report["device"]


def test_validation_diagnostics(used_device, used_probe):
    reports = []
    for item in [
        used_case(uq=[3, 1], uk=[17, 0]),
        used_case(1, [3, 0, 1], [17, 0, 33], [2, 0, 1], [7, 0, 17]),
        used_case(2, [3, 0, 1] * 14, [16384] * 42, [2, 0, 1] * 14, [16384, 0, 513] * 14),
        used_case(1, [0, 0], [0, 0], [0, 0], [0, 0]),
    ]:
        case = make_case(used_device, used_probe, item)
        samples = []
        for i in range(25):
            torch.npu.synchronize()
            start = time.perf_counter_ns()
            launch(used_device, case)
            torch.npu.synchronize()
            if i >= 5:
                samples.append((time.perf_counter_ns() - start) / 1000)
        check(case)
        reports.append(dict(item=item, wall_us=samples, median_us=statistics.median(samples)))
    (used_device["directory"] / f"timing-{used_device['dtype']}.json").write_text(
        json.dumps(
            dict(
                records=reports,
                cpu_affinity=sorted(os.sched_getaffinity(0)),
                scope="v5 validation/diagnostic writes/ctypes/stream registration/sync; no attention or public E2E",
            ),
            indent=2,
        )
        + "\n"
    )


@pytest.fixture(scope="module")
def v4_comparison(tmp_path_factory):
    from tests.mla.support.mla_metadata_build import build_probe
    from tests.mla.unit.dense.test_flash_attn_npu_v3_mla_metadata import load_probe
    from tests.mla.unit.mixed.test_flash_attn_npu_v3_mla_tnd_bsnd_metadata_aicpu import MixedArgs

    directory = Path(
        os.environ.get("MLA_USED_COMPARISON_DIR", tmp_path_factory.mktemp("mla-v4-comparison"))
    )
    manifest = build_consumers(directory / "consumer", mixed=True)
    with load_probe(
        build_probe(directory / "producer", mixed=True), directory / "producer"
    ) as probe:
        probe["lib"].launch_mixed_metadata.argtypes = [ct.c_void_p] * 4 + [ct.POINTER(MixedArgs)]
        probe["lib"].launch_mixed_metadata.restype = ct.c_int
        probe["lib"].mixed_metadata_host.argtypes = [ct.POINTER(MixedArgs), ct.c_void_p]
        probe["lib"].mixed_metadata_host.restype = ct.c_uint32
        yield manifest, probe


def test_validation_cost_vs_v4(used_device, used_probe, v4_comparison):
    from tests.mla.unit.mixed.test_flash_attn_npu_v3_mla_tnd_bsnd_metadata_device import (
        load_consumer as load_v4,
        make_case as make_v4,
        launch as launch_v4,
        check as check_v4,
    )

    manifest, producer = v4_comparison
    v4 = dict(lib=load_v4(manifest, used_device["dtype"]), dtype=used_device["dtype"])
    reports = []
    # Full Q and explicit KV-used have matching logical inputs in v4 and v5.
    # Both probes preallocate outputs, then validate and export diagnostic state.
    for q, k, capacity in [
        ([1, 1], [257, 257], 257),
        ([3, 0, 1], [17, 0, 33], 64),
        ([3, 0, 1] * 14, [16384, 0, 513] * 14, 16384),
        ([0, 0], [0, 0], 16384),
    ]:
        old = make_v4(v4, producer, q, k, capacity=capacity)
        new = make_case(used_device, used_probe, used_case(2, q, [capacity] * len(q), None, k))
        samples = {"v4": [], "v5": []}
        for i in range(45):
            order = ("v4", "v5") if i % 2 else ("v5", "v4")
            for name in order:
                torch.npu.synchronize()
                start = time.perf_counter_ns()
                if name == "v4":
                    launch_v4(v4, old)
                else:
                    launch(used_device, new)
                torch.npu.synchronize()
                if i >= 5:
                    samples[name].append((time.perf_counter_ns() - start) / 1000)
        check_v4(old)
        check(new)
        reports.append(
            dict(
                q=q,
                kv=k,
                capacity=capacity,
                wall_us=samples,
                median_us={n: statistics.median(v) for n, v in samples.items()},
            )
        )
    (used_device["directory"] / f"comparison-{used_device['dtype']}.json").write_text(
        json.dumps(
            dict(
                records=reports,
                cpu_affinity=sorted(os.sched_getaffinity(0)),
                scope="v5 vs v4 validation and diagnostic writes/ctypes/stream/sync; extra v5 allocation and used checks; no attention/E2E",
            ),
            indent=2,
        )
        + "\n"
    )
