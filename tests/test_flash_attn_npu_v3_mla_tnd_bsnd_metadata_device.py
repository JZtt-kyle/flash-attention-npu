"""910B release validation of mixed TND-Q/BSND-KV records, independently of attention."""

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

from tests.mla_bsnd_consumer_build import build_consumers
from tests.mla_metadata_build import ROOT, sha256
from tests.mla_metadata_error_probe import forbid_host_readback
from tests.test_flash_attn_npu_v3_mla_tnd_bsnd_metadata_aicpu import (
    mixed_probe as mixed_probe,
    arguments,
    device_inputs,
    enqueue,
    host_record,
    storage_for,
)
from tests.test_flash_attn_npu_v3_mla_tnd_bsnd_metadata_consumer import (
    mixed_artifacts as mixed_artifacts,
    mixed_oracle as mixed_oracle,
    Context,
    MIXED_CORRUPTIONS,
    SENTINEL,
    check_records,
    context_for,
    expected_schedule,
    metadata_cases,
    prefixes,
    mixed_case,
)


def load_consumer(manifest, dtype):
    assert manifest["ndebug"]
    for name, digest in manifest["sources"].items():
        assert sha256(ROOT / name) == digest, name
    row = manifest["libraries"]["fp16" if dtype == 1 else "bf16"]
    assert sha256(row["path"]) == row["sha256"]
    lib = ct.CDLL(row["path"])
    lib.launch_mixed_consumer.argtypes = [
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
    lib.launch_mixed_consumer.restype = ct.c_int
    return lib


@pytest.fixture(scope="module")
def mixed_consumer_build(tmp_path_factory):
    directory = Path(
        os.environ.get(
            "MLA_TND_BSND_CONSUMER_DEVICE_DIR", tmp_path_factory.mktemp("mla-tnd-device")
        )
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = os.environ.get("MLA_TND_BSND_CONSUMER_MANIFEST")
    manifest = (
        json.loads(Path(path).read_text()) if path else build_consumers(directory, mixed=True)
    )
    (directory / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return directory, manifest


@pytest.fixture(scope="module", params=(1, 2))
def mixed_device(mixed_consumer_build, request):
    directory, manifest = mixed_consumer_build
    return dict(
        lib=load_consumer(manifest, request.param), dtype=request.param, directory=directory
    )


def make_case(consumer, producer, q=(3, 0, 1), k=(17, 0, 33), *, values=None, worker=0, **shape):
    if values is None:
        p = producer["platform"]
        values = mixed_case(
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
        if tensor is not None:
            tensor.record_stream(stream)
    assert (
        consumer["lib"].launch_mixed_consumer(
            stream.npu_stream,
            case["metadata"].data_ptr() if address is None else address,
            case["metadata"].numel() if words is None else words,
            ct.byref(c),
            case["query"].data_ptr() if qaddress is None else qaddress,
            case["query"].numel() if qcount is None else qcount,
            (0 if case["kv"] is None else case["kv"].data_ptr()) if kaddress is None else kaddress,
            (0 if case["kv"] is None else case["kv"].numel()) if kcount is None else kcount,
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
                    expected[16 + core * 96 : 16 + core * 96 + 92] = decoded
            else:
                expected[16:-16:16] = 1
        assert torch.equal(actual, expected), (i, status, actual.tolist())
    actual_record = [w & 0xFFFFFFFF for w in case["metadata_storage"].cpu().tolist()]
    assert actual_record == [SENTINEL, *(case["expected"] if record is None else record), SENTINEL]


@pytest.mark.parametrize(
    "q,k,heads",
    [
        ([3, 0, 1], [17, 0, 33], 128),
        ([0, 1, 257, 0, 3], [513, 0, 8193, 16384, 0], 171),
        ([2] * 10, [511] * 10, 128),
        ([2] * 10, [511] + [0] * 9, 128),
        ([2] * 10, [511] + [0] * 9, 129),
        ([2] * 10, [511] + [0] * 9, 255),
        ([3, 0] * 21, [0, 16384] * 21, 171),
        ([], [], 128),
        ([0, 0], [0, 513], 128),
        ([3, 0, 1], [0, 0, 0], 128),
    ],
)
def test_valid_reuse_and_empty(mixed_device, mixed_probe, q, k, heads):
    case = make_case(mixed_device, mixed_probe, q, k, heads=heads, sq=512, capacity=16384)
    for checked in (False, True, True):
        with forbid_host_readback():
            launch(mixed_device, case, checked=checked)
        check(case)
    cq, ck = prefixes(case["values"])
    assert case["query"]._base.cpu().tolist() == [SENTINEL, *cq, SENTINEL]
    assert case["kv"]._base.cpu().tolist() == [SENTINEL, *ck, SENTINEL]


def test_aicpu_consumer_source_differential(mixed_device, mixed_probe, mixed_oracle):
    cases = metadata_cases(mixed_probe["platform"].aic, 2)
    rows = []
    for values in cases:
        case = make_case(mixed_device, mixed_probe, values=values)
        with forbid_host_readback():
            launch(mixed_device, case)
        check(case)
        rows.append(case["expected"])
    check_records(
        rows,
        cases,
        mixed_oracle,
        mixed_device["directory"],
        f"source-{mixed_device['dtype']}",
        mixed_device["dtype"],
    )


def test_corruption_and_descriptors_do_not_write(mixed_device, mixed_probe):
    case = make_case(mixed_device, mixed_probe, [1, 1], [257, 257], heads=8, capacity=257)
    good = case["metadata"].clone()
    for word, value, status in MIXED_CORRUPTIONS:
        case["metadata"].copy_(good)
        case["metadata"][word] = value if value < 2**31 else value - 2**32
        altered = case["expected"].copy()
        altered[word] = value
        with forbid_host_readback():
            launch(mixed_device, case, checked=False)
        check(case, status, record=altered)
    case["metadata"].copy_(good)
    for options, status in [
        (dict(address=0), 1),
        (dict(address=case["metadata"].data_ptr() + 2), 1),
        (dict(address=2**64 - 4), 1),
        (dict(words=401), 1),
        (dict(words=case["metadata"].numel() + 1), 1),
        (dict(qaddress=0), 12),
        (dict(kaddress=0), 11),
        (dict(qaddress=case["query"].data_ptr() + 2), 12),
        (dict(kaddress=case["kv"].data_ptr() + 2), 11),
        (dict(qcount=1), 12),
        (dict(kcount=2**64 - 1), 11),
    ]:
        launch(mixed_device, case, checked=False, **options)
        check(case, status)
    launch(mixed_device, case)
    check(case)


def test_each_prefix_stale_and_regeneration(mixed_device, mixed_probe):
    b = 2 * mixed_probe["platform"].aic + 1
    shape = dict(sq=512, capacity=16384, block_dim=mixed_probe["platform"].aic)
    case = make_case(mixed_device, mixed_probe, [3] * b, [513] * b, **shape)
    for axis, tensor in enumerate((case["query"], case["kv"])):
        good = tensor.clone()
        for i in range(tensor.numel()):
            tensor[i] += 1
            launch(mixed_device, case, checked=False)
            check(case, 12 if axis == 0 else 11)
            tensor.copy_(good)
    # Redistribute tokens while keeping totals, max bounds and launch budget.
    values = mixed_case(
        [3] * b,
        cq=[0, 1, *prefixes(case["values"])[0][2:]],
        klens=[513, 7, *prefixes(case["values"])[1][2:]],
        **shape,
    )
    expected = host_record(mixed_probe, values, mixed_device["dtype"])
    with forbid_host_readback():
        case["query"][1], case["kv"][1] = 1, 7
        enqueue(mixed_probe, case["args"], case["metadata_storage"], case["query"], case["kv"])
        launch(mixed_device, case)
    case["expected"] = expected
    check(case)


def test_context_and_zero_cost_suffix(mixed_device, mixed_probe):
    case = make_case(
        mixed_device, mixed_probe, [3, 0, 1, 7, 0], [513, 7, 1, 0, 16384], heads=171, capacity=16384
    )
    c = case["context"]
    for field, value, status in [
        ("dtype", 3 - mixed_device["dtype"], 4),
        ("batch", c.batch + 1, 1),
        ("total_q", c.total_q + 1, 5),
        ("max_q", c.max_q + 1, 5),
        ("capacity", 16385, 5),
        ("heads", 0, 5),
        ("aic", 0, 6),
        ("aiv", c.aic, 6),
        ("block_dim", c.block_dim + 1, 6),
        ("workspace", c.workspace - 1, 9),
        ("libapi", 2**64 - 1, 9),
    ]:
        changed = Context.from_buffer_copy(c)
        setattr(changed, field, value)
        launch(mixed_device, case, checked=False, context=changed, blocks=c.block_dim)
        check(case, status)
    for name, _ in c.memory._fields_:
        changed = Context.from_buffer_copy(c)
        setattr(changed.memory, name, 0)
        launch(mixed_device, case, checked=False, context=changed)
        check(case, 10)
    for batch, m in [(1, 0), (2, 1), (5, 1)]:
        for word, value in ((20, 1), (23, batch), (49, m), (75, 0)):
            case["metadata"][word] = value
            case["expected"][word] = value
        launch(mixed_device, case, checked=False)
        check(case, 8)


def test_cross_stream_alias_clone_and_allocator_lifetime(mixed_device, mixed_probe):
    producer, consumer = torch.npu.Stream(), torch.npu.Stream()
    pending = []
    for i in range(16):
        with torch.npu.stream(producer):
            case = make_case(mixed_device, mixed_probe, [3, 0, 1], [3, 3, 4], worker=i % 2)
            copied = case["metadata_storage"].clone()
            case["metadata_storage"], case["metadata"] = copied, copied[1:-1]
            case["query"] = case["query"].clone()
            case["kv"] = case["query"][1:]  # Legal overlapping views with different counts.
            ready = torch.npu.Event()
            ready.record(producer)
        with torch.npu.stream(consumer):
            consumer.wait_event(ready)
            with forbid_host_readback():
                launch(mixed_device, case)
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
            del case, copied
            torch.empty((32768,), device="npu", dtype=torch.int32).fill_(SENTINEL)
    for case in pending:
        check(case)


FAULTS = (
    "v1",
    "v2",
    "v3",
    "wide_start",
    "capacity",
    "words",
    "span",
    "stale_q",
    "stale_k",
    "missing_q",
    "count_k",
    "suffix",
    "variable_m",
    "workspace",
    "generation",
    "empty_stale",
    "empty_batch",
)


@pytest.mark.parametrize("fault", FAULTS)
def test_runtime_trap(mixed_device, mixed_probe, fault):
    directory = mixed_device["directory"] / f"error-{mixed_device['dtype']}-{fault}"
    directory.mkdir()
    torch.npu.synchronize()
    sync = {"stale_q": "event", "stale_k": "stream"}.get(fault, "device")
    command = [
        sys.executable,
        "-m",
        "tests.mla_tnd_bsnd_metadata_error_probe",
        str(mixed_device["directory"] / "build-manifest.json"),
        str(mixed_probe["directory"] / "build-manifest.json"),
        str(mixed_device["dtype"]),
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


def test_validation_diagnostics(mixed_device, mixed_probe):
    reports = []
    for q, k in [
        ([1, 1], [257, 257]),
        ([3, 0, 1], [17, 0, 33]),
        ([3, 0, 1] * 14, [16384, 0, 513] * 14),
        ([0, 0], [0, 0]),
    ]:
        case = make_case(mixed_device, mixed_probe, q, k, capacity=16384)
        samples = []
        for iteration in range(25):
            torch.npu.synchronize()
            start = time.perf_counter_ns()
            launch(mixed_device, case)
            torch.npu.synchronize()
            if iteration >= 5:
                samples.append((time.perf_counter_ns() - start) / 1000)
        check(case)
        reports.append(dict(query=q, kv=k, wall_us=samples, median_us=statistics.median(samples)))
    (mixed_device["directory"] / f"timing-{mixed_device['dtype']}.json").write_text(
        json.dumps(
            dict(
                records=reports,
                cpu_affinity=sorted(os.sched_getaffinity(0)),
                scope="v4 validation plus diagnostic writes/ctypes/stream registration/sync; no attention or public E2E",
            ),
            indent=2,
        )
        + "\n"
    )


@pytest.mark.parametrize("q,k,capacity", [([], [], 16384), ([3, 0, 1], [0, 0, 0], 0)])
def test_empty_capacity_and_null_kv(mixed_device, mixed_probe, q, k, capacity):
    case = make_case(mixed_device, mixed_probe, q, k, capacity=capacity, has_k=bool(k))
    launch(mixed_device, case)
    check(case)
    case["query"][-1] += 1
    for buffer in case["buffers"]:
        buffer.fill_(SENTINEL)
    launch(mixed_device, case, checked=False)
    check(case, 12)


def test_wide_physical_offsets(mixed_device, mixed_probe):
    batch = 2**18 + 2
    case = make_case(mixed_device, mixed_probe, [0] * batch, [0] * batch, capacity=16384)
    with forbid_host_readback():
        launch(mixed_device, case)
    check(case)
    assert expected_schedule(case["expected"])[-2:] == [32768, 1]
    # Dropping only the high word of the last physical start must be rejected.
    case["metadata"][-3] = 0
    case["expected"][-3] = 0
    for buffer in case["buffers"]:
        buffer.fill_(SENTINEL)
    launch(mixed_device, case, checked=False)
    check(case, 11)
    (mixed_device["directory"] / f"wide-{mixed_device['dtype']}.json").write_text(
        json.dumps(
            dict(
                batch=batch,
                capacity=batch * 16384,
                last_start=(batch - 1) * 16384,
                truncated_start_status=11,
            )
        )
        + "\n"
    )
