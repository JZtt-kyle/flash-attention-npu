"""910B release validation of compact TND records, independently of attention."""

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
from tests.mla.support.mla_metadata_error_probe import forbid_host_readback
from tests.mla.support.components.tnd_metadata_aicpu import (
    tnd_probe as tnd_probe,
    enqueue,
    host_record,
)
from tests.mla.unit.tnd.test_flash_attn_npu_v3_mla_tnd_metadata_consumer import (
    tnd_artifacts as tnd_artifacts,
    tnd_oracle as tnd_oracle,
    TND_CORRUPTIONS,
    check_records,
    metadata_cases,
)
from tests.mla.support.components.tnd_metadata_consumer import Context, SENTINEL, prefixes, tnd_case

from tests.mla.support.components.tnd_metadata_device import (
    check as check,
    launch as launch,
    load_consumer as load_consumer,
    make_case as make_case,
)


@pytest.fixture(scope="module")
def tnd_consumer_build(tmp_path_factory):
    directory = Path(
        os.environ.get("MLA_TND_CONSUMER_DEVICE_DIR", tmp_path_factory.mktemp("mla-tnd-device"))
    )
    directory.mkdir(parents=True, exist_ok=True)
    path = os.environ.get("MLA_TND_CONSUMER_MANIFEST")
    manifest = json.loads(Path(path).read_text()) if path else build_consumers(directory, tnd=True)
    (directory / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return directory, manifest


@pytest.fixture(scope="module", params=(1, 2))
def tnd_device(tnd_consumer_build, request):
    directory, manifest = tnd_consumer_build
    return dict(
        lib=load_consumer(manifest, request.param), dtype=request.param, directory=directory
    )


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
def test_valid_reuse_and_empty(tnd_device, tnd_probe, q, k, heads):
    case = make_case(tnd_device, tnd_probe, q, k, heads=heads, sq=512, sk=16384)
    for checked in (False, True, True):
        with forbid_host_readback():
            launch(tnd_device, case, checked=checked)
        check(case)
    cq, ck = prefixes(case["values"])
    assert case["query"]._base.cpu().tolist() == [SENTINEL, *cq, SENTINEL]
    assert case["kv"]._base.cpu().tolist() == [SENTINEL, *ck, SENTINEL]


def test_aicpu_consumer_source_differential(tnd_device, tnd_probe, tnd_oracle):
    cases = metadata_cases(tnd_probe["platform"].aic, 2)
    rows = []
    for values in cases:
        case = make_case(tnd_device, tnd_probe, values=values)
        with forbid_host_readback():
            launch(tnd_device, case)
        check(case)
        rows.append(case["expected"])
    check_records(
        rows,
        cases,
        tnd_oracle,
        tnd_device["directory"],
        f"source-{tnd_device['dtype']}",
        tnd_device["dtype"],
    )


def test_corruption_and_descriptors_do_not_write(tnd_device, tnd_probe):
    case = make_case(tnd_device, tnd_probe, [1, 1], [257, 257], heads=8)
    good = case["metadata"].clone()
    for word, value, status in TND_CORRUPTIONS:
        case["metadata"].copy_(good)
        case["metadata"][word] = value if value < 2**31 else value - 2**32
        altered = case["expected"].copy()
        altered[word] = value
        with forbid_host_readback():
            launch(tnd_device, case, checked=False)
        check(case, status, record=altered)
    case["metadata"].copy_(good)
    for options, status in [
        (dict(address=0), 1),
        (dict(address=case["metadata"].data_ptr() + 2), 1),
        (dict(address=2**64 - 4), 1),
        (dict(words=401), 1),
        (dict(words=case["metadata"].numel() + 1), 1),
        (dict(qaddress=0), 12),
        (dict(kaddress=0), 12),
        (dict(qaddress=case["query"].data_ptr() + 2), 12),
        (dict(kaddress=case["kv"].data_ptr() + 2), 12),
        (dict(qcount=1), 12),
        (dict(kcount=2**64 - 1), 12),
    ]:
        launch(tnd_device, case, checked=False, **options)
        check(case, status)
    launch(tnd_device, case)
    check(case)


def test_each_prefix_stale_and_regeneration(tnd_device, tnd_probe):
    b = 2 * tnd_probe["platform"].aic + 1
    shape = dict(sq=512, sk=16384, block_dim=tnd_probe["platform"].aic)
    case = make_case(tnd_device, tnd_probe, [3] * b, [513] * b, **shape)
    for tensor in (case["query"], case["kv"]):
        good = tensor.clone()
        for i in range(b + 1):
            tensor[i] += 1
            launch(tnd_device, case, checked=False)
            check(case, 12)
            tensor.copy_(good)
    # Redistribute tokens while keeping totals, max bounds and launch budget.
    values = tnd_case(
        [3] * b,
        [513] * b,
        cq=[0, 1, *prefixes(case["values"])[0][2:]],
        ck=[0, 7, *prefixes(case["values"])[1][2:]],
        **shape,
    )
    expected = host_record(tnd_probe, values, tnd_device["dtype"])
    with forbid_host_readback():
        case["query"][1], case["kv"][1] = 1, 7
        enqueue(tnd_probe, case["args"], case["metadata_storage"], case["query"], case["kv"])
        launch(tnd_device, case)
    case["expected"] = expected
    check(case)


def test_context_and_zero_cost_suffix(tnd_device, tnd_probe):
    case = make_case(tnd_device, tnd_probe, [3, 0, 1, 7, 0], [513, 7, 1, 0, 16384], heads=171)
    c = case["context"]
    for field, value, status in [
        ("dtype", 3 - tnd_device["dtype"], 4),
        ("batch", c.batch + 1, 1),
        ("total_q", c.total_q + 1, 5),
        ("total_k", c.total_k + 1, 5),
        ("max_q", c.max_q + 1, 5),
        ("max_k", 16385, 5),
        ("heads", 0, 5),
        ("aic", 0, 6),
        ("aiv", c.aic, 6),
        ("block_dim", c.block_dim + 1, 6),
        ("workspace", c.workspace - 1, 9),
        ("libapi", 2**64 - 1, 9),
    ]:
        changed = Context.from_buffer_copy(c)
        setattr(changed, field, value)
        launch(tnd_device, case, checked=False, context=changed, blocks=c.block_dim)
        check(case, status)
    for name, _ in c.memory._fields_:
        changed = Context.from_buffer_copy(c)
        setattr(changed.memory, name, 0)
        launch(tnd_device, case, checked=False, context=changed)
        check(case, 10)
    for batch, m in [(1, 0), (2, 1), (5, 1)]:
        for word, value in ((20, 1), (23, batch), (49, m), (75, 0)):
            case["metadata"][word] = value
            case["expected"][word] = value
        launch(tnd_device, case, checked=False)
        check(case, 8)


def test_cross_stream_alias_clone_and_allocator_lifetime(tnd_device, tnd_probe):
    producer, consumer = torch.npu.Stream(), torch.npu.Stream()
    pending = []
    for i in range(16):
        with torch.npu.stream(producer):
            case = make_case(tnd_device, tnd_probe, [3, 0, 1], [3, 0, 1], worker=i % 2)
            copied = case["metadata_storage"].clone()
            case["metadata_storage"], case["metadata"] = copied, copied[1:-1]
            case["query"] = case["query"].clone()
            case["kv"] = case["query"]  # Legal read-only input alias.
            ready = torch.npu.Event()
            ready.record(producer)
        with torch.npu.stream(consumer):
            consumer.wait_event(ready)
            with forbid_host_readback():
                launch(tnd_device, case)
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
def test_runtime_trap(tnd_device, tnd_probe, fault):
    directory = tnd_device["directory"] / f"error-{tnd_device['dtype']}-{fault}"
    directory.mkdir()
    torch.npu.synchronize()
    sync = {"stale_q": "event", "stale_k": "stream"}.get(fault, "device")
    command = [
        sys.executable,
        "-m",
        "tests.mla.support.mla_tnd_metadata_error_probe",
        str(tnd_device["directory"] / "build-manifest.json"),
        str(tnd_probe["directory"] / "build-manifest.json"),
        str(tnd_device["dtype"]),
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


def test_validation_diagnostics(tnd_device, tnd_probe):
    reports = []
    for q, k in [
        ([1, 1], [257, 257]),
        ([3, 0, 1], [17, 0, 33]),
        ([3, 0, 1] * 14, [16384, 0, 513] * 14),
        ([0, 0], [0, 0]),
    ]:
        case = make_case(tnd_device, tnd_probe, q, k)
        samples = []
        for iteration in range(25):
            torch.npu.synchronize()
            start = time.perf_counter_ns()
            launch(tnd_device, case)
            torch.npu.synchronize()
            if iteration >= 5:
                samples.append((time.perf_counter_ns() - start) / 1000)
        check(case)
        reports.append(dict(query=q, kv=k, wall_us=samples, median_us=statistics.median(samples)))
    (tnd_device["directory"] / f"timing-{tnd_device['dtype']}.json").write_text(
        json.dumps(
            dict(
                records=reports,
                cpu_affinity=sorted(os.sched_getaffinity(0)),
                scope="v3 validation plus diagnostic writes/ctypes/stream registration/sync; no attention or public E2E",
            ),
            indent=2,
        )
        + "\n"
    )
