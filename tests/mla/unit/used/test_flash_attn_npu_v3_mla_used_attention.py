"""v5 attention: physical allocations, independent effective source inputs and math."""

import itertools
import json
import os
import signal
import subprocess
import sys

import pytest
import torch
import torch_npu  # noqa: F401

from tests.mla.support.mla_metadata_error_probe import forbid_host_readback
from tests.mla.support.components.used_metadata_aicpu import used_probe as used_probe, enqueue
from tests.mla.support.components.used_metadata_consumer import (
    used_artifacts as used_artifacts,
    used_oracle as used_oracle,
    check_records,
    used_case,
)
from tests.mla.unit.used.test_flash_attn_npu_v3_mla_used_metadata_consumer import USED_CORRUPTIONS

from tests.mla.support.components.used_attention import (
    CASES as CASES,
    EMPTY_CASES as EMPTY_CASES,
    PERFORMANCE_CASES as PERFORMANCE_CASES,
    UsedSchedule as UsedSchedule,
    check_case as check_case,
    check_math as check_math,
    check_rejected as check_rejected,
    configure_producer as configure_producer,
    effective as effective,
    fault_case as fault_case,
    launch as launch,
    load_attention as load_attention,
    make_case as make_case,
    references as references,
    used_attention as used_attention,
    valid_outputs as valid_outputs,
)


@pytest.mark.parametrize("spec", CASES + EMPTY_CASES)
def test_accuracy(used_attention, used_probe, spec):
    check_case(used_attention, make_case(used_attention, used_probe, spec))


@pytest.mark.parametrize("layout,query_used", itertools.product(range(3), (0, 1)))
def test_global_empty_fills_allocation(used_attention, used_probe, layout, query_used):
    # The fixed FA3 host branch for a globally empty K allocation defines all
    # output slots, including query padding. Normal attention defines used rows.
    item = used_case(layout, [3, 3], [0, 0], [query_used] * 2, [0, 0], heads=8)
    case = make_case(used_attention, used_probe, item)
    check_case(used_attention, case)
    output, lse = [t.cpu() for t in case["outputs"]["migrated"]]
    assert (output == 0).all() and torch.isposinf(lse).all()


@pytest.mark.parametrize("layout,flags", itertools.product(range(3), range(4)))
def test_default_used_and_explicit_capacity(used_attention, used_probe, layout, flags):
    q, k = ([7, 7], [33, 33]) if layout == 0 else ([3, 0, 7], [33] * 3)
    item = used_case(layout, q, k, q if flags & 1 else None, k if flags & 2 else None, heads=129)
    check_case(used_attention, make_case(used_attention, used_probe, item))


@pytest.mark.parametrize("layout,scale", itertools.product(range(3), (0.0, -0.03125, 0.125)))
def test_scales_and_max_changes(used_attention, used_probe, layout, scale):
    check_case(
        used_attention,
        make_case(used_attention, used_probe, CASES[3 + layout], scale=scale, amplitude=3.0),
    )


def test_source_schedules(used_probe, used_oracle, tmp_path):
    probes = ({"dtype": torch.float16}, tmp_path)
    cases = [make_case(probes, used_probe, spec) for spec in CASES]
    check_records(
        [c["expected"] for c in cases],
        [c["item"] for c in cases],
        used_oracle,
        tmp_path,
        "attention",
        1,
    )


def test_corruption_precedes_attention(used_attention, used_probe):
    case = make_case(used_attention, used_probe, used_case(1, uq=[3, 1], uk=[17, 0], heads=8))
    good = case["metadata"].clone()
    for word, value, status in USED_CORRUPTIONS:
        case["metadata"].copy_(good)
        case["metadata"][word] = value if value < 2**31 else value - 2**32
        with forbid_host_readback():
            launch(used_attention, case, checked=False)
        check_rejected(case, status)


@pytest.mark.parametrize("axis", range(4))
def test_each_length_update(used_attention, used_probe, axis):
    case = make_case(
        used_attention, used_probe, used_case(1, [7, 7], [33, 33], [3, 1], [17, 0], heads=129)
    )
    # Same allocation totals; change one live input before metadata regeneration.
    case["inputs"][axis][1 if axis < 2 else 0] += 1
    launch(used_attention, case, checked=False)
    check_rejected(case, 12 if axis < 2 else 13)
    new = dict(case["item"])
    if axis < 2:
        name = ("q", "kv")[axis]
        new[name] = [new[name][0] + 1, new[name][1] - 1]
    else:
        name = ("uq", "uk")[axis - 2]
        new[name] = [new[name][0] + 1, new[name][1]]
    updated = make_case(used_attention, used_probe, new, worker=1)
    updated["metadata_storage"], updated["metadata"] = case["metadata_storage"], case["metadata"]
    updated["inputs"][axis] = case["inputs"][axis]
    with forbid_host_readback():
        enqueue(used_probe, updated["args"], updated["metadata_storage"], updated["inputs"], 1)
    check_case(used_attention, updated)


@pytest.mark.parametrize("offsets", [(0, 0, 0, 0), (16, 32, 48, 64)])
def test_offsets(used_attention, used_probe, offsets):
    for item in CASES[:3]:
        check_case(used_attention, make_case(used_attention, used_probe, item, offsets=offsets))


def test_alias_clone_and_device_schedule(used_attention, used_probe):
    case = make_case(
        used_attention, used_probe, used_case(1, [3, 1], [3, 1], [1, 1], [1, 1], heads=128)
    )
    case["inputs"][1] = case["inputs"][0]
    case["inputs"][3] = case["inputs"][2]
    case["metadata"] = case["metadata"].clone()
    check_case(used_attention, case)
    want = [t.clone() for t in case["outputs"]["migrated"]]
    case["schedule"] = UsedSchedule()  # Migrated execution must consume device v5 only.
    launch(used_attention, case)
    for a, b in zip(
        valid_outputs(case, [t.cpu() for t in case["outputs"]["migrated"]]),
        valid_outputs(case, [t.cpu() for t in want]),
    ):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_cross_stream_allocator_lifetime(used_attention, used_probe):
    producer, consumer = torch.npu.Stream(), torch.npu.Stream()
    pending = []
    for i in range(9):
        with torch.npu.stream(producer):
            case = make_case(used_attention, used_probe, CASES[i % 3], worker=i % 2)
            case["metadata"] = case["metadata"].clone()
            ready = torch.npu.Event()
            ready.record(producer)
        with torch.npu.stream(consumer):
            consumer.wait_event(ready)
            with forbid_host_readback():
                launch(used_attention, case)
                output = [t.clone() for t in case["outputs"]["migrated"]]
            reference = {key: case[key] for key in ("item", "host", "context", "scale")}
            pending.append((output, reference))
            del case
            for _ in range(4):
                torch.empty((32768,), dtype=torch.int32, device="npu").fill_(-1)
    for output, case in pending:
        check_math(valid_outputs(case, [t.cpu() for t in output]), references(case))


FAULTS = (
    "v4",
    "wide_start",
    "allocation",
    "flags",
    "stale_q",
    "stale_k",
    "stale_uq",
    "stale_uk",
    "count_uk",
    "align_uq",
    "workspace",
    "endpoint",
    "generation",
    "empty_q",
    "empty_kv",
    "empty_batch",
)


@pytest.mark.parametrize("fault", FAULTS)
def test_runtime_trap(used_attention, used_probe, fault):
    directory = used_attention[1] / ("trap-" + fault)
    directory.mkdir()
    torch.npu.synchronize()
    command = [
        sys.executable,
        "-m",
        "tests.mla.support.mla_used_attention_error_probe",
        str(used_attention[1]),
        str(used_probe["directory"] / "build-manifest.json"),
        fault,
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
    assert returncode == 0, (directory / "process.log").read_text()[-4000:]
    report = json.loads((directory / "runtime-error.json").read_text())
    assert "trap error" in report["error"] and report["elapsed_seconds"] < 15
    assert any(code in report["error"] for code in ("507015", "507016", "507035", "507036"))
