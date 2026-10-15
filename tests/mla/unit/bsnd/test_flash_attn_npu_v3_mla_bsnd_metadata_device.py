"""910B v2 validation only: no short-length attention or public dispatch."""

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
from tests.mla.support.mla_metadata_error_probe import forbid_host_readback
from tests.mla.support.components.bsnd_metadata import SENTINEL
from tests.mla.unit.bsnd.test_flash_attn_npu_v3_mla_bsnd_metadata_aicpu import (
    bsnd_probe as bsnd_probe,
)
from tests.mla.support.components.bsnd_metadata_aicpu import enqueue
from tests.mla.unit.bsnd.test_flash_attn_npu_v3_mla_bsnd_metadata_consumer import BSND_CORRUPTIONS
from tests.mla.support.components.metadata_consumer import Context

from tests.mla.support.components.bsnd_metadata_device import (
    check as check,
    launch as launch,
    load_consumer as load_consumer,
    make_case as make_case,
)


@pytest.fixture(scope="module", params=(1, 2))
def bsnd_device(tmp_path_factory, request):
    directory = Path(
        os.environ.get("MLA_BSND_CONSUMER_DEVICE_DIR", tmp_path_factory.mktemp("mla-bsnd-device"))
    )
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = os.environ.get("MLA_BSND_CONSUMER_MANIFEST")
    manifest = (
        json.loads(Path(manifest_path).read_text()) if manifest_path else build_consumers(directory)
    )
    (directory / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return dict(
        lib=load_consumer(manifest, request.param), dtype=request.param, directory=directory
    )


@pytest.mark.parametrize(
    "lengths,shape,aic",
    [
        (None, (2, 1, 8, 257), None),
        ([257, 257], (2, 1, 8, 257), None),
        ([17, 0], (2, 7, 16, 33), None),
        ([0, 1, 511, 512, 513, 8193, 16384], None, None),
        ([0] * 41, None, None),
        ([511] * 10, (10, 2, 129, 16384), None),
        ([8193] + [0] * 40, None, None),
        ([0, 1, 511, 512, 513] * 3 + [0], (16, 1, 1, 513), 15),
        ([], (0, 3, 171, 16384), None),
        ([0, 17], (2, 0, 171, 16384), None),
        ([0, 0], (2, 3, 171, 0), None),
    ],
)
def test_valid_and_reuse(bsnd_device, bsnd_probe, lengths, shape, aic):
    case = make_case(bsnd_device, bsnd_probe, lengths, shape=shape, aic=aic)
    for checked in (False, True, True):
        with forbid_host_readback():
            launch(bsnd_device, case, checked=checked)
        check(case)
    if lengths is not None:
        assert case["lengths"].cpu().tolist() == lengths


def test_corruption_does_not_write(bsnd_device, bsnd_probe):
    case = make_case(bsnd_device, bsnd_probe, [257, 257], shape=(2, 1, 8, 257))
    good = case["metadata"].clone()
    for word, value, status in BSND_CORRUPTIONS:
        case["metadata"].copy_(good)
        case["metadata"][word] = value if value < 2**31 else value - 2**32
        altered = case["expected"].copy()
        altered[word] = value
        with forbid_host_readback():
            launch(bsnd_device, case, checked=False)
        check(case, status, record=altered)
    case["metadata"].copy_(good)
    for options, status in [
        (dict(address=0), 1),
        (dict(address=case["metadata"].data_ptr() + 2), 1),
        (dict(words=401), 1),
        (dict(words=404), 1),
        (dict(count=1), 11),
        (dict(length_address=0), 11),
        (dict(length_address=case["lengths"].data_ptr() + 2), 11),
    ]:
        launch(bsnd_device, case, checked=False, **options)
        check(case, status)
    launch(bsnd_device, case)
    check(case)


def test_each_stale_length_and_regeneration(bsnd_device, bsnd_probe):
    lengths = [0, 1, 511, 512, 513, 8193, 16384] * 6
    # A static full-core launch supports changed device tilings without a host
    # read of usedCoreNum. Keep blockDim fixed in both the generator and caller.
    case = make_case(bsnd_device, bsnd_probe, lengths, block_dim=bsnd_probe["platform"].aic)
    for b in range(len(lengths)):
        case["lengths"][b] = (lengths[b] + 1) % 16385
        launch(bsnd_device, case, checked=False)
        check(case, 11)
        case["lengths"][b] = lengths[b]
    # Regenerate on-device after a genuine length change, then accept the new
    # record without any length readback or host synchronization in the chain.
    with forbid_host_readback():
        case["lengths"].zero_()
        enqueue(bsnd_probe, case["args"], case["metadata_storage"], case["lengths"])
        launch(bsnd_device, case)
    fresh = make_case(
        bsnd_device, bsnd_probe, [0] * len(lengths), block_dim=bsnd_probe["platform"].aic
    )
    case["expected"] = fresh["expected"]
    check(case)


def test_zero_suffix_checks_every_omitted_sample(bsnd_device, bsnd_probe):
    case = make_case(bsnd_device, bsnd_probe, [1, 0, 7, 0, 0], shape=(5, 3, 171, 513))
    altered = case["expected"].copy()
    for word, value in ((20, 1), (23, 1), (49, 0), (75, 0)):
        case["metadata"][word] = value
        altered[word] = value
    launch(bsnd_device, case, checked=False)
    check(case, 8, record=altered)
    case["metadata"][23] = altered[23] = 3
    case["metadata"][49] = altered[49] = 1
    launch(bsnd_device, case, checked=False)
    check(case, 8, record=altered)


def test_independent_context_and_launch_dimension(bsnd_device, bsnd_probe):
    case = make_case(bsnd_device, bsnd_probe, [17, 0], shape=(2, 7, 16, 33))
    original = case["context"]
    for field, value, expected in (
        ("dtype", 3 - bsnd_device["dtype"], 4),
        ("batch", 3, 1),
        ("sq", 8, 5),
        ("aic", 0, 6),
        ("aiv", original.aic, 6),
        ("block_dim", original.block_dim + 1, 6),
        ("workspace", original.workspace - 1, 9),
        ("libapi", 2**64 - 1, 9),
    ):
        context = Context.from_buffer_copy(original)
        setattr(context, field, value)
        launch(bsnd_device, case, checked=False, context=context, blocks=original.block_dim)
        check(case, expected)
    for field, _ in original.memory._fields_:
        context = Context.from_buffer_copy(original)
        setattr(context.memory, field, 0)
        launch(bsnd_device, case, checked=False, context=context)
        check(case, 10)


def test_cross_stream_clone_and_allocator_lifetime(bsnd_device, bsnd_probe):
    producer, consumer = torch.npu.Stream(), torch.npu.Stream()
    pending = []
    for i in range(12):
        lengths = [0, 513, 1, 0] if i % 2 else [8193] * 4
        with torch.npu.stream(producer):
            case = make_case(bsnd_device, bsnd_probe, lengths, worker=i % 2)
            # Use new equal lengths: the snapshot has no hidden ownership of
            # the generation tensor. Clone remains a normal offset view.
            copied = case["metadata_storage"].clone()
            case["metadata_storage"], case["metadata"] = copied, copied[1:-1]
            case["lengths"] = case["lengths"].clone()
            ready = torch.npu.Event()
            ready.record(producer)
        with torch.npu.stream(consumer):
            consumer.wait_event(ready)
            with forbid_host_readback():
                launch(bsnd_device, case)
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
    "header",
    "words",
    "snapshot",
    "stale",
    "missing",
    "count",
    "suffix",
    "partial_suffix",
    "workspace",
    "generation",
    "empty_stale",
)


@pytest.mark.parametrize("fault", FAULTS)
def test_runtime_trap(bsnd_device, bsnd_probe, fault):
    directory = bsnd_device["directory"] / f"error-{bsnd_device['dtype']}-{fault}"
    directory.mkdir()
    torch.npu.synchronize()
    sync = {"snapshot": "event", "stale": "stream"}.get(fault, "device")
    command = [
        sys.executable,
        "-m",
        "tests.mla.support.mla_bsnd_metadata_error_probe",
        str(bsnd_device["directory"] / "build-manifest.json"),
        str(bsnd_probe["directory"] / "build-manifest.json"),
        str(bsnd_device["dtype"]),
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
