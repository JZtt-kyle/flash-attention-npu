"""TND-Q/BSND-KV v4 attention: fixed CANN, independent math and guarded storage."""

import json
import os
from pathlib import Path

import pytest
import torch
import torch_npu  # noqa: F401

from tests.common.compare import assert_fa_close
from tests.mla.support.mla_dense_build import build_probes
from tests.mla.support.mla_metadata_error_probe import forbid_host_readback
from tests.mla.support.components.tnd_bsnd_metadata_aicpu import mixed_probe as mixed_probe, enqueue
from tests.mla.unit.mixed.test_flash_attn_npu_v3_mla_tnd_bsnd_metadata_consumer import (
    mixed_artifacts as mixed_artifacts,
    mixed_oracle as mixed_oracle,
    MIXED_CORRUPTIONS,
    check_records,
)
from tests.mla.support.components.tnd_bsnd_metadata_consumer import SENTINEL

from tests.mla.support.components.tnd_bsnd_attention import (
    CASES as CASES,
    MixedSchedule as MixedSchedule,
    PERFORMANCE_CASES as PERFORMANCE_CASES,
    check_case as check_case,
    configure_producer as configure_producer,
    launch as launch,
    load_attention as load_attention,
    make_case as make_case,
    references as references,
)


EMPTY_CASES = [
    ([], [], 64, 128, None),
    ([0, 0], [0, 513], 1025, 128, None),
    ([3, 0, 1], [0, 0, 0], 0, 171, None),
    ([0], [0], 0, 1, None),
]


@pytest.fixture(scope="module", params=("fp16", "bf16"))
def mixed_attention(tmp_path_factory, request):
    directory = (
        Path(
            os.environ.get(
                "MLA_TND_BSND_ATTENTION_DIR", tmp_path_factory.mktemp("mla-mixed-attention")
            )
        )
        / request.param
    )
    if os.environ.get("MLA_TND_BSND_ATTENTION_REUSE") != "1":
        build_probes(directory, request.param, mixed=True)
    return load_attention(directory)


@pytest.mark.parametrize("spec", CASES + EMPTY_CASES)
def test_accuracy(mixed_attention, mixed_probe, spec):
    check_case(mixed_attention, make_case(mixed_attention, mixed_probe, spec))


@pytest.mark.parametrize("scale", [0.0, -0.03125, 0.125])
def test_scales_and_max_changes(mixed_attention, mixed_probe, scale):
    check_case(
        mixed_attention,
        make_case(mixed_attention, mixed_probe, CASES[2], scale=scale, amplitude=3.0),
    )


def test_source_schedules(mixed_probe, mixed_oracle, tmp_path):
    dummy = ({"dtype": torch.float16}, tmp_path)
    cases = [make_case(dummy, mixed_probe, spec) for spec in CASES]
    check_records(
        [c["expected"] for c in cases],
        [c["values"] for c in cases],
        mixed_oracle,
        tmp_path,
        "attention",
        1,
    )


def test_corruption_precedes_attention(mixed_attention, mixed_probe):
    case = make_case(mixed_attention, mixed_probe, ([1, 1], [257, 257], 257, 8, None))
    good = case["metadata"].clone()
    for word, value, status in MIXED_CORRUPTIONS:
        case["metadata"].copy_(good)
        case["metadata"][word] = value if value < 2**31 else value - 2**32
        with forbid_host_readback():
            launch(mixed_attention, case, checked=False)
        assert (case["status"][::16].cpu() == status).all()
        assert (case["status"].view(-1, 16)[:, 1:].cpu() == SENTINEL).all()
        for t in case["outputs"]["migrated"]:
            assert (t.cpu() == 12345).all()
        assert (case["ws_storage"]["migrated"].cpu() == 205).all()
    case["metadata"].copy_(good)
    # Same allocation and same total capacity with changed segment boundaries.
    case["query"][1] = 0
    launch(mixed_attention, case, checked=False)
    assert (case["status"][::16].cpu() == 12).all()
    for t in case["outputs"]["migrated"]:
        assert (t.cpu() == 12345).all()
    assert (case["ws_storage"]["migrated"].cpu() == 205).all()


def test_streams_clone_and_regeneration(mixed_attention, mixed_probe):
    streams = [torch.npu.Stream(), torch.npu.Stream()]
    original = torch.npu.current_stream()
    case = make_case(mixed_attention, mixed_probe, ([3, 0, 1], [17, 0, 33], 64, 128, None))
    for i, stream in enumerate(streams):
        stream.wait_stream(original)
        with torch.npu.stream(stream):
            clone = case["metadata"].clone()
            other = dict(case, metadata=clone)
            with forbid_host_readback():
                launch(mixed_attention, other)
            del clone, other
            torch.empty_like(case["metadata"]).fill_(-1)
        original.wait_stream(stream)
    check_case(mixed_attention, case)
    # Update boundaries while retaining totals; all inputs are regenerated from
    # the new independent logical segments (old poison must not become data).
    updated = make_case(
        mixed_attention, mixed_probe, ([1, 2, 1], [16, 17, 17], 64, 128, None), worker=1
    )
    updated["metadata_storage"] = case["metadata_storage"]
    updated["metadata"] = case["metadata"]
    with forbid_host_readback():
        enqueue(
            mixed_probe,
            updated["args"],
            updated["metadata_storage"],
            updated["query"],
            updated["kv"],
            1,
        )
        launch(mixed_attention, updated)
    check_case(mixed_attention, updated)


def test_device_schedule_and_length_alias(mixed_attention, mixed_probe):
    case = make_case(mixed_attention, mixed_probe, ([1, 3], [1, 4], 7, 129, None))
    case["kv"] = case["query"][1:]
    check_case(mixed_attention, case)
    expected = [t.clone() for t in case["outputs"]["migrated"]]
    case["schedule"] = MixedSchedule()  # Only the fixed-source driver may use this.
    for t in case["outputs"]["migrated"]:
        t.fill_(12345)
    with forbid_host_readback():
        launch(mixed_attention, case)
    for actual, want in zip(case["outputs"]["migrated"], expected):
        torch.testing.assert_close(actual.cpu(), want.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize(
    "fault",
    [
        "v1",
        "v2",
        "v3",
        "words",
        "span",
        "wide_start",
        "capacity",
        "workspace",
        "stale_q",
        "stale_k",
        "missing_q",
        "count_k",
        "empty_stale",
        "empty_batch",
        "empty_kv",
        "suffix",
        "variable_m",
        "generation",
    ],
)
def test_runtime_rejection(mixed_attention, mixed_probe, fault):
    import subprocess
    import sys

    directory = mixed_attention[1] / ("trap-" + fault)
    directory.mkdir(exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "tests.mla.support.mla_tnd_bsnd_attention_error_probe",
        str(mixed_attention[1]),
        str(mixed_probe["directory"] / "build-manifest.json"),
        fault,
        str(directory),
    ]
    (directory / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    with (directory / "process.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=90)
    assert result.returncode == 0, (directory / "process.log").read_text()[-4000:]
    report = json.loads((directory / "runtime-error.json").read_text())
    assert report["elapsed_seconds"] < 15
    assert "trap error" in report["error"]
    assert any(code in report["error"] for code in ("507015", "507016", "507035", "507036"))


@pytest.mark.parametrize("offsets", [(0, 0, 0, 0), (16, 32, 48, 64)])
def test_aligned_and_zero_offsets(mixed_attention, mixed_probe, offsets):
    check_case(mixed_attention, make_case(mixed_attention, mixed_probe, CASES[1], offsets=offsets))


def test_pending_allocator_lifetime(mixed_attention, mixed_probe):
    producer, consumer = torch.npu.Stream(), torch.npu.Stream()
    pending = []
    for i in range(8):
        with torch.npu.stream(producer):
            case = make_case(mixed_attention, mixed_probe, CASES[0], worker=i % 2)
            ready = torch.npu.Event()
            ready.record(producer)
        with torch.npu.stream(consumer):
            consumer.wait_event(ready)
            with forbid_host_readback():
                launch(mixed_attention, case)
                outputs = [t.clone() for t in case["outputs"]["migrated"]]
            pending.append((outputs, case["inputs"], case["spec"], case["scale"]))
            del case
            for _ in range(4):
                torch.empty((32768,), dtype=torch.int32, device="npu").fill_(-1)
    for outputs, inputs, spec, scale in pending:
        refs = references(dict(inputs=inputs, spec=spec, scale=scale))
        for i, t in enumerate(outputs):
            assert_fa_close(t.cpu(), refs[i], refs[i + 2], name="lifetime")
