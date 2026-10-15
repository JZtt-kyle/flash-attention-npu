"""Compact TND v3 attention: fixed CANN, independent math and guarded storage."""

import hashlib
import json
import os
from pathlib import Path

import pytest
import torch
import torch_npu  # noqa: F401

from tests.common.compare import assert_fa_close
from tests.mla.support.mla_dense_build import build_probes
from tests.mla.support.mla_metadata_error_probe import forbid_host_readback
from tests.mla.support.components.tnd_metadata_aicpu import tnd_probe as tnd_probe, enqueue
from tests.mla.unit.tnd.test_flash_attn_npu_v3_mla_tnd_metadata_consumer import (
    tnd_artifacts as tnd_artifacts,
    tnd_oracle as tnd_oracle,
    TND_CORRUPTIONS,
    check_records,
)
from tests.mla.support.components.tnd_metadata_consumer import SENTINEL

from tests.mla.support.components.tnd_attention import (
    TndSchedule as TndSchedule,
    configure_producer as configure_producer,
    launch as launch,
    load_attention as load_attention,
    make_case as make_case,
    references as references,
)


# Q lengths, KV lengths, Hq, optional synthetic AIC count. The input capacity is
# the sum of the given segments, never B * max(lengths). Include total KV >16K.
CASES = [
    ([3, 0, 1], [17, 0, 33], 128, None),  # T01
    ([1, 7, 2], [1, 513, 31], 16, None),
    ([0, 3, 1, 0, 7], [33, 511, 0, 513, 1025], 171, None),
    ([1, 2, 3], [16384, 8193, 17], 8, None),
    ([1, 1, 1], [17, 129, 257], 129, None),
    ([2] * 10, [511] + [0] * 9, 128, 15),  # omitted zero-cost suffix
    ([3, 0, 1, 7, 0], [17, 65, 513, 0, 1], 171, 1),  # task ring crosses samples
    ([3, 0] * 21, [0, 513] * 21, 171, None),  # nonempty, zero total cost
    ([257, 1, 0, 3], [33, 513, 17, 1], 1, None),
    ([1, 1, 1, 1], [31, 32, 33, 65], 15, None),
    ([1, 1, 1, 1], [127, 128, 129, 511], 255, None),
    ([2, 0, 1], [512, 0, 513], 256, None),
    ([0, 1, 0, 0], [0, 0, 0, 1], 1, None),
]
PERFORMANCE_CASES = [CASES[i] for i in (0, 1, 2, 3, 4, 5, 6, 7, 8, 12)]
EMPTY_CASES = [
    ([], [], 128, None),
    ([0, 0], [0, 513], 128, None),
    ([3, 0, 1], [0, 0, 0], 171, None),
    ([0], [0], 1, None),
]


@pytest.fixture(scope="module", params=("fp16", "bf16"))
def tnd_attention(tmp_path_factory, request):
    directory = (
        Path(os.environ.get("MLA_TND_ATTENTION_DIR", tmp_path_factory.mktemp("mla-tnd-attention")))
        / request.param
    )
    if os.environ.get("MLA_TND_ATTENTION_REUSE") != "1":
        build_probes(directory, request.param, tnd=True)
    return load_attention(directory)


def check_case(probes, case, *, repeats=3):
    qs, ks, heads, _ = case["spec"]
    empty = (
        torch.cat([torch.full((q,), k == 0, dtype=torch.bool) for q, k in zip(qs, ks)])
        if qs
        else torch.empty(0, dtype=torch.bool)
    )
    source_ok = bool(sum(qs) and sum(ks))
    first = None
    for _ in range(repeats):
        with forbid_host_readback():
            if source_ok:
                launch(probes, case, "oracle")
            launch(probes, case)
        actual = [t.cpu() for t in case["outputs"]["migrated"]]
        if source_ok:
            source = [t.cpu() for t in case["outputs"]["oracle"]]
            torch.testing.assert_close(actual[0], source[0], rtol=0, atol=0)
            torch.testing.assert_close(actual[1][:, ~empty], source[1][~empty].T, rtol=0, atol=0)
            assert torch.isposinf(source[1][empty]).all()
        if first is not None:
            for a, b in zip(actual, first):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        first = actual
    refs = references(case)
    errors = []
    for i in range(2):
        if i == 0:
            if first[i].numel():
                assert_fa_close(first[i], refs[i], refs[i + 2], name="tnd-out")
            finite = torch.ones_like(refs[i], dtype=torch.bool)
        else:
            assert torch.equal(torch.isneginf(first[i]), torch.isneginf(refs[i]))
            assert torch.equal(torch.isposinf(first[i]), torch.isposinf(refs[i]))
            finite = torch.isfinite(refs[i])
            if finite.any():
                assert_fa_close(
                    first[i][finite], refs[i][finite], refs[i + 2][finite], name="tnd-lse"
                )
        errors.append(
            float((first[i][finite] - refs[i][finite]).abs().max()) if finite.any() else 0
        )
    for before, after, prefix, readonly in case["guards"]:
        got = after.cpu()
        if readonly:
            torch.testing.assert_close(got, before, rtol=0, atol=0, equal_nan=True)
        else:
            assert torch.equal(got[:prefix], before[:prefix])
            assert torch.equal(got[-17:], before[-17:])
    for buffer in case["ws_storage"].values():
        assert (buffer[:512].cpu() == 205).all() and (buffer[-512:].cpu() == 205).all()
    assert [w & 0xFFFFFFFF for w in case["metadata_storage"].cpu().tolist()] == [
        SENTINEL,
        *case["expected"],
        SENTINEL,
    ]
    report = dict(
        spec=case["spec"],
        scale=case["scale"],
        dtype=str(probes[0]["dtype"]),
        errors=errors,
        source_compared=source_ok,
        m_base=case["schedule"].tiling.m_base,
        output_sha256=hashlib.sha256(
            b"".join(t.contiguous().view(torch.uint8).numpy().tobytes() for t in first)
        ).hexdigest(),
    )
    with (probes[1] / "accuracy.jsonl").open("a") as f:
        f.write(json.dumps(report) + "\n")
    return report


@pytest.mark.parametrize("spec", CASES + EMPTY_CASES)
def test_accuracy(tnd_attention, tnd_probe, spec):
    check_case(tnd_attention, make_case(tnd_attention, tnd_probe, spec))


@pytest.mark.parametrize("scale", [0.0, -0.03125, 0.125])
def test_scales_and_max_changes(tnd_attention, tnd_probe, scale):
    check_case(
        tnd_attention, make_case(tnd_attention, tnd_probe, CASES[2], scale=scale, amplitude=3.0)
    )


def test_source_schedules(tnd_probe, tnd_oracle, tmp_path):
    dummy = ({"dtype": torch.float16}, tmp_path)
    cases = [make_case(dummy, tnd_probe, spec) for spec in CASES]
    check_records(
        [c["expected"] for c in cases],
        [c["values"] for c in cases],
        tnd_oracle,
        tmp_path,
        "attention",
        1,
    )


def test_corruption_precedes_attention(tnd_attention, tnd_probe):
    case = make_case(tnd_attention, tnd_probe, ([1, 1], [257, 257], 8, None))
    good = case["metadata"].clone()
    for word, value, status in TND_CORRUPTIONS:
        case["metadata"].copy_(good)
        case["metadata"][word] = value if value < 2**31 else value - 2**32
        with forbid_host_readback():
            launch(tnd_attention, case, checked=False)
        assert (case["status"][::16].cpu() == status).all()
        assert (case["status"].view(-1, 16)[:, 1:].cpu() == SENTINEL).all()
        for t in case["outputs"]["migrated"]:
            assert (t.cpu() == 12345).all()
        assert (case["ws_storage"]["migrated"].cpu() == 205).all()
    case["metadata"].copy_(good)
    # Same allocation and same total capacity with changed segment boundaries.
    case["query"][1] = 0
    launch(tnd_attention, case, checked=False)
    assert (case["status"][::16].cpu() == 12).all()
    for t in case["outputs"]["migrated"]:
        assert (t.cpu() == 12345).all()
    assert (case["ws_storage"]["migrated"].cpu() == 205).all()


def test_streams_clone_and_regeneration(tnd_attention, tnd_probe):
    streams = [torch.npu.Stream(), torch.npu.Stream()]
    original = torch.npu.current_stream()
    case = make_case(tnd_attention, tnd_probe, ([3, 0, 1], [17, 0, 33], 128, None))
    for i, stream in enumerate(streams):
        stream.wait_stream(original)
        with torch.npu.stream(stream):
            clone = case["metadata"].clone()
            other = dict(case, metadata=clone)
            with forbid_host_readback():
                launch(tnd_attention, other)
            del clone, other
            torch.empty_like(case["metadata"]).fill_(-1)
        original.wait_stream(stream)
    check_case(tnd_attention, case)
    # Update boundaries while retaining totals; all inputs are regenerated from
    # the new independent logical segments (old poison must not become data).
    updated = make_case(tnd_attention, tnd_probe, ([1, 2, 1], [16, 17, 17], 128, None), worker=1)
    updated["metadata_storage"] = case["metadata_storage"]
    updated["metadata"] = case["metadata"]
    with forbid_host_readback():
        enqueue(
            tnd_probe,
            updated["args"],
            updated["metadata_storage"],
            updated["query"],
            updated["kv"],
            1,
        )
        launch(tnd_attention, updated)
    check_case(tnd_attention, updated)


def test_device_schedule_and_prefix_alias(tnd_attention, tnd_probe):
    case = make_case(tnd_attention, tnd_probe, ([1, 3], [1, 3], 129, None))
    case["kv"] = case["query"]
    check_case(tnd_attention, case)
    expected = [t.clone() for t in case["outputs"]["migrated"]]
    case["schedule"] = TndSchedule()  # Only the fixed-source driver may use this.
    for t in case["outputs"]["migrated"]:
        t.fill_(12345)
    with forbid_host_readback():
        launch(tnd_attention, case)
    for actual, want in zip(case["outputs"]["migrated"], expected):
        torch.testing.assert_close(actual.cpu(), want.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize(
    "fault",
    [
        "v1",
        "v2",
        "words",
        "span",
        "workspace",
        "stale_q",
        "stale_k",
        "empty_stale",
        "empty_batch",
        "empty_kv",
        "suffix",
        "variable_m",
        "generation",
    ],
)
def test_runtime_rejection(tnd_attention, tnd_probe, fault):
    import subprocess
    import sys

    directory = tnd_attention[1] / ("trap-" + fault)
    directory.mkdir(exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "tests.mla.support.mla_tnd_attention_error_probe",
        str(tnd_attention[1]),
        str(tnd_probe["directory"] / "build-manifest.json"),
        fault,
        str(directory),
    ]
    (directory / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    with (directory / "process.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=90)
    assert result.returncode == 0, (directory / "process.log").read_text()[-4000:]
    report = json.loads((directory / "runtime-error.json").read_text())
    assert report["elapsed_seconds"] < 15
