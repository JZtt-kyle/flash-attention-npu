"""AICPU -> immutable metadata -> dense device consumer, without public dispatch."""

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

from tests.common.compare import assert_fa_close
from tests.mla.support.mla_ref import mla_reference_pair
from tests.mla.support.mla_dense_build import build_probes
from tests.mla.unit.dense.test_flash_attn_npu_v3_mla_dense import (
    check_case,
    launch,
    load_probes,
    make_case,
)
from tests.mla.unit.dense.test_flash_attn_npu_v3_mla_metadata import probe as probe
from tests.mla.unit.dense.test_flash_attn_npu_v3_mla_metadata import Args, Shape, enqueue
from tests.mla.unit.dense.test_flash_attn_npu_v3_mla_metadata_consumer import (
    CORRUPTIONS,
    SENTINEL,
    WORDS,
    Context,
    Memory,
)


@pytest.fixture(scope="module", params=["fp16", "bf16"])
def consumers(tmp_path_factory, request):
    directory = (
        Path(os.environ.get("MLA_DENSE_ARTIFACT_DIR", tmp_path_factory.mktemp("mla-md-dense")))
        / request.param
    )
    dtype = torch.float16 if request.param == "fp16" else torch.bfloat16
    return configure(build_probes(directory, request.param, metadata=True), directory, dtype)


def configure(libraries, directory, dtype):
    probes = load_probes(libraries, directory, dtype)
    for name in ("metadata", "metadata_checked"):
        lib = probes[0][name]
        lib.launch_dense_metadata.argtypes = [ct.c_void_p] * 10 + [
            ct.c_uint32,
            ct.c_void_p,
            ct.POINTER(Context),
            ct.c_float,
        ]
        lib.launch_dense_metadata.restype = ct.c_int
    return probes


def make_metadata_case(probes, probe, shape, *, scale=None, worker=0):
    case = make_case(probes, *shape, scale=scale, offset=1)
    d = case["details"]
    dtype = 1 if probes[0]["dtype"] == torch.float16 else 2
    case["context"] = Context(
        dtype, *shape, d[2], d[3], case["schedule"].block_dim, d[0], d[9], Memory(*d[4:9])
    )
    storage = torch.full((WORDS + 2,), SENTINEL, dtype=torch.int32, device="npu")
    case["metadata_storage"] = storage
    case["metadata"] = storage[1:-1]  # Odd int32 offset, only 4-byte aligned.
    case["status_storage"] = torch.full(
        ((3 * case["context"].block_dim + 2) * 16,), SENTINEL, dtype=torch.int32, device="npu"
    )
    case["status"] = case["status_storage"][16:-16]
    case["cache_seqlens"] = torch.full((shape[0],), shape[3], dtype=torch.int32, device="npu")
    args = Args(0, WORDS, dtype, Shape(*shape), probe["platform"])
    # No D2H or synchronization between the AICPU producer and the consumer.
    case["metadata_expected"] = enqueue(
        probe, args, storage, worker, cache_seqlens=case["cache_seqlens"]
    )
    return case


def launch_consumer(probes, case, *, words=WORDS, address=None, context=None, checked=True):
    tensors = [case["device"][n] for n in ("qv", "q", "v", "k")]
    tensors += [case["desc"], *case["outputs"]["migrated"], case["workspace"]["migrated"]]
    result = probes[0]["metadata_checked" if checked else "metadata"].launch_dense_metadata(
        torch.npu.current_stream().npu_stream,
        *(t.data_ptr() for t in tensors),
        case["metadata"].data_ptr() if address is None else address,
        words,
        None if checked else case["status"].data_ptr(),
        ct.byref(case["context"] if context is None else context),
        case["scale"],
    )
    assert result == 0, result


def assert_status(case, expected):
    words = case["status_storage"].cpu()
    expected_words = torch.full_like(words, SENTINEL)
    expected_words[16:-16:16] = expected
    assert torch.equal(words, expected_words), words.tolist()


def check_metadata_case(probes, case, repeats=3):
    def dispatch(lib, current, name):
        if name == "migrated":
            launch_consumer(probes, current)
        else:
            launch(lib, current, name)

    record = check_case(probes, case, repeats=repeats, launch_fn=dispatch)
    actual = case["metadata_storage"].cpu().tolist()
    assert actual[0] == actual[-1] == SENTINEL
    assert [x & 0xFFFFFFFF for x in actual[1:-1]] == case["metadata_expected"]
    return record


@pytest.mark.parametrize("shape", [(2, 1, 8, 257), (1, 7, 16, 65)])
@pytest.mark.parametrize("scale", [None, 0.0, -0.23, 0.23])
def test_d01_d02(consumers, probe, shape, scale):
    check_metadata_case(consumers, make_metadata_case(consumers, probe, shape, scale=scale))


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 1, 1),
        (1, 1, 15, 15),
        (1, 1, 16, 16),
        (1, 1, 17, 17),
        (1, 1, 127, 31),
        (1, 1, 128, 32),
        (1, 1, 129, 33),
        (1, 1, 255, 127),
        (1, 1, 256, 128),
        (1, 1, 128, 512),
        (2, 2, 128, 511),
        (2, 128, 2, 129),
        (2, 256, 1, 257),
        (10, 2, 128, 65),
        (19, 2, 128, 65),
        (20, 2, 128, 65),
        (21, 2, 128, 65),
        (41, 2, 128, 65),
    ],
)
def test_boundaries(consumers, probe, shape):
    check_metadata_case(consumers, make_metadata_case(consumers, probe, shape))


def test_reject_corrupt_records(consumers, probe):
    case = make_metadata_case(consumers, probe, (2, 1, 8, 257))
    # Expected statuses are shared with CPU tests; the NPU must leave ALL
    # output/workspace bytes unchanged, not merely their guard regions.
    good = case["metadata"].clone()
    outputs = [*case["outputs"]["migrated"], case["workspace"]["migrated"]]
    original = [t.cpu() for t in outputs]
    for word, value, expected in CORRUPTIONS:
        case["metadata"].copy_(good)
        case["metadata"][word] = value if value < 2**31 else value - 2**32
        launch_consumer(consumers, case, checked=False)
        assert_status(case, expected)
        for actual, before in zip(outputs, original):
            assert torch.equal(actual.cpu(), before), (word, value)
    for words, address in [
        (0, None),
        (WORDS - 1, None),
        (WORDS + 1, None),
        (WORDS, 0),
        (WORDS, case["metadata"].data_ptr() + 2),
    ]:
        launch_consumer(consumers, case, words=words, address=address, checked=False)
        assert_status(case, 1)
        for actual, before in zip(outputs, original):
            assert torch.equal(actual.cpu(), before)
    case["metadata"].copy_(good)
    for name, value, expected in [
        ("dtype", 3 - case["context"].dtype, 4),
        ("batch", 1, 5),
        ("sq", 2, 5),
        ("heads", 16, 5),
        ("sk", 65, 5),
        ("aic", 21, 6),
        ("aiv", 20, 6),
        ("workspace", 0, 9),
        ("libapi", 0, 9),
    ]:
        context = Context.from_buffer_copy(case["context"])
        setattr(context, name, value)
        launch_consumer(consumers, case, context=context, checked=False)
        assert_status(case, expected)
        for actual, before in zip(outputs, original):
            assert torch.equal(actual.cpu(), before)
    case["metadata"].copy_(good)
    check_metadata_case(consumers, case)  # Diagnostic rejection preserves the stream.
    (consumers[1] / "rejections.json").write_text(
        json.dumps(
            dict(
                corrupt_records=len(CORRUPTIONS),
                buffer_failures=5,
                context_failures=9,
                all_cores_checked=True,
                output_and_workspace_unchanged=True,
            )
        )
        + "\n"
    )


def test_reuse_scales_streams_and_inputs(consumers, probe):
    case = make_metadata_case(consumers, probe, (2, 1, 8, 257))
    ready = torch.npu.Event()
    ready.record()
    streams = [torch.npu.Stream(), torch.npu.Stream()]
    pending = []
    for i in range(8):
        stream = streams[i % 2]
        with torch.npu.stream(stream):
            stream.wait_event(ready)
            current = make_case(consumers, 2, 1, 8, 257, scale=(0, 0.23, -0.23, 576**-0.5)[i % 4])
            current.update(
                context=case["context"],
                metadata=case["metadata"],
                status=torch.empty_like(case["status"]),
            )
            current["inputs"]["qv"].mul_(1 + i / 16)
            current["device"]["qv"].copy_(current["inputs"]["qv"])
            launch_consumer(consumers, current)
            pending.append(current)
    for current in pending:
        refs = mla_reference_pair(**current["inputs"], softmax_scale=current["scale"])
        for i, out in enumerate(current["outputs"]["migrated"]):
            assert_fa_close(out.cpu(), refs[i], refs[i + 2], name=("out", "lse")[i])
    assert [x & 0xFFFFFFFF for x in case["metadata"].cpu().tolist()] == case["metadata_expected"]


def test_aicpu_failure_and_overwrite(consumers, probe):
    case = make_metadata_case(consumers, probe, (2, 1, 8, 257))
    for iteration in range(3):
        for shape, expected in [
            ((2, 1, 8, 257), 0),
            ((2, 1, 8, 16385), 3),
            ((1, 7, 16, 65), 5),
            ((2, 1, 8, 257), 0),
        ]:
            outputs = [*case["outputs"]["migrated"], case["workspace"]["migrated"]]
            original = [t.cpu() for t in outputs] if expected else []
            args = Args(0, WORDS, case["context"].dtype, Shape(*shape), probe["platform"])
            case["metadata_expected"] = enqueue(
                probe, args, case["metadata_storage"], iteration % 2
            )
            launch_consumer(consumers, case, checked=False)
            assert_status(case, expected)
            for actual, before in zip(outputs, original):
                assert torch.equal(actual.cpu(), before)
    check_metadata_case(consumers, case)


def test_cache_lengths_failure_and_recovery(consumers, probe):
    case = make_metadata_case(consumers, probe, (2, 1, 8, 257))
    check_metadata_case(consumers, case)

    outputs = [*case["outputs"]["migrated"], case["workspace"]["migrated"]]
    original = [t.cpu() for t in outputs]
    args = Args(0, WORDS, case["context"].dtype, Shape(*case["shape"]), probe["platform"])
    for value in (-1, 0, 1, 256, 258, 512, 513, 2**31 - 1):
        case["cache_seqlens"][1] = value
        enqueue(probe, args, case["metadata_storage"], cache_seqlens=case["cache_seqlens"])
        # A failed producer record is passed straight to the real device
        # consumer; no status copy or synchronization intervenes.
        launch_consumer(consumers, case, checked=False)
        assert_status(case, 3)  # GenerationFailed on every Cube/Vector core.
        words = case["metadata_storage"].cpu().tolist()
        assert words[0] == words[-1] == SENTINEL
        assert words[6:9] == [4, 0, 0]
        assert not any(words[9:-1])
        for actual, before in zip(outputs, original):
            assert torch.equal(actual.cpu(), before)
    case["cache_seqlens"].fill_(257)
    case["metadata_expected"] = enqueue(
        probe, args, case["metadata_storage"], cache_seqlens=case["cache_seqlens"]
    )
    check_metadata_case(consumers, case)


def test_checked_enqueue_without_host_readback(consumers, probe):
    from tests.mla.support.mla_metadata_error_probe import forbid_host_readback

    case = make_metadata_case(consumers, probe, (2, 1, 8, 257))
    args = Args(0, WORDS, case["context"].dtype, Shape(*case["shape"]), probe["platform"])
    with forbid_host_readback():
        for _ in range(8):
            enqueue(probe, args, case["metadata_storage"], cache_seqlens=case["cache_seqlens"])
            launch_consumer(consumers, case)
    check_metadata_case(consumers, case)


@pytest.mark.parametrize(
    "fault",
    [
        "null",
        "unaligned",
        "truncated",
        "magic",
        "version",
        "schedule_failure",
        "length_count",
        "length_negative",
        "length_zero",
        "length_short",
        "length_over_capacity",
        "dtype",
        "shape",
        "platform",
        "tiling",
        "endpoints",
        "workspace_high_word",
        "local_memory",
    ],
)
def test_checked_runtime_error(consumers, probe, fault):
    # Build once in the parent, then load the exact same hashed libraries into
    # fresh interpreters. An expected runtime failure must not poison pytest.
    directory = consumers[1] / ("error-" + fault)
    directory.mkdir()
    torch.npu.synchronize()
    sync = {"magic": "event", "length_short": "stream"}.get(fault, "device")
    command = [
        sys.executable,
        "-m",
        "tests.mla.support.mla_metadata_error_probe",
        str(consumers[1] / "build-manifest.json"),
        str(probe["directory"] / "build-manifest.json"),
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
                # The child deliberately bypasses NPU tensor destruction after
                # Trap. Clean its CANN compiler workers too, using only this
                # test's private process group (never the parent's group).
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
    # C1V2 may report the Cube or Vector exception first. Pin the documented
    # exception codes, require the trap diagnostic and exclude task timeouts.
    assert any(code in report["error"] for code in ("507015", "507016", "507035", "507036")), report
    assert "trap error" in report["error"], report
    assert report["elapsed_seconds"] < 15, report  # A real exception, not the op timeout.
    assert "910B" in report["device"]
