"""Production mixed strides: public v4 reuse and internal Q-used v5."""

from contextlib import nullcontext
import gc
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from tests.mla.support.mla_metadata_error_probe import forbid_host_readback
from tests.mla.support.components.bsnd_stride import NAMES, strides_for
from tests.mla.support.public_cases.bsnd_stride_production import storage_fault as storage_fault
from tests.mla.support.public_cases.contract import _forward
from tests.mla.e2e.mixed.test_flash_attn_npu_v3_mla_mixed_public import invoke as mixed_invoke
from tests.mla.support.public_cases.mixed_public import metadata
from tests.mla.support.components.tnd_stride import (
    BOUNDARY_CASES,
    EMPTY_CASES,
    LAYOUT_CASES,
    WIDE_CASES,
    launch,
    spec,
    stride_attention as stride_attention,
    stride_producer as stride_producer,
)
from tests.mla.support.components.used_attention import effective
from tests.mla.support.public_cases.varlen_used import check_result as check_result

from tests.mla.support.public_cases.mixed_stride_production import (
    PERFORMANCE_CASES as PERFORMANCE_CASES,
    PUBLIC_CASES as PUBLIC_CASES,
    forward_args as forward_args,
    production_case as production_case,
    public_args as public_args,
)


ACCURACY_CASES = (
    [c for c in LAYOUT_CASES + BOUNDARY_CASES + EMPTY_CASES if c["item"]["layout"] == 2]
    + PERFORMANCE_CASES
    + PUBLIC_CASES
)


def invoke(args, entry="public"):
    if args.get("seqused_q") is not None:
        return _forward("eager" if entry in ("public", "op") else entry, args)
    return mixed_invoke(entry, public_args(args))


def check_production(probes, case, entry="public"):
    args = forward_args(case)
    with forbid_host_readback():
        if sum(effective(case["item"])[0]) and sum(case["item"]["kv"]):
            assert launch(probes, case, "oracle") == 0
        record = metadata(public_args(args)) if case["config"]["version"] == 4 else None
        with patch.object(torch.Tensor, "contiguous", side_effect=AssertionError("input copy")):
            results = [invoke(args, entry)]
            results.append(invoke(dict(args, scheduler_metadata=record), entry))
    for result in results:
        errors = check_result(case, result, source=True)
    if record is not None:
        torch.testing.assert_close(record.cpu(), case["metadata"].cpu(), rtol=0, atol=0)
    for before, after in case["input_checks"]:
        torch.testing.assert_close(after.cpu(), before, rtol=0, atol=0, equal_nan=True)
    return errors


@pytest.mark.parametrize("config", ACCURACY_CASES)
def test_production_source_and_math(stride_attention, stride_producer, config, tmp_path):
    errors = check_production(
        stride_attention, production_case(stride_attention, stride_producer, config)
    )
    directory = Path(os.environ.get("MLA_MIXED_STRIDE_DIR", tmp_path))
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "accuracy.jsonl").open("a") as log:
        log.write(
            json.dumps(dict(config=config, dtype=str(stride_attention[0]["dtype"]), errors=errors))
            + "\n"
        )


@pytest.mark.parametrize("config", [c for c in LAYOUT_CASES if c["item"]["layout"] == 2])
def test_raw_cpp_source_and_math(stride_attention, stride_producer, config):
    check_production(
        stride_attention, production_case(stride_attention, stride_producer, config), "cpp"
    )


@pytest.mark.parametrize("config", [c for c in WIDE_CASES if c["item"]["layout"] == 2])
def test_wide_active_address(stride_attention, stride_producer, config):
    torch.npu.empty_cache()
    check_production(stride_attention, production_case(stride_attention, stride_producer, config))


@pytest.mark.parametrize("version", (4, 5))
@pytest.mark.parametrize("scale", (0.0, -0.125, 0.23))
def test_scale_source_and_math(stride_attention, stride_producer, version, scale):
    config = spec(version, 2, "ntd", uq=(2, 0, 1) if version == 5 else None, uk=(17, 0, 33))
    check_production(
        stride_attention, production_case(stride_attention, stride_producer, config, scale=scale)
    )


def inputs(dtype, profile="independent", qs=(7, 3), ks=(33, 17), capacity=33, used=False):
    args = {}
    for i, name in enumerate(NAMES):
        shape = (
            (1, sum(qs), 8, 64 if i % 2 == 0 else 512)
            if i < 2
            else (len(qs), capacity, 1, 64 if i % 2 == 0 else 512)
        )
        strides = strides_for(shape, "bnsd" if profile == "ntd" else profile, name)
        span = 0 if 0 in shape else 1 + sum((n - 1) * s for n, s in zip(shape, strides))
        base = torch.zeros(span + 32, dtype=dtype, device="npu")
        view = base.as_strided(shape, strides, i + 3)
        args[name] = view[0] if i < 2 else view
    args.update(
        cu_seqlens_q=torch.tensor(
            list(itertools.accumulate(qs, initial=0)), dtype=torch.int32, device="npu"
        ),
        max_seqlen_q=max(qs, default=0),
        seqused_k=torch.tensor(ks, dtype=torch.int32, device="npu"),
    )
    if used:
        args["seqused_q"] = torch.tensor(qs, dtype=torch.int32, device="npu")
    return args


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("fake", (False, True))
@pytest.mark.parametrize(
    "qs,ks,capacity",
    (((7, 3), (33, 17), 33), ((), (), 33), ((0, 0), (33, 17), 33), ((7, 3), (0, 0), 0)),
)
@pytest.mark.parametrize("entry", ("public", "op", "eager", "cpp"))
@pytest.mark.parametrize("used", (False, True))
def test_shapes_and_empty(dtype, fake, qs, ks, capacity, entry, used):
    with FakeTensorMode() if fake else nullcontext():
        args = inputs(dtype, qs=qs, ks=ks, capacity=capacity, used=used)
        out, lse, *_ = invoke(args, "fake" if fake and entry == "cpp" else entry)
        assert out.shape == (sum(qs), 8, 512) and out.dtype == dtype
        assert lse.shape == (8, sum(qs)) and lse.dtype == torch.float32
        if not fake:
            assert (out.cpu() == 0).all()
            expected = torch.tensor([float(k) for q, k in zip(qs, ks) for _ in range(q)]).log()
            if not capacity:
                expected.fill_(float("inf"))
            torch.testing.assert_close(lse.cpu(), expected.expand(8, -1))


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("length", (None, 0, 33))
@pytest.mark.parametrize("fake", (False, True))
def test_none_integer_lengths_and_offset_record(dtype, length, fake):
    with FakeTensorMode() if fake else nullcontext():
        args = public_args(inputs(dtype, "independent"))
        args["cache_seqlens"] = length
        record = metadata(args)
        storage = torch.full((record.numel() + 4,), 123456, dtype=torch.int32, device="npu")
        storage[1:-3].copy_(record)
        for md in (None, record.clone(), storage[1:-3]):
            with forbid_host_readback():
                out, lse, *_ = mixed_invoke("public", dict(args, scheduler_metadata=md))
            if not fake:
                assert (out.cpu() == 0).all()
                torch.testing.assert_close(
                    lse.cpu(), torch.full((8, 10), float(33 if length is None else length)).log()
                )
        if not fake:
            assert storage[0].cpu() == 123456 and (storage[-3:].cpu() == 123456).all()
            torch.testing.assert_close(storage[1:-3].cpu(), record.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("profile", ("ntd", "broadcast", "overlap", "matrix_65536", "head_65536"))
@pytest.mark.parametrize("mode", ("implicit", "reuse", "generated", "used"))
def test_compile_lengths_and_cross_view_metadata(dtype, profile, mode):
    torch._dynamo.reset()

    def call(args):
        if mode == "generated":
            args = dict(args, scheduler_metadata=metadata(public_args(args)))
        return invoke(args)[:2]

    compiled = torch.compile(call, backend="eager", fullgraph=True)
    args = inputs(dtype, used=mode == "used")
    for current, qs, ks, uq, scale in (
        (profile, (7, 3), (33, 17), (3, 1), -0.125),
        ("contiguous", (3, 7), (0, 33), (1, 0), 0.0),
        (profile, (7, 3), (17, 0), (7, 3), 0.23),
    ):
        args["cu_seqlens_q"].copy_(torch.tensor((0, qs[0], sum(qs)), dtype=torch.int32))
        args["seqused_k"].copy_(torch.tensor(ks, dtype=torch.int32))
        if mode == "used":
            args["seqused_q"].copy_(torch.tensor(uq, dtype=torch.int32))
        else:
            # Generate on the preceding views, then reuse on a different layout.
            record = metadata(public_args(args))
        views = inputs(dtype, current)
        args.update({n: views[n] for n in NAMES}, softmax_scale=scale)
        if mode == "reuse":
            args["scheduler_metadata"] = record
        if mode != "used":
            torch.testing.assert_close(
                metadata(public_args(args)).cpu(), record.cpu(), rtol=0, atol=0
            )
        # Eager/lifetime paths forbid host readback. Dynamo tracing inspects
        # Tensor methods, so compile coverage instead requires fullgraph capture.
        out, lse = compiled(args)
        start = 0
        for b in range(2):
            count = uq[b] if mode == "used" else qs[b]
            assert (out[start : start + count].cpu() == 0).all()
            torch.testing.assert_close(
                lse[:, start : start + count].cpu(), torch.full((8, count), float(ks[b])).log()
            )
            start += qs[b]


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize("mode", ("implicit", "reuse", "used"))
def test_stream_and_storage_lifetime(dtype, mode):
    creator, consumer = torch.npu.Stream(), torch.npu.Stream()
    results = []
    for step in range(4):
        with torch.npu.stream(creator):
            args = inputs(dtype, "ntd", ks=(33, 0), used=mode == "used")
            args["v"].fill_(step + 1)
            if mode == "used":
                args["seqused_q"].copy_(torch.tensor([3, 1], dtype=torch.int32))
            elif mode == "reuse":
                args["scheduler_metadata"] = metadata(public_args(args))
        consumer.wait_stream(creator)
        with torch.npu.stream(consumer), forbid_host_readback():
            results.append(invoke(args)[:2])
        del args
        gc.collect()
        with torch.npu.stream(creator):
            churn = [torch.full((8192,), -77, dtype=dtype, device="npu") for _ in range(16)]
        del churn
    consumer.synchronize()
    for step, (out, lse) in enumerate(results):
        first, second = (3, 1) if mode == "used" else (7, 3)
        torch.testing.assert_close(
            out[:first].cpu(), torch.full((first, 8, 512), step + 1, dtype=dtype)
        )
        assert (out[7 : 7 + second].cpu() == 0).all() and lse[
            :, 7 : 7 + second
        ].cpu().isneginf().all()
        torch.testing.assert_close(lse[:, :first].cpu(), torch.full((8, first), 33.0).log())
    creator.synchronize()


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("entry", ("public", "cpp", "op", "fake"))
@pytest.mark.parametrize("mode", ("implicit", "reuse", "used"))
def test_full_storage_rejected(name, entry, mode, storage_fault):
    with FakeTensorMode() if entry == "fake" else nullcontext():
        args = inputs(torch.float16, "row_65536", used=mode == "used")
        args["seqused_k"].zero_()
        if mode == "used":
            args["seqused_q"].zero_()
        elif mode == "reuse":
            args["scheduler_metadata"] = metadata(public_args(args))
        layout = "TND" if name in ("q", "qv") else "BSND"
        with (
            storage_fault(args[name]),
            forbid_host_readback(),
            pytest.raises(RuntimeError, match=layout + " input view exceeds"),
        ):
            invoke(args, entry)


@pytest.mark.parametrize("mode", ("implicit", "reuse", "used"))
@pytest.mark.parametrize("name", NAMES)
def test_compiled_storage_rechecked(mode, name, storage_fault):
    torch._dynamo.reset()
    args = inputs(torch.float16, "row_65536", used=mode == "used")
    if mode == "reuse":
        args["scheduler_metadata"] = metadata(public_args(args))
    compiled = torch.compile(lambda args: invoke(args), backend="eager", fullgraph=True)
    compiled(args)
    torch.npu.synchronize()
    with storage_fault(args[name]), pytest.raises(RuntimeError, match="input view exceeds"):
        compiled(args)


@pytest.mark.parametrize("kind", ("fp16", "bf16"))
@pytest.mark.parametrize(
    "version,fault",
    (
        (4, "q_start"),
        (4, "stale_q"),
        (4, "stale_k"),
        (4, "endpoint"),
        (4, "empty_q"),
        (4, "empty_k"),
        (5, "q_over"),
        (5, "zero_q_bad_k"),
        (5, "zero_k_bad_q"),
        (5, "empty_k"),
    ),
)
def test_strided_device_rejections(kind, version, fault, tmp_path):
    script = "mla_mixed_public_error_probe" if version == 4 else "mla_mixed_used_error_probe"
    directory = (
        Path(os.environ.get("MLA_MIXED_STRIDE_DIR", tmp_path)) / kind / f"trap-v{version}-{fault}"
    )
    directory.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "tests." + script, kind, fault, str(directory), "strided"]
    (directory / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    with (directory / "process.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=90)
    assert result.returncode == 0, (directory / "process.log").read_text()[-4000:]
    report = json.loads((directory / "runtime-error.json").read_text())
    assert report["input_layout"] == "strided" and "trap error" in report["error"]
    assert report["elapsed_seconds"] < 15
