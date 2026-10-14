"""Internal BSND Q-used production entry: source/math, validation and lifetime."""

from contextlib import nullcontext
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.mla.support.mla_metadata_error_probe import forbid_host_readback
from tests.mla.e2e.dense.test_flash_attn_npu_v3_mla_contract import _forward
from tests.mla.unit.used.test_flash_attn_npu_v3_mla_used_attention import (
    used_attention as used_attention,
    used_probe as used_probe,
    CASES,
    EMPTY_CASES,
    check_math,
    effective,
    launch,
    make_case,
    references,
    used_case,
    valid_outputs,
)
from tests.mla.unit.used.test_flash_attn_npu_v3_mla_used_metadata_consumer import (
    used_artifacts as used_artifacts,
    used_oracle as used_oracle,
    check_records,
)


# Fixed before profiling: U01, unequal/zero lengths, M/S2 tails and long KV.
# Production launches all platform cores, independently of the device schedule.
PERFORMANCE_CASES = [dict(c) for c in CASES[:14] if c["layout"] == 0] + [
    used_case(q=[3] * 43, kv=[513] * 43, uq=[2, 0] * 21 + [1], uk=[0, 513] * 21 + [0], heads=128),
    used_case(uq=[3, 1], heads=128),
    used_case(uq=[3, 1], uk=[0, 0], heads=128),
]
ACCURACY_CASES = (
    PERFORMANCE_CASES
    + [c for c in EMPTY_CASES if c["layout"] == 0]
    + [
        used_case(q=[3, 3], kv=[513, 513], uq=[1, 2], uk=[512, 513], heads=h)
        for h in (1, 15, 16, 17, 127, 129, 255, 256, 257)
    ]
    + [used_case(uq=[7, 7], uk=[33, 33], heads=8)]
)


def production_case(probes, producer, spec, **kwargs):
    spec = dict(spec, aic=producer["platform"].aic)
    spec["block_dim"] = spec["aic"] if sum(spec["q"]) and sum(spec["kv"]) else 0
    assert spec["layout"] == 0 and spec["uq"] is not None
    return make_case(probes, producer, spec, **kwargs)


def forward_args(case):
    c = case["context"]
    args = {
        name: tensor.view(
            c.batch,
            c.query.maximum if name in ("q", "qv") else c.kv.maximum,
            c.heads if name in ("q", "qv") else 1,
            tensor.shape[-1],
        )
        for name, tensor in case["devices"]["migrated"].items()
    }
    args.update(
        seqused_q=case["inputs"][2], seqused_k=case["inputs"][3], softmax_scale=case["scale"]
    )
    return args


def check_result(case, result, *, source=False):
    c = case["context"]
    assert result[0].shape == (c.batch, c.query.maximum, c.heads, 512)
    assert result[1].shape == (c.batch, c.heads, c.query.maximum)
    assert result[0].dtype == case["host"]["v"].dtype and result[1].dtype == torch.float32
    host = [t.cpu() for t in result[:2]]
    actual = valid_outputs(case, [host[0].reshape(-1, c.heads, 512), host[1]])
    errors = check_math(actual, references(case))
    if not sum(case["item"]["kv"]):
        assert (host[0] == 0).all() and host[1].isposinf().all()
    uq, uk = effective(case["item"])
    if source and sum(uq) and sum(case["item"]["kv"]):
        expected = valid_outputs(case, [t.cpu() for t in case["outputs"]["oracle"]], source=True)
        nonempty = torch.tensor([k > 0 for q, k in zip(uq, uk) for _ in range(q)], dtype=torch.bool)
        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[1][:, nonempty], expected[1][:, nonempty], rtol=0, atol=0)
        assert expected[1][:, ~nonempty].isposinf().all()
    return errors


@pytest.mark.parametrize("spec", ACCURACY_CASES)
@pytest.mark.parametrize("entry", ["eager", "cpp"])
def test_production_accuracy(used_attention, used_probe, spec, entry):
    case = production_case(used_attention, used_probe, spec)
    args = forward_args(case)
    with forbid_host_readback():
        if sum(spec["uq"]) and sum(spec["kv"]):
            launch(used_attention, case, "oracle")
        results = [_forward(entry, args) for _ in range(3)]
    for result in results:
        errors = check_result(case, result, source=True)
    assert len({r[0].data_ptr() for r in results}) == (3 if results[0][0].numel() else 1)
    for before, after, _, readonly in case["guards"]:
        if readonly:
            torch.testing.assert_close(after.cpu(), before, rtol=0, atol=0, equal_nan=True)
    for key, tensor in zip(("uq", "uk"), case["inputs"][2:]):
        if tensor is not None:
            assert tensor.cpu().tolist() == spec[key]
    with (used_attention[1] / "production-accuracy.jsonl").open("a") as log:
        log.write(json.dumps(dict(item=case["item"], entry=entry, errors=errors)) + "\n")


@pytest.mark.parametrize("scale", [None, 0.0, -0.03125, 0.125])
def test_scale(used_attention, used_probe, scale):
    case = production_case(
        used_attention, used_probe, PERFORMANCE_CASES[0], scale=scale, amplitude=3.0
    )
    args = forward_args(case)
    args["softmax_scale"] = scale
    with forbid_host_readback():
        launch(used_attention, case, "oracle")
        result = _forward("eager", args)
    check_result(case, result, source=True)


def test_production_source_schedules(used_probe, used_oracle, tmp_path):
    cases = [
        production_case(({"dtype": torch.float16}, tmp_path), used_probe, spec)
        for spec in PERFORMANCE_CASES
    ]
    check_records(
        [c["expected"] for c in cases],
        [c["item"] for c in cases],
        used_oracle,
        tmp_path,
        "production-bsnd-used",
        1,
    )


def inputs(dtype=torch.float16, batch=2, sq=7, sk=33):
    return dict(
        q=torch.zeros((batch, sq, 8, 64), dtype=dtype, device="npu"),
        qv=torch.zeros((batch, sq, 8, 512), dtype=dtype, device="npu"),
        k=torch.zeros((batch, sk, 1, 64), dtype=dtype, device="npu"),
        v=torch.zeros((batch, sk, 1, 512), dtype=dtype, device="npu"),
        seqused_q=torch.full((batch,), min(sq, 3), dtype=torch.int32, device="npu"),
        seqused_k=torch.full((batch,), min(sk, 17), dtype=torch.int32, device="npu"),
    )


@pytest.fixture(params=[torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def dtype(request):
    return request.param


@pytest.fixture(params=["eager", "cpp", "fake"])
def entry(request):
    with FakeTensorMode() if request.param == "fake" else nullcontext():
        yield request.param


@pytest.mark.parametrize("batch,sq,sk", [(2, 7, 33), (0, 7, 33), (2, 0, 33), (2, 7, 0)])
@pytest.mark.parametrize("kv_used", [False, True])
def test_shapes_and_empty(entry, dtype, batch, sq, sk, kv_used):
    args = inputs(dtype, batch, sq, sk)
    if not kv_used:
        args["seqused_k"] = None
    with forbid_host_readback():
        result = _forward(entry, args)
    assert result[0].shape == (batch, sq, 8, 512) and result[0].dtype == dtype
    assert result[1].shape == (batch, 8, sq) and result[1].dtype == torch.float32
    if entry != "cpp":
        assert all(t.numel() == 0 and t.device == args["q"].device for t in result[2:])
    if entry != "fake":
        used = min(sq, 3)
        assert (result[0][:, :used].cpu() == 0).all()
        if sk:
            expected = torch.full((batch, 8, used), float(min(sk, 17) if kv_used else sk)).log()
            torch.testing.assert_close(result[1][:, :, :used].cpu(), expected)
        else:
            assert (result[0].cpu() == 0).all() and result[1].cpu().isposinf().all()


@pytest.mark.parametrize("axis", ["q", "k"])
@pytest.mark.parametrize(
    "fault,match",
    [
        ("dtype", "dtype int32"),
        ("device", "same NPU device"),
        ("rank", "shape"),
        ("count", "shape"),
        ("stride", "contiguous"),
    ],
)
def test_used_descriptor_rejections(entry, dtype, axis, fault, match):
    args = inputs(dtype)
    key = "seqused_" + axis
    t = args[key]
    args[key] = {
        "dtype": lambda: t.to(torch.int64),
        "device": lambda: t.cpu(),
        "rank": lambda: t.unsqueeze(0),
        "count": lambda: t[:1],
        "stride": lambda: torch.empty(4, dtype=torch.int32, device="npu")[::2],
    }[fault]()
    with pytest.raises(RuntimeError, match=match):
        _forward(entry, args)


@pytest.mark.parametrize("name", ["q", "qv", "k", "v"])
@pytest.mark.parametrize("fault", ["stride", "gradient"])
def test_tensor_rejections(entry, dtype, name, fault):
    args = inputs(dtype)
    if fault == "stride":
        shape = list(args[name].shape)
        shape[-1] *= 2
        args[name] = torch.empty(shape, dtype=dtype, device="npu")[..., ::2]
    else:
        args[name].requires_grad_(True)
    with pytest.raises(
        RuntimeError, match="contiguous last dimension" if fault == "stride" else "gradients"
    ):
        _forward(entry, args)


def test_explicit_metadata_rejected(entry, dtype):
    args = inputs(dtype)
    args["scheduler_metadata"] = torch.empty(426, dtype=torch.int32, device="npu")
    with pytest.raises(RuntimeError, match="Q-used with explicit metadata"):
        _forward(entry, args)


def test_metadata_word_overflow(entry, dtype):
    args = inputs(dtype, batch=0, sq=0, sk=0)
    batch = ((1 << 32) - 1 - 410) // 8 + 1
    for name in ("q", "qv", "k", "v"):
        t = args[name]
        args[name] = torch.empty((batch, 0, *t.shape[2:]), dtype=dtype, device="npu")
    # Reject the static record bound before allocating or reading any lengths.
    with pytest.raises(RuntimeError, match="used metadata word count overflow"):
        _forward(entry, args)


def test_fullgraph_and_length_updates(dtype):
    args = inputs(dtype)
    compiled = torch.compile(api._flash_attn_forward, backend="eager", fullgraph=True)
    for q_used, k_used in (([3, 1], [17, 0]), ([1, 0], [0, 33]), ([7, 7], [33, 17])):
        args["seqused_q"].copy_(torch.tensor(q_used, dtype=torch.int32))
        args["seqused_k"].copy_(torch.tensor(k_used, dtype=torch.int32))
        # Dynamo enumerates Tensor methods while tracing; replacing them with
        # mocks breaks that introspection. Guard eager enqueue separately.
        with forbid_host_readback():
            eager = api._flash_attn_forward(**args)
        out, lse, *_ = compiled(**args)
        for b, (q, k) in enumerate(zip(q_used, k_used)):
            assert (out[b, :q].cpu() == 0).all()
            torch.testing.assert_close(lse[b, :, :q].cpu(), torch.full((8, q), float(k)).log())
            torch.testing.assert_close(out[b, :q], eager[0][b, :q], rtol=0, atol=0)
            torch.testing.assert_close(lse[b, :, :q], eager[1][b, :, :q], rtol=0, atol=0)


def test_alias_clone_offset_and_updates(used_attention, used_probe):
    case = production_case(used_attention, used_probe, used_case(uq=[3, 1], uk=[3, 1], heads=8))
    args = forward_args(case)
    # Both read-only used inputs can alias a contiguous view with a nonzero offset.
    args["seqused_k"] = args["seqused_q"]
    assert args["seqused_q"].storage_offset() > 0
    for length in (3, 2, 0):
        args["seqused_q"][0] = length
        case["item"]["uq"] = [length, 1]
        case["item"]["uk"] = [length, 1]
        for clone in (False, True):
            current = dict(args)
            if clone:
                current["seqused_q"] = args["seqused_q"].clone()
                current["seqused_k"] = args["seqused_k"].clone()
            with forbid_host_readback():
                result = _forward("eager", current)
            check_result(case, result)


def test_cross_stream_allocator_lifetime(dtype):
    creator, consumer = torch.npu.Stream(), torch.npu.Stream()
    consumer.wait_stream(torch.npu.current_stream())
    results = []
    for step in range(8):
        with torch.npu.stream(creator):
            args = inputs(dtype)
            args["v"].fill_(step + 1)
            args["seqused_q"][1] = 1
            args["seqused_k"][1] = 0
        consumer.wait_stream(creator)
        with torch.npu.stream(consumer), forbid_host_readback():
            results.append(_forward("eager", args))
        del args
        # Reuse the allocating stream immediately, before consuming outputs.
        with torch.npu.stream(creator):
            churn = [
                torch.full((2, 7, 8, 512), torch.nan, dtype=dtype, device="npu") for _ in range(12)
            ]
            del churn
    consumer.synchronize()
    for step, (out, lse, *_) in enumerate(results):
        torch.testing.assert_close(out[0, :3].cpu(), torch.full((3, 8, 512), step + 1, dtype=dtype))
        assert (out[1, :1].cpu() == 0).all() and lse[1, :, :1].cpu().isneginf().all()
        torch.testing.assert_close(lse[0, :, :3].cpu(), torch.full((8, 3), 17.0).log())
    creator.synchronize()


@pytest.mark.parametrize("kind", ["fp16", "bf16"])
@pytest.mark.parametrize(
    "fault",
    [
        "q_negative",
        "q_over",
        "k_negative",
        "k_over",
        "zero_q_bad_k",
        "zero_k_bad_q",
        "empty_q",
        "empty_k",
    ],
)
def test_runtime_trap(kind, fault, tmp_path):
    base = Path(os.environ.get("MLA_BSND_USED_DIR", tmp_path)) / kind / ("trap-" + fault)
    base.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "tests.mla.support.mla_bsnd_used_error_probe",
        kind,
        fault,
        str(base),
    ]
    (base / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    with (base / "process.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=90)
    assert result.returncode == 0, (base / "process.log").read_text()[-4000:]
    report = json.loads((base / "runtime-error.json").read_text())
    assert "trap error" in report["error"] and report["elapsed_seconds"] < 15
