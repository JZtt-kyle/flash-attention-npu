"""Public TND used: allocation offsets, v5 validation, accuracy and lifetime."""

from contextlib import nullcontext
import itertools
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


# Fixed before sampling. Native host accepts Hq<=128; accuracy also checks
# larger heads. Include both optional used axes, unequal/zero lengths and tails.
PERFORMANCE_CASES = [
    dict(c, heads=min(c["heads"], 128)) for c in CASES[:14] if c["layout"] == 1
] + [
    used_case(1, [7, 3], [33, 513], [3, 1], None, heads=16),
    used_case(1, [7, 3], [33, 513], None, [17, 257], heads=16),
    used_case(1, [3, 0, 1], [33, 17, 65], [2, 0, 1], [0, 0, 0], heads=128),
    used_case(1, [3] * 10, [511] * 10, [3] * 10, [511] * 10, heads=128),
    used_case(1, [3, 0] * 21, [17, 513] * 21, [2, 0] * 21, [0, 513] * 21, heads=128),
]
ACCURACY_CASES = (
    PERFORMANCE_CASES
    + [c for c in EMPTY_CASES if c["layout"] == 1]
    + [c for c in CASES[14:] if c["layout"] == 1]
    + [
        used_case(1, [3, 2], [513, 33], [2, 1], [513, 17], heads=h)
        for h in (1, 15, 16, 17, 127, 129, 255, 256, 257)
    ]
)


def production_case(probes, producer, spec, **kwargs):
    spec = dict(spec, aic=producer["platform"].aic)
    spec["block_dim"] = spec["aic"] if sum(spec["q"]) and sum(spec["kv"]) else 0
    assert spec["layout"] == 1 and (spec["uq"] is not None or spec["uk"] is not None)
    return make_case(probes, producer, spec, **kwargs)


def forward_args(case):
    c = case["context"]
    return dict(
        **case["devices"]["migrated"],
        cu_seqlens_q=case["inputs"][0],
        cu_seqlens_k=case["inputs"][1],
        seqused_q=case["inputs"][2],
        seqused_k=case["inputs"][3],
        max_seqlen_q=c.query.maximum,
        max_seqlen_k=c.kv.maximum,
        softmax_scale=case["scale"],
    )


def invoke(entry, args):
    if entry.endswith("public"):
        return api.flash_attn_varlen_func(**args, return_attn_probs=True)
    if entry.endswith("op"):
        return torch.ops.flash_attn_npu_3._mla_tnd_forward(**args)
    return _forward(entry, args)


def check_result(case, result, *, source=False):
    item, heads = case["item"], case["context"].heads
    assert result[0].shape == (sum(item["q"]), heads, 512)
    assert result[1].shape == (heads, sum(item["q"]))
    assert result[0].dtype == case["host"]["v"].dtype and result[1].dtype == torch.float32
    host = [t.cpu() for t in result[:2]]
    actual = valid_outputs(case, host)
    errors = check_math(actual, references(case))
    if not sum(item["kv"]):
        assert (host[0] == 0).all() and host[1].isposinf().all()
    uq, uk = effective(item)
    if source and sum(uq) and sum(item["kv"]):
        expected = valid_outputs(case, [t.cpu() for t in case["outputs"]["oracle"]], source=True)
        nonempty = torch.tensor([k > 0 for q, k in zip(uq, uk) for _ in range(q)], dtype=torch.bool)
        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[1][:, nonempty], expected[1][:, nonempty], rtol=0, atol=0)
        assert expected[1][:, ~nonempty].isposinf().all()
    return errors


@pytest.mark.parametrize("spec", ACCURACY_CASES)
@pytest.mark.parametrize("entry", ["public", "cpp"])
def test_production_accuracy(used_attention, used_probe, spec, entry):
    case = production_case(used_attention, used_probe, spec)
    args = forward_args(case)
    with forbid_host_readback():
        if sum(effective(spec)[0]) and sum(spec["kv"]):
            launch(used_attention, case, "oracle")
        results = [invoke(entry, args) for _ in range(3)]
    for result in results:
        errors = check_result(case, result, source=True)
    assert len({r[0].data_ptr() for r in results}) == (3 if results[0][0].numel() else 1)
    for before, after, _, readonly in case["guards"]:
        if readonly:
            torch.testing.assert_close(after.cpu(), before, rtol=0, atol=0, equal_nan=True)
    for tensor, values in zip(
        case["inputs"],
        (
            list(itertools.accumulate(spec["q"], initial=0)),
            list(itertools.accumulate(spec["kv"], initial=0)),
            spec["uq"],
            spec["uk"],
        ),
    ):
        if tensor is not None:
            assert tensor.cpu().tolist() == values
    with (used_attention[1] / "production-varlen-accuracy.jsonl").open("a") as log:
        log.write(json.dumps(dict(item=case["item"], entry=entry, errors=errors)) + "\n")


@pytest.mark.parametrize("scale", [None, 0.0, -0.03125, 0.125])
def test_scale(used_attention, used_probe, scale):
    case = production_case(
        used_attention, used_probe, PERFORMANCE_CASES[0], scale=scale, amplitude=3.0
    )
    args = dict(forward_args(case), softmax_scale=scale)
    with forbid_host_readback():
        launch(used_attention, case, "oracle")
        result = invoke("public", args)
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
        "production-varlen-used",
        1,
    )


def inputs(dtype=torch.float16, qs=(7, 3), ks=(33, 17)):
    return dict(
        q=torch.zeros((sum(qs), 8, 64), dtype=dtype, device="npu"),
        qv=torch.zeros((sum(qs), 8, 512), dtype=dtype, device="npu"),
        k=torch.zeros((sum(ks), 1, 64), dtype=dtype, device="npu"),
        v=torch.zeros((sum(ks), 1, 512), dtype=dtype, device="npu"),
        cu_seqlens_q=torch.tensor(
            list(itertools.accumulate(qs, initial=0)), dtype=torch.int32, device="npu"
        ),
        cu_seqlens_k=torch.tensor(
            list(itertools.accumulate(ks, initial=0)), dtype=torch.int32, device="npu"
        ),
        seqused_q=torch.tensor([min(q, 2) for q in qs], dtype=torch.int32, device="npu"),
        seqused_k=torch.tensor([min(k, 9) for k in ks], dtype=torch.int32, device="npu"),
        max_seqlen_q=max(qs, default=0) + 3,
        max_seqlen_k=max(ks, default=0) + 7,
    )


@pytest.fixture(params=[torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def dtype(request):
    return request.param


@pytest.fixture(params=["public", "op", "eager", "cpp", "fake", "fake-public", "fake-op"])
def entry(request):
    with FakeTensorMode() if request.param.startswith("fake") else nullcontext():
        yield request.param


@pytest.mark.parametrize(
    "qs,ks", [((7, 3), (33, 17)), ((), ()), ((0, 0), (33, 17)), ((7, 3), (0, 0))]
)
@pytest.mark.parametrize("used", ["q", "k", "both"])
def test_shapes_empty_and_defaults(entry, dtype, qs, ks, used):
    args = inputs(dtype, qs, ks)
    if used != "both":
        args["seqused_" + ("q" if used == "k" else "k")] = None
    with forbid_host_readback():
        result = invoke(entry, args)
    assert result[0].shape == (sum(qs), 8, 512) and result[0].dtype == dtype
    assert result[1].shape == (8, sum(qs)) and result[1].dtype == torch.float32
    if entry in ("public", "fake-public"):
        assert isinstance(api.flash_attn_varlen_func(**args), torch.Tensor)
    if entry in ("eager", "fake"):
        assert all(t.numel() == 0 and t.device == args["q"].device for t in result[2:])
    if not entry.startswith("fake"):
        pos = 0
        for q, k in zip(qs, ks):
            uq, uk = q if used == "k" else min(q, 2), k if used == "q" else min(k, 9)
            assert (result[0][pos : pos + uq].cpu() == 0).all()
            expected = (
                torch.full((8, uq), float(uk)).log() if sum(ks) else torch.full((8, uq), torch.inf)
            )
            torch.testing.assert_close(result[1][:, pos : pos + uq].cpu(), expected)
            pos += q
        if not sum(ks):
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
        invoke(entry, args)


def test_explicit_metadata_rejected(dtype):
    for entry in ("public", "op", "eager", "cpp", "fake"):
        with FakeTensorMode() if entry == "fake" else nullcontext():
            args = inputs(dtype)
            args["scheduler_metadata"] = torch.empty(426, dtype=torch.int32, device="npu")
            with pytest.raises(
                (RuntimeError, TypeError), match="explicit metadata|scheduler_metadata"
            ):
                invoke(entry, args)


@pytest.mark.parametrize("entry", ["public", "eager"])
def test_fullgraph_four_array_updates(dtype, entry):
    args = inputs(dtype)
    fn = api.flash_attn_varlen_func if entry == "public" else api._flash_attn_forward
    compiled = torch.compile(fn, backend="eager", fullgraph=True)
    if entry == "public":
        args["return_attn_probs"] = True
    for qs, ks, uq, uk in (
        ([7, 3], [33, 17], [3, 1], [17, 0]),
        ([4, 6], [19, 31], [1, 3], [0, 31]),
        ([0, 10], [0, 50], [0, 10], [0, 50]),
    ):
        args["max_seqlen_k"] = 50
        for name, values in (
            ("cu_seqlens_q", list(itertools.accumulate(qs, initial=0))),
            ("cu_seqlens_k", list(itertools.accumulate(ks, initial=0))),
            ("seqused_q", uq),
            ("seqused_k", uk),
        ):
            args[name].copy_(torch.tensor(values, dtype=torch.int32))
        with forbid_host_readback():
            eager = fn(**args)
        out, lse, *_ = compiled(**args)
        pos = 0
        for capacity, q, k in zip(qs, uq, uk):
            assert (out[pos : pos + q].cpu() == 0).all()
            torch.testing.assert_close(
                lse[:, pos : pos + q].cpu(), torch.full((8, q), float(k)).log()
            )
            torch.testing.assert_close(out[pos : pos + q], eager[0][pos : pos + q], rtol=0, atol=0)
            torch.testing.assert_close(
                lse[:, pos : pos + q], eager[1][:, pos : pos + q], rtol=0, atol=0
            )
            pos += capacity


def test_alias_clone_offset_and_updates(used_attention, used_probe):
    case = production_case(
        used_attention, used_probe, used_case(1, [7, 3], [7, 3], [3, 1], [3, 1], heads=8)
    )
    args = forward_args(case)
    args["cu_seqlens_k"] = args["cu_seqlens_q"]
    args["seqused_k"] = args["seqused_q"]
    assert all(args[n].storage_offset() > 0 for n in ("cu_seqlens_q", "seqused_q"))
    for length in (3, 2, 0):
        args["seqused_q"][0] = length
        case["item"].update(uq=[length, 1], uk=[length, 1])
        for clone in (False, True):
            current = dict(args)
            if clone:
                for name in ("cu_seqlens_q", "cu_seqlens_k", "seqused_q", "seqused_k"):
                    current[name] = args[name].clone()
            with forbid_host_readback():
                result = invoke("public", current)
            check_result(case, result)


def test_four_array_updates_preserve_physical_offsets(used_attention, used_probe):
    case = production_case(
        used_attention, used_probe, used_case(1, [7, 3], [33, 17], [7, 3], [33, 17], heads=8)
    )
    args = dict(forward_args(case), max_seqlen_q=10, max_seqlen_k=50)
    for qs, ks, uq, uk in (
        ([4, 6], [19, 31], [1, 3], [0, 31]),
        ([0, 10], [0, 50], [0, 7], [0, 17]),
    ):
        case["item"].update(q=qs, kv=ks, uq=uq, uk=uk)
        for name, values in (
            ("cu_seqlens_q", list(itertools.accumulate(qs, initial=0))),
            ("cu_seqlens_k", list(itertools.accumulate(ks, initial=0))),
            ("seqused_q", uq),
            ("seqused_k", uk),
        ):
            args[name].copy_(torch.tensor(values, dtype=torch.int32))
        with forbid_host_readback():
            result = invoke("public", args)
        check_result(case, result)


@pytest.mark.parametrize("entry", ["fake", "fake-public", "fake-op"])
def test_metadata_word_overflow(entry):
    with FakeTensorMode():
        args = inputs(qs=(), ks=())
        batch = ((1 << 32) - 1 - 410) // 8 + 1
        for name in ("cu_seqlens_q", "cu_seqlens_k"):
            args[name] = torch.empty(batch + 1, dtype=torch.int32, device="npu")
        with pytest.raises(RuntimeError, match="used metadata word count overflow"):
            invoke(entry, args)


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
            results.append(invoke("public", args))
        del args
        with torch.npu.stream(creator):
            churn = [
                torch.full(shape, -123, dtype=t, device="npu")
                for _ in range(12)
                for shape, t in (((10, 8, 512), dtype), ((2,), torch.int32), ((3,), torch.int32))
            ]
            del churn
    consumer.synchronize()
    for step, (out, lse) in enumerate(results):
        torch.testing.assert_close(out[:2].cpu(), torch.full((2, 8, 512), step + 1, dtype=dtype))
        assert (out[7:8].cpu() == 0).all() and lse[:, 7:8].cpu().isneginf().all()
        torch.testing.assert_close(lse[:, :2].cpu(), torch.full((8, 2), 9.0).log())
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
        "q_start",
        "k_tail",
        "q_decrease",
        "k_max",
    ],
)
def test_runtime_trap(kind, fault, tmp_path):
    base = Path(os.environ.get("MLA_VARLEN_USED_DIR", tmp_path)) / kind / ("trap-" + fault)
    base.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "tests.mla.support.mla_varlen_used_error_probe",
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
