"""Internal mixed Q-used: independent Q allocation and physical BSND cache slots."""

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
from tests.mla.e2e.dense.test_flash_attn_npu_v3_mla_contract import (
    _forward,
    dtype as dtype,
    entry as entry,
)
from tests.mla.e2e.mixed.test_flash_attn_npu_v3_mla_mixed_public import inputs as mixed_inputs
from tests.mla.unit.used.test_flash_attn_npu_v3_mla_used_attention import (
    used_attention as used_attention,
    used_probe as used_probe,
    CASES,
    EMPTY_CASES,
    effective,
    launch,
    make_case,
    used_case,
)
from tests.mla.unit.used.test_flash_attn_npu_v3_mla_used_metadata_consumer import (
    used_artifacts as used_artifacts,
    used_oracle as used_oracle,
    check_records,
)
from tests.mla.e2e.tnd.test_flash_attn_npu_v3_mla_varlen_used import check_result as check_result


# Fixed before profiling: mixed T01, unequal/zero lengths, tails, long KV,
# more batches than cores, optional/full KV-used and a multi-block query.
PERFORMANCE_CASES = [dict(c) for c in CASES[:14] if c["layout"] == 2] + [
    used_case(2, [7, 3], [513] * 2, [3, 1], None, heads=16),
    used_case(2, [3, 0, 1], [33] * 3, [2, 0, 1], [0, 0, 0], heads=128),
    used_case(2, [260, 3, 1], [513] * 3, [257, 1, 0], [33, 513, 0], heads=1),
    used_case(2, [7, 3], [513] * 2, [7, 3], [513] * 2, heads=16),
]
ACCURACY_CASES = (
    PERFORMANCE_CASES
    + [c for c in EMPTY_CASES if c["layout"] == 2]
    + [
        used_case(2, [3, 2, 5], [513] * 3, [1, 1, 2], [127, 128, 129], heads=h)
        for h in (1, 15, 16, 17, 127, 129, 255, 256, 257)
    ]
    + [used_case(2, [3, 7], [513] * 2, [1, 2], [17, 257], heads=16, sq=64)]
)


def production_case(probes, producer, spec, **kwargs):
    spec = dict(spec, aic=producer["platform"].aic)
    spec["block_dim"] = spec["aic"] if sum(spec["q"]) and sum(spec["kv"]) else 0
    assert spec["layout"] == 2 and spec["uq"] is not None
    return make_case(probes, producer, spec, **kwargs)


def forward_args(case):
    c = case["context"]
    args = dict(case["devices"]["migrated"])
    for name in ("k", "v"):
        args[name] = args[name].view(c.batch, c.kv.maximum, 1, args[name].shape[-1])
    return dict(
        **args,
        cu_seqlens_q=case["inputs"][0],
        seqused_q=case["inputs"][2],
        seqused_k=case["inputs"][3],
        max_seqlen_q=c.query.maximum,
        softmax_scale=case["scale"],
    )


@pytest.mark.parametrize("spec", ACCURACY_CASES)
@pytest.mark.parametrize("entry", ["eager", "cpp"])
def test_production_accuracy(used_attention, used_probe, spec, entry):
    case = production_case(used_attention, used_probe, spec)
    args = forward_args(case)
    with forbid_host_readback():
        if sum(effective(spec)[0]) and sum(spec["kv"]):
            launch(used_attention, case, "oracle")
        results = [_forward(entry, args) for _ in range(3)]
    for result in results:
        errors = check_result(case, result, source=True)
    assert len({r[0].data_ptr() for r in results}) == (3 if results[0][0].numel() else 1)
    for before, after, _, readonly in case["guards"]:
        if readonly:
            torch.testing.assert_close(after.cpu(), before, rtol=0, atol=0, equal_nan=True)
    for name, values in (
        ("cu_seqlens_q", list(itertools.accumulate(spec["q"], initial=0))),
        ("seqused_q", spec["uq"]),
        ("seqused_k", spec["uk"]),
    ):
        if args[name] is not None:
            assert args[name].cpu().tolist() == values
    with (used_attention[1] / "production-mixed-used-accuracy.jsonl").open("a") as log:
        log.write(json.dumps(dict(item=case["item"], entry=entry, errors=errors)) + "\n")


@pytest.mark.parametrize("scale", [None, 0.0, -0.03125, 0.125])
def test_scale(used_attention, used_probe, scale):
    case = production_case(
        used_attention, used_probe, PERFORMANCE_CASES[0], scale=scale, amplitude=3.0
    )
    with forbid_host_readback():
        launch(used_attention, case, "oracle")
        result = _forward("eager", dict(forward_args(case), softmax_scale=scale))
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
        "production-mixed-used",
        1,
    )


def inputs(dtype=torch.float16, qs=(7, 3), capacity=33):
    args = mixed_inputs(dtype, qs, [min(capacity, 9)] * len(qs), capacity, heads=8)
    args["seqused_k"] = args.pop("cache_seqlens")
    args["seqused_q"] = torch.tensor(
        [123, *[min(q, 2) for q in qs], 456], dtype=torch.int32, device="npu"
    )[1:-1]
    return args


@pytest.mark.parametrize(
    "qs,capacity", [((7, 3), 33), ((), 0), ((), 33), ((0, 0), 33), ((7, 3), 0)]
)
@pytest.mark.parametrize("kv_used", [False, True])
def test_shapes_empty_and_defaults(entry, dtype, qs, capacity, kv_used):
    args = inputs(dtype, qs, capacity)
    if not kv_used:
        args["seqused_k"] = None
    with forbid_host_readback():
        result = _forward(entry, args)
    assert result[0].shape == (sum(qs), 8, 512) and result[0].dtype == dtype
    assert result[1].shape == (8, sum(qs)) and result[1].dtype == torch.float32
    if entry != "cpp":
        assert all(t.numel() == 0 and t.device == args["q"].device for t in result[2:])
    if entry != "fake":
        pos = 0
        for q in qs:
            uq, uk = min(q, 2), min(capacity, 9) if kv_used else capacity
            assert (result[0][pos : pos + uq].cpu() == 0).all()
            expected = (
                torch.full((8, uq), float(uk)).log() if capacity else torch.full((8, uq), torch.inf)
            )
            torch.testing.assert_close(result[1][:, pos : pos + uq].cpu(), expected)
            pos += q
        if capacity == 0:
            assert (result[0].cpu() == 0).all() and result[1].cpu().isposinf().all()


@pytest.mark.parametrize("name", ["cu_seqlens_q", "seqused_q", "seqused_k"])
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
def test_length_descriptor_rejections(entry, dtype, name, fault, match):
    args = inputs(dtype)
    t = args[name]
    args[name] = {
        "dtype": lambda: t.to(torch.int64),
        "device": lambda: t.cpu(),
        "rank": lambda: t.unsqueeze(0),
        "count": lambda: t[:1],
        "stride": lambda: torch.empty(t.numel() * 2, dtype=torch.int32, device="npu")[::2],
    }[fault]()
    with pytest.raises(RuntimeError, match=match):
        _forward(entry, args)


@pytest.mark.parametrize("words", [416, 426])
def test_explicit_metadata_rejected(entry, dtype, words):
    args = inputs(dtype)
    args["scheduler_metadata"] = torch.empty(words, dtype=torch.int32, device="npu")
    with pytest.raises(RuntimeError, match="Q-used with explicit metadata"):
        _forward(entry, args)


@pytest.mark.parametrize("fault", ["max_q", "max_k", "stride", "split", "out", "grad"])
def test_unconnected_combinations(entry, dtype, fault):
    args = inputs(dtype)
    if fault == "max_q":
        args["max_seqlen_q"] = None
    elif fault == "max_k":
        args["max_seqlen_k"] = 33
    elif fault == "stride":
        args["k"] = torch.empty((2, 33, 1, 128), dtype=dtype, device="npu")[..., ::2]
    elif fault == "split":
        args["num_splits"] = 0
    elif fault == "out":
        args["out_"] = torch.empty((10, 8, 512), dtype=dtype, device="npu")
    else:
        args["qv"].requires_grad_(True)
    with pytest.raises(
        RuntimeError, match="max_seqlen_q|not implemented yet|gradient|contiguous last dimension"
    ):
        _forward(entry, args)


def test_public_and_private_kvcache_contract_unchanged(entry, dtype):
    args = inputs(dtype)
    args["cache_seqlens"] = args.pop("seqused_k")
    with pytest.raises(RuntimeError, match="Unknown keyword argument 'seqused_q'"):
        torch.ops.flash_attn_npu_3._mla_tnd_bsnd_forward(**args)
    args["k_cache"], args["v_cache"] = args.pop("k"), args.pop("v")
    with pytest.raises(TypeError, match="seqused_q"):
        api.flash_attn_with_kvcache(**args, num_splits=1)


def test_metadata_word_overflow():
    with FakeTensorMode():
        batch = ((1 << 32) - 1 - 410) // 8 + 1
        args = inputs(qs=(), capacity=0)
        args["k"] = torch.empty((batch, 0, 1, 64), dtype=torch.float16, device="npu")
        args["v"] = torch.empty((batch, 0, 1, 512), dtype=torch.float16, device="npu")
        args["cu_seqlens_q"] = torch.empty(batch + 1, dtype=torch.int32, device="npu")
        args["seqused_k"] = None
        args["seqused_q"] = torch.empty(batch, dtype=torch.int32, device="npu")
        with pytest.raises(RuntimeError, match="used metadata word count overflow"):
            _forward("fake", args)


def test_wide_cache_fake():
    # Physical BSND KV spans use 64 bits even though Q prefixes are int32.
    with FakeTensorMode():
        batch, capacity = (1 << 18) + 1, 16384
        args = inputs(qs=(1,), capacity=0)
        for name, width in (("k", 64), ("v", 512)):
            args[name] = torch.empty((batch, capacity, 1, width), dtype=torch.float16, device="npu")
        args["cu_seqlens_q"] = torch.empty(batch + 1, dtype=torch.int32, device="npu")
        args["seqused_q"] = torch.empty(batch, dtype=torch.int32, device="npu")
        args["seqused_k"] = None
        out, lse, *_ = _forward("fake", args)
        assert out.shape == (1, 8, 512) and lse.shape == (8, 1)


def test_fullgraph_prefix_and_used_updates(dtype):
    args = inputs(dtype)
    compiled = torch.compile(api._flash_attn_forward, backend="eager", fullgraph=True)
    for qs, uq, uk in (
        ([7, 3], [3, 1], [17, 0]),
        ([4, 6], [1, 3], [0, 31]),
        ([0, 10], [0, 10], [0, 33]),
    ):
        for name, values in (
            ("cu_seqlens_q", list(itertools.accumulate(qs, initial=0))),
            ("seqused_q", uq),
            ("seqused_k", uk),
        ):
            args[name].copy_(torch.tensor(values, dtype=torch.int32))
        with forbid_host_readback():
            eager = _forward("eager", args)
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
        used_attention, used_probe, used_case(2, [7, 3], [7, 7], [3, 1], [3, 1], heads=8)
    )
    args = forward_args(case)
    args["seqused_k"] = args["seqused_q"]
    assert all(args[n].storage_offset() > 0 for n in ("cu_seqlens_q", "seqused_q", "k", "v"))
    for length in (3, 2, 0):
        args["seqused_q"][0] = length
        case["item"].update(uq=[length, 1], uk=[length, 1])
        for clone in (False, True):
            current = dict(args)
            if clone:
                for name in ("cu_seqlens_q", "seqused_q", "seqused_k"):
                    current[name] = args[name].clone()
            with forbid_host_readback():
                result = _forward("eager", current)
            check_result(case, result)


def test_updates_preserve_query_and_cache_offsets(used_attention, used_probe):
    case = production_case(
        used_attention, used_probe, used_case(2, [7, 3], [33, 33], [7, 3], [33, 33], heads=8)
    )
    args = dict(forward_args(case), max_seqlen_q=10)
    for qs, uq, uk in (([4, 6], [1, 3], [0, 31]), ([0, 10], [0, 7], [0, 17])):
        case["item"].update(q=qs, uq=uq, uk=uk)
        for name, values in (
            ("cu_seqlens_q", list(itertools.accumulate(qs, initial=0))),
            ("seqused_q", uq),
            ("seqused_k", uk),
        ):
            args[name].copy_(torch.tensor(values, dtype=torch.int32))
        with forbid_host_readback():
            result = _forward("eager", args)
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
        with torch.npu.stream(creator):
            churn = [
                torch.full(shape, -123, dtype=t, device="npu")
                for _ in range(12)
                for shape, t in (
                    ((10, 8, 512), dtype),
                    ((66, 1, 512), dtype),
                    ((2,), torch.int32),
                    ((3,), torch.int32),
                )
            ]
            del churn
    consumer.synchronize()
    for step, (out, lse, *_) in enumerate(results):
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
        "empty_q_bad_k",
        "empty_k_bad_q",
        "q_start",
        "q_tail",
        "q_decrease",
        "q_max",
        "empty_batch_prefix",
    ],
)
def test_runtime_trap(kind, fault, tmp_path):
    base = Path(os.environ.get("MLA_MIXED_USED_DIR", tmp_path)) / kind / ("trap-" + fault)
    base.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "tests.mla.support.mla_mixed_used_error_probe",
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
