"""Public mixed kvcache: v4 reuse, source accuracy and stream lifetime."""

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
from tests.common.compare import assert_fa_close
from tests.mla.support.mla_metadata_error_probe import forbid_host_readback
from tests.mla.support.public_cases.contract import _forward
from tests.mla.support.components.tnd_bsnd_attention import CASES, launch, make_case, references
from tests.mla.unit.mixed.test_flash_attn_npu_v3_mla_tnd_bsnd_attention import (
    EMPTY_CASES,
    mixed_attention as mixed_attention,
    mixed_probe as mixed_probe,
)

from tests.mla.support.public_cases.mixed_public import (
    dtype as dtype,
    inputs as inputs,
    metadata as metadata,
    public as public,
)


def invoke(entry, args):
    entry = entry.removesuffix("_fake")
    if entry == "public":
        return public(args)
    if entry == "op":
        return torch.ops.flash_attn_npu_3._mla_tnd_bsnd_forward(**args)
    args = dict(args)
    args["seqused_k"] = args.pop("cache_seqlens", None)
    return _forward(entry, args)


@pytest.fixture(params=["public", "op", "eager", "cpp", "public_fake", "op_fake", "eager_fake"])
def entry(request):
    with FakeTensorMode() if request.param.endswith("fake") else nullcontext():
        yield request.param


@pytest.mark.parametrize("spec", [(q, k, c, h, None) for q, k, c, h, _ in CASES + EMPTY_CASES])
@pytest.mark.parametrize("scale", [None, 0.0, -0.125])
def test_public_source_accuracy(mixed_attention, mixed_probe, spec, scale):
    case = make_case(mixed_attention, mixed_probe, spec, scale=scale)
    qs, ks, capacity, heads, _ = spec
    args = dict(
        case["device"],
        cu_seqlens_q=case["query"],
        cache_seqlens=case["kv"],
        max_seqlen_q=max(qs, default=0) + 3,
        softmax_scale=scale,
    )
    for name in ("k", "v"):
        args[name] = args[name].view(len(qs), capacity, 1, args[name].shape[-1])
    record = metadata(args)
    storage = torch.full((record.numel() + 4,), 123456, dtype=torch.int32, device="npu")
    storage[1:-3].copy_(record)
    before = {
        n: t.cpu().view(torch.uint8).clone() for n, t in args.items() if isinstance(t, torch.Tensor)
    }
    md_before = storage.cpu().clone()
    if sum(qs) and capacity:
        launch(mixed_attention, case, "oracle")
    refs = references(case)
    nonempty = torch.tensor([k > 0 for q, k in zip(qs, ks) for _ in range(q)], dtype=torch.bool)
    for md in (None, record.clone(), storage[1:-3]):
        with forbid_host_readback():
            result = public(dict(args, scheduler_metadata=md))
        assert len(result) == 4
        assert result[0].shape == (sum(qs), heads, 512) and result[1].shape == (heads, sum(qs))
        assert all(t.shape == (0,) and t.device == args["q"].device for t in result[2:])
        actual = [t.cpu() for t in result[:2]]
        for i in range(2):
            assert torch.equal(actual[i].isposinf(), refs[i].isposinf())
            assert torch.equal(actual[i].isneginf(), refs[i].isneginf())
            if actual[i].numel():
                assert_fa_close(actual[i], refs[i], refs[i + 2], name="public-mixed-" + str(i))
        if sum(qs) and capacity:
            source = [t.cpu() for t in case["outputs"]["oracle"]]
            torch.testing.assert_close(actual[0], source[0], rtol=0, atol=0)
            torch.testing.assert_close(
                actual[1][:, nonempty], source[1][nonempty].T, rtol=0, atol=0
            )
            assert source[1][~nonempty].isposinf().all()
    for name, value in before.items():
        assert torch.equal(args[name].cpu().view(torch.uint8), value)
    torch.testing.assert_close(storage.cpu(), md_before, rtol=0, atol=0)
    with (mixed_attention[1] / "public-accuracy.jsonl").open("a") as f:
        f.write(
            json.dumps(dict(spec=spec, scale=scale, source_compared=bool(sum(qs) and capacity)))
            + "\n"
        )


@pytest.mark.parametrize(
    "qs,ks,capacity",
    [
        ([3, 0, 1], [17, 0, 33], 64),
        ([], [], 16384),
        ([0, 0], [1, 513], 1024),
        ([3, 0, 1], [0, 0, 0], 0),
        ([3, 0, 1], [0, 0, 0], 64),
    ],
)
@pytest.mark.parametrize("explicit", [False, True])
def test_entries_empty_and_returns(entry, dtype, qs, ks, capacity, explicit):
    args = inputs(dtype, qs, ks, capacity)
    if explicit:
        args["scheduler_metadata"] = metadata(args)
    with forbid_host_readback():
        result = invoke(entry, args)
    assert result[0].shape == (sum(qs), 4, 512) and result[0].dtype == dtype
    assert result[1].shape == (4, sum(qs)) and result[1].dtype == torch.float32
    if entry.startswith("public"):
        assert isinstance(public(args, lse=False), torch.Tensor)
        assert len(result) == 4 and all(t.numel() == 0 for t in result[2:])
    if not entry.endswith("fake"):
        assert (result[0].cpu() == 0).all()
        want = torch.tensor([float(k) for q, k in zip(qs, ks) for _ in range(q)]).log()
        if capacity == 0:
            want.fill_(torch.inf)
        torch.testing.assert_close(result[1].cpu(), want.expand(4, -1))


@pytest.mark.parametrize("length", [None, 0, 1, 63, 64])
@pytest.mark.parametrize("fake", [False, True])
def test_none_integer_lengths(dtype, length, fake):
    with FakeTensorMode() if fake else nullcontext():
        args = inputs(dtype)
        args["cache_seqlens"] = length
        record = metadata(args)
        for md in (None, record):
            with forbid_host_readback():
                result = public(dict(args, scheduler_metadata=md))
            if not fake:
                assert (result[0].cpu() == 0).all()
                torch.testing.assert_close(
                    result[1].cpu(),
                    torch.full((4, 4), float(64 if length is None else length)).log(),
                )
        for invalid in (-1, 65, 1 << 40):
            with pytest.raises(RuntimeError, match="between zero and KV capacity"):
                public(dict(args, cache_seqlens=invalid))


def test_direct_none_lengths(entry, dtype):
    args = inputs(dtype)
    args["cache_seqlens"] = None
    record = metadata(args)
    for md in (None, record):
        with forbid_host_readback():
            result = invoke(entry, dict(args, scheduler_metadata=md))
        if not entry.endswith("fake"):
            assert (result[0].cpu() == 0).all()
            torch.testing.assert_close(result[1].cpu(), torch.full((4, 4), 64.0).log())


@pytest.mark.parametrize("name", ["q", "k", "v", "qv"])
def test_noncontiguous_last_dimension_rejected(entry, dtype, name):
    args = inputs(dtype)
    shape = list(args[name].shape)
    shape[-1] *= 2
    args[name] = torch.zeros(shape, dtype=dtype, device="npu")[..., ::2]
    with pytest.raises(RuntimeError, match="contiguous last dimension"):
        invoke(entry, args)


@pytest.mark.parametrize(
    "fault,match",
    [
        ("q_dtype", "dtype int32"),
        ("q_device", "same NPU device"),
        ("q_rank", "shape"),
        ("q_count", "shape"),
        ("q_stride", "contiguous"),
        ("k_dtype", "dtype int32"),
        ("k_device", "same NPU device"),
        ("k_rank", "shape"),
        ("k_count", "shape"),
        ("k_stride", "contiguous"),
        ("md_dtype", "dtype int32"),
        ("md_device", "same NPU device"),
        ("md_rank", "shape"),
        ("md_count", "shape"),
        ("md_stride", "contiguous"),
        ("max_negative", "max_seqlen bounds"),
        ("max_over", "max_seqlen bounds"),
        ("grad_q", "gradients"),
        ("grad_k", "gradients"),
        ("grad_v", "gradients"),
        ("grad_qv", "gradients"),
        ("scale_nan", "finite in FP32"),
        ("scale_over", "finite in FP32"),
        ("kv_capacity", "max_seqlen bounds"),
        ("cache_batch", "shape"),
    ],
)
def test_static_rejections(entry, dtype, fault, match):
    args = inputs(dtype)
    args["scheduler_metadata"] = torch.empty(421, dtype=torch.int32, device="npu")
    group, detail = fault.split("_")
    if group in ("q", "k", "md"):
        key = {"q": "cu_seqlens_q", "k": "cache_seqlens", "md": "scheduler_metadata"}[group]
        t = args[key]
        if detail == "dtype":
            args[key] = t.to(torch.int64)
        elif detail == "device":
            args[key] = t.cpu()
        elif detail == "rank":
            args[key] = t.unsqueeze(0)
        elif detail == "count":
            args[key] = t[:-1]
        else:
            args[key] = torch.empty(t.numel() * 2, dtype=torch.int32, device="npu")[::2]
    elif group == "max":
        args["max_seqlen_q"] = -1 if detail == "negative" else (1 << 30) + 1
    elif group == "grad":
        args[detail].requires_grad_(True)
    elif group == "scale":
        args["softmax_scale"] = float("nan") if detail == "nan" else 1e39
    else:
        for n in ("k", "v"):
            args[n] = (
                args[n][:2]
                if group == "cache"
                else torch.empty((3, 16385, 1, args[n].shape[-1]), dtype=dtype, device="npu")
            )
    with pytest.raises((RuntimeError, ValueError), match=match):
        invoke(entry, args)


@pytest.mark.parametrize("entry", ["public", "eager", "cpp", "public_fake", "eager_fake"])
@pytest.mark.parametrize(
    "fault",
    [
        "max_missing",
        "max_k",
        "split0",
        "split2",
        "pack",
        "margin",
        "chunk",
        "page",
        "append",
        "descale",
        "rotary",
    ],
)
def test_unconnected_options(entry, fault):
    with FakeTensorMode() if entry.endswith("fake") else nullcontext():
        args = inputs()
        if fault == "max_missing":
            args["max_seqlen_q"] = None
        elif fault == "max_k":
            args["max_seqlen_k"] = 64
        elif fault.startswith("split"):
            args["num_splits"] = int(fault[-1])
        elif fault == "page":
            args["page_table"] = torch.zeros((3, 1), dtype=torch.int32, device="npu")
        elif fault == "append":
            if entry.startswith("public"):
                with pytest.raises(RuntimeError, match="not implemented yet"):
                    api.flash_attn_with_kvcache(
                        args["q"],
                        args["k"],
                        args["v"],
                        k=args["k"][:, :1],
                        qv=args["qv"],
                        cu_seqlens_q=args["cu_seqlens_q"],
                        max_seqlen_q=6,
                        num_splits=1,
                    )
                return
            args["k_new"] = args["k"][:, :1]
        else:
            key, value = {
                "pack": ("pack_gqa", False),
                "margin": ("sm_margin", 1),
                "chunk": ("attention_chunk", 1),
                "descale": ("q_descale", torch.ones(1, device="npu")),
                "rotary": ("rotary_interleaved", False),
            }[fault]
            args[key] = value
        # Public kvcache intentionally has no max_seqlen_k.
        error = TypeError if entry.startswith("public") and fault == "max_k" else RuntimeError
        with pytest.raises(error, match="max_seqlen_q|not implemented yet|unexpected keyword"):
            invoke(entry, args)


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
def test_updates_streams_allocator_and_aliases(dtype, compiled, explicit):
    def run(args):
        record = metadata(args) if explicit else None
        return public(dict(args, scheduler_metadata=record))[:2]

    compute = torch.compile(run, backend="eager", fullgraph=True) if compiled else run
    producer, consumer = torch.npu.Stream(), torch.npu.Stream()
    args = inputs(dtype)
    producer.wait_stream(torch.npu.current_stream())
    results = []
    for qp, lengths in [
        ([0, 3, 3, 4], [17, 0, 33]),
        ([0, 0, 4, 4], [64, 3, 0]),
        ([0, 1, 1, 4], [0, 64, 0]),
        ([0, 2, 3, 4], [64, 1, 32]),
    ]:
        producer.wait_stream(consumer)
        with torch.npu.stream(producer):
            args["cu_seqlens_q"].copy_(torch.tensor(qp, dtype=torch.int32, device="npu"))
            args["cache_seqlens"].copy_(torch.tensor(lengths, dtype=torch.int32, device="npu"))
            temporary = {
                n: t.clone() if isinstance(t, torch.Tensor) else t for n, t in args.items()
            }
            record = metadata(temporary)
        consumer.wait_stream(producer)
        with torch.npu.stream(consumer):
            with forbid_host_readback():
                results.append(
                    (public(dict(temporary, scheduler_metadata=record))[:2], qp, lengths)
                )
            with nullcontext() if compiled else forbid_host_readback():
                results.append((compute(temporary), qp, lengths))
        del temporary, record
        with torch.npu.stream(producer):
            trash = [torch.full((8192,), -17.0, dtype=dtype, device="npu") for _ in range(16)]
        del trash
    for (out, lse), qp, lengths in results:
        assert (out.cpu() == 0).all()
        want = torch.tensor(
            [float(lengths[b]) for b in range(3) for _ in range(qp[b + 1] - qp[b])]
        ).log()
        torch.testing.assert_close(lse.cpu(), want.expand(4, -1))
    # Read-only prefix and KV length views may overlap, including odd offsets.
    args["cache_seqlens"] = args["cu_seqlens_q"][1:]
    assert args["cu_seqlens_q"].storage_offset() == 1
    before = args["cu_seqlens_q"]._base.cpu().clone()
    result = run(args)
    torch.testing.assert_close(
        result[1].cpu(), torch.tensor([2.0, 2.0, 3.0, 4.0]).log().expand(4, -1)
    )
    torch.testing.assert_close(args["cu_seqlens_q"]._base.cpu(), before)


@pytest.mark.parametrize("entry", ["public", "op", "eager"])
@pytest.mark.parametrize(
    "fault", ["total_q", "heads", "words", "rows", "legal_wide_capacity", "legal_compact_q"]
)
def test_fake_allocation_bounds(entry, fault):
    with FakeTensorMode():
        args = inputs()
        batch, capacity, total, heads, max_q = 3, 64, 4, 4, 6
        if fault == "total_q":
            total = 1 << 31
        elif fault == "heads":
            heads = ((1 << 32) - 1) // 512 + 1
        elif fault == "words":
            batch, capacity, total = ((1 << 32) - 1 - 406) // 5 + 1, 0, 0
        elif fault == "rows":
            heads, max_q = 128, 1 << 30
        elif fault == "legal_wide_capacity":
            batch, capacity, total = (1 << 32) // 16384 + 2, 16384, 0
        else:
            batch, capacity, total, heads, max_q = 1 << 20, 0, 1, 1, 1 << 30
        for n, shape in (
            ("q", (total, heads, 64)),
            ("qv", (total, heads, 512)),
            ("k", (batch, capacity, 1, 64)),
            ("v", (batch, capacity, 1, 512)),
        ):
            args[n] = torch.empty(shape, dtype=torch.float16, device="npu")
        args.update(
            cu_seqlens_q=torch.empty(batch + 1, dtype=torch.int32, device="npu"),
            cache_seqlens=None,
            max_seqlen_q=max_q,
        )
        if fault.startswith("legal"):
            result = invoke(entry, args)
            assert result[0].shape == (total, heads, 512)
        else:
            with pytest.raises(RuntimeError, match="word count overflow|max_seqlen bounds"):
                invoke(entry, args)


TRAP_FAULTS = [
    "q_start",
    "q_decrease",
    "q_tail",
    "q_max",
    "k_negative",
    "k_over",
    "stale_q",
    "stale_k",
    "stale_none",
    "stale_integer",
    "magic",
    "version",
    "dtype",
    "shape",
    "capacity",
    "kv_high",
    "endpoint",
    "workspace",
    "empty_b",
    "empty_q",
    "empty_k",
]


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("fault", TRAP_FAULTS)
def test_public_runtime_trap(dtype, fault, tmp_path):
    base = Path(os.environ.get("MLA_MIXED_KVCACHE_DIR", tmp_path)) / dtype / ("trap-" + fault)
    base.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "tests.mla.support.mla_mixed_public_error_probe",
        dtype,
        fault,
        str(base),
    ]
    (base / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    with (base / "process.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=90)
    assert result.returncode == 0, (base / "process.log").read_text()[-4000:]
    report = json.loads((base / "runtime-error.json").read_text())
    assert "trap error" in report["error"] and report["elapsed_seconds"] < 15
