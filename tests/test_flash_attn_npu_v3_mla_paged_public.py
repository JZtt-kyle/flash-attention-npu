"""BSND read-only PA kvcache: public/raw/fake contract and source/math checks."""

from contextlib import nullcontext
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.mla_metadata_error_probe import forbid_host_readback
from tests.test_flash_attn_npu_v3_mla_contract import _forward
from tests import test_flash_attn_npu_v3_mla_paged_bsnd_production as production
from tests.test_flash_attn_npu_v3_mla_paged_bsnd_production import (
    component,
    generate,
    make_case,
    small_inputs,
    paged_attention as paged_attention,
    paged_probe as paged_probe,
    paged_artifacts as paged_artifacts,
    paged_oracles as paged_oracles,
    storage_fault as storage_fault,
)


# Same page/length matrix as production, using full BSND Q (no public Q-used).
# Keep native FD cases and fix the matrix before collecting any timings.
PERFORMANCE_CASES = [dict(c, uq=None) for c in production.PERFORMANCE_CASES]


def public(args, lse=True):
    args = dict(args)
    args["k_cache"], args["v_cache"] = args.pop("k"), args.pop("v")
    args.setdefault("num_splits", 1)
    return api.flash_attn_with_kvcache(**args, return_softmax_lse=lse)


def invoke(entry, args):
    entry = entry.removesuffix("_fake")
    if entry == "public":
        return public(args)
    if entry == "op":
        return torch.ops.flash_attn_npu_3._mla_paged_bsnd_forward(**args)
    args = dict(args)
    args["seqused_k"] = args.pop("cache_seqlens", None)
    return _forward(entry, args)


@pytest.fixture(params=("public", "op", "eager", "cpp", "public_fake", "op_fake", "eager_fake"))
def entry(request):
    with FakeTensorMode() if request.param.endswith("fake") else nullcontext():
        yield request.param


@pytest.fixture(params=(torch.float16, torch.bfloat16), ids=("fp16", "bf16"))
def dtype(request):
    return request.param


def check_constant(result, lengths, value=3):
    out, lse = result[:2]
    assert out.dtype in (torch.float16, torch.bfloat16) and lse.dtype == torch.float32
    assert lse.shape == (out.shape[0], out.shape[2], out.shape[1])
    for b, n in enumerate(lengths):
        torch.testing.assert_close(out[b].cpu(), torch.full_like(out[b].cpu(), value if n else 0))
        torch.testing.assert_close(lse[b].cpu(), torch.full(lse[b].shape, float(n)).log())


@pytest.mark.parametrize("page", range(16, 1025, 16))
def test_all_page_sizes_at_each_entry(entry, dtype, page):
    args = small_inputs(dtype)
    args["k"] = torch.zeros((7, page, 1, 64), dtype=dtype, device="npu")
    args["v"] = torch.full((7, page, 1, 512), 3, dtype=dtype, device="npu")
    args["cache_seqlens"] = torch.tensor([page + 1, 0], dtype=torch.int32, device="npu")
    args["page_table"][0] = torch.tensor([5, 1, -1, -1], dtype=torch.int32, device="npu")
    args["page_table"][1].fill_(-1)
    args["scheduler_metadata"] = generate(args)
    with forbid_host_readback():
        result = invoke(entry, args)
    assert result[0].shape == (2, 3, 8, 512) and result[1].shape == (2, 8, 3)
    if not entry.endswith("fake"):
        check_constant(result, [page + 1, 0])


@pytest.mark.parametrize("length", (None, 0, 113, 448, "tensor"))
@pytest.mark.parametrize("reuse", (False, True))
@pytest.mark.parametrize("fake", (False, True))
def test_lengths_return_and_offset_record(dtype, length, reuse, fake):
    with FakeTensorMode() if fake else nullcontext():
        args = small_inputs(dtype)
        args["v"].fill_(3)
        expected = [113, 0] if length == "tensor" else [448 if length is None else length] * 2
        args["cache_seqlens"] = torch.tensor(expected, dtype=torch.int32, device="npu")
        if reuse:
            record = generate(args)
            storage = torch.empty(record.numel() + 2, dtype=torch.int32, device="npu")
            storage[1:-1].copy_(record)
            args["scheduler_metadata"] = storage[1:-1]
        if length != "tensor":
            args["cache_seqlens"] = length
        with forbid_host_readback():
            result = public(args)
            only_out = public(args, lse=False)
        assert len(result) == 4 and only_out.shape == result[0].shape
        for accumulator in result[2:]:
            assert accumulator.numel() == 0 and accumulator.device == result[0].device
            assert accumulator.dtype == torch.get_default_dtype()
        if not fake:
            check_constant(result, expected)
            torch.testing.assert_close(only_out, result[0], rtol=0, atol=0)


@pytest.mark.parametrize("length", (-1, 449, 1 << 40))
def test_invalid_integer_before_materialization(dtype, length):
    args = small_inputs(dtype)
    args["cache_seqlens"] = length
    with patch.object(torch, "full", side_effect=AssertionError("invalid length materialized")):
        with pytest.raises(RuntimeError, match="KV capacity"):
            public(args)


@pytest.mark.parametrize("sq,pool,width", ((0, 7, 4), (3, 0, 0), (3, 7, 0), (3, 0, 4), (3, 7, 4)))
def test_empty_shapes_and_zero_kv(entry, dtype, sq, pool, width):
    args = small_inputs(dtype, sq=sq, pool=pool, width=width)
    args["cache_seqlens"].zero_()
    args["page_table"].fill_(-1)
    with forbid_host_readback():
        result = invoke(entry, dict(args, scheduler_metadata=generate(args)))
    assert result[0].shape == (2, sq, 8, 512)
    if not entry.endswith("fake"):
        check_constant(result, [0, 0])


def test_empty_batch(entry, dtype):
    args = small_inputs(dtype)
    for name in ("q", "qv", "page_table", "cache_seqlens"):
        args[name] = args[name][:0]
    result = invoke(entry, dict(args, scheduler_metadata=generate(args)))
    assert result[0].shape == (0, 3, 8, 512) and result[1].shape == (0, 8, 3)


def check_case(probes, case, entry="public", metadata=None):
    args = production.forward_args(case)
    assert args.pop("seqused_q") is None
    with (
        forbid_host_readback(),
        patch.object(torch.Tensor, "contiguous", side_effect=AssertionError("input copy")),
    ):
        if production.source_available(case):
            assert component.launch(probes, case, "oracle") == 0
        record = generate(args) if metadata is None else metadata
        implicit = invoke(entry, args)
        actual = invoke(entry, dict(args, scheduler_metadata=record))
    report = production.check_result(case, actual[:2])
    for a, b in zip(implicit[:2], actual[:2]):
        torch.testing.assert_close(a.cpu(), b.cpu(), rtol=0, atol=0)
    assert [x & 0xFFFFFFFF for x in record.cpu().tolist()] == case["expected"]
    for before, after in case["input_checks"]:
        torch.testing.assert_close(after.cpu(), before, rtol=0, atol=0, equal_nan=True)
    report.update(
        config=case["config"], entry=entry, scale=case["scale"], expected_metadata=case["expected"]
    )
    target = probes[1] / "public-accuracy"
    target.mkdir(exist_ok=True)
    digest = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()[:20]
    torch.save(
        dict(
            host=case["host"],
            pages=case["table_host"],
            actual=[t.cpu() for t in actual[:2]],
            source=[t.cpu() for t in case["outputs"]["oracle"]],
            report=report,
        ),
        target / (digest + ".pt"),
    )
    with (target / "results.jsonl").open("a") as f:
        f.write(json.dumps(report) + "\n")
    return report


@pytest.mark.parametrize("page", range(16, 1025, 16))
@pytest.mark.parametrize("profile", ("contiguous", "pa02_k", "pa02_v", "pa02_both"))
def test_source_and_math(paged_attention, paged_probe, page, profile):
    check_case(
        paged_attention,
        make_case(paged_attention, paged_probe, component.config(page, profile, uq=None)),
    )


@pytest.mark.parametrize(
    "profile",
    (
        "bnsd",
        "independent",
        "broadcast",
        "overlap",
        "table_broadcast",
        "table_overlap",
        "wide_k_0",
        "wide_v_1",
        "table_wide",
    ),
)
@pytest.mark.parametrize("kind", ("public", "cpp"))
def test_strides_and_reuse(paged_attention, paged_probe, profile, kind):
    torch.npu.empty_cache()
    lengths = (17, 17) if profile == "table_broadcast" else (17, 1)
    old = make_case(
        paged_attention, paged_probe, component.config(16, "contiguous", uq=None, lengths=lengths)
    )
    args = production.forward_args(old)
    record = generate(args)
    new = make_case(
        paged_attention,
        paged_probe,
        component.config(
            16,
            profile,
            uq=None,
            lengths=lengths,
            pool=2 if profile.startswith("wide_") else 11,
        ),
        scale=-0.125,
    )
    check_case(paged_attention, new, kind, record)


@pytest.mark.parametrize("mode", ("implicit", "generated", "reused", "integer", "none"))
def test_compile_and_length_updates(dtype, mode):
    torch._dynamo.reset()

    def call(args):
        args = dict(args)
        if mode == "generated":
            args["scheduler_metadata"] = generate(args)
        return public(args)

    compiled = torch.compile(call, backend="eager", fullgraph=True)
    args = small_inputs(dtype)
    args["v"].fill_(3)
    for lengths in ([113, 17], [0, 0], [224, 113]):
        if not isinstance(args["cache_seqlens"], torch.Tensor):
            args["cache_seqlens"] = torch.empty(2, dtype=torch.int32, device="npu")
        args["cache_seqlens"].copy_(torch.tensor(lengths, dtype=torch.int32))
        if mode == "reused":
            args["scheduler_metadata"] = generate(args)
        elif mode == "integer":
            args["cache_seqlens"] = lengths[0]
            lengths = [lengths[0]] * 2
        elif mode == "none":
            args["cache_seqlens"] = None
            lengths = [448] * 2
        check_constant(compiled(args), lengths)


@pytest.mark.parametrize("reuse", (False, True))
def test_stream_lifetime(dtype, reuse):
    creator, consumer = torch.npu.Stream(), torch.npu.Stream()
    results = []
    for step in range(4):
        with torch.npu.stream(creator):
            args = small_inputs(dtype)
            args["v"].fill_(step + 1)
            if reuse:
                args["scheduler_metadata"] = generate(args)
        consumer.wait_stream(creator)
        with torch.npu.stream(consumer), forbid_host_readback():
            results.append(public(args))
        del args
        gc.collect()
        with torch.npu.stream(creator):
            churn = [torch.full((8192,), -77, dtype=dtype, device="npu") for _ in range(16)]
        del churn
    consumer.synchronize()
    for step, result in enumerate(results):
        check_constant(result, [113, 0], value=step + 1)
    creator.synchronize()


@pytest.mark.parametrize(
    "name", ("q", "qv", "k", "v", "page_table", "cache_seqlens", "scheduler_metadata")
)
def test_complete_storage(entry, name, storage_fault):
    args = small_inputs()
    args["cache_seqlens"].zero_()
    args["scheduler_metadata"] = generate(args)
    faulty = args[name]
    if name in ("cache_seqlens", "scheduler_metadata"):
        faulty = torch.empty_strided(
            (2, faulty.numel()), (2 * faulty.numel(), 1), dtype=torch.int32, device="npu"
        )
        faulty[1].copy_(args[name])
        args[name] = faulty[1]
    with (
        storage_fault(faulty),
        forbid_host_readback(),
        pytest.raises(RuntimeError, match="storage"),
    ):
        invoke(entry, args)


@pytest.mark.parametrize(
    "name", ("q", "qv", "k", "v", "page_table", "cache_seqlens", "scheduler_metadata")
)
def test_compiled_storage_rechecked(name, storage_fault):
    torch._dynamo.reset()
    args = small_inputs()
    args["scheduler_metadata"] = generate(args)
    faulty = args[name]
    if name in ("cache_seqlens", "scheduler_metadata"):
        faulty = torch.empty_strided(
            (2, faulty.numel()), (2 * faulty.numel(), 1), dtype=torch.int32, device="npu"
        )
        faulty[1].copy_(args[name])
        args[name] = faulty[1]
    compiled = torch.compile(public, backend="eager", fullgraph=True)
    compiled(args)
    torch.npu.synchronize()
    with storage_fault(faulty), pytest.raises(RuntimeError, match="storage"):
        compiled(args)


@pytest.mark.parametrize(
    "fault",
    (
        "page",
        "capacity",
        "table_rank",
        "table_batch",
        "table_dtype",
        "table_device",
        "table_stride",
        "length_dtype",
        "length_shape",
        "length_device",
        "length_stride",
        "metadata_dtype",
        "metadata_shape",
        "metadata_device",
        "metadata_stride",
        "q_rank",
        "qv_shape",
        "q_inner",
        "dtype",
        "device",
        "grad",
        "nan",
        "inf",
        "overflow",
    ),
)
def test_host_rejections(entry, fault):
    args = small_inputs()
    if fault == "page":
        args["k"], args["v"] = args["k"][:, :17], args["v"][:, :17]
    elif fault == "capacity":
        args["page_table"] = torch.zeros((2, 147), dtype=torch.int32, device="npu")
    elif fault == "q_rank":
        args["q"] = args["q"][0]
    elif fault == "qv_shape":
        args["qv"] = args["qv"][:, :1]
    elif fault == "q_inner":
        args["q"] = torch.empty((2, 3, 8, 128), dtype=torch.float16, device="npu")[..., ::2]
    elif fault == "dtype":
        args["v"] = args["v"].float()
    elif fault == "device":
        args["v"] = args["v"].cpu()
    elif fault == "grad":
        args["qv"].requires_grad_()
    elif fault in ("nan", "inf", "overflow"):
        args["softmax_scale"] = {"nan": float("nan"), "inf": float("inf"), "overflow": 1e100}[fault]
    else:
        prefix, change = fault.split("_")
        name = {"table": "page_table", "length": "cache_seqlens", "metadata": "scheduler_metadata"}[
            prefix
        ]
        if prefix == "metadata":
            args[name] = generate(args)
        tensor = args[name]
        if change == "dtype":
            tensor = tensor.long()
        elif change == "device":
            tensor = tensor.cpu()
        elif change in ("shape", "batch"):
            tensor = tensor[:1]
        elif change == "rank":
            tensor = tensor[0]
        else:
            tensor = torch.empty(
                (*tensor.shape[:-1], 2 * tensor.shape[-1]), dtype=torch.int32, device="npu"
            )[..., ::2]
        args[name] = tensor
    with forbid_host_readback(), pytest.raises(RuntimeError, match="MLA|rank"):
        invoke(entry, args)


@pytest.mark.parametrize(
    "fault",
    (
        "causal",
        "window",
        "split0",
        "split2",
        "softcap",
        "pack",
        "margin",
        "chunk",
        "append",
        "rotary",
        "remap",
        "leftpad",
        "max_q",
    ),
)
@pytest.mark.parametrize("kind", ("public", "eager", "cpp", "public_fake", "eager_fake"))
def test_unconnected_options(kind, fault):
    with FakeTensorMode() if kind.endswith("fake") else nullcontext():
        args = small_inputs()
        is_public = kind.startswith("public")
        if fault == "window":
            args.update({"window_size": (1, -1)} if is_public else {"window_size_left": 1})
        elif fault.startswith("split"):
            args["num_splits"] = int(fault[-1])
        elif fault == "append":
            if is_public:
                with pytest.raises(RuntimeError, match="not implemented yet"):
                    api.flash_attn_with_kvcache(
                        args["q"],
                        args["k"],
                        args["v"],
                        k=args["k"][:1],
                        qv=args["qv"],
                        page_table=args["page_table"],
                        num_splits=1,
                    )
                return
            args["k_new"] = args["k"][:1]
        else:
            name, value = {
                "causal": ("causal", True),
                "softcap": ("softcap", 1.0),
                "pack": ("pack_gqa", False),
                "margin": ("sm_margin", 1),
                "chunk": ("attention_chunk", 1),
                "rotary": ("rotary_interleaved", False),
                "remap": (
                    "cache_batch_idx" if is_public else "kv_batch_idx",
                    args["cache_seqlens"],
                ),
                "leftpad": ("cache_leftpad" if is_public else "leftpad_k", args["cache_seqlens"]),
                "max_q": ("max_seqlen_q", 3),
            }[fault]
            args[name] = value
        with pytest.raises(RuntimeError, match="not implemented yet"):
            invoke(kind, args)


@pytest.mark.parametrize("kind", ("fp16", "bf16"))
@pytest.mark.parametrize(
    "fault",
    (
        "page_negative",
        "page_over",
        "empty_q_page",
        "stale_k",
        "bad_k",
        "empty_pool",
        "magic",
        "dtype",
        "empty_payload",
        "v1",
        "v2",
        "v3",
        "v4",
        "v5",
        "page_size",
        "capacity",
    ),
)
def test_public_release_trap(kind, fault, tmp_path):
    directory = Path(os.environ.get("MLA_PAGED_KVCACHE_DIR", tmp_path)) / kind / ("trap-" + fault)
    directory.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "tests.mla_paged_bsnd_production_error_probe",
        kind,
        fault,
        str(directory),
        "public",
    ]
    (directory / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    with (directory / "process.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=120)
    assert result.returncode == 0, (directory / "process.log").read_text()[-4000:]
    report = json.loads((directory / "runtime-error.json").read_text())
    assert "trap error" in report["error"] and report["elapsed_seconds"] < 15
