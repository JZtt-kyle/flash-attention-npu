"""Public compact TND: implicit v3, source/math accuracy and stream lifetime."""

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
from tests.mla.e2e.dense.test_flash_attn_npu_v3_mla_contract import _forward, _inputs
from tests.mla.unit.tnd.test_flash_attn_npu_v3_mla_tnd_attention import (
    CASES,
    EMPTY_CASES,
    launch,
    make_case,
    references,
    tnd_attention as tnd_attention,
    tnd_probe as tnd_probe,
)


def public(args, lse=True):
    return api.flash_attn_varlen_func(**args, return_attn_probs=lse)


def invoke(entry, args):
    if entry == "public":
        return public(args)
    if entry == "op":
        return torch.ops.flash_attn_npu_3._mla_tnd_forward(**args)
    return _forward(entry, args)[:2]


@pytest.mark.parametrize("spec", [(q, k, h, None) for q, k, h, _ in CASES + EMPTY_CASES])
@pytest.mark.parametrize("scale", [None, 0.0, -0.125])
def test_public_source_accuracy(tnd_attention, tnd_probe, spec, scale):
    case = make_case(tnd_attention, tnd_probe, spec, scale=scale)
    qs, ks, heads, _ = spec
    args = dict(
        **case["device"],
        cu_seqlens_q=case["query"],
        cu_seqlens_k=case["kv"],
        max_seqlen_q=max(qs, default=0) + 3,
        max_seqlen_k=min(16384, max(ks, default=0) + 7),
        softmax_scale=scale,
    )
    before = {
        n: t.cpu().view(torch.uint8).clone() for n, t in args.items() if isinstance(t, torch.Tensor)
    }
    with forbid_host_readback():
        actual = public(args)
        if sum(qs) and sum(ks):
            launch(tnd_attention, case, "oracle")
    assert actual[0].shape == (sum(qs), heads, 512)
    assert actual[1].shape == (heads, sum(qs))
    actual = [t.cpu() for t in actual]
    refs = references(case)
    for i in range(2):
        assert torch.equal(actual[i].isposinf(), refs[i].isposinf())
        assert torch.equal(actual[i].isneginf(), refs[i].isneginf())
        if actual[i].numel():
            assert_fa_close(actual[i], refs[i], refs[i + 2], name="public-tnd-" + str(i))
    if sum(qs) and sum(ks):
        nonempty = torch.cat([torch.full((q,), k > 0) for q, k in zip(qs, ks)])
        source = [t.cpu() for t in case["outputs"]["oracle"]]
        torch.testing.assert_close(actual[0], source[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[1][:, nonempty], source[1][nonempty].T, rtol=0, atol=0)
        assert source[1][~nonempty].isposinf().all()
    for name, value in before.items():
        assert torch.equal(args[name].cpu().view(torch.uint8), value)
    with (tnd_attention[1] / "public-accuracy.jsonl").open("a") as f:
        f.write(
            json.dumps(dict(spec=spec, scale=scale, source_compared=bool(sum(qs) and sum(ks))))
            + "\n"
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("entry", ["public", "op", "eager", "cpp"])
@pytest.mark.parametrize("fake", [False, True])
def test_entries_and_return(dtype, entry, fake):
    with FakeTensorMode() if fake else nullcontext():
        args = _inputs(layout="tnd", dtype=dtype)
        out, lse = invoke("fake" if fake and entry == "cpp" else entry, args)
        assert out.shape == (4, 4, 512) and out.dtype == dtype
        assert lse.shape == (4, 4) and lse.dtype == torch.float32
        if entry == "public":
            assert isinstance(public(args, lse=False), torch.Tensor)
        if not fake:
            assert (out.cpu() == 0).all()
            torch.testing.assert_close(
                lse.cpu(), torch.tensor([9.0, 9.0, 9.0, 8.0]).log().expand(4, -1)
            )


@pytest.mark.parametrize("entry", ["public", "op", "eager", "cpp", "fake"])
@pytest.mark.parametrize(
    "fault,match",
    [
        ("q_dtype", "dtype int32"),
        ("k_dtype", "dtype int32"),
        ("q_device", "same NPU device"),
        ("k_device", "same NPU device"),
        ("q_rank", "matching shape"),
        ("k_count", "matching shape"),
        ("both_empty", "matching shape"),
        ("q_stride", "contiguous"),
        ("k_stride", "contiguous"),
        ("max_q_negative", "max_seqlen bounds"),
        ("max_k_negative", "max_seqlen bounds"),
        ("max_q_over", "max_seqlen bounds"),
        ("max_k_over", "max_seqlen bounds"),
        ("grad_q", "gradients"),
        ("grad_k", "gradients"),
        ("grad_v", "gradients"),
        ("grad_qv", "gradients"),
        ("scale_nan", "finite in FP32"),
        ("scale_over", "finite in FP32"),
    ],
)
def test_static_rejections(entry, fault, match):
    with FakeTensorMode() if entry == "fake" else nullcontext():
        args = _inputs(layout="tnd")
        if fault[:2] in ("q_", "k_"):
            name = "cu_seqlens_" + fault[0]
            t = args[name]
            if fault.endswith("dtype"):
                args[name] = t.to(torch.int64)
            elif fault.endswith("device"):
                args[name] = t.cpu()
            elif fault.endswith("rank"):
                args[name] = t.unsqueeze(0)
            elif fault.endswith("count"):
                args[name] = t[:-1]
            else:
                args[name] = torch.zeros(6, dtype=torch.int32, device="npu")[::2]
        elif fault == "both_empty":
            for name in ("cu_seqlens_q", "cu_seqlens_k"):
                args[name] = args[name][:0]
        elif fault.startswith("max_"):
            name = "max_seqlen_" + fault[4]
            args[name] = (
                -1 if fault.endswith("negative") else ((1 << 30) + 1 if fault[4] == "q" else 16385)
            )
        elif fault.startswith("grad_"):
            args[fault[5:]].requires_grad_(True)
        else:
            args["softmax_scale"] = float("nan") if fault == "scale_nan" else 1e39
        with pytest.raises((RuntimeError, ValueError), match=match):
            invoke(entry, args)


@pytest.mark.parametrize("entry", ["public", "eager", "cpp", "fake"])
@pytest.mark.parametrize(
    "fault",
    [
        "split0",
        "split2",
        "softcap",
        "pack",
        "margin",
        "chunk",
        "descale",
        "metadata",
    ],
)
def test_unconnected_options(entry, fault):
    with FakeTensorMode() if entry == "fake" else nullcontext():
        args = _inputs(layout="tnd")
        if fault.startswith("split"):
            args["num_splits"] = int(fault[-1])
        elif fault == "descale":
            args["q_descale"] = torch.ones(1, device="npu")
        elif fault == "metadata":
            # The public varlen signature has no metadata parameter.
            if entry == "public":
                with pytest.raises(TypeError):
                    public(
                        dict(
                            args,
                            scheduler_metadata=torch.empty(413, dtype=torch.int32, device="npu"),
                        )
                    )
                return
            args["scheduler_metadata"] = torch.empty(413, dtype=torch.int32, device="npu")
        else:
            key, value = {
                "softcap": ("softcap", 1.0),
                "pack": ("pack_gqa", False),
                "margin": ("sm_margin", 1),
                "chunk": ("attention_chunk", 1),
            }[fault]
            args[key] = value
        with pytest.raises(RuntimeError, match="not implemented yet"):
            invoke(entry, args)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("compiled", [False, True])
def test_prefix_updates_cross_stream_and_allocator(dtype, compiled):
    compute = torch.compile(public, backend="eager", fullgraph=True) if compiled else public
    producer, consumer = torch.npu.Stream(), torch.npu.Stream()
    args = _inputs(layout="tnd", dtype=dtype)
    producer.wait_stream(torch.npu.current_stream())
    results = []
    for qprefix, kprefix in [
        ([0, 3, 4], [0, 9, 17]),
        ([0, 0, 4], [0, 17, 17]),
        ([0, 4, 4], [0, 17, 17]),
        ([0, 1, 4], [0, 1, 17]),
    ]:
        producer.wait_stream(consumer)
        with torch.npu.stream(producer):
            args["cu_seqlens_q"].copy_(torch.tensor(qprefix, dtype=torch.int32, device="npu"))
            args["cu_seqlens_k"].copy_(torch.tensor(kprefix, dtype=torch.int32, device="npu"))
            temporary = {
                n: t.clone() if isinstance(t, torch.Tensor) else t for n, t in args.items()
            }
            temporary.update(max_seqlen_q=4, max_seqlen_k=17)
        consumer.wait_stream(producer)
        with torch.npu.stream(consumer):
            with nullcontext() if compiled else forbid_host_readback():
                result = compute(temporary)
            results.append((result, qprefix, kprefix))
        del temporary
        with torch.npu.stream(producer):
            trash = [torch.full((8192,), -17.0, dtype=dtype, device="npu") for _ in range(16)]
        del trash
    for (out, lse), qp, kp in results:
        assert (out.cpu() == 0).all()
        want = torch.empty(4)
        for b in range(2):
            want[qp[b] : qp[b + 1]] = torch.tensor(float(kp[b + 1] - kp[b])).log()
        torch.testing.assert_close(lse.cpu(), want.expand(4, -1))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_alias_prefixes_with_offsets(dtype):
    args = _inputs(layout="tnd", dtype=dtype)
    args["k"], args["v"] = args["k"][:4], args["v"][:4]
    storage = torch.tensor([1234, 0, 3, 4, 1234], dtype=torch.int32, device="npu")
    args.update(cu_seqlens_q=storage[1:4], cu_seqlens_k=storage[1:4], max_seqlen_k=3)
    with forbid_host_readback():
        out, lse = public(args)
    assert (out.cpu() == 0).all()
    torch.testing.assert_close(lse.cpu(), torch.tensor([3.0, 3.0, 3.0, 1.0]).log().expand(4, -1))
    assert storage.cpu().tolist() == [1234, 0, 3, 4, 1234]


@pytest.mark.parametrize("fault", ["total_q", "total_k", "heads", "words"])
@pytest.mark.parametrize("entry", ["public", "op", "fake"])
def test_fake_allocation_bounds(fault, entry):
    with FakeTensorMode():
        args = _inputs(layout="tnd")
        if fault == "words":
            for n in ("cu_seqlens_q", "cu_seqlens_k"):
                args[n] = torch.empty(
                    ((1 << 32) - 1 - 405) // 4 + 2, dtype=torch.int32, device="npu"
                )
        else:
            total_q = (1 << 31) if fault == "total_q" else 4
            total_k = (1 << 31) if fault == "total_k" else 17
            heads = ((1 << 32) - 1) // 512 + 1 if fault == "heads" else 4
            for n, total, h, d in (
                ("q", total_q, heads, 64),
                ("qv", total_q, heads, 512),
                ("k", total_k, 1, 64),
                ("v", total_k, 1, 512),
            ):
                args[n] = torch.empty((total, h, d), dtype=torch.float16, device="npu")
        with pytest.raises(RuntimeError, match="word count overflow|max_seqlen bounds"):
            invoke(entry, args)


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize(
    "fault",
    [
        "q_start",
        "k_start",
        "q_decrease",
        "k_decrease",
        "q_tail",
        "k_tail",
        "q_max",
        "k_max",
        "empty_q",
        "empty_k",
        "empty_b",
    ],
)
def test_public_runtime_trap(dtype, fault, tmp_path):
    base = Path(os.environ.get("MLA_VARLEN_PUBLIC_DIR", tmp_path)) / dtype / ("trap-" + fault)
    base.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "tests.mla.support.mla_varlen_public_error_probe",
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
