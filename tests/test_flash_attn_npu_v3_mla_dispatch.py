"""Dense MLA dispatcher: independent boundaries and public option rejection."""

from contextlib import nullcontext
import math

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.test_flash_attn_npu_v3_mla_contract import _forward, _inputs
from tests.test_flash_attn_npu_v3_mla_kvcache import generate


@pytest.fixture(params=[torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def dtype(request):
    return request.param


@pytest.mark.parametrize("reuse", [False, True])
def test_dispatcher_schema_and_outputs(dtype, reuse):
    args = _inputs(dtype=dtype)
    lengths = torch.full((2,), 17, dtype=torch.int32, device="npu")
    record = generate(args, lengths) if reuse else None
    values = (*(args[n] for n in ("q", "k", "v", "qv")), -0.23, lengths, record)
    op = torch.ops.flash_attn_npu_3._mla_dense_forward.default
    torch.library.opcheck(
        op,
        values,
        test_utils=("test_schema", "test_faketensor", "test_autograd_registration"),
    )
    expected = _forward(
        "cpp", dict(args, softmax_scale=-0.23, seqused_k=lengths, scheduler_metadata=record)
    )
    # Check normal dispatch, no-grad and inference-mode dispatch keys.
    for mode in (nullcontext(), torch.no_grad(), torch.inference_mode()):
        with mode:
            actual = op(*values)
        for got, want in zip(actual, expected[:2]):
            torch.testing.assert_close(got, want, rtol=0, atol=0)


@pytest.mark.parametrize("fake", [False, True])
@pytest.mark.parametrize("public", [False, True])
@pytest.mark.parametrize(
    "fault,message",
    [
        ("qv_shape", "qv shape"),
        ("dtype", "same dtype"),
        ("device", "same device"),
        ("stride", "noncontiguous BSND"),
        ("grad", "gradients"),
        ("shape", "outside B>0"),
        ("scale", "finite in FP32"),
        ("length_dtype", "cache_seqlens must have dtype"),
        ("metadata_device", "scheduler_metadata must be on the same"),
        ("metadata_shape", "scheduler_metadata must have shape"),
        ("q_rank", "q rank"),
        ("k_rank", "k/v rank"),
        ("qk_dim", "q and k head dimensions"),
        ("kv_shape", "k/v shapes"),
        ("last_stride", "contiguous last dimension"),
    ],
)
def test_dispatcher_rejects_invalid_inputs(dtype, fake, public, fault, message):
    with FakeTensorMode() if fake else nullcontext():
        args = _inputs(dtype=dtype)
        args.pop("softmax_scale")
        if fault == "qv_shape":
            args["qv"] = args["qv"][:, :-1]
        elif fault == "dtype":
            args["v"] = args["v"].float()
        elif fault == "device":
            args["v"] = torch.empty(args["v"].shape, dtype=dtype, device="cpu")
        elif fault == "stride":
            args = _inputs(dtype=dtype, strided=True)
        elif fault == "grad":
            args["q"].requires_grad_(True)
        elif fault == "shape":
            args["k"] = torch.empty((2, 16385, 1, 64), dtype=dtype, device="npu")
            args["v"] = torch.empty((2, 16385, 1, 512), dtype=dtype, device="npu")
        elif fault == "scale":
            args["softmax_scale"] = 1e100
        elif fault == "length_dtype":
            args["cache_seqlens"] = torch.empty(2, dtype=torch.int64, device="npu")
        elif fault == "metadata_device":
            args["scheduler_metadata"] = torch.empty(401, dtype=torch.int32, device="cpu")
        elif fault == "metadata_shape":
            args["scheduler_metadata"] = torch.empty(400, dtype=torch.int32, device="npu")
        elif fault in ("q_rank", "k_rank"):
            name = fault.removesuffix("_rank")
            args[name] = args[name][0]
        elif fault == "qk_dim":
            args["k"] = args["k"][..., :32].contiguous()
        elif fault == "kv_shape":
            args["v"] = args["v"][:, :-1]
        elif fault == "last_stride":
            args["qv"] = torch.empty((2, 3, 4, 1024), dtype=dtype, device="npu")[..., ::2]
        with pytest.raises(RuntimeError, match=message):
            if public:
                api.flash_attn_with_kvcache(
                    args.pop("q"), args.pop("k"), args.pop("v"), **args, num_splits=1
                )
            else:
                torch.ops.flash_attn_npu_3._mla_dense_forward(**args)


@pytest.mark.parametrize("fake", [False, True])
def test_public_dense_cache_rejects_cpu_tensors(dtype, fake):
    with FakeTensorMode() if fake else nullcontext():
        args = _inputs(dtype=dtype)
        tensors = {name: args[name].cpu() for name in ("q", "k", "v", "qv")}
        with pytest.raises(RuntimeError, match="MLA: inputs must be NPU tensors"):
            api.flash_attn_with_kvcache(
                tensors["q"], tensors["k"], tensors["v"], qv=tensors["qv"], num_splits=1
            )


@pytest.mark.parametrize("fake", [False, True])
@pytest.mark.parametrize("public", [False, True])
@pytest.mark.parametrize(
    "scale",
    [
        float("inf"),
        -float("inf"),
        float("nan"),
        1e100,
        math.nextafter(float(torch.finfo(torch.float32).max), math.inf),
        -math.nextafter(float(torch.finfo(torch.float32).max), math.inf),
    ],
)
def test_dispatcher_nonfinite_scale_boundary(dtype, fake, public, scale):
    with FakeTensorMode() if fake else nullcontext():
        args = _inputs(dtype=dtype)
        with pytest.raises(RuntimeError, match="softmax_scale must be finite in FP32"):
            if public:
                api.flash_attn_with_kvcache(
                    args["q"],
                    args["k"],
                    args["v"],
                    qv=args["qv"],
                    softmax_scale=scale,
                    num_splits=1,
                )
            else:
                torch.ops.flash_attn_npu_3._mla_dense_forward(**dict(args, softmax_scale=scale))


@pytest.mark.parametrize("fake", [False, True])
@pytest.mark.parametrize("public", [False, True])
@pytest.mark.parametrize("sign", [-1, 1])
def test_dispatcher_accepts_finite_scale_limit(dtype, fake, public, sign):
    with FakeTensorMode() if fake else nullcontext():
        args = _inputs(dtype=dtype)
        scale = sign * float(torch.finfo(torch.float32).max)
        if public:
            outputs = api.flash_attn_with_kvcache(
                args["q"],
                args["k"],
                args["v"],
                qv=args["qv"],
                softmax_scale=scale,
                num_splits=1,
                return_softmax_lse=True,
            )
        else:
            outputs = torch.ops.flash_attn_npu_3._mla_dense_forward(
                **dict(args, softmax_scale=scale)
            )
        assert outputs[0].shape == (2, 3, 4, 512) and outputs[1].shape == (2, 4, 3)
        if not fake:
            torch.testing.assert_close(outputs[0].cpu(), torch.zeros_like(args["qv"], device="cpu"))
            torch.testing.assert_close(outputs[1].cpu(), torch.full((2, 4, 3), math.log(17)))


@pytest.mark.parametrize("fake", [False, True])
@pytest.mark.parametrize("name", ["q", "k", "v", "qv"])
@pytest.mark.parametrize("inference", [False, True])
def test_dispatcher_rejects_gradients_without_recording(dtype, fake, name, inference):
    with FakeTensorMode() if fake else nullcontext():
        args = _inputs(dtype=dtype)
        args[name].requires_grad_(True)
        with torch.inference_mode() if inference else torch.no_grad():
            with pytest.raises(RuntimeError, match="gradients"):
                torch.ops.flash_attn_npu_3._mla_dense_forward(**args)


@pytest.mark.parametrize("fake", [False, True])
@pytest.mark.parametrize(
    "name,value",
    [
        ("causal", True),
        ("window_size", (3, 0)),
        ("attention_chunk", 4),
        ("softcap", 1.0),
        ("num_splits", 0),
        ("pack_gqa", False),
        ("sm_margin", 1),
        ("rotary_interleaved", False),
        ("max_seqlen_q", 3),
        *[
            (n, "tensor")
            for n in (
                "k",
                "v",
                "cu_seqlens_k_new",
                "cache_batch_idx",
                "cache_leftpad",
                "rotary_cos",
                "rotary_sin",
                "rotary_seqlens",
                "q_descale",
                "k_descale",
                "v_descale",
            )
        ],
    ],
)
def test_public_dispatch_rejects_unsupported_options(dtype, fake, name, value):
    with FakeTensorMode() if fake else nullcontext():
        args = _inputs(dtype=dtype)
        options = dict(qv=args["qv"], num_splits=1)
        options[name] = args["q"] if value == "tensor" else value
        with pytest.raises(RuntimeError, match="MLA compressed forward is not implemented yet"):
            api.flash_attn_with_kvcache(args["q"], args["k"], args["v"], **options)


@pytest.mark.parametrize("fake", [False, True])
def test_public_empty_accumulators_follow_default_dtype(dtype, fake):
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with FakeTensorMode() if fake else nullcontext():
            args = _inputs(dtype=dtype)
            outputs = api.flash_attn_with_kvcache(
                args["q"],
                args["k"],
                args["v"],
                qv=args["qv"],
                num_splits=1,
                return_softmax_lse=True,
            )
            assert outputs[0].dtype == dtype and outputs[1].dtype == torch.float32
            for empty in outputs[2:]:
                assert empty.shape == (0,) and empty.dtype == torch.float64
                assert empty.device == args["q"].device
    finally:
        torch.set_default_dtype(previous)
