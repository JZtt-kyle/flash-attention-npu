"""910B dense MLA online accumulation, currently bounded to KV <= 512."""

from contextlib import nullcontext

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.common.compare import assert_fa_close
from tests.common.mla_ref import mla_reference_pair
from tests.mla_cann_baseline import Case, make_inputs, require_910b
from tests.test_flash_attn_npu_v3_mla_contract import _forward, _inputs
from tests.test_flash_attn_npu_v3_mla_short import test_short_accuracy as _check_accuracy


@pytest.fixture(scope="module", autouse=True)
def device():
    require_910b("npu:0")


CASES = [
    Case("kv129-one-head", "compressed", 1, 1, 129, 1, 1),
    Case("kv255-odd-heads", "compressed", 2, 3, 255, 7, 1),
    Case("kv256", "compressed", 2, 1, 256, 8, 1),
    Case("D01-default", "compressed", 2, 1, 257, 8, 1),
    Case("D01-scale", "compressed", 2, 1, 257, 8, 1, scale=0.23),
    Case("D01-zero", "compressed", 2, 1, 257, 8, 1, scale=0.0),
    Case("kv383", "compressed", 1, 2, 383, 31, 1),
    Case("kv384", "compressed", 2, 7, 384, 128, 1),
    Case("kv385", "compressed", 1, 1, 385, 17, 1),
    Case("kv511-q1", "compressed", 1, 1, 511, 16, 1),
    Case("kv511-q7", "compressed", 1, 7, 511, 16, 1),
    Case("kv512", "compressed", 2, 3, 512, 17, 1),
    Case("task-reuse", "compressed", 2, 65, 257, 32, 1),
    Case("scale-fp32", "compressed", 2, 3, 511, 16, 1, scale=0.230013),
    Case("negative", "compressed", 1, 7, 511, 17, 1, scale=-0.130017),
]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_online_accuracy(case, dtype):
    _check_accuracy(case, dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("biases", [(-80, -20, 35, 100), (100, 35, -20, -80), (-80, 100, -20, 35)])
def test_online_changing_maximum(dtype, biases):
    # Scores are controlled by one content coordinate. Opposite query heads
    # exercise both increasing and decreasing maxima, including exp underflow.
    case = Case("maximum", "compressed", 1, 3, 385, 17, 1, scale=1.0)
    cpu = make_inputs(case, dtype, 20260924)
    cpu["q"].zero_()
    cpu["qv"].zero_()
    cpu["qv"][..., ::2, 0] = 1
    cpu["qv"][..., 1::2, 0] = -1
    for tile, bias in enumerate(biases):
        cpu["v"][:, tile * 128 : (tile + 1) * 128, :, 0] = bias
    refs = mla_reference_pair(**cpu, softmax_scale=case.scale)
    actual = api.flash_attn_func(
        **{name: tensor.npu() for name, tensor in cpu.items()},
        softmax_scale=case.scale,
        return_attn_probs=True,
    )
    for name, tensor, ref, pt in zip(("out", "lse"), actual, refs[:2], refs[2:]):
        assert torch.isfinite(tensor).all()
        assert_fa_close(tensor.cpu(), ref, pt, name=name)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_online_contiguous_storage_offset(dtype):
    cpu = make_inputs(Case("offset", "compressed", 2, 3, 511, 17, 1), dtype, 20260924)
    refs = mla_reference_pair(**cpu)
    inputs, guards = {}, []
    for name, tensor in cpu.items():
        storage = torch.full((tensor.numel() + 2,), float("nan"), dtype=dtype, device="npu")
        view = storage[1:-1].view(tensor.shape)
        view.copy_(tensor)
        assert view.is_contiguous() and view.storage_offset() == 1
        inputs[name] = view
        guards.append(storage)
    actual = api.flash_attn_func(**inputs, return_attn_probs=True)
    for name, tensor, ref, pt in zip(("out", "lse"), actual, refs[:2], refs[2:]):
        assert_fa_close(tensor.cpu(), ref, pt, name=name)
    for storage in guards:
        assert storage[0].isnan() and storage[-1].isnan()


@pytest.mark.parametrize("entry", ["eager", "fake", "cpp"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_online_entries(entry, dtype):
    with FakeTensorMode() if entry == "fake" else nullcontext():
        args = _inputs(dtype=dtype)
        args["k"] = torch.zeros((2, 512, 1, 64), dtype=dtype, device="npu")
        args["v"] = torch.zeros((2, 512, 1, 512), dtype=dtype, device="npu")
        out, lse, oa, la = _forward(entry, args)
        assert out.shape == (2, 3, 4, 512) and out.dtype == dtype
        assert lse.shape == (2, 4, 3) and lse.dtype == torch.float32
        assert oa.numel() == la.numel() == 0
        if entry != "fake":
            torch.testing.assert_close(out, torch.zeros_like(out), rtol=0, atol=0)
            torch.testing.assert_close(lse, torch.full_like(lse, 512.0).log())


@pytest.mark.parametrize("entry", ["eager", "fake", "cpp"])
@pytest.mark.parametrize("length", [513, 8193])
def test_longer_kv_rejected(entry, length):
    with FakeTensorMode() if entry == "fake" else nullcontext():
        args = _inputs()
        args["k"] = torch.empty((2, length, 1, 64), dtype=torch.float16, device="npu")
        args["v"] = torch.empty((2, length, 1, 512), dtype=torch.float16, device="npu")
        with pytest.raises(RuntimeError, match="outside nonempty KV"):
            _forward(entry, args)


@pytest.mark.parametrize("fake", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("length", [17, 257, 511])
def test_dense_public_entries(fake, dtype, length):
    with FakeTensorMode() if fake else nullcontext():
        args = _inputs(dtype=dtype)
        args["k"] = torch.zeros((2, length, 1, 64), dtype=dtype, device="npu")
        args["v"] = torch.zeros((2, length, 1, 512), dtype=dtype, device="npu")
        out, lse = api.flash_attn_func(**args, return_attn_probs=True)
        assert out.shape == (2, 3, 4, 512) and out.dtype == dtype
        assert lse.shape == (2, 4, 3) and lse.dtype == torch.float32
        if not fake:
            torch.testing.assert_close(out, torch.zeros_like(out), rtol=0, atol=0)
            torch.testing.assert_close(lse, torch.full_like(lse, float(length)).log())


@pytest.mark.parametrize("fake", [False, True])
@pytest.mark.parametrize(
    "option,value",
    [
        ("num_splits", 0),
        ("num_splits", 2),
        ("pack_gqa", False),
        ("pack_gqa", True),
        ("sm_margin", 1),
        ("causal", True),
        ("window_size", (0, -1)),
        ("window_size", (-1, 512)),
        ("attention_chunk", 1),
        ("softcap", 10.0),
        ("softcap", -1.0),
    ],
)
def test_dense_public_options_rejected(fake, option, value):
    with FakeTensorMode() if fake else nullcontext():
        args = _inputs()
        args[option] = value
        with pytest.raises(RuntimeError, match="forward is not implemented yet"):
            api.flash_attn_func(**args)


@pytest.mark.parametrize("fake", [False, True])
@pytest.mark.parametrize("name", ["q_descale", "k_descale", "v_descale"])
def test_dense_public_descale_rejected(fake, name):
    with FakeTensorMode() if fake else nullcontext():
        args = _inputs()
        args[name] = torch.ones((2, 4), dtype=torch.float32, device="npu")
        with pytest.raises(RuntimeError, match="forward is not implemented yet"):
            api.flash_attn_func(**args)


@pytest.mark.parametrize("fake", [False, True])
@pytest.mark.parametrize("name", ["q", "k", "v", "qv"])
def test_dense_public_gradient_rejected(fake, name):
    with FakeTensorMode() if fake else nullcontext():
        args = _inputs()
        args[name].requires_grad_(True)
        with pytest.raises(RuntimeError, match="gradients"):
            api.flash_attn_func(**args)


@pytest.mark.parametrize("fake", [False, True])
@pytest.mark.parametrize("scale", [float("inf"), float("nan")])
def test_dense_public_nonfinite_scale_rejected(fake, scale):
    with FakeTensorMode() if fake else nullcontext():
        args = _inputs()
        args["softmax_scale"] = scale
        with pytest.raises(RuntimeError, match="must be finite"):
            api.flash_attn_func(**args)
