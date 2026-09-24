"""910B short dense MLA acceptance; references never call the device kernel."""

from contextlib import nullcontext

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.common.compare import assert_fa_close
from tests.common.mla_ref import mla_reference_pair
from tests.mla_cann_baseline import Case, make_inputs, require_910b
from tests.test_flash_attn_npu_v3_mla_contract import _forward, _inputs


@pytest.fixture(scope="module", autouse=True)
def device():
    require_910b("npu:0")


# D02 is inherited unchanged. The remaining cases exercise the short-path
# head/sequence tails, both vector lanes, per-core task reuse, and FP32 scale.
CASES = [
    Case("D02", "compressed", 1, 7, 65, 16, 1),
    Case("D02-q1", "compressed", 1, 1, 65, 16, 1),
    Case("one", "compressed", 1, 1, 1, 1, 1),
    Case("odd-heads", "compressed", 2, 3, 15, 7, 1),
    Case("kv16", "compressed", 2, 1, 16, 8, 1),
    Case("kv17", "compressed", 2, 3, 17, 17, 1),
    Case("kv127", "compressed", 1, 2, 127, 31, 1),
    Case("kv128", "compressed", 2, 7, 128, 128, 1),
    Case("task-reuse", "compressed", 2, 65, 65, 32, 1),
    Case("scale", "compressed", 2, 3, 65, 16, 1, scale=0.230013),
    Case("zero", "compressed", 2, 3, 65, 16, 1, scale=0.0),
    Case("negative", "compressed", 1, 7, 65, 17, 1, scale=-0.130017),
]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_short_accuracy(case, dtype):
    cpu = make_inputs(case, dtype, 20260924)
    refs = mla_reference_pair(**cpu, softmax_scale=case.scale)
    inputs = {name: tensor.npu() for name, tensor in cpu.items()}
    before = {name: tensor.clone() for name, tensor in inputs.items()}
    actual = api.flash_attn_func(**inputs, softmax_scale=case.scale, return_attn_probs=True)
    assert actual[0].shape == inputs["qv"].shape and actual[0].dtype == dtype
    assert actual[1].shape == (case.batch, case.heads_q, case.seqlen_q)
    assert actual[1].dtype == torch.float32
    for name, tensor, ref, pt in zip(("out", "lse"), actual, refs[:2], refs[2:]):
        assert torch.isfinite(tensor).all()
        assert_fa_close(tensor.cpu(), ref, pt, name=name)
    repeated = api.flash_attn_func(**inputs, softmax_scale=case.scale, return_attn_probs=True)
    for first, second in zip(actual, repeated):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    for name, tensor in inputs.items():
        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_short_contiguous_storage_offset(dtype):
    case = Case("offset", "compressed", 2, 3, 65, 17, 1)
    cpu = make_inputs(case, dtype, 20260924)
    refs = mla_reference_pair(**cpu)
    inputs = {}
    for name, tensor in cpu.items():
        storage = torch.empty(tensor.numel() + 1, dtype=dtype, device="npu")
        view = storage[1:].view(tensor.shape)
        view.copy_(tensor)
        assert view.is_contiguous() and view.storage_offset() == 1
        inputs[name] = view
    actual = api.flash_attn_func(**inputs, return_attn_probs=True)
    for name, tensor, ref, pt in zip(("out", "lse"), actual, refs[:2], refs[2:]):
        assert_fa_close(tensor.cpu(), ref, pt, name=name)


@pytest.mark.parametrize("entry", ["eager", "fake", "cpp"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_short_entries(entry, dtype):
    with FakeTensorMode() if entry == "fake" else nullcontext():
        args = _inputs(dtype=dtype)
        out, lse, oa, la = _forward(entry, args)
        assert out.shape == (2, 3, 4, 512) and out.dtype == dtype
        assert lse.shape == (2, 4, 3) and lse.dtype == torch.float32
        assert oa.numel() == la.numel() == 0
        if entry != "fake":
            torch.testing.assert_close(out, torch.zeros_like(out), rtol=0, atol=0)
            torch.testing.assert_close(lse, torch.full_like(lse, 17.0).log())


def test_short_bypasses_ordinary_metadata(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("MLA reached ordinary FA metadata generation")

    monkeypatch.setattr(api, "get_scheduler_metadata", unexpected)
    result = api.flash_attn_func(**_inputs())
    assert result.shape == (2, 3, 4, 512)


@pytest.mark.parametrize("entry", ["eager", "fake", "cpp"])
@pytest.mark.parametrize("name", ["q", "k", "v", "qv"])
def test_short_grad_rejected(entry, name):
    with FakeTensorMode() if entry == "fake" else nullcontext():
        args = _inputs()
        args[name].requires_grad_(True)
        with pytest.raises(RuntimeError, match="gradients"):
            _forward(entry, args)


OPTIONS = [
    ("num_splits", 0),
    ("num_splits", 2),
    ("pack_gqa", False),
    ("pack_gqa", True),
    ("sm_margin", 1),
    ("causal", True),
    ("window_size_left", 0),
    ("window_size_right", 128),
    ("attention_chunk", 1),
    ("softcap", -1.0),
    ("softcap", 10.0),
    ("rotary_interleaved", False),
    ("max_seqlen_q", 3),
    ("max_seqlen_k", 17),
]


@pytest.mark.parametrize("entry", ["eager", "fake", "cpp"])
@pytest.mark.parametrize("name,value", OPTIONS)
def test_short_options_rejected(entry, name, value):
    with FakeTensorMode() if entry == "fake" else nullcontext():
        args = _inputs()
        args[name] = value
        with pytest.raises(RuntimeError, match="forward is not implemented yet"):
            _forward(entry, args)


@pytest.mark.parametrize("entry", ["eager", "fake", "cpp"])
@pytest.mark.parametrize(
    "name",
    [
        "k_new",
        "v_new",
        "out_",
        "cu_seqlens_k_new",
        "seqused_q",
        "seqused_k",
        "kv_batch_idx",
        "leftpad_k",
        "rotary_cos",
        "rotary_sin",
        "seqlens_rotary",
        "q_descale",
        "k_descale",
        "v_descale",
        "scheduler_metadata",
    ],
)
def test_short_tensor_options_rejected(entry, name):
    with FakeTensorMode() if entry == "fake" else nullcontext():
        args = _inputs()
        args[name] = torch.empty(1, dtype=torch.int32, device="npu")
        with pytest.raises(RuntimeError, match="forward is not implemented yet"):
            _forward(entry, args)


@pytest.mark.parametrize("entry", ["eager", "fake", "cpp"])
@pytest.mark.parametrize("scale", [float("inf"), float("nan")])
def test_short_nonfinite_scale(entry, scale):
    with FakeTensorMode() if entry == "fake" else nullcontext():
        args = _inputs()
        args["softmax_scale"] = scale
        with pytest.raises(RuntimeError, match="must be finite"):
            _forward(entry, args)


@pytest.mark.parametrize("entry", ["eager", "fake", "cpp"])
def test_short_batch_mismatch(entry):
    with FakeTensorMode() if entry == "fake" else nullcontext():
        args = _inputs()
        args["k"], args["v"] = args["k"][:1], args["v"][:1]
        with pytest.raises(RuntimeError, match="batch dimensions"):
            _forward(entry, args)
