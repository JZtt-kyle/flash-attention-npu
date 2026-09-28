"""910B dense MLA online accumulation, currently bounded to KV <= 1024."""

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
    Case("kv513-one-head", "compressed", 1, 1, 513, 1, 1),
    Case("kv513-q1", "compressed", 1, 1, 513, 16, 1),
    Case("kv513-q7", "compressed", 1, 7, 513, 16, 1),
    Case("kv639", "compressed", 2, 3, 639, 7, 1),
    Case("kv640", "compressed", 2, 1, 640, 8, 1),
    Case("kv641", "compressed", 1, 2, 641, 31, 1),
    Case("kv767", "compressed", 1, 3, 767, 17, 1),
    Case("kv768", "compressed", 2, 7, 768, 128, 1),
    Case("kv769", "compressed", 1, 1, 769, 17, 1),
    Case("kv895", "compressed", 1, 3, 895, 7, 1),
    Case("kv896", "compressed", 2, 1, 896, 8, 1),
    Case("kv897", "compressed", 1, 2, 897, 31, 1),
    Case("kv1023-q1", "compressed", 1, 1, 1023, 16, 1),
    Case("kv1023-q7", "compressed", 1, 7, 1023, 16, 1),
    Case("kv1024-q1", "compressed", 1, 1, 1024, 16, 1),
    Case("kv1024-q7", "compressed", 1, 7, 1024, 16, 1),
    Case("long-task-reuse-odd", "compressed", 2, 33, 769, 17, 1),
    Case("long-task-reuse-even", "compressed", 2, 33, 1024, 32, 1),
    Case("long-scale-fp32", "compressed", 2, 3, 1023, 16, 1, scale=0.230013),
    Case("long-zero", "compressed", 2, 3, 1024, 17, 1, scale=0.0),
    Case("long-negative", "compressed", 1, 7, 1023, 17, 1, scale=-0.130017),
]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_online_accuracy(case, dtype):
    _check_accuracy(case, dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("heads", [2, 4, 6, 8, 10, 11, 12, 13, 14, 15, 16])
def test_online_output_rows(heads, dtype):
    # On C1V2 this covers one through eight output rows per vector lane,
    # including unequal lane sizes around the rescale traversal boundary.
    _check_accuracy(Case("output-rows", "compressed", 1, 3, 383, heads, 1), dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("length", [129, 383, 1024])
def test_online_cached_query_components(dtype, length):
    # Select content coordinates on both sides of every Cube K-slice boundary,
    # plus the first/last RoPE coordinate. More tasks than cores force cached Q
    # to change, including a one-head tail and both L1 KV buffers.
    case = Case("cached-query", "compressed", 2, 21, length, 17, 1, scale=0.23)
    cpu = make_inputs(case, dtype, 20260924)
    cpu["q"].zero_()
    cpu["qv"].zero_()
    coordinates = [0, 127, 128, 255, 256, 383, 384, 511, 512, 575]
    for row in range(case.batch * case.seqlen_q * case.heads_q):
        coordinate = coordinates[row % len(coordinates)]
        component = cpu["qv"] if coordinate < 512 else cpu["q"]
        index = coordinate if coordinate < 512 else coordinate - 512
        component.view(-1, component.shape[-1])[row, index] = 1 if row % 2 else -1
    refs = mla_reference_pair(**cpu, softmax_scale=case.scale)
    inputs = {name: tensor.npu() for name, tensor in cpu.items()}
    first = api.flash_attn_func(**inputs, softmax_scale=case.scale, return_attn_probs=True)
    second = api.flash_attn_func(**inputs, softmax_scale=case.scale, return_attn_probs=True)
    for name, tensor, repeated, ref, pt in zip(("out", "lse"), first, second, refs[:2], refs[2:]):
        assert_fa_close(tensor.cpu(), ref, pt, name=name)
        torch.testing.assert_close(tensor, repeated, rtol=0, atol=0)
    for name, tensor in inputs.items():
        torch.testing.assert_close(tensor.cpu(), cpu[name], rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("length", [129, 383, 1024])
def test_online_output_columns(dtype, length):
    # Distinguish both sides of every output slice boundary while reusing each
    # core for full and one-head tasks, with odd/even numbers of KV tiles.
    case = Case("output-columns", "compressed", 2, 21, length, 17, 1, scale=0.23)
    cpu = make_inputs(case, dtype, 20260928)
    cpu["qv"].zero_()  # RoPE controls weights independently of output values.
    columns = [0, 127, 128, 255, 256, 383, 384, 511]
    for index, column in enumerate(columns):
        values = (torch.arange(length) % (index + 2) - index / 2) * (-1 if index % 2 else 1)
        cpu["v"][..., column] = values[None, :, None]
    refs = mla_reference_pair(**cpu, softmax_scale=case.scale)
    inputs = {name: tensor.npu() for name, tensor in cpu.items()}
    results = [
        api.flash_attn_func(**inputs, softmax_scale=case.scale, return_attn_probs=True)
        for _ in range(2)
    ]
    for name, tensor, repeated, ref, pt in zip(("out", "lse"), *results, refs[:2], refs[2:]):
        assert_fa_close(tensor.cpu(), ref, pt, name=name)
        torch.testing.assert_close(tensor, repeated, rtol=0, atol=0)
    for name, tensor in inputs.items():
        torch.testing.assert_close(tensor.cpu(), cpu[name], rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("length", [527, 528, 529, 575, 576, 577])
def test_online_reduction_tail(length, dtype):
    # Exercise both the Cube's 16-column padding and the in-place reduction's
    # 64-element boundary, including a one-row head tail.
    _check_accuracy(Case("reduction-tail", "compressed", 1, 3, length, 17, 1), dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "biases",
    [
        (-80, -20, 35, 100),
        (100, 35, -20, -80),
        (-80, 100, -20, 35),
        (-80, -60, -40, -20, 20, 40, 60, 100),
        (100, 60, 40, 20, -20, -40, -60, -80),
        (-80, 100, -20, 35, 120, -120, 0, 150),
    ],
)
def test_online_changing_maximum(dtype, biases):
    # Scores are controlled by one content coordinate. Opposite query heads
    # exercise both increasing and decreasing maxima, including exp underflow.
    case = Case("maximum", "compressed", 1, 3, (len(biases) - 1) * 128 + 1, 17, 1, scale=1.0)
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
@pytest.mark.parametrize("length", [511, 1023])
def test_online_contiguous_storage_offset(dtype, length):
    cpu = make_inputs(Case("offset", "compressed", 2, 3, length, 17, 1), dtype, 20260924)
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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_online_queued_calls(dtype):
    # Queue mixed odd/even tile counts without synchronizing each call. Reuse
    # each core across query/head tasks, then check every retained result.
    expected, inputs = [], []
    for seed, length in enumerate((513, 1024, 129, 1023, 257, 640, 65, 897)):
        cpu = make_inputs(Case("queued", "compressed", 1, 17, length, 17, 1), dtype, seed)
        expected.append(mla_reference_pair(**cpu))
        inputs.append({name: tensor.npu() for name, tensor in cpu.items()})
    results = [api.flash_attn_func(**args, return_attn_probs=True) for args in inputs]
    for actual, refs in zip(results, expected):
        for name, tensor, ref, pt in zip(("out", "lse"), actual, refs[:2], refs[2:]):
            assert_fa_close(tensor.cpu(), ref, pt, name=name)


@pytest.mark.parametrize("entry", ["eager", "fake", "cpp"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("length", [512, 513, 1024])
def test_online_entries(entry, dtype, length):
    with FakeTensorMode() if entry == "fake" else nullcontext():
        args = _inputs(dtype=dtype)
        args["k"] = torch.zeros((2, length, 1, 64), dtype=dtype, device="npu")
        args["v"] = torch.zeros((2, length, 1, 512), dtype=dtype, device="npu")
        out, lse, oa, la = _forward(entry, args)
        assert out.shape == (2, 3, 4, 512) and out.dtype == dtype
        assert lse.shape == (2, 4, 3) and lse.dtype == torch.float32
        assert oa.numel() == la.numel() == 0
        if entry != "fake":
            torch.testing.assert_close(out, torch.zeros_like(out), rtol=0, atol=0)
            torch.testing.assert_close(lse, torch.full_like(lse, float(length)).log())


@pytest.mark.parametrize("entry", ["eager", "fake", "cpp"])
@pytest.mark.parametrize("length", [1025, 8193])
def test_longer_kv_rejected(entry, length):
    with FakeTensorMode() if entry == "fake" else nullcontext():
        args = _inputs()
        args["k"] = torch.empty((2, length, 1, 64), dtype=torch.float16, device="npu")
        args["v"] = torch.empty((2, length, 1, 512), dtype=torch.float16, device="npu")
        with pytest.raises(RuntimeError, match="outside nonempty KV"):
            _forward(entry, args)


@pytest.mark.parametrize("fake", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("length", [17, 257, 511, 513, 1023, 1024])
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
