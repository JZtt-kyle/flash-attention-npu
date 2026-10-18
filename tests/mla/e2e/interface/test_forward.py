"""Public entry composition against the unchanged C++ forward contract."""

from contextlib import nullcontext
import inspect

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from flash_attn_npu_3 import flash_attn_npu_interface as api


# Named combinations exercise the entry-specific argument/return adapters.
# Numerical, stride, length and metadata matrices remain owned by each behavior.
CASES = [
    ("func", 1, False, (-1, -1), None),
    ("func", 0, True, (-1, -1), -0.125),
    ("varlen", 1, False, (-1, -1), None),
    ("varlen", 3, True, (2, 1), 0.0),
    ("cache", 1, False, (-1, -1), None),
    ("mixed", 1, True, (2, 1), 0.125),
    ("paged", 0, False, (2, 1), None),
    ("paged_tnd", 3, True, (-1, -1), -0.125),
]


def inputs(layout, dtype):
    packed = layout in ("varlen", "mixed", "paged_tnd")
    qshape = (6, 2, 64) if packed else (2, 3, 2, 64)
    kshape = (64, 1, 64) if layout == "varlen" else (2, 32, 1, 64)
    if layout in ("paged", "paged_tnd"):
        kshape = (4, 16, 1, 64)
    args = dict(
        q=torch.randn(qshape, dtype=dtype, device="npu") * 0.1,
        k=torch.randn(kshape, dtype=dtype, device="npu") * 0.1,
        v=torch.randn((*kshape[:-1], 512), dtype=dtype, device="npu") * 0.1,
        qv=torch.randn((*qshape[:-1], 512), dtype=dtype, device="npu") * 0.1,
    )
    if packed:
        args.update(
            cu_seqlens_q=torch.tensor([0, 3, 6], dtype=torch.int32, device="npu"),
            max_seqlen_q=3,
        )
    if layout == "varlen":
        args.update(
            cu_seqlens_k=torch.tensor([0, 32, 64], dtype=torch.int32, device="npu"),
            max_seqlen_k=32,
            seqused_q=torch.tensor([3, 1], dtype=torch.int32, device="npu"),
        )
    if layout != "func":
        args["seqused_k"] = torch.tensor([32, 7], dtype=torch.int32, device="npu")
    if layout in ("paged", "paged_tnd"):
        args["page_table"] = torch.tensor([[2, 0], [3, 1]], dtype=torch.int32, device="npu")
    return args


def public(layout, args, with_lse=True):
    args = dict(args)
    if layout == "func":
        return api.flash_attn_func(**args, return_attn_probs=with_lse)
    if layout == "varlen":
        return api.flash_attn_varlen_func(**args, return_attn_probs=with_lse)
    args["k_cache"] = args.pop("k")
    args["v_cache"] = args.pop("v")
    args["cache_seqlens"] = args.pop("seqused_k", None)
    return api.flash_attn_with_kvcache(**args, return_softmax_lse=with_lse)


def raw(args):
    args = dict(args)
    args["window_size_left"], args["window_size_right"] = args.pop("window_size")
    bound = inspect.signature(api._flash_attn_forward_fake).bind(**args)
    bound.apply_defaults()
    return api.flash_attn_npu_3.fwd(*bound.arguments.values())


def assert_outputs(actual, expected, args):
    # Output slots beyond Q-used are unspecified. Compare only the used rows.
    for index, (left, right) in enumerate(zip(actual, expected)):
        assert left.shape == right.shape and left.dtype == right.dtype
        if "seqused_q" in args:
            left, right = (
                x[[0, 1, 2, 3]] if index == 0 else x[:, [0, 1, 2, 3]] for x in (left, right)
            )
        torch.testing.assert_close(left, right, rtol=0, atol=0)


@pytest.mark.parametrize("layout,splits,causal,window,scale", CASES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("fake", [False, True], ids=["real", "fake"])
def test_entry_contract(layout, splits, causal, window, scale, dtype, fake):
    with FakeTensorMode() if fake else nullcontext():
        args = inputs(layout, dtype)
        args.update(num_splits=splits, causal=causal, window_size=window, softmax_scale=scale)
        result = public(layout, args)
        assert len(result) == (2 if layout in ("func", "varlen") else 4)
        assert result[0].shape == (*args["q"].shape[:-1], 512)
        assert result[0].dtype == dtype and result[1].dtype == torch.float32
        if len(result) == 4:
            for tensor in result[2:]:
                assert tensor.device == args["q"].device
                assert tensor.shape[0] == (26 if splits == 0 else splits if splits > 1 else 0)
                assert tensor.dtype == torch.float32
        output_only = public(layout, args, with_lse=False)
        assert isinstance(output_only, torch.Tensor) and output_only.shape == result[0].shape
        if not fake:
            expected = raw(args)
            assert_outputs(result[:2], expected[:2], args)
            assert_outputs((output_only,), result[:1], args)
            if len(result) == 4 and splits != 1:
                for actual, reference in zip(result[2:], expected[2:]):
                    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
            if scale is None:
                explicit = public(layout, dict(args, softmax_scale=576**-0.5))
                assert_outputs(result[:2], explicit[:2], args)


@pytest.mark.parametrize("layout", ["func", "varlen", "cache"])
def test_default_compile(layout):
    args = inputs(layout, torch.bfloat16)
    # Omit split and scale: dense/varlen default to 1, cache defaults to 0.
    compiled = torch.compile(
        lambda q: public(layout, dict(args, q=q)), backend="eager", fullgraph=True
    )
    expected = public(layout, args)
    result = compiled(args["q"])
    assert len(result) == len(expected)
    assert_outputs(result[:2], expected[:2], dict(args, num_splits=0 if layout == "cache" else 1))


@pytest.mark.parametrize("layout", ["func", "varlen", "cache"])
@pytest.mark.parametrize(
    "name,value",
    [
        ("attention_chunk", 1),
        ("num_splits", -1),
        ("pack_gqa", "invalid"),
        ("sm_margin", -1),
        ("q_descale", 1),
    ],
)
@pytest.mark.parametrize("fake", [False, True], ids=["real", "fake"])
def test_unsupported_option_names(layout, name, value, fake):
    with FakeTensorMode() if fake else nullcontext():
        args = inputs(layout, torch.float16)
        args[name] = value
        match = name if name in ("pack_gqa", "sm_margin") else f"not implemented yet for {name}"
        with pytest.raises(RuntimeError, match=match):
            public(layout, args)


@pytest.mark.parametrize("layout", ["cache", "mixed", "paged", "paged_tnd"])
def test_integer_cache_lengths(layout):
    args = inputs(layout, torch.float16)
    args.update(num_splits=1, window_size=(2, 1), causal=True)
    explicit = public(
        layout, dict(args, seqused_k=torch.full((2,), 7, dtype=torch.int32, device="npu"))
    )
    integer = public(layout, dict(args, seqused_k=7))
    assert_outputs(integer[:2], explicit[:2], args)
    with pytest.raises(RuntimeError, match="cache_seqlens must be between zero"):
        public(layout, dict(args, seqused_k=33))
