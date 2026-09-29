"""MLA supported subset and rejections through eager, fake and raw C++ entries."""

from contextlib import nullcontext
import inspect

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from flash_attn_npu_3 import flash_attn_npu_interface as api


@pytest.fixture(scope="module", autouse=True)
def require_910b():
    assert "Ascend910B" in torch.npu.get_device_name(0)


@pytest.fixture(params=["eager", "fake", "cpp"])
def entry(request):
    with FakeTensorMode() if request.param == "fake" else nullcontext():
        yield request.param


@pytest.fixture(params=[torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def dtype(request):
    return request.param


def _tensor(shape, dtype, strided=False):
    if not strided:
        return torch.zeros(shape, dtype=dtype, device="npu")
    storage = torch.zeros(
        (*(2 * size + 1 for size in shape[:-1]), shape[-1]), dtype=dtype, device="npu"
    )
    return storage[(*(slice(1, None, 2) for _ in shape[:-1]), slice(None))]


def _inputs(representation="compressed", layout="dense", dtype=torch.float16, strided=False):
    dq, dv, hkv = (64, 512, 1) if representation == "compressed" else (192, 128, 2)
    varlen_q = layout in ("tnd", "paged_tnd")
    q_shape = (4, 4, dq) if varlen_q else (2, 3, 4, dq)
    k_shape = (17, hkv, dq) if layout == "tnd" else (2, 17, hkv, dq)
    if layout.startswith("paged"):
        k_shape = (4, 16, hkv, dq)
    args = dict(
        q=_tensor(q_shape, dtype, strided),
        k=_tensor(k_shape, dtype, strided),
        v=_tensor((*k_shape[:-1], dv), dtype, strided),
        qv=_tensor((*q_shape[:-1], dv), dtype, strided) if representation == "compressed" else None,
        softmax_scale=0.23,
    )
    if varlen_q:
        args.update(
            cu_seqlens_q=torch.tensor([0, 3, 4], dtype=torch.int32, device="npu"), max_seqlen_q=3
        )
    if layout == "tnd":
        args.update(
            cu_seqlens_k=torch.tensor([0, 9, 17], dtype=torch.int32, device="npu"), max_seqlen_k=9
        )
    if layout.startswith("paged"):
        args["page_table"] = torch.tensor([[2, 0], [3, 1]], dtype=torch.int32, device="npu")
    return args


def _forward(entry, args):
    if entry == "cpp":
        bound = inspect.signature(api._flash_attn_forward_fake).bind(**args)
        bound.apply_defaults()
        return api.flash_attn_npu_3.fwd(*bound.arguments.values())
    return api._flash_attn_forward(**args)


@pytest.mark.parametrize("representation", ["compressed", "prefill"])
@pytest.mark.parametrize("layout", ["dense", "tnd", "paged", "paged_tnd"])
@pytest.mark.parametrize("strided", [False, True])
def test_mla_supported_subset_and_rejections(entry, representation, layout, dtype, strided):
    args = _inputs(representation, layout, dtype, strided)
    if strided:
        for name in ("q", "k", "v"):
            assert args[name].stride(-1) == 1
            assert not args[name].is_contiguous()
            assert args[name].storage_offset() > 0
    if representation == "compressed" and layout == "dense" and not strided:
        out, lse, *_ = _forward(entry, args)
        assert out.shape == (2, 3, 4, 512) and out.dtype == dtype
        assert lse.shape == (2, 4, 3) and lse.dtype == torch.float32
        if entry != "fake":
            torch.testing.assert_close(out.cpu(), torch.zeros(out.shape, dtype=out.dtype))
            torch.testing.assert_close(lse.cpu(), torch.full(lse.shape, 17.0).log())
    else:
        with pytest.raises(
            RuntimeError, match=f"MLA {representation} forward is not implemented yet"
        ):
            _forward(entry, args)


INVALID = [
    ("q_rank", "q rank"),
    ("k_rank", "k/v rank"),
    ("qk_dim", "q and k head dimensions"),
    ("kv_shape", "k/v shapes"),
    ("qv_shape", "qv shape"),
    ("qv_rank", "qv shape"),
    ("dtype", "FP16 or BF16"),
    ("qv_dtype", "same dtype"),
    ("v_dtype", "same dtype"),
    ("qv_device", "same device"),
    ("v_device", "same device"),
    ("q_stride", "contiguous last dimension"),
    ("k_stride", "contiguous last dimension"),
    ("v_stride", "contiguous last dimension"),
    ("qv_stride", "contiguous last dimension"),
    ("zero_heads", "positive KV head count"),
    ("compressed_heads", "one KV head"),
    ("qv_equal_dims", "Dq=64 and Dv=512"),
    ("missing_qv", "without qv requires Dq=192 and Dv=128"),
    ("page_and_cu_k", "cannot be combined"),
]


@pytest.mark.parametrize("case,message", INVALID, ids=[row[0] for row in INVALID])
def test_mla_invalid_inputs(entry, case, message, dtype):
    args = _inputs(layout="tnd" if case == "page_and_cu_k" else "dense", dtype=dtype)
    if case in ("q_rank", "k_rank", "qv_rank"):
        name = case.removesuffix("_rank")
        args[name] = args[name][0]
    elif case == "qk_dim":
        args["k"] = args["k"][..., :32]
    elif case == "kv_shape":
        args["v"] = args["v"][:, :-1]
    elif case == "qv_shape":
        args["qv"] = args["qv"][:, :-1]
    elif case == "dtype":
        for name in ("q", "k", "v", "qv"):
            args[name] = args[name].float()
    elif case.endswith("_dtype"):
        name = case.removesuffix("_dtype")
        args[name] = args[name].to(torch.bfloat16 if dtype == torch.float16 else torch.float16)
    elif case.endswith("_device"):
        name = case.removesuffix("_device")
        args[name] = args[name].cpu()
    elif case.endswith("_stride"):
        name = case.removesuffix("_stride")
        tensor = args[name]
        args[name] = _tensor((*tensor.shape[:-1], tensor.shape[-1] * 2), tensor.dtype)[..., ::2]
    elif case == "zero_heads":
        args["k"], args["v"] = args["k"][..., :0, :], args["v"][..., :0, :]
    elif case == "compressed_heads":
        args["k"] = args["k"].expand(-1, -1, 2, -1)
        args["v"] = args["v"].expand(-1, -1, 2, -1)
    elif case == "qv_equal_dims":
        # The old implementation silently ran ordinary attention for this qv.
        args["v"], args["qv"] = args["v"][..., :64], args["qv"][..., :64]
    elif case == "missing_qv":
        args["qv"] = None
    elif case == "page_and_cu_k":
        args["page_table"] = torch.zeros((2, 2), dtype=torch.int32, device="npu")
    with pytest.raises(RuntimeError, match=message):
        _forward(entry, args)


@pytest.mark.parametrize("representation", ["compressed", "prefill"])
@pytest.mark.parametrize("sq,sk", [(0, 3), (3, 0), (0, 0)])
def test_mla_empty_is_still_unimplemented(entry, representation, sq, sk, dtype):
    args = _inputs(representation, dtype=dtype)
    args["q"] = args["q"][:, :sq]
    args["k"], args["v"] = args["k"][:, :sk], args["v"][:, :sk]
    if args["qv"] is not None:
        args["qv"] = args["qv"][:, :sq]
    with pytest.raises(RuntimeError, match="forward is not implemented yet"):
        _forward(entry, args)


@pytest.mark.parametrize("representation", ["compressed", "prefill"])
@pytest.mark.parametrize("public_entry", ["dense", "varlen", "kvcache"])
@pytest.mark.parametrize("fake", [False, True])
def test_public_rejects_before_metadata_or_dispatch(
    monkeypatch, representation, public_entry, fake, dtype
):
    def unexpected(*args, **kwargs):
        pytest.fail("MLA reached ordinary metadata/kernel dispatch")

    monkeypatch.setattr(api, "get_scheduler_metadata", unexpected)
    supported = representation == "compressed" and public_entry in ("dense", "kvcache")
    if not supported:
        monkeypatch.setattr(api, "_flash_attn_forward", unexpected)
    with FakeTensorMode() if fake else nullcontext():
        args = _inputs(representation, "tnd" if public_entry == "varlen" else "dense", dtype=dtype)
        if supported:
            if public_entry == "dense":
                out, lse = api.flash_attn_func(**args, return_attn_probs=True)
            else:
                args["k_cache"], args["v_cache"] = args.pop("k"), args.pop("v")
                out, lse, *_ = api.flash_attn_with_kvcache(
                    **args, num_splits=1, return_softmax_lse=True
                )
            assert out.shape == (2, 3, 4, 512) and lse.shape == (2, 4, 3)
            return
        with pytest.raises(RuntimeError, match="forward is not implemented yet"):
            if public_entry == "dense":
                api.flash_attn_func(**args)
            elif public_entry == "varlen":
                api.flash_attn_varlen_func(**args)
            else:
                args["k_cache"], args["v_cache"] = args.pop("k"), args.pop("v")
                api.flash_attn_with_kvcache(**args)


@pytest.mark.parametrize("representation", ["compressed", "prefill"])
@pytest.mark.parametrize("option", ["split", "mask", "softcap", "schedule", "append", "grad"])
def test_mla_options_cannot_fall_through(entry, representation, option, dtype):
    args = _inputs(representation, dtype=dtype)
    if option == "split":
        args["num_splits"] = 256
    elif option == "mask":
        args.update(causal=True, window_size_left=2, window_size_right=0)
    elif option == "softcap":
        args["softcap"] = 15.0
    elif option == "schedule":
        args.update(
            pack_gqa=True,
            sm_margin=1,
            scheduler_metadata=torch.zeros(2384, dtype=torch.uint8, device="npu"),
        )
    elif option == "append":
        args.update(k_new=args["k"][:, :1], v_new=args["v"][:, :1])
    elif option == "grad":
        for name in ("q", "k", "v", "qv"):
            if args[name] is not None:
                args[name].requires_grad_(True)
    before = {name: t.detach().clone() for name, t in args.items() if isinstance(t, torch.Tensor)}
    with pytest.raises(RuntimeError, match="forward is not implemented yet"):
        _forward(entry, args)
    if entry != "fake":
        for name, tensor in before.items():
            torch.testing.assert_close(args[name], tensor, rtol=0, atol=0)


@pytest.mark.parametrize("dims", [(64, 512), (192, 128), (64, 128)])
@pytest.mark.parametrize("has_softcap", [False, True])
def test_mla_metadata_rejected(entry, dims, dtype, has_softcap):
    args = dict(
        batch_size=2,
        max_seqlen_q=3,
        max_seqlen_k=17,
        num_heads_q=4,
        num_heads_kv=1,
        headdim=dims[0],
        headdim_v=dims[1],
        qkv_dtype=dtype,
        cache_seqlens=torch.full((2,), 17, dtype=torch.int32, device="npu"),
        has_softcap=has_softcap,
    )
    with pytest.raises(RuntimeError, match="MLA scheduler metadata is not implemented yet"):
        api.get_scheduler_metadata(**args)
    args.update(
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        cu_seqlens_k_new=None,
        seqused_q=None,
        cache_leftpad=None,
        page_size=None,
        max_seqlen_k_new=0,
        causal=False,
        window_size_left=-1,
        window_size_right=-1,
        attention_chunk=0,
        softcap=0.0,
        num_splits=0,
        pack_gqa=None,
        sm_margin=0,
        softmax_scale=None,
    )
    with pytest.raises(RuntimeError, match="MLA scheduler metadata is not implemented yet"):
        if entry == "cpp":
            bound = inspect.signature(api._get_scheduler_metadata_fake).bind(**args)
            api.flash_attn_npu_3.get_scheduler_metadata(*bound.arguments.values())
        else:
            api._get_scheduler_metadata_op(**args)


@pytest.mark.parametrize(
    "option",
    [
        "k_new",
        "v_new",
        "out_",
        "cu_seqlens_k_new",
        "seqused_q",
        "max_seqlen_q",
        "max_seqlen_k",
        "kv_batch_idx",
        "leftpad_k",
        "rotary_cos",
        "rotary_sin",
        "seqlens_rotary",
        "q_descale",
        "k_descale",
        "v_descale",
        "rotary_interleaved",
        "causal",
        "window_size_left",
        "window_size_right",
        "attention_chunk",
        "softcap",
        "num_splits",
        "pack_gqa",
        "sm_margin",
    ],
)
def test_each_unsupported_dense_option(entry, option, dtype):
    args = _inputs(dtype=dtype)
    if option in ("max_seqlen_q", "max_seqlen_k"):
        args[option] = 17
    elif option in ("rotary_interleaved", "pack_gqa"):
        args[option] = False
    elif option == "causal":
        args[option] = True
    elif option in ("window_size_left", "window_size_right", "num_splits"):
        args[option] = 0
    elif option in ("attention_chunk", "softcap", "sm_margin"):
        args[option] = 1
    else:
        args[option] = torch.empty(1, device="npu")
    with pytest.raises(RuntimeError, match="MLA compressed forward is not implemented yet"):
        _forward(entry, args)


@pytest.mark.parametrize("shape", [(0, 1, 1), (1, 257, 1), (1, 2, 129)])
def test_dense_query_limits(entry, shape, dtype):
    args = _inputs(dtype=dtype)
    args["q"] = torch.empty((*shape, 64), dtype=dtype, device="npu")
    args["qv"] = torch.empty((*shape, 512), dtype=dtype, device="npu")
    args["k"] = args["k"][: shape[0]]
    args["v"] = args["v"][: shape[0]]
    with pytest.raises(RuntimeError, match="outside B>0"):
        _forward(entry, args)


def test_dense_kv_limit(entry, dtype):
    args = _inputs(dtype=dtype)
    args["k"] = torch.empty((2, 513, 1, 64), dtype=dtype, device="npu")
    args["v"] = torch.empty((2, 513, 1, 512), dtype=dtype, device="npu")
    with pytest.raises(RuntimeError, match="outside B>0"):
        _forward(entry, args)


def test_dense_batch_mismatch(entry, dtype):
    args = _inputs(dtype=dtype)
    args["k"], args["v"] = args["k"][:1], args["v"][:1]
    with pytest.raises(RuntimeError, match="dense batch sizes must match"):
        _forward(entry, args)


@pytest.mark.parametrize("scale", [float("inf"), -float("inf"), float("nan"), 1e39])
def test_dense_nonfinite_scale(entry, scale, dtype):
    args = _inputs(dtype=dtype)
    args["softmax_scale"] = scale
    with pytest.raises(RuntimeError, match="softmax_scale must be finite"):
        _forward(entry, args)


def test_fake_batch_field_overflow(dtype):
    with FakeTensorMode():
        args = {
            name: torch.empty((2**32, 1, 1, dim), dtype=dtype, device="npu")
            for name, dim in (("q", 64), ("k", 64), ("v", 512), ("qv", 512))
        }
        with pytest.raises(RuntimeError, match="outside B>0"):
            api._flash_attn_forward(**args)


@pytest.mark.parametrize("layout", ["dense", "tnd"])
def test_ordinary_fake_output_and_metadata_shapes(layout, dtype):
    with FakeTensorMode():
        args = _inputs(layout=layout, dtype=dtype)
        args["v"] = args["k"]
        args["qv"] = None
        out, lse, out_accum, lse_accum = api._flash_attn_forward(**args)
        assert out.shape == args["q"].shape and out.dtype == dtype
        expected_lse = (4, 4) if layout == "tnd" else (2, 4, 3)
        assert lse.shape == expected_lse and lse.dtype == torch.float32
        assert out_accum.numel() == lse_accum.numel() == 0
        metadata = api.get_scheduler_metadata(
            2,
            3,
            17,
            4,
            1,
            64,
            torch.empty(2, dtype=torch.int32, device="npu"),
        )
        assert metadata.shape == (2384,) and metadata.dtype == torch.uint8
