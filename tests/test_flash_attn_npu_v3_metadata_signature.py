"""Official FA3 metadata arguments and the ordinary-FA keyword compatibility layer.

Signature source: Dao-AILab/flash-attention d15f1531a460ba456f41b01a774f33ab2db8febf,
hopper/flash_attn_interface.py::get_scheduler_metadata. MLA generation remains
explicitly rejected until the public producer/consumer path is connected.
"""

from contextlib import nullcontext
import inspect

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.test_flash_attn_npu_v3_metadata import _assert_bsnd_matches_ref


OFFICIAL_PARAMETERS = (
    "batch_size",
    "max_seqlen_q",
    "max_seqlen_k",
    "num_heads_q",
    "num_heads_kv",
    "headdim",
    "cache_seqlens",
    "qkv_dtype",
    "headdim_v",
    "cu_seqlens_q",
    "cu_seqlens_k_new",
    "cache_leftpad",
    "page_size",
    "max_seqlen_k_new",
    "causal",
    "window_size",
    "attention_chunk",
    "has_softcap",
    "num_splits",
    "pack_gqa",
    "sm_margin",
)


@pytest.fixture(scope="module", autouse=True)
def require_910b():
    assert "Ascend910B" in torch.npu.get_device_name(0)


@pytest.fixture(params=[torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def dtype(request):
    return request.param


@pytest.fixture(params=["public", "op", "cpp", "public_fake", "op_fake"])
def entry(request):
    with FakeTensorMode() if request.param.endswith("fake") else nullcontext():
        yield request.param


def arguments(dtype):
    return dict(
        batch_size=2,
        max_seqlen_q=3,
        max_seqlen_k=128,
        num_heads_q=4,
        num_heads_kv=2,
        headdim=128,
        cache_seqlens=torch.full((2,), 128, dtype=torch.int32, device="npu"),
        qkv_dtype=dtype,
        num_splits=1,
    )


def metadata(entry, args):
    if entry.startswith("public"):
        return api.get_scheduler_metadata(**args)
    bound = inspect.signature(api.get_scheduler_metadata).bind(**args)
    bound.apply_defaults()
    internal = dict(bound.arguments)
    internal["headdim_v"] = internal["headdim_v"] or internal["headdim"]
    internal["window_size_left"], internal["window_size_right"] = internal.pop("window_size")
    internal.update(cu_seqlens_k=None, seqused_q=None)
    if entry == "cpp":
        ordered = inspect.signature(api._get_scheduler_metadata_fake).bind(**internal)
        return api.flash_attn_npu_3.get_scheduler_metadata(*ordered.arguments.values())
    return api._get_scheduler_metadata_op(**internal)


def test_official_positional_prefix_and_keyword_only_compatibility():
    parameters = inspect.signature(api.get_scheduler_metadata).parameters
    assert tuple(parameters) == (*OFFICIAL_PARAMETERS, "softcap", "softmax_scale")
    assert all(
        parameters[n].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD for n in OFFICIAL_PARAMETERS
    )
    defaults = (
        torch.bfloat16,
        None,
        None,
        None,
        None,
        None,
        0,
        False,
        (-1, -1),
        0,
        False,
        0,
        None,
        0,
    )
    assert all(parameters[n].default is inspect.Parameter.empty for n in OFFICIAL_PARAMETERS[:7])
    assert tuple(parameters[n].default for n in OFFICIAL_PARAMETERS[7:]) == defaults
    for name in ("softcap", "softmax_scale"):
        assert parameters[name].kind == inspect.Parameter.KEYWORD_ONLY
        assert parameters[name].default is None
    # Supplying the former final scale position must not bind to a new option.
    with pytest.raises(TypeError):
        api.get_scheduler_metadata(*([None] * (len(OFFICIAL_PARAMETERS) + 1)))
    fake = inspect.signature(api._get_scheduler_metadata_fake).parameters
    schema = torch.ops.flash_attn_npu_3._get_scheduler_metadata.default._schema
    assert [a.name for a in schema.arguments] == list(fake)
    assert str(next(a.type for a in schema.arguments if a.name == "has_softcap")) == "bool"


@pytest.mark.parametrize(
    "has_softcap,softcap", [(False, None), (False, 0.0), (False, 15.0), (True, 15.0)]
)
def test_ordinary_metadata_compatibility(entry, dtype, has_softcap, softcap):
    args = arguments(dtype)
    actual = metadata(
        entry, dict(args, has_softcap=has_softcap, softcap=softcap, softmax_scale=0.23)
    )
    expected = api.get_scheduler_metadata(**args, softcap=softcap or 0.0, softmax_scale=0.23)
    assert actual.shape == expected.shape == (2384,)
    assert actual.dtype == expected.dtype == torch.uint8
    assert actual.device == expected.device and actual.is_contiguous()
    if not entry.endswith("fake"):
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
        if entry != "cpp":
            assert actual._fa_scheduler_params == expected._fa_scheduler_params
            assert actual._fa_scheduler_params["softcap"] == (softcap or 0.0)


@pytest.mark.parametrize(
    "options,message",
    [
        (dict(has_softcap=True), "not implemented yet without the legacy softcap keyword"),
        (dict(has_softcap=True, softcap=0.0), "conflicts with legacy softcap=0.0"),
        (dict(softcap=-1.0), "finite and non-negative"),
        (dict(softcap=float("inf")), "finite and non-negative"),
        (dict(softcap=float("-inf")), "finite and non-negative"),
        (dict(softcap=float("nan")), "finite and non-negative"),
    ],
)
def test_unsupported_and_conflicting_softcap_is_rejected(entry, dtype, options, message):
    with pytest.raises(RuntimeError, match=message):
        metadata(entry, dict(arguments(dtype), **options))


@pytest.mark.parametrize("mode", ["official", "legacy", "flag_and_legacy"])
@pytest.mark.parametrize("scale", [None, 0.0, 0.23, -0.11])
def test_metadata_forward_precision(dtype, mode, scale):
    args = arguments(dtype)
    cap = 0.0 if mode == "official" else 15.0
    options = dict(softmax_scale=scale)
    if mode != "official":
        options["softcap"] = cap
    if mode == "flag_and_legacy":
        options["has_softcap"] = True
    if mode in ("official", "flag_and_legacy"):
        # Exercise every official position, including has_softcap and the
        # three subsequent scheduling options, through the real custom op.
        bound = inspect.signature(api.get_scheduler_metadata).bind(**args)
        bound.apply_defaults()
        bound.arguments["has_softcap"] = options.pop("has_softcap", False)
        md = api.get_scheduler_metadata(
            *(bound.arguments[n] for n in OFFICIAL_PARAMETERS), **options
        )
    else:
        md = api.get_scheduler_metadata(**args, **options)
    generator = torch.Generator().manual_seed(20260929)
    q, k, v = [
        torch.randn(shape, generator=generator, dtype=torch.float32).to(dtype).to("npu")
        for shape in ((2, 3, 4, 128), (2, 128, 2, 128), (2, 128, 2, 128))
    ]
    out, lse, *_ = api.flash_attn_with_kvcache(
        q,
        k,
        v,
        cache_seqlens=args["cache_seqlens"],
        softmax_scale=scale,
        softcap=cap,
        num_splits=1,
        scheduler_metadata=md,
        return_softmax_lse=True,
    )
    _assert_bsnd_matches_ref(
        out,
        lse,
        q,
        (k.cpu(), v.cpu()),
        batch_size=2,
        q_seqlen=3,
        num_heads=4,
        head_size=128,
        scale=128**-0.5 if scale is None else scale,
        data_type=dtype,
        is_causal=False,
        softcap=cap,
    )


@pytest.mark.parametrize("legacy", [False, True])
def test_official_metadata_torch_compile(dtype, legacy):
    options = (
        dict(has_softcap=False)
        if not legacy
        else dict(has_softcap=True, softcap=15.0, softmax_scale=0.23)
    )

    def generate(lengths):
        return api.get_scheduler_metadata(
            2, 3, 128, 4, 2, 128, lengths, dtype, num_splits=1, **options
        )

    compiled = torch.compile(generate, backend="eager", fullgraph=True)
    for lengths in ([128, 96], [64, 128], [17, 31]):
        tensor = torch.tensor(lengths, dtype=torch.int32, device="npu")
        actual, expected = compiled(tensor), generate(tensor)
        assert actual.shape == (2384,) and actual.dtype == torch.uint8
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
        assert actual._fa_scheduler_params == expected._fa_scheduler_params
