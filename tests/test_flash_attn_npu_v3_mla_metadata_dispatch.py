"""Dense metadata dispatcher: independent validation, schemas and graph capture."""

from contextlib import nullcontext
import inspect

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.test_flash_attn_npu_v3_metadata_signature import metadata
from tests.test_flash_attn_npu_v3_mla_kvcache import malformed, metadata_args


@pytest.fixture(params=[torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def dtype(request):
    return request.param


def arguments(dtype, shape=(2, 3, 4, 17)):
    public = metadata_args(dtype, shape)
    return {
        name: public[name] for name in inspect.signature(api._mla_dense_metadata_fake).parameters
    }


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 1, 1),
        (2, 7, 16, 65),
        (21, 2, 128, 511),
        (1, 257, 1, 17),
        (2, 3, 171, 65),
        (10, 2, 129, 33),
    ],
)
def test_metadata_dispatcher_schema_and_record(dtype, shape):
    args = arguments(dtype, shape)
    op = torch.ops.flash_attn_npu_3._mla_dense_metadata.default
    assert [a.name for a in op._schema.arguments] == list(args)
    torch.library.opcheck(
        op,
        (),
        args,
        test_utils=("test_schema", "test_faketensor", "test_autograd_registration"),
    )
    expected = metadata("cpp", metadata_args(dtype, shape))
    records = []
    for mode in (nullcontext(), torch.no_grad(), torch.inference_mode()):
        with mode:
            actual = op(**args)
        assert actual.dtype == torch.int32 and actual.shape == (401,)
        assert actual.device == args["cache_seqlens"].device and actual.is_contiguous()
        assert not actual.requires_grad and not hasattr(actual, "_fa_scheduler_params")
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
        records.append(actual)
    assert len({t.data_ptr() for t in records}) == len(records)


@pytest.mark.parametrize("fake", [False, True])
@pytest.mark.parametrize(
    "name,value",
    [
        ("batch_size", -1),
        ("batch_size", 2**32),
        ("max_seqlen_q", -1),
        ("max_seqlen_q", 2**30 - 63),
        ("max_seqlen_q", 2**30 + 1),
        ("max_seqlen_k", -1),
        ("max_seqlen_k", 16385),
        ("num_heads_q", -1),
        ("num_heads_q", 0),
        ("num_heads_q", 2**32 - 1),
        ("num_heads_q", 2**23),
        ("qkv_dtype", torch.float32),
        ("qkv_dtype", torch.int32),
    ],
)
def test_metadata_dispatcher_shape_and_dtype_boundary(dtype, fake, name, value):
    with FakeTensorMode() if fake else nullcontext():
        args = arguments(dtype)
        args[name] = value
        message = "qkv_dtype must be FP16 or BF16" if name == "qkv_dtype" else "dense shape bounds"
        with pytest.raises(RuntimeError, match=message):
            torch.ops.flash_attn_npu_3._mla_dense_metadata(**args)


@pytest.mark.parametrize("fake", [False, True])
@pytest.mark.parametrize("fault", ["dtype", "device", "rank", "size", "stride"])
def test_metadata_dispatcher_length_tensor_boundary(dtype, fake, fault):
    with FakeTensorMode() if fake else nullcontext():
        args = arguments(dtype)
        args["cache_seqlens"] = malformed(args["cache_seqlens"], fault)
        with pytest.raises(RuntimeError, match="MLA: cache_seqlens"):
            torch.ops.flash_attn_npu_3._mla_dense_metadata(**args)


@pytest.mark.parametrize("fake", [False, True])
def test_metadata_dispatcher_rejects_tensor_byte_overflow(dtype, fake):
    with FakeTensorMode() if fake else nullcontext():
        args = arguments(dtype)
        args.update(batch_size=2**24, max_seqlen_q=2**30, num_heads_q=1)
        with pytest.raises(RuntimeError, match="dense shape bounds"):
            torch.ops.flash_attn_npu_3._mla_dense_metadata(**args)


@pytest.mark.parametrize("fake", [False, True])
@pytest.mark.parametrize("device", ["cpu", "npu"])
@pytest.mark.parametrize("inference", [False, True])
def test_metadata_dispatcher_rejects_float_lengths_with_grad(dtype, fake, device, inference):
    with FakeTensorMode() if fake else nullcontext():
        args = arguments(dtype)
        args["cache_seqlens"] = torch.ones(2, device=device, requires_grad=True)
        with torch.inference_mode() if inference else torch.no_grad():
            with pytest.raises(RuntimeError, match="cache_seqlens must have dtype int32"):
                torch.ops.flash_attn_npu_3._mla_dense_metadata(**args)


@pytest.mark.parametrize("public", [False, True])
def test_metadata_dispatcher_compile_reads_current_lengths(dtype, public):
    args = metadata_args(dtype)

    def run(lengths):
        if public:
            return api.get_scheduler_metadata(**dict(args, cache_seqlens=lengths))
        return torch.ops.flash_attn_npu_3._mla_dense_metadata(2, 3, 17, 4, dtype, lengths)

    compiled = torch.compile(run, backend="eager", fullgraph=True)
    lengths = args["cache_seqlens"]
    records = []
    for length in (17, 0, 16, 18, -1, 17):
        lengths.fill_(length)
        record = compiled(lengths)
        torch.testing.assert_close(record.cpu(), run(lengths).cpu(), rtol=0, atol=0)
        assert record.cpu()[5].item() == (0 if length == 17 else 4)
        if length != 17:
            assert record.cpu()[8:].count_nonzero().item() == 0
        records.append(record)
    assert len({r.data_ptr() for r in records}) == len(records)
    torch.testing.assert_close(records[0].cpu(), records[-1].cpu(), rtol=0, atol=0)
