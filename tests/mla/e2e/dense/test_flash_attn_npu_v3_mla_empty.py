"""FA3 global-empty capacities: values, dispatch, metadata and stream lifetime."""

from contextlib import nullcontext
import itertools

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.mla.support.mla_metadata_error_probe import forbid_host_readback
from tests.test_flash_attn_npu_v3_metadata_signature import metadata as call_metadata
from tests.mla.e2e.dense.test_flash_attn_npu_v3_mla_contract import _forward
from tests.mla.e2e.dense.test_flash_attn_npu_v3_mla_kvcache import cached, generate, metadata_args
from tests.mla.e2e.dense.test_flash_attn_npu_v3_mla_kvcache import (
    test_public_runtime_error as runtime_error_case,
)


SHAPES = [
    (b, sq, 17, sk) for b, sq, sk in itertools.product((0, 2), (0, 3), (0, 17)) if not b * sq * sk
] + [(2, 257, 3, 0), (2, 0, 128, 16384)]


def inputs(shape, dtype, offset=False):
    b, sq, heads, sk = shape
    result = {}
    for name, size in {
        "q": (b, sq, heads, 64),
        "qv": (b, sq, heads, 512),
        "k": (b, sk, 1, 64),
        "v": (b, sk, 1, 512),
    }.items():
        count = size[0] * size[1] * size[2] * size[3]
        # Poison all available inputs: empty attention must never read them.
        storage = torch.full(
            (count + 2 if offset else count,), float("nan"), dtype=dtype, device="npu"
        )
        result[name] = (storage[1:-1] if offset else storage).view(size)
    return result


def check(outputs, shape, dtype, fake=False):
    b, sq, heads, sk = shape
    out, lse = outputs[:2]
    assert out.shape == (b, sq, heads, 512) and out.dtype == dtype
    assert lse.shape == (b, heads, sq) and lse.dtype == torch.float32
    assert out.device.type == lse.device.type == "npu"
    if not fake:
        assert out.cpu().count_nonzero() == 0
        assert not lse.cpu().isnan().any()
        if sk == 0:
            assert lse.cpu().isposinf().all()
    for accumulator in outputs[2:]:
        assert accumulator.shape == (0,) and accumulator.dtype == torch.get_default_dtype()
        assert accumulator.device == out.device


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("fake", [False, True])
@pytest.mark.parametrize("offset", [False, True])
def test_empty_public_and_direct_forward(dtype, shape, fake, offset):
    with FakeTensorMode() if fake else nullcontext():
        device = inputs(shape, dtype, offset)
        lengths = torch.full((shape[0],), shape[3], dtype=torch.int32, device="npu")
        record = generate(device, lengths)
        # The wire record may be cloned into a 4-byte-aligned offset view.
        copied = torch.empty(record.numel() + 2, dtype=torch.int32, device="npu")[1:-1]
        copied.copy_(record)
        before = None if fake else copied.cpu().clone()
        with forbid_host_readback():
            results = [
                api.flash_attn_func(**device, num_splits=1, return_attn_probs=True),
                cached(device, lengths=lengths),
                cached(device, copied, lengths),
                cached(device, copied, shape[3]),
                torch.ops.flash_attn_npu_3._mla_dense_forward(**device, scheduler_metadata=copied),
                _forward("fake" if fake else "cpp", dict(**device, scheduler_metadata=copied))[:2],
            ]
            out_only = cached(device, copied, return_lse=False)
        for result in results:
            check(result, shape, dtype, fake)
        assert out_only.shape == results[0][0].shape
        if not fake:
            assert out_only.cpu().count_nonzero() == 0
            torch.testing.assert_close(copied.cpu(), before, rtol=0, atol=0)
            assert all(t.cpu().isnan().all() for t in device.values())


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("entry", ["public", "op", "cpp", "public_fake", "op_fake"])
def test_empty_metadata_entries(dtype, shape, entry):
    with FakeTensorMode() if entry.endswith("fake") else nullcontext():
        args = metadata_args(dtype, shape)
        record = call_metadata(entry, args)
        assert record.shape == (401 + shape[0],) and record.dtype == torch.int32
        if not entry.endswith("fake"):
            words = record.cpu().tolist()
            assert words[:8] == [
                0x4D4C4131,
                2,
                2,
                401 + shape[0],
                1 if dtype == torch.float16 else 2,
                0,
                0,
                0,
            ]
            assert words[8:17] == [shape[0], 1, shape[2], shape[1], shape[3], 512, 64, 0, 0]
            assert words[20:23] == [1, 1, 1]
            assert not any(words[23:401])
            assert words[401:] == [shape[3]] * shape[0]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(2, 0, 17, 16384), (2, 257, 3, 0), (0, 3, 17, 17)])
def test_empty_compile_and_stream_lifetime(dtype, shape):
    def run(q, k, v, qv, lengths):
        data = dict(q=q, k=k, v=v, qv=qv)
        return cached(data, generate(data, lengths), lengths)

    compiled = torch.compile(run, backend="eager", fullgraph=True)
    produced = torch.npu.Stream()
    consumed = torch.npu.Stream()
    results = []
    for i in range(4):
        with torch.npu.stream(produced):
            device = inputs(shape, dtype, offset=True)
            lengths = torch.full((shape[0],), shape[3], dtype=torch.int32, device="npu")
            record = generate(device, lengths)
        consumed.wait_stream(produced)
        with torch.npu.stream(consumed):
            results.append(compiled(**device, lengths=lengths))
            results.append(cached(device, record, lengths, scale=(0.0, -0.23, 0.23, None)[i]))
        del device, lengths, record
        # Trigger allocator reuse while the independent consumer stream runs.
        with torch.npu.stream(produced):
            trash = torch.empty(32768, dtype=torch.int32, device="npu")
            trash.fill_(i)
    consumed.synchronize()
    for result in results:
        check(result, shape, dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(2, 0, 4, 17), (2, 3, 4, 0), (0, 3, 4, 17)])
@pytest.mark.parametrize("fake", [False, True])
def test_empty_does_not_bypass_boundary_checks(dtype, shape, fake):
    with FakeTensorMode() if fake else nullcontext():
        device = inputs(shape, dtype)
        for options, message in [
            ({"causal": True}, "not implemented"),
            ({"num_splits": 0}, "not implemented"),
            ({"scheduler_metadata": torch.empty(400, dtype=torch.int32, device="npu")}, "shape"),
            ({"cache_seqlens": torch.empty(shape[0], dtype=torch.int64, device="npu")}, "int32"),
        ]:
            with pytest.raises(RuntimeError, match=message):
                api.flash_attn_with_kvcache(
                    device["q"],
                    device["k"],
                    device["v"],
                    qv=device["qv"],
                    **(dict(num_splits=1) | options),
                )
        with pytest.raises(RuntimeError, match="gradients"):
            torch.ops.flash_attn_npu_3._mla_dense_forward(
                **dict(device, qv=device["qv"].requires_grad_())
            )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "fault",
    [
        f"{empty}:{fault}"
        for empty in ("emptyq", "emptyk", "emptyb")
        for fault in ("magic", "version", "kind", "shape", "workspace")
    ]
    + [
        f"{empty}:{mode}-{value}"
        for empty in ("emptyq", "emptyk")
        for mode in ("generated", "implicit", "reused")
        for value in ("negative", "over")
    ],
)
def test_empty_runtime_rejection(dtype, fault, tmp_path):
    runtime_error_case(dtype, fault, tmp_path)
