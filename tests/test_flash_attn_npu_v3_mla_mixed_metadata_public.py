"""Official TND-query metadata generation and its device-only v4 protocol."""

from contextlib import nullcontext
import inspect
import itertools
import json
import os
from pathlib import Path
import statistics
import time

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.mla_metadata_error_probe import forbid_host_readback
from tests.test_flash_attn_npu_v3_metadata_signature import metadata
from tests.test_flash_attn_npu_v3_mla_kvcache import malformed
from tests.test_flash_attn_npu_v3_mla_tnd_bsnd_metadata_aicpu import (
    mixed_probe as mixed_probe,
    host_record,
)
from tests.test_flash_attn_npu_v3_mla_tnd_bsnd_metadata import (
    mixed_artifacts as mixed_artifacts,
    mixed_oracle as mixed_oracle,
    check_records,
    metadata_cases,
)


@pytest.fixture(params=[torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def dtype(request):
    return request.param


@pytest.fixture(
    params=["public", "op", "cpp", "dispatch", "public_fake", "op_fake", "dispatch_fake"]
)
def entry(request):
    with FakeTensorMode() if request.param.endswith("fake") else nullcontext():
        yield request.param


def arguments(dtype, q=(3, 0, 1), k=(17, 0, 33), *, capacity=64, max_q=7, heads=128):
    def offset_tensor(values):
        return torch.tensor([123, *values, 456], dtype=torch.int32, device="npu")[1:-1]

    return dict(
        batch_size=len(q),
        max_seqlen_q=max_q,
        max_seqlen_k=capacity,
        num_heads_q=heads,
        num_heads_kv=1,
        headdim=64,
        headdim_v=512,
        qkv_dtype=dtype,
        num_splits=1,
        cache_seqlens=offset_tensor(k),
        cu_seqlens_q=offset_tensor(list(itertools.accumulate(q, initial=0))),
    )


def call(entry, args):
    if entry.startswith("dispatch"):
        names = inspect.signature(api._mla_tnd_bsnd_metadata_fake).parameters
        return torch.ops.flash_attn_npu_3._mla_tnd_bsnd_metadata(**{n: args[n] for n in names})
    return metadata(entry, args)


@pytest.mark.parametrize(
    "q,k,capacity",
    [
        ([3, 0, 1], [17, 0, 33], 64),
        ([], [], 0),
        ([], [], 16384),
        ([0, 0], [1, 513], 1024),
        ([3, 0, 1], [0, 0, 0], 0),
        ([3, 0, 1], [0, 0, 0], 64),
    ],
)
def test_entries_shape_empty_and_offsets(entry, dtype, q, k, capacity):
    args = arguments(dtype, q, k, capacity=capacity)
    with forbid_host_readback():
        result = call(entry, args)
    assert result.shape == (406 + 5 * len(q),) and result.dtype == torch.int32
    assert result.device == args["cache_seqlens"].device and result.is_contiguous()
    assert not result.requires_grad and not hasattr(result, "_fa_scheduler_params")
    if entry.endswith("fake"):
        return
    row = result.cpu().tolist()
    assert row[:8] == [0x4D4C4131, 4, 4, len(row), 1 if dtype == torch.float16 else 2, 0, 0, 0]
    assert row[401:406] == [sum(q), 7, capacity, len(q) * capacity, 0]
    cq = list(itertools.accumulate(q, initial=0))
    assert row[406:] == [w for b in range(len(q)) for w in (cq[b], b * capacity, 0, q[b], k[b])]
    if not sum(q) or capacity == 0:
        assert row[21] == 1  # A single validation core, even with a positive max-Q bound.
        assert row[367:391] == [0] * 24
    assert args["cu_seqlens_q"]._base.cpu().tolist() == [123, *cq, 456]
    assert args["cache_seqlens"]._base.cpu().tolist() == [123, *k, 456]


def test_public_records_match_cpu_and_pinned_source(dtype, mixed_probe, mixed_oracle):
    p = mixed_probe["platform"]
    cases = metadata_cases(p.aic, 2)
    rows = []
    for c in cases:
        c[6:11] = [getattr(p.memory, key) for key in ("l1", "ub", "l0a", "l0b", "l0c")]
        c[11], c[12] = p.libapi, p.block_dims[p.aic - 1]
        tail = 16 + c[15]
        cq, kv = c[tail + 5 :], c[16:tail]
        q = [b - a for a, b in zip(cq, cq[1:])]
        args = arguments(dtype, q, kv, capacity=c[3], max_q=c[1], heads=c[2])
        with forbid_host_readback():
            record = api.get_scheduler_metadata(**args)
        row = [x & 0xFFFFFFFF for x in record.cpu().tolist()]
        assert row == host_record(mixed_probe, c, 1 if dtype == torch.float16 else 2)
        rows.append(row)
    check_records(
        rows,
        cases,
        mixed_oracle,
        mixed_probe["directory"],
        f"public-{dtype}",
        1 if dtype == torch.float16 else 2,
    )


@pytest.mark.parametrize(
    "name,value",
    [
        ("batch_size", -1),
        ("batch_size", (2**32 - 1 - 406) // 5 + 1),
        ("max_seqlen_q", -1),
        ("max_seqlen_q", 2**30 + 1),
        ("max_seqlen_q", 2**25),
        ("max_seqlen_k", -1),
        ("max_seqlen_k", 16385),
        ("num_heads_q", 0),
        ("num_heads_q", 2**23),
        ("qkv_dtype", torch.float32),
    ],
)
def test_static_shape_dtype_bounds(entry, dtype, name, value):
    args = arguments(dtype)
    args[name] = value
    with pytest.raises(RuntimeError, match="MLA:"):
        call(entry, args)


@pytest.mark.parametrize("name", ["cache_seqlens", "cu_seqlens_q"])
@pytest.mark.parametrize("fault", ["dtype", "device", "rank", "size", "stride"])
def test_length_tensor_boundary(entry, dtype, name, fault):
    args = arguments(dtype)
    args[name] = malformed(args[name], fault)
    with pytest.raises(RuntimeError, match=f"MLA: {name}"):
        call(entry, args)


@pytest.mark.parametrize("name", ["cache_seqlens", "cu_seqlens_q"])
@pytest.mark.parametrize("fake", [False, True])
@pytest.mark.parametrize("inference", [False, True])
def test_dispatch_rejects_floating_lengths_with_grad(dtype, name, fake, inference):
    with FakeTensorMode() if fake else nullcontext():
        args = arguments(dtype)
        args[name] = torch.ones(args[name].shape, device="npu", requires_grad=True)
        with torch.inference_mode() if inference else torch.no_grad():
            with pytest.raises(RuntimeError, match=f"{name} must have dtype int32"):
                call("dispatch", args)


def test_fake_prefix_device_mismatch(dtype):
    with FakeTensorMode():
        args = arguments(dtype)
        args["cu_seqlens_q"] = torch.empty(4, device="npu:1", dtype=torch.int32)
        for kind in ("public_fake", "op_fake", "dispatch_fake"):
            with pytest.raises(RuntimeError, match="cu_seqlens_q must be on the same NPU device"):
                call(kind, args)


@pytest.mark.parametrize("value", [None, 0, 64])
def test_generator_requires_tensor_cache_lengths(dtype, value):
    args = arguments(dtype)
    args["cache_seqlens"] = value
    with pytest.raises((RuntimeError, TypeError), match="cache_seqlens|[Tt]ensor"):
        api.get_scheduler_metadata(**args)


@pytest.mark.parametrize("extra", ["cu_seqlens_k", "seqused_q"])
@pytest.mark.parametrize("kind", ["op", "cpp", "op_fake"])
def test_internal_length_options_still_rejected(dtype, kind, extra):
    with FakeTensorMode() if kind.endswith("fake") else nullcontext():
        args = arguments(dtype)
        bound = inspect.signature(api.get_scheduler_metadata).bind(**args)
        bound.apply_defaults()
        internal = dict(bound.arguments)
        internal["window_size_left"], internal["window_size_right"] = internal.pop("window_size")
        internal.update(cu_seqlens_k=None, seqused_q=None)
        internal[extra] = args["cache_seqlens"]
        with pytest.raises(RuntimeError, match="MLA scheduler metadata is not implemented yet"):
            if kind == "cpp":
                ordered = inspect.signature(api._get_scheduler_metadata_fake).bind(**internal)
                api.flash_attn_npu_3.get_scheduler_metadata(*ordered.arguments.values())
            else:
                api._get_scheduler_metadata_op(**internal)


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_heads_kv", 2),
        ("num_splits", 0),
        ("num_splits", 2),
        ("attention_chunk", 1),
        ("has_softcap", True),
        ("pack_gqa", False),
        ("sm_margin", 1),
        ("page_size", 17),
        ("max_seqlen_k_new", 1),
        ("softcap", 0.0),
        ("softmax_scale", 0.23),
        ("cu_seqlens_k_new", "tensor"),
        ("cache_leftpad", "tensor"),
    ],
)
@pytest.mark.parametrize("kind", ["public", "op", "cpp", "public_fake", "op_fake"])
def test_unsupported_options(dtype, kind, field, value):
    with FakeTensorMode() if kind.endswith("fake") else nullcontext():
        args = arguments(dtype)
        args[field] = args["cache_seqlens"] if value == "tensor" else value
        with pytest.raises(RuntimeError, match="MLA"):
            call(kind, args)


@pytest.mark.parametrize(
    "cq,kv,max_q,sk",
    [
        ([1, 3, 4], [17, 33], 7, 64),
        ([0, 3, 2], [17, 33], 7, 64),
        ([0, 3, -1], [17, 33], 7, 64),
        ([0, 3, -(2**31)], [17, 33], 7, 64),
        ([0, 8, 8], [17, 33], 7, 64),
        ([0, 3, 4], [-1, 33], 7, 64),
        ([0, 3, 4], [17, 65], 7, 64),
        ([0, 3, 4], [17, 2**31 - 1], 7, 64),
        ([1], [], 7, 0),
        ([0, 0, 0], [0, 1], 0, 0),
    ],
)
def test_invalid_values_write_failure_then_recover(dtype, cq, kv, max_q, sk):
    args = arguments(dtype, [0] * len(kv), kv, capacity=sk, max_q=max_q)
    args["cu_seqlens_q"].copy_(torch.tensor(cq, dtype=torch.int32, device="npu"))
    with forbid_host_readback():
        bad = api.get_scheduler_metadata(**args)
        args["cu_seqlens_q"].zero_()
        args["cache_seqlens"].zero_()
        good = api.get_scheduler_metadata(**args)
    row = bad.cpu().tolist()
    assert row[5:8] == [2, 1, 0] and not any(row[8:])
    assert good.cpu()[5:8].tolist() == [0, 0, 0]


def test_prefix_updates_streams_allocator_and_aliases(dtype):
    producer, consumer = torch.npu.Stream(), torch.npu.Stream()
    results = []
    for q in ([3, 0, 1], [0, 0, 0], [1, 2, 4]):
        with torch.npu.stream(producer), forbid_host_readback():
            args = arguments(dtype, q, [17, 0, 33])
            # Both descriptors may alias the same allocation, at independent offsets.
            args["cache_seqlens"] = args["cu_seqlens_q"][:3]
            record = api.get_scheduler_metadata(**args)
            del args
            torch.empty((2048,), dtype=torch.int32, device="npu").fill_(-77)
        with torch.npu.stream(consumer), forbid_host_readback():
            consumer.wait_stream(producer)
            record.record_stream(consumer)
            results.append(record.clone())
            del record
    torch.npu.current_stream().wait_stream(consumer)
    for q, record in zip(([3, 0, 1], [0, 0, 0], [1, 2, 4]), results):
        cq = list(itertools.accumulate(q, initial=0))
        row = record.cpu().tolist()
        assert row[401] == sum(q) and row[5] == 0
        assert row[406:] == [w for b in range(3) for w in (cq[b], b * 64, 0, q[b], cq[b])]


def test_schema_compile_and_current_device_lengths(dtype):
    args = arguments(dtype)
    op = torch.ops.flash_attn_npu_3._mla_tnd_bsnd_metadata.default
    names = inspect.signature(api._mla_tnd_bsnd_metadata_fake).parameters
    torch.library.opcheck(
        op,
        (),
        {n: args[n] for n in names},
        test_utils=("test_schema", "test_faketensor", "test_autograd_registration"),
    )

    def run(q, k):
        return api.get_scheduler_metadata(**dict(args, cu_seqlens_q=q, cache_seqlens=k))

    compiled = torch.compile(run, backend="eager", fullgraph=True)
    records = []
    for cq, kv in [
        ([0, 3, 3, 4], [17, 0, 33]),
        ([0, 0, 0, 0], [0, 0, 0]),
        ([0, 1, 5, 7], [64, 2, 0]),
        ([0, 1, 5, 7], [65, 2, 0]),
    ]:
        args["cu_seqlens_q"].copy_(torch.tensor(cq, dtype=torch.int32, device="npu"))
        args["cache_seqlens"].copy_(torch.tensor(kv, dtype=torch.int32, device="npu"))
        record = compiled(args["cu_seqlens_q"], args["cache_seqlens"])
        torch.testing.assert_close(
            record.cpu(), run(args["cu_seqlens_q"], args["cache_seqlens"]).cpu(), rtol=0, atol=0
        )
        assert record.cpu()[5].item() == (0 if max(kv) <= 64 else 2)
        records.append(record)
    assert len({r.data_ptr() for r in records}) == len(records)
    assert records[0].cpu()[401].item() == 4


def test_large_physical_capacity_and_fake_allocation_bounds(dtype):
    b, sk = 262146, 16384
    args = arguments(dtype, [], [], capacity=sk, max_q=1, heads=1)
    args.update(
        batch_size=b,
        cu_seqlens_q=torch.zeros(b + 1, device="npu", dtype=torch.int32),
        cache_seqlens=torch.zeros(b, device="npu", dtype=torch.int32),
    )
    args["cu_seqlens_q"][-1] = 1
    row = api.get_scheduler_metadata(**args).cpu().tolist()
    assert row[5] == 0 and row[404:406] == [(b * sk) % 2**32, (b * sk) // 2**32]
    assert row[-5:] == [0, sk, 1, 1, 0]
    with FakeTensorMode():
        b = (2**32 - 1 - 406) // 5
        args = arguments(dtype, [], [], max_q=2**30, heads=1)
        args.update(
            batch_size=b,
            cu_seqlens_q=torch.empty(b + 1, device="npu", dtype=torch.int32),
            cache_seqlens=torch.empty(b, device="npu", dtype=torch.int32),
        )
        result = api.get_scheduler_metadata(**args)
        assert result.numel() == 406 + 5 * b


def test_generation_timing(dtype, tmp_path):
    args = arguments(dtype, [2] * 21, [511] * 21, capacity=513, max_q=7)
    for _ in range(5):
        api.get_scheduler_metadata(**args)
    torch.npu.synchronize()
    samples = []
    for _ in range(20):
        start = time.perf_counter_ns()
        result = api.get_scheduler_metadata(**args)
        torch.npu.synchronize()
        samples.append((time.perf_counter_ns() - start) / 1000)
    assert result.cpu()[5].item() == 0
    directory = Path(os.environ.get("MLA_MIXED_PUBLIC_DIR", tmp_path))
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"generation-{dtype}.json").write_text(
        json.dumps(
            dict(
                dtype=str(dtype),
                warmup=5,
                wall_us=samples,
                median_us=statistics.median(samples),
                scope="Public generation, allocation, event handshake and synchronization; no forward",
            ),
            indent=2,
        )
        + "\n"
    )
