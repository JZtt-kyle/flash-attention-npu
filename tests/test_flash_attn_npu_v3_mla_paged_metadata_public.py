"""Public v6/v10/v11 generation: geometry, device lengths and consumer boundary."""

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
from tests.test_flash_attn_npu_v3_mla_contract import _forward, _inputs
from tests.test_flash_attn_npu_v3_mla_kvcache import malformed
from tests.test_flash_attn_npu_v3_mla_paged_metadata import decode_record
from tests.test_flash_attn_npu_v3_mla_paged_metadata_aicpu import (
    paged_probe as paged_probe,
    host_record,
)
from tests.test_flash_attn_npu_v3_mla_paged_metadata_consumer import (
    paged_consumer as paged_consumer,
    context_for,
    decode,
)
from tests.test_flash_attn_npu_v3_mla_paged_schedule import (
    paged_artifacts as paged_artifacts,
    paged_oracles as paged_oracles,
    execute,
    paged_case,
)


@pytest.fixture(params=[torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def dtype(request):
    return request.param


@pytest.fixture(params=[0, 2], ids=["bsnd", "tnd"])
def layout(request):
    return request.param


@pytest.fixture(
    params=["public", "op", "cpp", "dispatch", "public_fake", "op_fake", "dispatch_fake"]
)
def entry(request):
    with FakeTensorMode() if request.param.endswith("fake") else nullcontext():
        yield request.param


@pytest.fixture(scope="module")
def evidence(tmp_path_factory):
    target = os.environ.get("MLA_PAGED_PUBLIC_DIR")
    directory = Path(target) if target else tmp_path_factory.mktemp("mla-paged-public")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def arguments(
    dtype, layout, q=None, kv=(17, 0, 113), *, page=112, capacity=224, max_q=7, heads=128
):
    if q is None:
        q = [7, 7, 7] if layout == 0 else [3, 0, 1]

    def offset(values):
        return torch.tensor([123, *values, 456], device="npu", dtype=torch.int32)[1:-1]

    return dict(
        batch_size=len(q),
        max_seqlen_q=max_q,
        max_seqlen_k=capacity,
        num_heads_q=heads,
        num_heads_kv=1,
        headdim=64,
        headdim_v=512,
        qkv_dtype=dtype,
        page_size=page,
        cache_seqlens=offset(kv),
        cu_seqlens_q=offset(list(itertools.accumulate(q, initial=0))) if layout == 2 else None,
        num_splits=1,
    )


def call(entry, args):
    if entry.startswith("dispatch"):
        names = inspect.signature(api._mla_paged_metadata_fake).parameters
        return torch.ops.flash_attn_npu_3._mla_paged_metadata(
            **{n: args[n] for n in names if n in args}
        )
    return metadata(entry, args)


def check_snapshot(result, args, q, kv):
    row = [w & 0xFFFFFFFF for w in result.cpu().tolist()]
    assert row[:8] == [
        0x4D4C4131,
        6,
        6,
        408 + 5 * len(q),
        1 if args["qkv_dtype"] == torch.float16 else 2,
        0,
        0,
        0,
    ]
    total = sum(q)
    assert row[401:408] == [
        int(args["cu_seqlens_q"] is not None),
        total % 2**32,
        total // 2**32,
        args["max_seqlen_q"],
        0,
        args["page_size"],
        args["max_seqlen_k"],
    ]
    starts = list(itertools.accumulate(q, initial=0))
    assert row[408:] == [
        w for b in range(len(q)) for w in (starts[b] % 2**32, starts[b] // 2**32, q[b], q[b], kv[b])
    ]
    full_launch = args["batch_size"] and args["max_seqlen_q"] and args["max_seqlen_k"]
    assert row[21] == (row[17] if full_launch else 1)
    return row


@pytest.mark.parametrize("page", range(16, 1025, 16))
def test_all_pages_entries_and_offsets(entry, dtype, layout, page):
    q = [7, 7, 7] if layout == 0 else [3, 0, 1]
    kv = [page + 1, 0, 2 * page - 3]
    args = arguments(dtype, layout, q, kv, page=page, capacity=3 * page)
    with forbid_host_readback():
        record = call(entry, args)
    assert record.shape == (423,) and record.dtype == torch.int32 and record.is_contiguous()
    assert record.device == args["cache_seqlens"].device
    assert not record.requires_grad and not hasattr(record, "_fa_scheduler_params")
    if not entry.endswith("fake"):
        row = check_snapshot(record, args, q, kv)
        assert row[12] == 2 * page  # Source S2 is based on used pages, not capacity.
        assert args["cache_seqlens"]._base.cpu().tolist() == [123, *kv, 456]
        if layout == 2:
            assert args["cu_seqlens_q"]._base.cpu().tolist() == [123, 0, 3, 3, 4, 456]


@pytest.mark.parametrize("case", ["empty_batch", "zero_capacity", "zero_q", "zero_kv"])
def test_empty_records(entry, dtype, layout, case):
    q, kv, max_q, capacity = [7, 7], [17, 0], 7, 224
    if case == "empty_batch":
        q, kv = [], []
    elif case == "zero_capacity":
        kv, capacity = [0, 0], 0
    elif case == "zero_q":
        q, max_q = [0, 0], 0 if layout == 0 else 7
    else:
        kv = [0, 0]
    args = arguments(dtype, layout, q, kv, capacity=capacity, max_q=max_q)
    # Actual empty allocations also exercise null data_ptr() at B=0.
    if not q:
        args["cache_seqlens"] = torch.empty(0, device="npu", dtype=torch.int32)
    with forbid_host_readback():
        record = call(entry, args)
    assert record.shape == (408 + 5 * len(q),) and record.dtype == torch.int32
    if not entry.endswith("fake"):
        row = check_snapshot(record, args, q, kv)
        assert row[20] == 1 and not any(row[367:401])


def test_complete_records_against_cpu_and_fixed_source(
    dtype, layout, paged_probe, paged_oracles, evidence
):
    p = paged_probe["platform"]
    items = []
    for page in range(16, 1025, 16):
        capacity = 16384 // page * page
        for length in sorted({1, page - 1, page, page + 1, 449, 513, capacity}):
            items.append(
                paged_case(
                    layout,
                    [7, 7, 7] if layout == 0 else [7, 0, 3],
                    [length, 0, min(17, length)],
                    page=page,
                    capacity=capacity,
                    sq=7,
                    aic=p.aic,
                    ratio=2,
                    libapi=p.libapi,
                    block_dim=p.block_dims[p.aic - 1],
                    **{name: getattr(p.memory, name) for name in ("l1", "ub", "l0a", "l0b", "l0c")},
                )
            )
    source, native = execute(paged_oracles[0], items), execute(paged_oracles[0], items, native=True)
    rows = []
    for item, expected in zip(items, source):
        args = arguments(
            dtype, layout, item["q"], item["uk"], page=item["page"], capacity=item["sk"]
        )
        with forbid_host_readback():
            record = api.get_scheduler_metadata(**args)
        row = check_snapshot(record, args, item["q"], item["uk"])
        code = 1 if dtype == torch.float16 else 2
        assert row == host_record(paged_probe, item, code)
        decode_record(row, item, code, expected)
        rows.append(row)
    (evidence / f"source-{dtype}-{layout}.json").write_text(
        json.dumps(
            dict(
                inputs=items,
                records=rows,
                source=source,
                native=native,
                native_non_fd=sum(row[5] == 0 for row in native),
                scope="Public generator vs CPU serializer and independent pinned PA/non-FD source; no forward",
            )
        )
        + "\n"
    )


@pytest.mark.parametrize(
    "name,value",
    [
        ("batch_size", -1),
        ("batch_size", (2**32 - 1 - 408) // 5 + 1),
        ("max_seqlen_q", -1),
        ("max_seqlen_q", 2**30 + 1),
        ("max_seqlen_q", 2**25),
        ("max_seqlen_k", -1),
        ("max_seqlen_k", 16464),
        ("max_seqlen_k", 223),
        ("num_heads_q", 0),
        ("num_heads_q", 2**23),
        ("qkv_dtype", torch.float32),
        *[("page_size", p) for p in (-1, 0, 15, 17, 1025, 2**63 - 1)],
    ],
)
def test_static_bounds(entry, dtype, layout, name, value):
    args = arguments(dtype, layout)
    args[name] = value
    with pytest.raises(RuntimeError, match="MLA:"):
        call(entry, args)


@pytest.mark.parametrize("name", ["cache_seqlens", "cu_seqlens_q"])
@pytest.mark.parametrize("fault", ["dtype", "device", "rank", "size", "stride"])
def test_length_tensor_boundary(entry, dtype, name, fault):
    args = arguments(dtype, 2)
    args[name] = malformed(args[name], fault)
    with pytest.raises(RuntimeError, match=f"MLA: {name}"):
        call(entry, args)


@pytest.mark.parametrize("name", ["cache_seqlens", "cu_seqlens_q"])
@pytest.mark.parametrize("fake", [False, True])
@pytest.mark.parametrize("inference", [False, True])
def test_floating_lengths_with_grad(dtype, name, fake, inference):
    with FakeTensorMode() if fake else nullcontext():
        args = arguments(dtype, 2)
        args[name] = torch.ones(args[name].shape, device="npu", requires_grad=True)
        with torch.inference_mode() if inference else torch.no_grad():
            with pytest.raises(RuntimeError, match=f"{name} must have dtype int32"):
                call("dispatch", args)


def test_fake_prefix_device_mismatch(dtype):
    with FakeTensorMode():
        args = arguments(dtype, 2)
        args["cu_seqlens_q"] = torch.empty(4, device="npu:1", dtype=torch.int32)
        for kind in ("public_fake", "op_fake", "dispatch_fake"):
            with pytest.raises(RuntimeError, match="cu_seqlens_q must be on the same NPU device"):
                call(kind, args)


@pytest.mark.parametrize("value", [None, 0, 224])
def test_generator_requires_tensor_lengths(dtype, layout, value):
    args = arguments(dtype, layout)
    args["cache_seqlens"] = value
    with pytest.raises((RuntimeError, TypeError), match="cache_seqlens|[Tt]ensor"):
        api.get_scheduler_metadata(**args)


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_heads_kv", 2),
        ("num_splits", 0),
        ("num_splits", 2),
        ("window_size", (0, -1)),
        ("attention_chunk", 1),
        ("has_softcap", True),
        ("pack_gqa", False),
        ("pack_gqa", True),
        ("sm_margin", 1),
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
        args = arguments(dtype, 2)
        args[field] = args["cache_seqlens"] if value == "tensor" else value
        with pytest.raises(
            RuntimeError, match="MLA (scheduler|causal) metadata is not implemented yet"
        ):
            call(kind, args)


def test_causal_bsnd_all_metadata_entries(dtype):
    for fake in (False, True):
        with FakeTensorMode() if fake else nullcontext():
            entries = (
                ("public_fake", "op_fake", "dispatch_fake")
                if fake
                else ("public", "op", "cpp", "dispatch")
            )
            for kv in ([3, 225], [0, 0], [17, 33]):
                args = dict(arguments(dtype, 0, [7, 7], kv, capacity=336), causal=True)
                with forbid_host_readback():
                    records = [call(entry, args) for entry in entries]
                for record in records:
                    assert record.shape == (418,) and record.dtype == torch.int32
                    assert record.device == args["cache_seqlens"].device and record.is_contiguous()
                if not fake:
                    rows = [[v & 0xFFFFFFFF for v in r.cpu().tolist()] for r in records]
                    assert all(r == rows[0] for r in rows)
                    assert rows[0][:6] == [
                        0x4D4C4131,
                        10,
                        10,
                        418,
                        1 if dtype == torch.float16 else 2,
                        0,
                    ]
                    assert rows[0][408:] == [0, 0, 7, 7, kv[0], 7, 0, 7, 7, kv[1]]


def test_causal_metadata_failure(dtype):
    for kv in ([-1, 0], [337, 0]):
        args = dict(arguments(dtype, 0, [7, 7], kv, capacity=336), causal=True)
        with forbid_host_readback():
            records = [call(entry, args) for entry in ("public", "op", "cpp", "dispatch")]
        for record in records:
            row = record.cpu().tolist()
            assert row[1:3] == [10, 10] and row[5] != 0 and not any(row[8:])
    for fake in (False, True):
        with FakeTensorMode() if fake else nullcontext():
            args = dict(arguments(dtype, 2), causal=True)
            entries = (
                ("public_fake", "op_fake", "dispatch_fake")
                if fake
                else ("public", "op", "cpp", "dispatch")
            )
            for entry in entries:
                record = call(entry, args)
                assert record.shape == (408 + 5 * args["batch_size"],)
                if not fake:
                    assert record[1].item() == record[2].item() == 11


@pytest.mark.parametrize("extra", ["cu_seqlens_k", "seqused_q"])
@pytest.mark.parametrize("kind", ["op", "cpp", "op_fake"])
def test_internal_options_rejected(dtype, kind, extra):
    with FakeTensorMode() if kind.endswith("fake") else nullcontext():
        args = arguments(dtype, 2)
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
    "cq,kv,max_q,capacity",
    [
        ([1, 3, 4], [17, 33], 7, 224),
        ([0, 3, 2], [17, 33], 7, 224),
        ([0, 3, -1], [17, 33], 7, 224),
        ([0, 3, -(2**31)], [17, 33], 7, 224),
        ([0, 8, 8], [17, 33], 7, 224),
        ([0, 3, 4], [-1, 33], 7, 224),
        ([0, 3, 4], [17, 225], 7, 224),
        ([0, 3, 4], [17, 2**31 - 1], 7, 224),
        ([1], [], 7, 0),
        ([0, 0, 0], [0, 1], 0, 0),
    ],
)
@pytest.mark.parametrize("kind", ["public", "op", "cpp", "dispatch"])
def test_invalid_device_values_clear_payload_and_recover(dtype, kind, cq, kv, max_q, capacity):
    args = arguments(dtype, 2, [0] * len(kv), kv, max_q=max_q, capacity=capacity)
    args["cu_seqlens_q"].copy_(torch.tensor(cq, dtype=torch.int32, device="npu"))
    with forbid_host_readback():
        bad = call(kind, args)
        args["cu_seqlens_q"].zero_()
        args["cache_seqlens"].zero_()
        good = call(kind, args)
    row = bad.cpu().tolist()
    assert row[5] == 2 and row[6] != 0 and row[7] == 0 and not any(row[8:])
    check_snapshot(good, args, [0] * len(kv), [0] * len(kv))


@pytest.mark.parametrize("kv,capacity", [([-1, 0], 224), ([0, 225], 224), ([0, 1], 0)])
def test_bsnd_zero_q_still_checks_kv(dtype, kv, capacity):
    args = arguments(dtype, 0, [0, 0], kv, max_q=0, capacity=capacity)
    with forbid_host_readback():
        record = api.get_scheduler_metadata(**args)
    row = record.cpu().tolist()
    assert row[5:8] == [2, 1, 0] and not any(row[8:])


def test_stream_updates_and_allocator_lifetime(dtype, layout):
    consumer, pending = torch.npu.Stream(), []
    for producer in (torch.npu.Stream(), torch.npu.Stream()):
        for i in range(16):
            q = [7, 7, 7] if layout == 0 else [i % 7, 0, 7 - i % 7]
            kv = [i, 0, 224 - i]
            with torch.npu.stream(producer), forbid_host_readback():
                args = arguments(dtype, layout, q, kv)
                if layout == 2 and i % 2 == 0:
                    args["cache_seqlens"] = args["cu_seqlens_q"][:3]
                    kv = [0, q[0], q[0]]
                record = api.get_scheduler_metadata(**args)
                args["cache_seqlens"].zero_()
                if layout == 2:
                    args["cu_seqlens_q"].zero_()
                del args
                torch.empty(32768, device="npu", dtype=torch.int32).fill_(-77)
            with torch.npu.stream(consumer), forbid_host_readback():
                consumer.wait_stream(producer)
                record.record_stream(consumer)
                pending.append((record.clone(), q, kv))
                del record
    torch.npu.current_stream().wait_stream(consumer)
    for record, q, kv in pending:
        check_snapshot(record, arguments(dtype, layout, q, kv), q, kv)


def test_schema_compile_and_length_updates(dtype, layout):
    args = arguments(dtype, layout)
    op = torch.ops.flash_attn_npu_3._mla_paged_metadata.default
    names = inspect.signature(api._mla_paged_metadata_fake).parameters
    assert [a.name for a in op._schema.arguments] == list(names)
    torch.library.opcheck(
        op,
        (),
        {n: args[n] for n in names if n in args},
        test_utils=("test_schema", "test_faketensor", "test_autograd_registration"),
    )

    def run(cu_q, kv):
        return api.get_scheduler_metadata(**dict(args, cu_seqlens_q=cu_q, cache_seqlens=kv))

    compiled = torch.compile(run, backend="eager", fullgraph=True)
    records = []
    for cq, kv in [
        ([0, 3, 3, 4], [17, 0, 113]),
        ([0, 0, 0, 0], [0, 0, 0]),
        ([0, 1, 5, 7], [224, 2, 0]),
        ([0, 1, 5, 7], [225, 2, 0]),
    ]:
        if layout == 2:
            args["cu_seqlens_q"].copy_(torch.tensor(cq, dtype=torch.int32, device="npu"))
        args["cache_seqlens"].copy_(torch.tensor(kv, dtype=torch.int32, device="npu"))
        result = compiled(args["cu_seqlens_q"], args["cache_seqlens"])
        torch.testing.assert_close(
            result.cpu(), run(args["cu_seqlens_q"], args["cache_seqlens"]).cpu(), rtol=0, atol=0
        )
        assert result.cpu()[5].item() == (0 if max(kv) <= 224 else 2)
        records.append(result)
    assert len({r.data_ptr() for r in records}) == len(records)
    check_snapshot(records[0], args, [7, 7, 7] if layout == 0 else [3, 0, 1], [17, 0, 113])


def test_wide_allocation_and_fake_bounds(dtype):
    batch = 65538
    args = arguments(dtype, 0, [65536] * batch, [0] * batch, max_q=65536, heads=1)
    row = check_snapshot(api.get_scheduler_metadata(**args), args, [65536] * batch, [0] * batch)
    assert row[403] == row[-4] == 1
    with FakeTensorMode():
        batch = (2**32 - 1 - 408) // 5
        args = arguments(dtype, 2, [], [], max_q=2**30, heads=1)
        args.update(
            batch_size=batch,
            cache_seqlens=torch.empty(batch, device="npu", dtype=torch.int32),
            cu_seqlens_q=torch.empty(batch + 1, device="npu", dtype=torch.int32),
        )
        assert api.get_scheduler_metadata(**args).numel() == 408 + 5 * batch
        # The same rectangular BSND geometry would exceed the tensor byte bound.
        args["cu_seqlens_q"] = None
        with pytest.raises(RuntimeError, match="paged metadata shape bounds"):
            api.get_scheduler_metadata(**args)


@pytest.mark.parametrize("kind", ["eager", "cpp", "fake", "public"])
def test_v6_forward_protocol_boundary(dtype, layout, kind):
    with FakeTensorMode() if kind == "fake" else nullcontext():
        args = _inputs(layout="paged_tnd" if layout == 2 else "paged", dtype=dtype)
        q, kv = ([3, 1] if layout == 2 else [3, 3]), [17, 32]
        options = arguments(dtype, layout, q, kv, page=16, capacity=32, max_q=3, heads=4)
        record = api.get_scheduler_metadata(**options)
        for paged in (True, False):
            current = dict(args)
            if not paged:
                current.pop("page_table")
                current["k"] = torch.empty(2, 32, 1, 64, dtype=dtype, device="npu")
                current["v"] = torch.empty(2, 32, 1, 512, dtype=dtype, device="npu")
            # Both PA query layouts consume v6; nonpaged consumers reject it.
            with (
                nullcontext()
                if paged
                else pytest.raises(RuntimeError, match="not implemented yet|scheduler_metadata")
            ):
                if kind == "public":
                    api.flash_attn_with_kvcache(
                        current["q"],
                        current["k"],
                        current["v"],
                        qv=current["qv"],
                        page_table=current.get("page_table"),
                        cu_seqlens_q=current.get("cu_seqlens_q"),
                        max_seqlen_q=current.get("max_seqlen_q"),
                        cache_seqlens=options["cache_seqlens"],
                        scheduler_metadata=record,
                        num_splits=1,
                    )
                else:
                    _forward(
                        kind,
                        dict(
                            current, seqused_k=options["cache_seqlens"], scheduler_metadata=record
                        ),
                    )


@pytest.mark.parametrize("empty_kv", [False, True])
def test_public_records_current_lengths_and_old_protocols(
    dtype, layout, empty_kv, paged_probe, paged_consumer
):
    q = [7, 7] if layout == 0 else [3, 0]
    kv = [0, 0] if empty_kv else [17, 113]
    args = arguments(dtype, layout, q, kv)
    row = check_snapshot(api.get_scheduler_metadata(**args), args, q, kv)
    p = paged_probe["platform"]
    item = paged_case(
        layout,
        q,
        kv,
        page=112,
        capacity=224,
        sq=7,
        aic=p.aic,
        ratio=2,
        libapi=p.libapi,
        block_dim=p.block_dims[p.aic - 1],
        **{name: getattr(p.memory, name) for name in ("l1", "ub", "l0a", "l0b", "l0c")},
    )
    context = context_for(row, item, 1 if dtype == torch.float16 else 2)
    assert decode(paged_consumer, row, context, item)[0] == 0
    # Fresh public records must still be checked against current lengths.
    assert decode(paged_consumer, row, context, dict(item, uk=[1, kv[1]]))[0] != 0
    if layout == 2:
        assert decode(paged_consumer, row, context, dict(item, cq=[0, 2, 3]))[0] != 0
    old = api.get_scheduler_metadata(**dict(args, page_size=None)).cpu().tolist()
    assert decode(paged_consumer, old, context, item)[0] != 0
    # Matching the v6 size alone must not make an old header acceptable.
    padded = old + [0] * (len(row) - len(old))
    padded[3] = len(row)
    assert decode(paged_consumer, padded, context, item)[0] == 2


def test_generation_timing(dtype, layout, evidence):
    reports = []
    for page in (16, 112, 128, 256, 528, 1024):
        args = arguments(
            dtype,
            layout,
            [2] * 10,
            [1025] * 10,
            page=page,
            capacity=(1025 + page - 1) // page * page,
            max_q=2,
        )
        samples = []
        for i in range(25):
            torch.npu.synchronize()
            start = time.perf_counter_ns()
            record = api.get_scheduler_metadata(**args)
            torch.npu.synchronize()
            if i >= 5:
                samples.append((time.perf_counter_ns() - start) / 1000)
        check_snapshot(record, args, [2] * 10, [1025] * 10)
        reports.append(dict(page=page, wall_us=samples, median_us=statistics.median(samples)))
    (evidence / f"timing-{dtype}-{layout}.json").write_text(
        json.dumps(
            dict(
                records=reports,
                warmup=5,
                scope="Public metadata allocation, AICPU generation and sync; no PA forward/E2E",
            )
        )
        + "\n"
    )


def test_causal_tnd_metadata_entries_prefix_updates(dtype):
    for fake in (False, True):
        with FakeTensorMode() if fake else nullcontext():
            args = dict(arguments(dtype, 2, [7, 0, 3], [3, 17, 225], capacity=336), causal=True)
            entries = (
                ("public_fake", "op_fake", "dispatch_fake")
                if fake
                else ("public", "op", "cpp", "dispatch")
            )
            for prefix in (
                [0, 7, 7, 10],
                [0, 3, 10, 10],
                [0, 0, 0, 0],
                [0, 7, 6, 10],
                [1, 7, 7, 10],
                [0, 7, 7, 15],
            ):
                args["cu_seqlens_q"].copy_(torch.tensor(prefix, dtype=torch.int32))
                with forbid_host_readback():
                    records = [call(entry, args) for entry in entries]
                if fake:
                    assert all(r.shape == (423,) for r in records)
                    continue
                rows = [[x & 0xFFFFFFFF for x in r.cpu().tolist()] for r in records]
                assert all(row == rows[0] for row in rows)
                row = rows[0]
                assert row[1:3] == [11, 11]
                valid = prefix[0] == 0 and all(0 <= b - a <= 7 for a, b in zip(prefix, prefix[1:]))
                if valid:
                    assert row[5] == 0 and row[401:408] == [1, prefix[-1], 0, 7, 0, 112, 336]
                    expected = [
                        x
                        for b, (a, z) in enumerate(zip(prefix, prefix[1:]))
                        for x in (a, 0, z - a, z - a, (3, 17, 225)[b])
                    ]
                    assert row[408:] == expected
                else:
                    assert row[5] != 0 and not any(row[8:])
