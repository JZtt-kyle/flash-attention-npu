"""TND public INT8 behavior, including mask, real FD partials and graph replay."""

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.mla.support.public_cases.int8_tnd import (
    TND_CASES,
    make_case,
    kwargs,
    metadata,
    invoke,
    equal,
    check_math,
    check_source,
)
from tests.mla.support.mla_int8_cases import SOURCE_CASES
from tests.mla.support.public_cases.graph import capture
from tests.mla.support.mla_metadata_error_probe import forbid_host_readback
from tests.mla.support.mla_int8_layout import preserve_view


@pytest.mark.parametrize(
    "index,splits,view",
    [(11, 1, "query-gap"), (12, 0, "scale-gap"), (18, 0, "allocation-gap"), (20, 1, "table-gap")],
)
def test_daily(index, splits, view):
    c = make_case(SOURCE_CASES[index], view, splits, uniform=True)
    with forbid_host_readback():
        record = metadata(c)
        result = invoke(c, record)
    equal(result, invoke(c))
    check_math(c, result)
    if splits == 1:
        assert result[2].numel() == result[3].numel() == 0


@pytest.mark.parametrize("spec", TND_CASES, ids=lambda s: s[0])
def test_source(spec):
    check_source(make_case(spec))


@pytest.mark.parametrize(
    "view",
    [
        "query-gap",
        "scale-gap",
        "allocation-gap",
        "page-gap",
        "row-gap",
        "wide-row",
        "broadcast",
        "table-gap",
    ],
)
def test_views(view):
    check_source(make_case(SOURCE_CASES[12], view))


@pytest.mark.parametrize(
    "index,pattern", [(12, "qs"), (12, "ks"), (13, "vs"), (12, "extreme"), (13, "head-scale")]
)
def test_public_scale_values(index, pattern):
    c = make_case(SOURCE_CASES[index])
    if pattern in ("qs", "ks", "vs"):
        c["raw"][pattern].zero_()
    elif pattern == "extreme":
        c["raw"]["qv"].fill_(-128)
        c["raw"]["v"].fill_(127)
    else:
        c["raw"]["qs"].copy_(torch.linspace(-0.02, 0.02, c["spec"][3]).expand_as(c["raw"]["qs"]))
    c["tensors"] = {key: preserve_view(t) for key, t in c["raw"].items()}
    check_source(c, label=pattern)


@pytest.mark.parametrize("lengths", [None, 128])
@pytest.mark.parametrize("record_view", ["clone", "offset"])
def test_optional_lengths_and_record_views(lengths, record_view):
    c = make_case(("full-pages", "TND", (1, 1), 16, (128, 128), 0), uniform=True)
    record = metadata(c)
    if record_view == "clone":
        record = record.clone()
    else:
        storage = torch.empty(record.numel() + 16, dtype=record.dtype, device=record.device)
        storage[8:-8].copy_(record)
        record = storage[8:-8]
    args = kwargs(c, record)
    args["cache_seqlens"] = lengths
    result = api.flash_attn_with_kvcache(**args)
    equal(result, invoke(c))
    check_math(c, result)


def test_fake_compile():
    c = make_case(SOURCE_CASES[12])
    args = kwargs(c)
    expected = invoke(c)
    with FakeTensorMode() as mode:
        fake = {
            k: mode.from_tensor(t) if isinstance(t, torch.Tensor) else t for k, t in args.items()
        }
        result = api.flash_attn_with_kvcache(**fake)
        assert [(t.shape, t.dtype) for t in result] == [(t.shape, t.dtype) for t in expected]
        assert metadata(
            dict(
                c,
                tensors={k: mode.from_tensor(t) for k, t in c["tensors"].items()},
                lengths=mode.from_tensor(c["lengths"]),
                prefix=mode.from_tensor(c["prefix"]),
            )
        ).shape == (469,)
    compiled = torch.compile(api.flash_attn_with_kvcache, backend="eager", fullgraph=True)
    equal(compiled(**args), expected)
    c["tensors"]["vs"].mul_(2)
    equal(compiled(**args), invoke(c))
    assert not torch.equal(invoke(c)[0].cpu(), expected[0].cpu())


@pytest.mark.parametrize("index", [11, 12, 13, 15, 16, 20])
def test_output_only(index):
    c = make_case(SOURCE_CASES[index])
    record = metadata(c)
    expected = invoke(c, record)[0]
    torch.testing.assert_close(invoke(c, record, lse=False), expected, rtol=0, atol=0)
    torch.testing.assert_close(invoke(c, lse=False), expected, rtol=0, atol=0)
    if index != 12:
        return
    args = kwargs(c, record, lse=False)
    with FakeTensorMode() as mode:
        fake = {
            k: mode.from_tensor(t) if isinstance(t, torch.Tensor) else t for k, t in args.items()
        }
        result = api.flash_attn_with_kvcache(**fake)
        assert result.shape == expected.shape and result.dtype == expected.dtype
    compiled = torch.compile(api.flash_attn_with_kvcache, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(**args), expected, rtol=0, atol=0)
    graph, output = capture(lambda: invoke(c, record, lse=False))
    c["tensors"]["vs"].mul_(2)
    graph.replay()
    torch.npu.synchronize()
    torch.testing.assert_close(output, invoke(c, record)[0], rtol=0, atol=0)
    assert not torch.equal(output.cpu(), expected.cpu())
    graph.reset()


@pytest.mark.parametrize("mode", ["implicit", "reuse", "generate", "external"])
@pytest.mark.parametrize("index", [11, 12, 13])
def test_graph(mode, index):
    c = make_case(SOURCE_CASES[index])
    record = metadata(c) if mode in ("reuse", "external") else None

    def call():
        return invoke(c, metadata(c) if mode == "generate" else record)

    graph, result = capture(call)
    baseline = call()[0].cpu()
    for iteration in range(3):
        for key, multiplier in (("qs", 0.75), ("ks", 0.9), ("vs", 1.5)):
            c["tensors"][key].mul_(multiplier)
        c["tensors"]["qv"].zero_() if iteration == 2 else None
        if mode != "reuse":
            c["lengths"].fill_((129, 0, 2)[iteration])
            if index == 12:
                c["prefix"].copy_(torch.tensor((0, 1, 5, 7), dtype=torch.int32, device="npu"))
            if mode == "external":
                record.copy_(metadata(c))
        graph.replay()
        torch.npu.synchronize()
        equal(result, call())
    assert not torch.equal(result[0].cpu(), baseline)
    graph.reset()


def test_streams_and_reuse():
    c = make_case(SOURCE_CASES[12], "query-gap")
    record = metadata(c)
    expected = invoke(c, record)
    ready = torch.npu.Event()
    ready.record()
    work = []
    for _ in range(3):
        stream = torch.npu.Stream()
        with torch.npu.stream(stream):
            stream.wait_event(ready)
            work.append((stream, invoke(c, record)))
    del record, c
    junk = [torch.empty(1024 * 1024, device="npu") for _ in range(4)]
    for stream, result in work:
        stream.synchronize()
        equal(result, expected)
    assert len({result[0].data_ptr() for _, result in work}) == 3
    del junk


def test_first_mask_use_inside_capture():
    c = make_case(SOURCE_CASES[12])
    expected = invoke(c)
    torch.npu.synchronize()
    stream = torch.npu.Stream()
    graph, result = capture(lambda: invoke(c), stream=stream, warmup=False)
    graph.replay()
    torch.npu.synchronize()
    equal(result, expected)
    graph.reset()


@pytest.mark.parametrize("empty", ["query", "batch", "pool"])
def test_empty(empty):
    c = make_case(SOURCE_CASES[12])
    args = kwargs(c)
    if empty == "query":
        for k in ("q", "qv", "mla_q_scale"):
            args[k] = args[k][:0]
        args["cu_seqlens_q"] = torch.zeros_like(c["prefix"])
    elif empty == "batch":
        for k in ("q", "qv", "mla_q_scale", "page_table", "cache_seqlens"):
            args[k] = args[k][:0]
        args["cu_seqlens_q"] = c["prefix"][:1]
    else:
        for k in ("k_cache", "v_cache"):
            args[k] = args[k][:0]
        args["cache_seqlens"].zero_()
    result = api.flash_attn_with_kvcache(**args)
    torch.npu.synchronize()
    if empty == "pool":
        assert torch.count_nonzero(result[0].cpu()) == 0
        assert torch.isneginf(result[1].cpu()).all()
    else:
        assert result[0].numel() == result[1].numel() == 0


@pytest.mark.parametrize("heads,lengths", [(1, (1, 0, 1)), (128, (32, 1, 0)), (8, (0, 0, 0))])
@pytest.mark.parametrize("splits", [0, 1])
def test_allocation_corners(heads, lengths, splits):
    spec = ("allocation-corner", "TND", lengths, heads, (1, 0, 129), 3 if max(lengths) > 1 else 0)
    c = make_case(spec, "allocation-gap", splits, uniform=True)
    record = metadata(c)
    out = invoke(c, record)
    equal(out, invoke(c))
    check_math(c, out)
    assert out[0].shape == (sum(lengths), heads, 512)
    assert out[1].shape == (heads, sum(lengths))


def test_prefix_reuse_binds_full_allocation():
    c = make_case(uniform=True)
    original = metadata(c)
    for prefix in ((0, 0, 4, 7), (0, 3, 3, 7), (0, 2, 6, 7)):
        c["prefix"].copy_(torch.tensor(prefix, dtype=torch.int32, device="npu"))
        lengths = tuple(b - a for a, b in zip(prefix, prefix[1:]))
        c["spec"] = (*c["spec"][:2], lengths, *c["spec"][3:])
        record = metadata(c)
        result = invoke(c, record)
        equal(result, invoke(c))
        check_math(c, result)
    equal(invoke(c, original), invoke(c))


def test_fast_fd():
    check_source(make_case(SOURCE_CASES[16]), label="fast-fd-tnd")


@pytest.mark.parametrize("qlens,klens", [((2, 0), (0, 129)), ((0, 1, 1), (129, 0, 0))])
@pytest.mark.parametrize("splits", [0, 1])
def test_disjoint_nonempty_allocations(qlens, klens, splits):
    c = make_case(("disjoint", "TND", qlens, 8, klens, 3 if max(qlens) > 1 else 0), splits=splits)
    result = invoke(c, metadata(c))
    assert torch.count_nonzero(result[0].cpu()) == 0
    assert torch.isneginf(result[1].cpu()).all()
    if splits == 0:
        assert torch.count_nonzero(result[2].cpu()) == 0
        assert torch.isneginf(result[3].cpu()).all()


@pytest.mark.parametrize("length", [129, 16385])
@pytest.mark.parametrize("splits", [0, 1])
def test_trailing_empty_segments(length, splits):
    # Include two trailing empty KV segments, an empty Q with nonempty KV,
    # and a final nonempty Q. Long KV exhausts the numerical core partition.
    c = make_case(
        ("trailing-empty", "TND", (2, 1, 1, 0, 2), 8, (length, 0, 0, 129, 0), 3),
        splits=splits,
        uniform=True,
    )
    result = invoke(c, metadata(c))
    equal(result, invoke(c))
    check_math(c, result)
    assert torch.count_nonzero(result[0][2:].cpu()) == 0
    assert torch.isneginf(result[1][:, 2:].cpu()).all()


@pytest.mark.parametrize("prefix", [(1, 2, 6, 7), (0, -1, 6, 7), (0, 3, 2, 7), (0, 5, 6, 7)])
def test_producer_invalid_prefix(prefix):
    c = make_case()
    c["prefix"].copy_(torch.tensor(prefix, dtype=torch.int32, device="npu"))
    record = metadata(c).cpu()
    assert record[1] == 49 and record[4] == 1
