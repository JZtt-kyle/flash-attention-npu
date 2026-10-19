"""BSND public INT8 behavior, including mask, real FD partials and graph replay."""

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.mla.support.public_cases.int8_bsnd import (
    BSND_CASES,
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
    [(3, 1, "query-gap"), (8, 0, "scale-gap"), (19, 1, "row-gap"), (17, 0, "table-gap")],
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


@pytest.mark.parametrize("spec", BSND_CASES, ids=lambda s: s[0])
def test_source(spec):
    check_source(make_case(spec))


@pytest.mark.parametrize(
    "view", ["query-gap", "scale-gap", "page-gap", "row-gap", "wide-row", "broadcast", "table-gap"]
)
def test_views(view):
    check_source(make_case(SOURCE_CASES[8], view))


@pytest.mark.parametrize(
    "index,pattern", [(8, "qs"), (8, "ks"), (6, "vs"), (8, "extreme"), (6, "head-scale")]
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
    c = make_case(SOURCE_CASES[2], uniform=True)
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
    c = make_case(SOURCE_CASES[8])
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
            )
        ).shape == (464,)
    compiled = torch.compile(api.flash_attn_with_kvcache, backend="eager", fullgraph=True)
    equal(compiled(**args), expected)
    c["tensors"]["vs"].mul_(2)
    equal(compiled(**args), invoke(c))
    assert not torch.equal(invoke(c)[0].cpu(), expected[0].cpu())


@pytest.mark.parametrize("mode", ["implicit", "reuse", "generate"])
@pytest.mark.parametrize("index", [6, 8, 16])
def test_graph(mode, index):
    c = make_case(SOURCE_CASES[index])
    record = metadata(c) if mode == "reuse" else None

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
        graph.replay()
        torch.npu.synchronize()
        equal(result, call())
    assert not torch.equal(result[0].cpu(), baseline)
    graph.reset()


def test_streams_and_reuse():
    c = make_case(SOURCE_CASES[8], "query-gap")
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
    c = make_case(SOURCE_CASES[8])
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
    c = make_case(SOURCE_CASES[8])
    args = kwargs(c)
    if empty == "query":
        for k in ("q", "qv", "mla_q_scale"):
            args[k] = args[k][:, :0]
    elif empty == "batch":
        for k in ("q", "qv", "mla_q_scale", "page_table", "cache_seqlens"):
            args[k] = args[k][:0]
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


@pytest.mark.parametrize(
    "query,length,splits",
    [
        pytest.param(1, 129, 0, id="decode-auto-tail"),
        pytest.param(1, 16385, 0, id="decode-auto-long-tail"),
        pytest.param(2, 129, 0, id="causal-auto-tail"),
        pytest.param(2, 16385, 0, id="causal-auto-long-tail"),
        pytest.param(1, 129, 1, id="decode-single-tail"),
        pytest.param(2, 16385, 1, id="causal-single-long-tail"),
    ],
)
def test_trailing_empty_segments(query, length, splits):
    # Balanced tiling can end before the final empty batches. Keep short and
    # core-exhausting long KV, plus the unchanged non-balanced single split.
    c = make_case(
        ("bsnd-trailing-empty", "BSND", (query,) * 4, 8, (length, 0, 0, 0), 3 if query > 1 else 0),
        splits=splits,
        uniform=True,
    )
    result = invoke(c, metadata(c))
    equal(result, invoke(c))
    check_math(c, result)
    assert torch.count_nonzero(result[0][1:].cpu()) == 0
    assert torch.isneginf(result[1][1:].cpu()).all()
    if splits == 0:
        assert torch.count_nonzero(result[2][:, 1:].cpu()) == 0
        assert torch.isneginf(result[3][:, 1:].cpu()).all()
