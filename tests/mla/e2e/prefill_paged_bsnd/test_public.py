"""Paged prefill numerical, public/fake/compile and lifetime acceptance."""

import contextlib
import gc
import inspect
import json
import os
from pathlib import Path
import resource
import time
import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.mla.support.public_cases.prefill_paged_bsnd import make_case, metadata, invoke, compare
from tests.mla.support.public_cases.contract import _forward


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize(
    "splits,mask,page", [(1, "none", 16), (2, "causal", 112), (0, "window", 528)]
)
def test_representative(dtype, splits, mask, page):
    c = make_case(dtype, splits=splits, mask=mask, page=page, strided=True, lengths=[1057, 517])
    record = metadata(c)
    assert record.cpu()[1:3].tolist() == [30, 30]
    compare(c, invoke(c, record), record)


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("splits", [0, 1, 3])
@pytest.mark.parametrize("mask", ["none", "causal", "window"])
def test_daily(dtype, splits, mask):
    c = make_case(dtype, (2, 7, 6, 65, 2), splits, mask, True, [65, 0], page=16)
    record = metadata(c)
    first = invoke(c, record)
    compare(c, first, record)
    storage = torch.full((record.numel() + 2,), 1234, dtype=torch.int32, device="npu")
    clone = storage[1:-1]
    clone.copy_(record.clone())
    for a, b in zip(first, invoke(c, clone)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert storage[[0, -1]].cpu().tolist() == [1234, 1234]


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize(
    "shape,splits,page",
    [((0, 7, 4, 31, 2), 0, 16), ((2, 0, 4, 31, 2), 1, 112), ((2, 7, 4, 0, 2), 2, 1024)],
)
def test_empty(dtype, shape, splits, page):
    c = make_case(dtype, shape, splits, page=page)
    record = metadata(c)
    compare(c, invoke(c, record), record)


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize(
    "shape,splits,mask,page",
    [
        ((1, 1, 4, 8193, 1), 0, "none", 16),
        ((1, 7, 8, 8193, 1), 2, "none", 112),
        ((1, 16, 12, 4097, 1), 1, "none", 528),
        ((1, 7, 8, 1025, 2), 0, "causal", 1024),
        ((1, 17, 8, 3, 8), 3, "causal", 16),
        ((1, 17, 2, 31, 2), 256, "window", 112),
        ((1, 1, 4, 16384, 1), 0, "none", 1024),
    ],
)
def test_tile_and_page_boundaries(dtype, shape, splits, mask, page):
    c = make_case(dtype, shape, splits, mask, True, page=page)
    record = metadata(c)
    compare(c, invoke(c, record), record)


def test_fake():
    with FakeTensorMode():
        c = make_case(splits=0)
        r = invoke(c, metadata(c))
        assert r[0].shape == (2, 17, 8, 128) and r[2].shape == (26, 2, 8, 17, 128)


def test_compile():
    c = make_case("bf16", (2, 7, 4, 65, 2), 2, "window", page=16)
    compiled = torch.compile(
        lambda q, k, v, pages, lengths: api.flash_attn_with_kvcache(
            q,
            k,
            v,
            page_table=pages,
            cache_seqlens=lengths,
            return_softmax_lse=True,
            **c["options"],
        ),
        backend="eager",
        fullgraph=True,
    )
    compare(c, compiled(**c["t"], pages=c["page_table"], lengths=c["lengths"]), metadata(c))
    c["lengths"].zero_()
    c["klens"] = [0, 0]
    compare(c, compiled(**c["t"], pages=c["page_table"], lengths=c["lengths"]), metadata(c))


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_default_and_integer_lengths(dtype):
    # Capacity is table width * P, not the physical pool or one page.
    c = make_case(dtype, (2, 7, 4, 80, 2), 2, "window", page=16)
    record = metadata(c)
    for length in (None, 80, 33):
        c["klens"] = [80 if length is None else length] * 2
        c["lengths"].fill_(c["klens"][0])
        if length == 33:
            record = metadata(c)
        result = api.flash_attn_with_kvcache(
            c["t"]["q"],
            c["t"]["k"],
            c["t"]["v"],
            page_table=c["page_table"],
            cache_seqlens=length,
            scheduler_metadata=record,
            return_softmax_lse=True,
            **c["options"],
        )
        compare(c, result, record)


@pytest.mark.parametrize("entry", ["eager", "cpp", "fake"])
@pytest.mark.parametrize("fault", ["zero", "unaligned", "capacity", "tnd", "length-dtype"])
def test_metadata_rejection(entry, fault):
    with FakeTensorMode() if entry == "fake" else contextlib.nullcontext():
        lengths = torch.zeros(2, dtype=torch.int32, device="npu")
        args = dict(
            batch_size=2,
            max_seqlen_q=7,
            max_seqlen_k=80,
            num_heads_q=4,
            num_heads_kv=2,
            headdim=192,
            headdim_v=128,
            cache_seqlens=lengths,
            page_size=16,
        )
        if fault in ("zero", "unaligned"):
            args["page_size"] = 0 if fault == "zero" else 15
        elif fault == "capacity":
            args["max_seqlen_k"] = 79
        elif fault == "tnd":
            args["cu_seqlens_q"] = torch.tensor([0, 7, 14], dtype=torch.int32, device="npu")
        else:
            args["cache_seqlens"] = lengths.long()
        with pytest.raises((RuntimeError, ValueError), match="MLA"):
            if entry == "cpp":
                bound = inspect.signature(api._get_scheduler_metadata_fake).bind(**args)
                bound.apply_defaults()
                api.flash_attn_npu_3.get_scheduler_metadata(*bound.arguments.values())
            else:
                api.get_scheduler_metadata(**args)


@pytest.mark.parametrize("entry", ["eager", "cpp", "fake"])
@pytest.mark.parametrize("splits", [0, 1, 3])
def test_raw_entries(entry, splits):
    with FakeTensorMode() if entry == "fake" else contextlib.nullcontext():
        c = make_case("bf16", (1, 3, 4, 17, 2), splits, "causal", page=16)
        record = metadata(c)
        result = _forward(
            entry,
            dict(
                **c["t"],
                page_table=c["page_table"],
                seqused_k=c["lengths"],
                scheduler_metadata=record,
                num_splits=splits,
                causal=True,
            ),
        )
        assert result[0].shape == (1, 3, 4, 128)
        if entry != "fake":
            compare(c, result, record)


@pytest.mark.parametrize("entry", ["eager", "cpp", "fake"])
@pytest.mark.parametrize(
    "fault",
    [
        "page-size",
        "page-dtype",
        "page-rank",
        "page-batch",
        "page-stride",
        "kv-heads",
        "last-stride",
        "metadata-size",
        "tnd-pa",
    ],
)
def test_static_rejection(entry, fault):
    with FakeTensorMode() if entry == "fake" else contextlib.nullcontext():
        c = make_case(shape=(1, 3, 4, 17, 2), page=16)
        q, k, v = c["t"].values()
        args = dict(q=q, k=k, v=v, page_table=c["page_table"], seqused_k=c["lengths"], num_splits=1)
        if fault == "page-size":
            args.update(k=k[:, :15], v=v[:, :15])
        elif fault == "page-dtype":
            args["page_table"] = c["page_table"].long()
        elif fault == "page-rank":
            args["page_table"] = c["page_table"].flatten()
        elif fault == "page-batch":
            args["page_table"] = c["page_table"].expand(2, -1)
        elif fault == "page-stride":
            args["page_table"] = torch.zeros((1, 4), dtype=torch.int32, device="npu")[:, ::2]
        elif fault == "kv-heads":
            args["v"] = v[:, :, :1]
        elif fault == "last-stride":
            args["k"] = torch.empty((*k.shape[:-1], 384), dtype=k.dtype, device=k.device)[..., ::2]
        elif fault == "metadata-size":
            args["scheduler_metadata"] = torch.empty(4, dtype=torch.int32, device="npu")
        elif fault == "tnd-pa":
            args.update(
                q=q.flatten(0, 1),
                cu_seqlens_q=torch.tensor([0, 3], dtype=torch.int32, device="npu"),
                max_seqlen_q=3,
            )
        with pytest.raises((RuntimeError, ValueError), match="MLA|FlashAttention"):
            _forward(entry, args)


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_content_scale_page_and_stride_reuse(dtype):
    c = make_case(dtype, (2, 7, 4, 65, 2), 0, "window", True, page=16)
    record = metadata(c)
    first = invoke(c, record)
    # Equivalent contiguous inputs, table storage offset and new valid physical IDs.
    c["t"] = {n: t.contiguous() for n, t in c["t"].items()}
    c["page_table"] = c["page_table"].clone()
    for a, b in zip(first, invoke(c, record)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    c["t"]["v"].mul_(0.5)
    c["raw"]["v"].mul_(0.5)
    c["options"]["softmax_scale"] *= 0.7
    compare(c, invoke(c, record), record)
    c["page_table"].fill_(0)
    for n in ("k", "v"):
        p = c["t"][n][0].cpu()
        c["raw"][n] = p.repeat(2, c["shape"][3] // 16, 1, 1)
    compare(c, invoke(c, record), record)


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_q_used_and_broadcast(dtype):
    c = make_case(dtype, (2, 7, 4, 65, 2), 2, "causal", page=16)
    c["qlens"] = [3, 0]
    uq = torch.tensor(c["qlens"], dtype=torch.int32, device="npu")
    result = torch.ops.flash_attn_npu_3._mla_prefill_bsnd_forward(
        **c["t"],
        num_splits=2,
        causal=True,
        cache_seqlens=c["lengths"],
        seqused_q=uq,
        page_table=c["page_table"],
    )
    record = metadata(c)
    # Only allocation identity stays rectangular; reference uses raw Q-used.
    compare(c, result, record)
    c["t"]["k"] = c["t"]["k"][:1, :1, :1].expand_as(c["t"]["k"])
    c["t"]["v"] = c["t"]["v"][:1, :1, :1].expand_as(c["t"]["v"])
    for name in ("k", "v"):
        c["raw"][name] = c["t"][name][0, 0, 0].cpu().expand_as(c["raw"][name])
    c["qlens"] = [7, 7]
    compare(c, invoke(c, record), record)


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_lifetime(dtype):
    c = make_case(dtype, (2, 7, 4, 65, 2), 0, "window", True, page=16)
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        record = metadata(c)
        outputs = [invoke(c, record) for _ in range(4)]
        kept = record.clone()
    expected = {k: v for k, v in c.items() if k not in ("t", "page_table", "lengths")}
    del record, c
    gc.collect()
    pressure = torch.empty((128 * 1024,), dtype=torch.float32, device="npu")
    pressure.fill_(123)
    torch.npu.current_stream().wait_stream(stream)
    compare(expected, outputs[-1], kept)
    for out in outputs[:-1]:
        for a, b in zip(out, outputs[-1]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_cost_boundary(dtype, tmp_path):
    started = time.monotonic()
    c = make_case(dtype, (2, 257, 8, 4097, 2), 0, "window", page=112)
    record = metadata(c)
    compare(c, invoke(c, record), record)
    report = dict(
        seconds=time.monotonic() - started,
        maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        shape=c["shape"],
        page=c["page"],
    )
    (Path(os.environ.get("MLA_VALIDATION_RESULT", tmp_path)) / f"cost-{dtype}.json").write_text(
        json.dumps(report) + "\n"
    )
