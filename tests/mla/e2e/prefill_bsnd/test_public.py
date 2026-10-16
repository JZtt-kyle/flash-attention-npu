"""192/128 BSND public geometry, mask/split dispatch and metadata lifecycle."""

import pytest
import torch
import contextlib
import gc
import json
import os
from pathlib import Path
import resource
import time
from concurrent.futures import ThreadPoolExecutor
from torch._subclasses.fake_tensor import FakeTensorMode
from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.mla.support.public_cases.prefill_bsnd import make_case, metadata, invoke, compare
from tests.mla.support.public_cases.contract import _forward


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("splits,mask", [(1, "none"), (2, "causal"), (0, "window")])
def test_representative(dtype, splits, mask):
    c = make_case(dtype, splits=splits, mask=mask, strided=True)
    record = metadata(c)
    compare(c, invoke(c, record), record)


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("splits", [0, 1, 3])
@pytest.mark.parametrize("mask", ["none", "causal", "window"])
def test_daily(dtype, splits, mask):
    c = make_case(
        dtype, shape=(2, 7, 6, 65, 2), splits=splits, mask=mask, strided=True, lengths=[65, 0]
    )
    record = metadata(c)
    first = invoke(c, record)
    compare(c, first, record)
    storage = torch.full((record.numel() + 2,), 1234, dtype=torch.int32, device=record.device)
    cloned = storage[1:-1]
    cloned.copy_(record.clone())
    for a, b in zip(first, invoke(c, cloned)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert storage[[0, -1]].cpu().tolist() == [1234, 1234]


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize(
    "shape,splits", [((0, 7, 4, 31, 2), 0), ((2, 0, 4, 31, 2), 1), ((2, 7, 4, 0, 2), 2)]
)
def test_empty(dtype, shape, splits):
    c = make_case(dtype, shape, splits)
    record = metadata(c)
    compare(c, invoke(c, record), record)


def test_fake():
    with FakeTensorMode():
        c = make_case(splits=0)
        r = invoke(c, metadata(c))
        assert r[0].shape == (2, 17, 8, 128) and r[2].shape == (26, 2, 8, 17, 128)


@pytest.mark.parametrize("entry", ["eager", "cpp", "fake"])
def test_single_split_default_dtype(entry):
    before = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with FakeTensorMode() if entry == "fake" else contextlib.nullcontext():
            c = make_case(shape=(1, 1, 2, 3, 1))
            result = _forward(entry, dict(**c["t"], num_splits=1))
            assert result[1].dtype == torch.float32
            assert all(t.shape == (0,) and t.dtype == torch.float64 for t in result[2:])
    finally:
        torch.set_default_dtype(before)


def test_compile():
    c = make_case("bf16", shape=(2, 7, 4, 65, 2), splits=2, mask="window")
    compiled = torch.compile(
        lambda q, k, v, lengths: api.flash_attn_with_kvcache(
            q, k, v, cache_seqlens=lengths, return_softmax_lse=True, **c["options"]
        ),
        backend="eager",
        fullgraph=True,
    )
    compare(c, compiled(**c["t"], lengths=c["lengths"]), metadata(c))
    c["lengths"].zero_()
    c["klens"] = [0, 0]
    compare(c, compiled(**c["t"], lengths=c["lengths"]), metadata(c))


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize(
    "shape,splits,mask",
    [
        ((1, 1, 4, 8193, 1), 0, "none"),  # M32, N8192, actual native FD
        ((1, 7, 8, 8193, 1), 2, "none"),  # M64, N4096 and two fragments
        ((1, 16, 12, 4097, 1), 1, "none"),  # M128, N2048
        ((1, 7, 8, 1025, 2), 0, "causal"),  # masked M256/N1024
        ((1, 17, 8, 3, 8), 3, "causal"),  # MHA and leading empty rows
        ((1, 17, 2, 31, 2), 256, "window"),  # FA3 split limit, neutral tail fragments
        ((1, 1, 4, 16384, 1), 0, "none"),  # KV capacity boundary
    ],
)
def test_tile_and_split_boundaries(dtype, shape, splits, mask):
    c = make_case(dtype, shape, splits, mask)
    record = metadata(c)
    compare(c, invoke(c, record), record)


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_cost_boundary(dtype):
    started = time.monotonic()
    c = make_case(dtype, (1, 257, 8, 4097, 2), 8, "window", strided=True)
    record = metadata(c)
    compare(c, invoke(c, record), record)
    if os.environ.get("MLA_VALIDATION_RESULT"):
        (Path(os.environ["MLA_VALIDATION_RESULT"]) / f"cost-{dtype}.json").write_text(
            json.dumps(
                dict(
                    shape=c["shape"],
                    splits=8,
                    mask="window",
                    seconds=time.monotonic() - started,
                    process_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                ),
                indent=2,
            )
            + "\n"
        )


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("entry", ["eager", "cpp", "fake"])
def test_entry_adapters(dtype, entry):
    with FakeTensorMode() if entry == "fake" else contextlib.nullcontext():
        c = make_case(dtype, (2, 7, 6, 65, 2), 2, "window")
        record = metadata(c)
        baseline = invoke(c, record)
        args = dict(
            **c["t"],
            num_splits=2,
            window_size_left=7,
            window_size_right=2,
            seqused_k=c["lengths"],
            softmax_scale=c["options"]["softmax_scale"],
        )
        result = _forward(entry, args)
        for actual, expected in zip(result, baseline):
            assert actual.shape == expected.shape and actual.dtype == expected.dtype
            if entry != "fake":
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        full_capacity = _forward(entry, dict(args, seqused_k=None, scheduler_metadata=record))
        for actual, expected in zip(full_capacity, baseline):
            assert actual.shape == expected.shape and actual.dtype == expected.dtype
            if entry != "fake":
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        func = api.flash_attn_func(**c["t"], return_attn_probs=True, **c["options"])
        if entry != "fake":
            for actual, expected in zip(func, baseline[:2]):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for lengths in (None, 65):
                result = api.flash_attn_with_kvcache(
                    c["t"]["q"],
                    c["t"]["k"],
                    c["t"]["v"],
                    cache_seqlens=lengths,
                    scheduler_metadata=record,
                    return_softmax_lse=True,
                    **c["options"],
                )
                for actual, expected in zip(result, baseline):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize(
    "window,causal,scale", [((-1, 1), False, 0.0), ((2, -1), False, -0.125), ((7, 2), True, 0.23)]
)
def test_window_scale_and_reuse(dtype, window, causal, scale):
    c = make_case(dtype, (2, 7, 4, 65, 2), 0)
    c["options"].update(window_size=window, causal=causal, softmax_scale=scale)
    record = metadata(c)
    compare(c, invoke(c, record), record)
    c["options"]["softmax_scale"] = 0.0625
    for name, value in c["raw"].items():
        value.mul_(0.5)
        c["t"][name].copy_(value)
    compare(c, invoke(c, record), record)


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_used_query_and_broadcast(dtype):
    c = make_case(dtype, (2, 7, 4, 65, 2), 2, "causal")
    for name in ("k", "v"):
        c["raw"][name] = c["raw"][name][:, :1, :1].expand_as(c["raw"][name])
        c["t"][name] = c["raw"][name][:, :1, :1].contiguous().npu().expand_as(c["t"][name])
    c["qlens"] = [3, 0]
    used = torch.tensor(c["qlens"], dtype=torch.int32, device="npu")
    result = _forward(
        "cpp", dict(**c["t"], num_splits=2, causal=True, seqused_q=used, seqused_k=c["lengths"])
    )
    compare(c, result, metadata(c))


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_streams_lengths_and_lifetime(dtype):
    from tests.mla.support.mla_metadata_error_probe import forbid_host_readback

    parent = torch.npu.current_stream()
    streams = [torch.npu.Stream(), torch.npu.Stream()]
    kept = []
    for stream in streams:
        with torch.npu.stream(stream):
            stream.wait_stream(parent)
            c = make_case(dtype, (2, 7, 4, 65, 2), 0, "window", strided=True)
            record = metadata(c)
            snapshot = record.cpu()
            with forbid_host_readback():
                first = invoke(c, record)
                c["lengths"].zero_()
                second = invoke(c, metadata(c))
            kept.append(
                (
                    {k: c[k] for k in ("raw", "shape", "qlens", "klens", "options")},
                    first,
                    second,
                    snapshot,
                )
            )
            del c, record
    gc.collect()
    pressure = [torch.empty(4096, device="npu") for _ in range(64)]
    for stream in streams:
        stream.synchronize()
    for c, first, second, record in kept:
        compare(c, first, record)
        assert (second[0] == 0).all() and second[1].isneginf().all()
        assert (second[2] == 0).all() and second[3].isneginf().all()
    assert len(pressure) == 64


def test_concurrent_host_launches():
    def run(index):
        torch.npu.set_device(0)
        c = make_case("fp16" if index % 2 else "bf16", (2, 7, 4, 65, 2), index, "window")
        record = metadata(c)
        compare(c, invoke(c, record), record)

    with ThreadPoolExecutor(3) as pool:
        list(pool.map(run, [0, 1, 2]))


@pytest.mark.parametrize("entry", ["eager", "cpp", "fake"])
@pytest.mark.parametrize(
    "fault",
    [
        "grad",
        "heads",
        "dtype",
        "last_stride",
        "length_dtype",
        "length_shape",
        "metadata_shape",
        "q_used_metadata",
        "split",
        "page_size",
    ],
)
def test_rejections(entry, fault):
    with FakeTensorMode() if entry == "fake" else contextlib.nullcontext():
        c = make_case(shape=(2, 7, 4, 65, 2))
        args = dict(**c["t"], seqused_k=c["lengths"])
        if fault == "grad":
            args["q"].requires_grad_(True)
        elif fault == "heads":
            args["q"] = args["q"][:, :, :3]
        elif fault == "dtype":
            args["v"] = args["v"].float()
        elif fault == "last_stride":
            args["q"] = torch.empty((2, 7, 4, 384), dtype=args["q"].dtype, device="npu")[..., ::2]
        elif fault == "length_dtype":
            args["seqused_k"] = c["lengths"].long()
        elif fault == "length_shape":
            args["seqused_k"] = c["lengths"][:1]
        elif fault == "metadata_shape":
            args["scheduler_metadata"] = metadata(c)[:-1]
        elif fault == "q_used_metadata":
            args.update(seqused_q=c["lengths"], scheduler_metadata=metadata(c))
        elif fault == "split":
            args["num_splits"] = 257
        elif fault == "page_size":
            # This cache has 65 tokens per page, outside the supported alignment.
            args["page_table"] = torch.zeros((2, 1), dtype=torch.int32, device="npu")
        with pytest.raises(RuntimeError):
            _forward(entry, args)
        if fault == "page_size":
            # Real/fake validate page geometry at different boundaries.
            with pytest.raises(RuntimeError):
                api.flash_attn_with_kvcache(
                    c["t"]["q"], c["t"]["k"], c["t"]["v"], page_table=args["page_table"]
                )
