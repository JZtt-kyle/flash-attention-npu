"""Complete compact prefill entry, allocation, numeric and asynchronous lifecycle contract."""

import contextlib
import gc
import json
import os
from pathlib import Path
import resource
import time
import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.mla.support.public_cases.prefill_tnd import make_case, metadata, invoke, public, compare
from tests.mla.support.public_cases.contract import _forward


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("splits,mask", [(1, "none"), (2, "causal"), (0, "window")])
def test_representative(dtype, splits, mask):
    c = make_case(
        dtype, splits=splits, mask=mask, strided=True, used_q=[3, 0, 15], used_k=[513, 0, 3]
    )
    r = metadata(c)
    compare(c, invoke(c, r), r)
    compare(c, public(c), r)


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("splits", [0, 1, 3])
@pytest.mark.parametrize("mask", ["none", "causal", "window"])
def test_daily(dtype, splits, mask):
    c = make_case(
        dtype,
        q=[3, 0, 7],
        k=[65, 1, 0],
        heads=6,
        splits=splits,
        mask=mask,
        strided=True,
        used_q=[2, 0, 7],
    )
    r = metadata(c)
    baseline = invoke(c, r)
    compare(c, baseline, r)
    compare(c, public(c), r)
    storage = torch.full((r.numel() + 2,), 1234, dtype=torch.int32, device="npu")
    view = storage[1:-1]
    view.copy_(r.clone())
    for a, b in zip(baseline, invoke(c, view)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert storage[[0, -1]].cpu().tolist() == [1234, 1234]


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize(
    "q,k,maxima,splits",
    [
        ((), (), (7, 31), 0),
        ((0, 0), (31, 3), (7, 31), 1),
        ((7, 2), (0, 0), (7, 31), 2),
        ((0,), (0,), (0, 0), 0),
    ],
)
def test_empty(dtype, q, k, maxima, splits):
    c = make_case(dtype, q=q, k=k, maxima=maxima, splits=splits)
    r = metadata(c)
    compare(c, invoke(c, r), r)
    compare(c, public(c), r)


@pytest.mark.parametrize("splits", [0, 1, 256])
def test_fake(splits):
    with FakeTensorMode():
        c = make_case(splits=splits)
        result = invoke(c, metadata(c))
        assert result[0].shape == (24, 8, 128) and result[1].shape == (8, 24)
        assert result[2].shape == ((0,) if splits == 1 else (splits or 26, 8, 24, 128))
        assert public(c)[0].shape == result[0].shape


def test_fake_compact_budget():
    # A large declared maximum must not impose rectangular B*maxQ storage.
    with FakeTensorMode():
        q = torch.empty((1, 1, 192), dtype=torch.float16, device="npu")
        v = torch.empty((1, 1, 128), dtype=q.dtype, device=q.device)
        prefix = torch.empty(1_000_001, dtype=torch.int32, device=q.device)
        result = torch.ops.flash_attn_npu_3._mla_prefill_tnd_forward(
            q, q, v, prefix, prefix, 1 << 30, 1, 0
        )
        assert result[0].shape == (1, 1, 128)
        assert result[2].shape == (26, 1, 1, 128)


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_declared_maxima(dtype):
    c = make_case(dtype, q=[3, 7], k=[65, 31], maxima=(64, 2048), splits=3, mask="window")
    r = metadata(c)
    compare(c, invoke(c, r), r)
    compare(c, public(c), r)


def test_compile():
    c = make_case(
        "bf16", q=[3, 7], k=[65, 31], splits=0, mask="window", used_q=[2, 6], used_k=[65, 7]
    )
    f = torch.compile(
        lambda q, k, v, uq, uk: api.flash_attn_varlen_func(
            q,
            k,
            v,
            **dict(c["lengths"], seqused_q=uq, seqused_k=uk),
            **c["options"],
            return_attn_probs=True,
        ),
        backend="eager",
        fullgraph=True,
    )
    compare(c, f(**c["t"], uq=c["lengths"]["seqused_q"], uk=c["lengths"]["seqused_k"]), metadata(c))
    c["lengths"]["seqused_k"].zero_()
    c["klens"] = [0, 0]
    compare(c, f(**c["t"], uq=c["lengths"]["seqused_q"], uk=c["lengths"]["seqused_k"]), metadata(c))


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize(
    "q,k,h,hk,splits,mask",
    [
        ((1, 2), (16384, 513), 8, 1, 0, "none"),
        ((129, 3), (517, 1025), 8, 2, 2, "window"),
        ((17, 1), (3, 65), 8, 8, 1, "causal"),
        ((7, 1), (31, 1), 2, 2, 256, "window"),
    ],
)
def test_boundaries(dtype, q, k, h, hk, splits, mask):
    c = make_case(dtype, q=q, k=k, heads=h, kv_heads=hk, splits=splits, mask=mask, strided=True)
    r = metadata(c)
    compare(c, invoke(c, r), r)
    compare(c, public(c), r)


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_cost_boundary(dtype):
    start = time.monotonic()
    c = make_case(dtype, q=[257, 1], k=[4097, 3], splits=8, mask="window", strided=True)
    r = metadata(c)
    compare(c, invoke(c, r), r)
    if os.environ.get("MLA_VALIDATION_RESULT"):
        (Path(os.environ["MLA_VALIDATION_RESULT"]) / f"cost-{dtype}.json").write_text(
            json.dumps(
                dict(
                    seconds=time.monotonic() - start,
                    process_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    shape=c["shape"],
                )
            )
        )


@pytest.mark.parametrize("entry", ["eager", "cpp", "fake"])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_entry_adapters(entry, dtype):
    with FakeTensorMode() if entry == "fake" else contextlib.nullcontext():
        c = make_case(dtype, q=[3, 7], k=[65, 31], splits=2, mask="window", used_q=[1, 7])
        r = metadata(c)
        result = _forward(
            entry,
            dict(
                **c["t"],
                **c["lengths"],
                num_splits=2,
                window_size_left=7,
                window_size_right=2,
                scheduler_metadata=r,
            ),
        )
        if entry != "fake":
            compare(c, result, r)
        else:
            assert result[2].shape == (2, 8, 10, 128)


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize(
    "window,causal,scale", [((-1, 1), False, 0.0), ((2, -1), False, -0.125), ((7, 2), True, 0.23)]
)
def test_scale_broadcast_reuse(dtype, window, causal, scale):
    c = make_case(dtype, q=[3, 7], k=[65, 31], splits=0, used_q=[2, 5])
    c["options"].update(window_size=window, causal=causal, softmax_scale=scale)
    for name in ("k", "v"):
        c["raw"][name] = c["raw"][name][:1, :1].expand_as(c["raw"][name])
        c["t"][name] = c["raw"][name][:1, :1].contiguous().npu().expand_as(c["t"][name])
    r = metadata(c)
    compare(c, invoke(c, r), r)
    c["options"]["softmax_scale"] = 0.0625
    compare(c, invoke(c, r), r)


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_streams_and_lifetime(dtype):
    from tests.mla.support.mla_metadata_error_probe import forbid_host_readback

    parent = torch.npu.current_stream()
    streams = [torch.npu.Stream(), torch.npu.Stream()]
    kept = []
    for stream in streams:
        with torch.npu.stream(stream):
            stream.wait_stream(parent)
            c = make_case(dtype, q=[3, 7], k=[65, 31], splits=0, mask="window", used_k=[65, 31])
            r = metadata(c)
            snapshot = r.cpu()
            with forbid_host_readback():
                first = invoke(c, r)
                c["lengths"]["seqused_k"].zero_()
                second = public(c)
            kept.append(
                (
                    {k: c[k] for k in ("raw", "shape", "q", "k", "qlens", "klens", "options")},
                    first,
                    second,
                    snapshot,
                )
            )
            del c, r
    gc.collect()
    pressure = [torch.empty(4096, device="npu") for _ in range(64)]
    for stream in streams:
        stream.synchronize()
    for c, first, second, record in kept:
        compare(c, first, record)
        assert (second[0] == 0).all() and second[1].isneginf().all()
    assert len(pressure) == 64


@pytest.mark.parametrize("entry", ["eager", "cpp", "fake"])
@pytest.mark.parametrize(
    "fault",
    [
        "max",
        "missing_max_q",
        "missing_max_k",
        "prefix_dtype",
        "prefix_shape",
        "prefix_stride",
        "used_dtype",
        "used_shape",
        "heads",
        "split",
        "metadata",
        "grad",
        "cache_batch",
        "last_stride",
    ],
)
def test_rejections(entry, fault):
    with FakeTensorMode() if entry == "fake" else contextlib.nullcontext():
        c = make_case(q=[3, 7], k=[65, 31])
        args = dict(**c["t"], **c["lengths"])
        if fault == "max":
            args["max_seqlen_k"] = 16385
        elif fault.startswith("missing_max_"):
            args["max_seqlen_" + fault[-1]] = None
        elif fault == "prefix_dtype":
            args["cu_seqlens_q"] = args["cu_seqlens_q"].long()
        elif fault == "prefix_shape":
            args["cu_seqlens_k"] = args["cu_seqlens_k"][:2]
        elif fault == "prefix_stride":
            args["cu_seqlens_q"] = torch.zeros(6, dtype=torch.int32, device="npu")[::2]
        elif fault == "used_dtype":
            args["seqused_q"] = torch.ones(2, device="npu")
        elif fault == "used_shape":
            args["seqused_k"] = torch.ones(1, dtype=torch.int32, device="npu")
        elif fault == "heads":
            args["q"] = args["q"][:, :3]
        elif fault == "split":
            args["num_splits"] = 257
        elif fault == "metadata":
            args["scheduler_metadata"] = metadata(c)[:-1]
        elif fault == "grad":
            args["v"].requires_grad_(True)
        elif fault == "cache_batch":
            # One cache batch cannot match the two query allocation segments.
            args["k"] = args["k"].unsqueeze(0)
            args["v"] = args["v"].unsqueeze(0)
            args["cu_seqlens_k"] = None
        elif fault == "last_stride":
            args["q"] = torch.empty((10, 8, 384), dtype=args["q"].dtype, device="npu")[..., ::2]
        with pytest.raises(RuntimeError):
            _forward(entry, args)
