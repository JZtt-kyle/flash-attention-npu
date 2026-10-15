"""Mixed split production, public metadata reuse, accumulators and allocation addresses."""

import copy
import gc
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.mla.e2e.window_mixed.test_flash_attn_npu_v3_mla_window_mixed_public import (
    prepare as prepare_window,
)
from tests.mla.e2e.split_tnd.test_public import reference as reference_tnd, compare as compare_tnd
from tests.mla.e2e.dense.test_flash_attn_npu_v3_mla_contract import _forward
from tests.mla.support.mla_metadata_error_probe import forbid_host_readback
from tests.mla.support.mla_split_mixed_cases import spec, DAILY_CASES, REPRESENTATIVES
from tests.mla.e2e.bsnd.test_flash_attn_npu_v3_mla_bsnd_stride_production import (
    storage_fault as storage_fault,
)


def prepare(dtype, s, *, poison=True):
    c = prepare_window(dtype, copy.deepcopy(s), poison=poison)
    c["args"]["num_splits"] = s["splits"]
    return c


def metadata(c, entry="public"):
    a = c["args"]
    lengths = a["cache_seqlens"]
    if lengths is None or isinstance(lengths, int):
        lengths = torch.full(
            (a["k"].shape[0],),
            a["k"].shape[1] if lengths is None else lengths,
            device=a["q"].device,
            dtype=torch.int32,
        )
    if entry == "op":
        return torch.ops.flash_attn_npu_3._mla_split_mixed_metadata(
            a["k"].shape[0],
            a["max_seqlen_q"],
            a["k"].shape[1],
            a["q"].shape[1],
            a["q"].dtype,
            lengths,
            a["cu_seqlens_q"],
            *a["window_size"],
            a["causal"],
            a["num_splits"],
        )
    from tests.test_flash_attn_npu_v3_metadata_signature import metadata as generate

    return generate(
        entry,
        dict(
            batch_size=a["k"].shape[0],
            max_seqlen_q=a["max_seqlen_q"],
            max_seqlen_k=a["k"].shape[1],
            num_heads_q=a["q"].shape[1],
            num_heads_kv=1,
            headdim=64,
            headdim_v=512,
            qkv_dtype=a["q"].dtype,
            cache_seqlens=lengths,
            cu_seqlens_q=a["cu_seqlens_q"],
            window_size=a["window_size"],
            causal=a["causal"],
            num_splits=a["num_splits"],
        ),
    )


def call(args, entry="public", record=None, *, lse=True):
    a = dict(args)
    if record is not None:
        a["scheduler_metadata"] = record
    a.setdefault("scheduler_metadata", None)
    if entry == "public":
        assert a.pop("seqused_q", None) is None
        a["k_cache"], a["v_cache"] = a.pop("k"), a.pop("v")
        return api.flash_attn_with_kvcache(**a, return_softmax_lse=lse)
    left, right = a.pop("window_size")
    if entry == "op":
        return torch.ops.flash_attn_npu_3._mla_split_mixed_forward(
            **a, window_left=left, window_right=right
        )
    a["seqused_k"] = a.pop("cache_seqlens")
    return _forward(entry, dict(a, window_size_left=left, window_size_right=right))


def reference(c, tile=None):
    if tile is None:
        md = metadata(c).cpu()
        tile = (
            (int(md[15]), int(md[16]))
            if sum(c["spec"]["q"]) and sum(c["spec"]["kv"])
            else (256, 512)
        )
    mapped = dict(c, args=dict(c["args"], max_seqlen_k=c["spec"]["capacity"]))
    return reference_tnd(mapped, tile)


def compare(c, result, refs=None):
    return compare_tnd(c, result, reference(c) if refs is None else refs)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("s", REPRESENTATIVES)
def test_representative(dtype, s):
    start = time.monotonic()
    c = prepare(dtype, s)
    r = call(c["args"], "op")
    compare(c, r)
    if os.environ.get("MLA_VALIDATION_RESULT"):
        path = (
            Path(os.environ["MLA_VALIDATION_RESULT"])
            / f"cost-{dtype}-{s['splits']}-{s['capacity']}.json"
        )
        path.write_text(
            json.dumps(
                dict(
                    spec=s,
                    seconds=time.monotonic() - start,
                    accumulator_bytes=sum(t.nbytes for t in r[2:]),
                ),
                indent=2,
            )
            + "\n"
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("case", range(len(DAILY_CASES)))
def test_daily(dtype, case):
    c = prepare(dtype, DAILY_CASES[case])
    refs = reference(c)
    with forbid_host_readback():
        record = metadata(c)
        saved = record.clone()
        caches = [c["args"][name].clone() for name in ("k", "v")]
        result = call(c["args"])
        reused = call(c["args"], record=record)
    compare(c, result, refs)
    compare(c, reused, refs)
    assert torch.equal(record.cpu(), saved.cpu())
    for name, before in zip(("k", "v"), caches):
        torch.testing.assert_close(c["args"][name], before, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "s",
    [
        spec(q=(0, 0), k=(3, 7), capacity=7),
        spec(q=(3, 7), k=(0, 0), capacity=0),
        spec(q=(), k=(), capacity=0),
        spec(uq=[0, 0, 0]),
        spec(k=(0, 0, 0)),
        spec(q=(65, 1), k=(513, 3), capacity=513, h=3, splits=193, window=(0, 0)),
        spec(q=(3, 2), k=(513, 17), capacity=513, splits=255, window=(-1, 1)),
        spec(q=(3, 7), k=(17, 257), capacity=513, uq=[1, 2], sq=64, window=(5, -1)),
        spec(q=(0, 0), k=(3, 7), capacity=7, sq=64),
        spec(q=(7, 3), k=(33, 17), h=3, stride="ntd", causal=True),
        spec(q=(3, 1), k=(17, 7), h=3, stride="row_65536"),
        spec(q=(1, 2), k=(3, 33), h=257, window=(2, 1)),
    ],
)
def test_boundaries(dtype, s):
    c = prepare(dtype, s)
    compare(c, call(c["args"], "op"))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("query_empty", [False, True])
def test_empty(dtype, query_empty):
    c = prepare(
        dtype,
        spec(
            q=(0, 0) if query_empty else (3, 1),
            k=(7, 3) if query_empty else (0, 0),
            capacity=7 if query_empty else 0,
            h=2,
        ),
    )
    refs = reference(c)
    compare(c, call(c["args"]), refs)
    compare(c, call(c["args"], record=metadata(c)), refs)


@pytest.mark.parametrize("entry", ["public", "eager", "cpp", "op"])
@pytest.mark.parametrize("mask", [0, 1, 2])
def test_entries(entry, mask):
    c = prepare(torch.float16, DAILY_CASES[mask])
    compare(c, call(c["args"], entry, metadata(c)))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_lengths_scales_views_and_reuse(dtype):
    s = spec(q=(5, 3), k=(33, 33), capacity=33, h=3, window=(2, 1))
    record = metadata(prepare(dtype, s))
    saved = record.cpu().clone()
    for stride, scale in [("independent", None), ("ntd", 0.0), ("row_65536", -0.1)]:
        c = prepare(dtype, dict(s, stride=stride, scale=scale))
        refs = reference(c)
        for lengths in [None, 33, c["args"]["cache_seqlens"]]:
            a = dict(c["args"], cache_seqlens=lengths)
            compare(c, call(a, record=record), refs)
            torch.testing.assert_close(
                call(a, record=record, lse=False), call(a, record=record)[0], rtol=0, atol=0
            )
    assert torch.equal(record.cpu(), saved)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_reuse_updates_and_lifetime(dtype):
    c = prepare(dtype, DAILY_CASES[2], poison=False)
    # Same storage, new allocation segments and used lengths: regenerate v21.
    c["spec"].update(q=[5, 2, 3], uk=[3, 1, 0])
    c["args"]["cu_seqlens_q"].copy_(torch.tensor([0, 5, 7, 10], dtype=torch.int32))
    c["args"]["cache_seqlens"].copy_(torch.tensor([3, 1, 0], dtype=torch.int32))
    refs = reference(c)
    streams = [torch.npu.Stream(), torch.npu.Stream()]
    results = []
    for stream in streams:
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream), forbid_host_readback():
            record = metadata(c)
            result = call(c["args"], record=record)
            results.append(result)
            del record
    args = c.pop("args")
    del args
    gc.collect()
    trash = [torch.empty(1024, dtype=torch.int32, device="npu") for _ in range(32)]
    for stream in streams:
        stream.synchronize()
    for r in results:
        compare(c, r, refs)
    del trash


def test_fake():
    with FakeTensorMode():
        c = prepare(torch.float16, DAILY_CASES[2])
        md = metadata(c)
        r = call(c["args"], record=md)
        assert md.shape == (446,) and len(r) == 4
        assert [t.shape for t in r] == [(10, 3, 512), (3, 10), (8, 3, 10, 512), (8, 3, 10)]
        assert all(t.dtype == torch.float32 for t in r[1:])


def test_compile():
    c = prepare(torch.bfloat16, DAILY_CASES[2])
    fn = torch.compile(
        lambda a: call(a, record=metadata(dict(args=a))), backend="eager", fullgraph=True
    )
    compare(c, fn(c["args"]))


@pytest.mark.parametrize("fake", [False, True])
@pytest.mark.parametrize(
    "fault",
    [
        "split-one",
        "split-over",
        "split-negative",
        "scale",
        "used-size",
        "prefix-dtype",
        "metadata-size",
        "grad",
        "stride",
        "max-q",
    ],
)
def test_reject(fake, fault):
    with FakeTensorMode() if fake else nullcontext():
        c = prepare(torch.float16, DAILY_CASES[0])
        a = c["args"]
        record = None
        if fault.startswith("split"):
            a["num_splits"] = {
                "split-one": 1,
                "split-over": 257,
                "split-negative": -1,
            }[fault]
        elif fault == "scale":
            a["softmax_scale"] = float("nan")
        elif fault == "used-size":
            a["seqused_q"] = torch.zeros(1, dtype=torch.int32, device="npu")
        elif fault == "prefix-dtype":
            a["cu_seqlens_q"] = a["cu_seqlens_q"].to(torch.int64)
        elif fault == "metadata-size":
            record = torch.empty(1, dtype=torch.int32, device="npu")
        elif fault == "grad":
            a["q"].requires_grad_()
        elif fault == "stride":
            a["q"] = torch.empty((*a["q"].shape[:-1], 128), device="npu", dtype=a["q"].dtype)[
                ..., ::2
            ]
        elif fault == "max-q":
            a["max_seqlen_q"] = -1
        with pytest.raises(RuntimeError, match="MLA"):
            call(a, "op", record)


def test_fake_accumulator_overflow():
    with FakeTensorMode():
        total, heads, batch = 2**31 - 1, 2**20, 2**23
        q = torch.empty_strided((total, heads, 64), (0, 0, 1), dtype=torch.float16, device="npu")
        qv = torch.empty_strided((total, heads, 512), (0, 0, 1), dtype=torch.float16, device="npu")
        k = torch.empty((batch, 1, 1, 64), dtype=torch.float16, device="npu")
        v = torch.empty((batch, 1, 1, 512), dtype=torch.float16, device="npu")
        cu = torch.empty(batch + 1, dtype=torch.int32, device="npu")
        with pytest.raises(RuntimeError, match="MLA: mixed split.*overflow"):
            torch.ops.flash_attn_npu_3._mla_split_mixed_forward(q, k, v, qv, cu, 256, 256)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("fake", [False, True])
def test_metadata_entries(dtype, fake):
    with FakeTensorMode() if fake else nullcontext():
        c = prepare(dtype, DAILY_CASES[2])
        records = []
        for entry in ("public", "op") if fake else ("public", "op", "cpp"):
            with forbid_host_readback():
                md = metadata(c, entry)
            assert md.shape == (446,) and md.dtype == torch.int32
            records.append(md)
        if not fake:
            for md in records:
                assert md.cpu().tolist()[:6] == [
                    0x4D4C4131,
                    21,
                    21,
                    446,
                    1 if dtype == torch.float16 else 2,
                    0,
                ]
                assert torch.equal(md.cpu(), records[0].cpu())


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_metadata_failure_recovery(dtype):
    c = prepare(dtype, DAILY_CASES[2])
    a = c["args"]
    valid = metadata(c).cpu()
    for name, index, value in [
        ("cu_seqlens_q", 0, 1),
        ("cu_seqlens_q", 2, 6),
        ("cu_seqlens_q", 3, -1),
        ("cu_seqlens_q", 1, 8),
        ("cache_seqlens", 0, -1),
        ("cache_seqlens", 1, 34),
    ]:
        saved = a[name].clone()
        a[name][index] = value
        with forbid_host_readback():
            md = metadata(c)
        words = md.cpu().tolist()
        assert words[1:3] == [21, 21] and words[5] != 0 and not any(words[8:])
        a[name].copy_(saved)
        assert torch.equal(metadata(c).cpu(), valid)


@pytest.mark.parametrize("fake", [False, True])
def test_storage_and_metadata_descriptors(fake, storage_fault):
    with FakeTensorMode() if fake else nullcontext():
        c = prepare(torch.float16, dict(DAILY_CASES[2], stride="row_65536"))
        for name in ("q", "k", "v", "qv"):
            tensor = c["args"][name]
            # The fault keeps compact bytes. A dense transposed Q view would
            # still fit, so require actual holes in every injected input.
            assert (
                1 + sum((n - 1) * s for n, s in zip(tensor.shape, tensor.stride())) > tensor.numel()
            )
            for entry in ("public", "eager", "op") if fake else ("public", "eager", "cpp", "op"):
                with (
                    storage_fault(c["args"][name]),
                    pytest.raises(RuntimeError, match="input view exceeds"),
                ):
                    call(c["args"], entry)
        for name, value in (
            ("cu_seqlens_q", c["args"]["cu_seqlens_q"].to(torch.int64)),
            ("cu_seqlens_q", c["args"]["cu_seqlens_q"][:-1]),
            ("cu_seqlens_q", torch.empty(8, dtype=torch.int32, device="npu")[::2]),
            ("cache_seqlens", torch.zeros(1, dtype=torch.int32, device="npu")),
            ("cache_seqlens", c["args"]["cache_seqlens"].to(torch.int64)),
        ):
            bad = dict(c, args=dict(c["args"], **{name: value}))
            with pytest.raises(RuntimeError, match="MLA"):
                metadata(bad)


def test_compiled_storage(storage_fault):
    c = prepare(torch.float16, dict(DAILY_CASES[2], stride="row_65536"))
    fn = torch.compile(lambda a: call(a), backend="eager", fullgraph=True)
    fn(c["args"])
    torch.npu.synchronize()
    for name in ("q", "k", "v", "qv"):
        tensor = c["args"][name]
        assert 1 + sum((n - 1) * s for n, s in zip(tensor.shape, tensor.stride())) > tensor.numel()
        with (
            storage_fault(c["args"][name]),
            pytest.raises(RuntimeError, match="input view exceeds"),
        ):
            fn(c["args"])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_raw_dispatch_masks(dtype):
    # Both independently callable raw dispatchers changed; preserve their three
    # mask branches here even when public-entry matrix reduction is enabled.
    for s in DAILY_CASES:
        c = prepare(dtype, s)
        refs = reference(c)
        for entry in ("eager", "cpp"):
            compare(c, call(c["args"], entry), refs)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_internal_used_masks(dtype):
    for s in DAILY_CASES:
        s = dict(s, uq=[min(1, q) for q in s["q"]])
        c = prepare(dtype, s)
        refs = reference(c)
        for entry in ("op", "cpp"):
            compare(c, call(c["args"], entry), refs)
