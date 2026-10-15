"""v23 production and public PA split; independent raw-page/math mapping."""

from contextlib import nullcontext
import gc
import json
import os
from pathlib import Path
import time

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.common.compare import assert_fa_close
from tests.mla.support.mla_split_bsnd_reference import references
from tests.mla.support.mla_metadata_error_probe import forbid_host_readback
from tests.mla.support.mla_split_paged_bsnd_cases import (
    DAILY_CASES,
    REPRESENTATIVES,
    BOUNDARIES,
    spec,
)
from tests.mla.e2e.bsnd.test_flash_attn_npu_v3_mla_bsnd_stride_production import (
    storage_fault as storage_fault,
)
from tests.mla.e2e.dense.test_flash_attn_npu_v3_mla_contract import _forward


def prepare(dtype, s):
    b, q, h, p, cap = len(s["lengths"]), s["q"], s["heads"], s["page"], s["capacity"]
    columns = cap // p
    pool = b * columns + 2
    g = torch.Generator().manual_seed(10723)
    raw = {
        n: torch.randn(shape, generator=g, dtype=dtype) * 0.1
        for n, shape in (
            ("q", (b, q, h, 64)),
            ("qv", (b, q, h, 512)),
            ("k", (pool, p, 1, 64)),
            ("v", (pool, p, 1, 512)),
        )
    }
    table = torch.arange(b * columns, dtype=torch.int32).reshape(b, columns).flip(1) + 1
    for bi, length in enumerate(s["lengths"]):
        for token in range(length, cap):
            for name in ("k", "v"):
                raw[name][(bi + 1) * columns - token // p, token % p] = torch.nan
    args = {}
    for i, (name, x) in enumerate(raw.items()):
        if s["stride"]:
            strides = (
                x.shape[1] * (x.shape[2] + 1) * (x.shape[3] + 16) + 32,
                (x.shape[2] + 1) * (x.shape[3] + 16),
                x.shape[3] + 16,
                1,
            )
            count = 1 + sum(max(0, n - 1) * st for n, st in zip(x.shape, strides))
            backing = torch.empty(count + 32, dtype=dtype, device="npu")
            value = backing.as_strided(x.shape, strides, 16)
            value.copy_(x)
        else:
            value = x.to("npu")
        args[name] = value
    if s["stride"]:
        backing = torch.empty((b, columns + 3), dtype=torch.int32, device="npu")
        device_table = backing[:, 1 : columns + 1]
        device_table.copy_(table)
    else:
        device_table = table.to("npu")
    args.update(
        page_table=device_table,
        cache_seqlens=torch.tensor(s["lengths"], dtype=torch.int32, device="npu"),
        seqused_q=None
        if s["used"] is None
        else torch.tensor(s["used"], dtype=torch.int32, device="npu"),
        num_splits=s["splits"],
        causal=s["causal"],
        window_size=s["window"],
        softmax_scale=s["scale"],
    )
    return dict(args=args, raw=raw, table=table, spec=s)


def metadata(c, entry="public"):
    a, s = c["args"], c["spec"]
    lengths = a["cache_seqlens"]
    if lengths is None or isinstance(lengths, int):
        lengths = torch.full(
            (a["q"].shape[0],),
            s["capacity"] if lengths is None else lengths,
            dtype=torch.int32,
            device=a["q"].device,
        )
    if entry == "op":
        return torch.ops.flash_attn_npu_3._mla_paged_metadata(
            a["q"].shape[0],
            s["q"],
            s["capacity"],
            s["heads"],
            s["page"],
            a["q"].dtype,
            lengths,
            None,
            s["causal"],
            *s["window"],
            s["splits"],
        )
    from tests.test_flash_attn_npu_v3_metadata_signature import metadata as generate

    return generate(
        entry,
        dict(
            batch_size=a["q"].shape[0],
            max_seqlen_q=s["q"],
            max_seqlen_k=s["capacity"],
            num_heads_q=s["heads"],
            num_heads_kv=1,
            headdim=64,
            headdim_v=512,
            qkv_dtype=a["q"].dtype,
            cache_seqlens=lengths,
            page_size=s["page"],
            causal=s["causal"],
            window_size=s["window"],
            num_splits=s["splits"],
        ),
    )


def call(a, entry="public", record=None, *, lse=True):
    a = dict(a)
    if record is not None:
        a["scheduler_metadata"] = record
    if entry == "public":
        assert a.pop("seqused_q", None) is None
        a["k_cache"], a["v_cache"] = a.pop("k"), a.pop("v")
        return api.flash_attn_with_kvcache(**a, return_softmax_lse=lse)
    left, right = a.pop("window_size")
    if entry == "op":
        return torch.ops.flash_attn_npu_3._mla_split_paged_bsnd_forward(
            **a, window_left=left, window_right=right
        )
    a["seqused_k"] = a.pop("cache_seqlens")
    return _forward(entry, dict(a, window_size_left=left, window_size_right=right))


def reference(c):
    s = c["spec"]
    b = len(s["lengths"])
    p = s["page"]
    cap = s["capacity"]
    raw = {n: c["raw"][n] for n in ("q", "qv")}
    for name, d in [("k", 64), ("v", 512)]:
        raw[name] = torch.zeros((b, cap, 1, d), dtype=c["raw"][name].dtype)
        for bi, length in enumerate(s["lengths"]):
            for token in range(length):
                raw[name][bi, token] = c["raw"][name][int(c["table"][bi, token // p]), token % p]
    # Geometry comes from the independently tested producer. Used queries do
    # not change PA source M selection, which uses original allocation rows.
    md = metadata(c).cpu()
    m, n = (
        int(md[15]) or 256,
        int(md[16]) or min(cap, 512 // p * p if p < 512 and 512 % p else 512) or 1,
    )
    refs = references(
        raw,
        [s["q"]] * b if s["used"] is None else s["used"],
        s["lengths"],
        s["splits"],
        m,
        n,
        s["causal"],
        s["window"],
        576**-0.5 if s["scale"] is None else s["scale"],
    )
    if cap == 0:
        refs = {k: v.clone() for k, v in refs.items()}
        refs["lse_high"].fill_(-torch.inf)
        refs["lse_low"].fill_(-torch.inf)
    return refs


def compare(c, result, refs=None):
    refs = reference(c) if refs is None else refs
    s = c["spec"]
    used = [s["q"]] * len(s["lengths"]) if s["used"] is None else s["used"]
    for i, name in enumerate(("out", "lse", "accum", "accum_lse")):
        got = result[i].cpu()
        assert got.shape == refs[name + "_high"].shape
        assert got.dtype == (c["raw"]["q"].dtype if i == 0 else torch.float32)
        for b, q in enumerate(used):
            idx = (
                (b, slice(q))
                if i == 0
                else (b, slice(None), slice(q))
                if i == 1
                else (slice(None), b, slice(None), slice(q))
            )
            assert_fa_close(
                got[idx],
                refs[name + "_high"][idx],
                refs[name + "_low"][idx],
                name="paged_split_" + name,
            )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("s", REPRESENTATIVES)
def test_representative(dtype, s):
    start = time.monotonic()
    c = prepare(dtype, s)
    r = call(c["args"], "op")
    compare(c, r)
    if os.environ.get("MLA_VALIDATION_RESULT"):
        path = Path(os.environ["MLA_VALIDATION_RESULT"]) / f"cost-{dtype}-{s['page']}.json"
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
        md = metadata(c)
        saved = md.clone()
        before = [c["args"][n].clone() for n in ("k", "v", "page_table")]
        results = [call(c["args"]), call(c["args"], record=md)]
    for r in results:
        compare(c, r, refs)
    assert torch.equal(md.cpu(), saved.cpu())
    for n, x in zip(("k", "v", "page_table"), before):
        torch.testing.assert_close(c["args"][n], x, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("s", BOUNDARIES)
def test_boundary(dtype, s):
    c = prepare(dtype, s)
    compare(c, call(c["args"], "op"))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("empty", ["q", "kv", "batch"])
def test_empty(dtype, empty):
    s = spec(
        q=0 if empty == "q" else 3,
        lengths=() if empty == "batch" else (0, 0),
        capacity=0 if empty == "kv" else 224,
    )
    c = prepare(dtype, s)
    refs = reference(c)
    compare(c, call(c["args"]), refs)
    compare(c, call(c["args"], record=metadata(c)), refs)


def test_fake():
    with FakeTensorMode():
        c = prepare(torch.float16, DAILY_CASES[2])
        md = metadata(c)
        assert md.shape == (428,)
        for lengths in (None, c["spec"]["capacity"], c["args"]["cache_seqlens"]):
            r = call(dict(c["args"], cache_seqlens=lengths), record=md)
            assert [tuple(t.shape) for t in r] == [
                (2, 5, 3, 512),
                (2, 3, 5),
                (8, 2, 3, 5, 512),
                (8, 2, 3, 5),
            ]


def test_compile():
    c = prepare(torch.bfloat16, DAILY_CASES[2])
    fn = torch.compile(lambda a: call(a), backend="eager", fullgraph=True)
    compare(c, fn(c["args"]))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_dispatch_masks_and_used(dtype):
    for s in DAILY_CASES:
        c = prepare(dtype, s)
        refs = reference(c)
        for entry in ("eager", "cpp", "op"):
            compare(c, call(c["args"], entry, metadata(c)), refs)
        c = prepare(dtype, dict(s, used=[1, 0]))
        refs = reference(c)
        for entry in ("op", "cpp"):
            compare(c, call(c["args"], entry), refs)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_reuse_lengths_scales_pages(dtype):
    s = spec(lengths=(224, 224), capacity=224, stride=True)
    c = prepare(dtype, s)
    md = metadata(c)
    saved = md.cpu()
    for scale in (None, 0.0, -0.1):
        d = prepare(dtype, dict(s, stride=scale is None, scale=scale))
        refs = reference(d)
        for lengths in (None, 224, d["args"]["cache_seqlens"]):
            a = dict(d["args"], cache_seqlens=lengths)
            compare(d, call(a, record=md), refs)
        # Arbitrary legal table permutation is a new physical input, not stale metadata.
        d["table"] = d["table"].flip(1)
        d["args"]["page_table"].copy_(d["table"])
        compare(d, call(d["args"], record=md))
        torch.testing.assert_close(
            call(d["args"], record=md, lse=False), call(d["args"], record=md)[0], rtol=0, atol=0
        )
    assert torch.equal(md.cpu(), saved)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_metadata_failure_recovery_and_lifetime(dtype):
    c = prepare(dtype, DAILY_CASES[2])
    good = metadata(c).cpu()
    a = c["args"]
    for value in (-1, c["spec"]["capacity"] + 1):
        saved = a["cache_seqlens"].clone()
        a["cache_seqlens"][0] = value
        with forbid_host_readback():
            bad = metadata(c)
        words = bad.cpu().tolist()
        assert words[1:3] == [23, 23] and words[5] != 0 and not any(words[8:])
        a["cache_seqlens"].copy_(saved)
        assert torch.equal(metadata(c).cpu(), good)
    c["spec"] = dict(c["spec"], lengths=[17, 0])
    a["cache_seqlens"][0] = 17
    refs = reference(c)
    streams = [torch.npu.Stream(), torch.npu.Stream()]
    results = []
    for stream in streams:
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream), forbid_host_readback():
            md = metadata(c)
            results.append(call(a, record=md))
            del md
    del a
    c.pop("args")
    gc.collect()
    trash = [torch.empty(1024, dtype=torch.int32, device="npu") for _ in range(32)]
    for stream in streams:
        stream.synchronize()
    for r in results:
        compare(c, r, refs)
    del trash


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("fake", [False, True])
def test_metadata_entries(dtype, fake):
    with FakeTensorMode() if fake else nullcontext():
        c = prepare(dtype, DAILY_CASES[2])
        records = [
            metadata(c, entry) for entry in (("public", "op") if fake else ("public", "op", "cpp"))
        ]
        for md in records:
            assert md.shape == (428,) and md.dtype == torch.int32
            if not fake:
                assert md.cpu().tolist()[:6] == [
                    0x4D4C4131,
                    23,
                    23,
                    428,
                    1 if dtype == torch.float16 else 2,
                    0,
                ]
                assert torch.equal(md.cpu(), records[0].cpu())


@pytest.mark.parametrize("fake", [False, True])
@pytest.mark.parametrize(
    "fault",
    [
        "zero",
        "one",
        "over",
        "negative",
        "grad",
        "scale",
        "stride",
        "table-stride",
        "table-dtype",
        "used-size",
        "metadata-size",
        "used-reuse",
    ],
)
def test_reject(fake, fault):
    with FakeTensorMode() if fake else nullcontext():
        c = prepare(torch.float16, DAILY_CASES[0])
        a = c["args"]
        md = None
        if fault in ("zero", "one", "over", "negative"):
            a["num_splits"] = dict(zero=0, one=1, over=257, negative=-1)[fault]
        elif fault == "grad":
            a["q"].requires_grad_()
        elif fault == "scale":
            a["softmax_scale"] = float("inf")
        elif fault == "stride":
            a["q"] = torch.empty((*a["q"].shape[:-1], 128), dtype=a["q"].dtype, device="npu")[
                ..., ::2
            ]
        elif fault == "table-stride":
            a["page_table"] = torch.zeros((2, 4), dtype=torch.int32, device="npu")[:, ::2]
        elif fault == "table-dtype":
            a["page_table"] = a["page_table"].to(torch.int64)
        elif fault == "used-size":
            a["seqused_q"] = torch.zeros(1, dtype=torch.int32, device="npu")
        elif fault == "metadata-size":
            md = torch.zeros(1, dtype=torch.int32, device="npu")
        elif fault == "used-reuse":
            md = metadata(c)
            a["seqused_q"] = torch.ones(2, dtype=torch.int32, device="npu")
        with pytest.raises(RuntimeError, match="MLA"):
            call(a, "op", md)


@pytest.mark.parametrize("fake", [False, True])
def test_storage(fake, storage_fault):
    with FakeTensorMode() if fake else nullcontext():
        c = prepare(torch.float16, DAILY_CASES[2])
        for name in ("q", "qv", "k", "v", "page_table"):
            for entry in ("public", "op") if fake else ("public", "op", "cpp"):
                with storage_fault(c["args"][name]), pytest.raises(RuntimeError, match="MLA"):
                    call(c["args"], entry)


def test_compiled_storage(storage_fault):
    c = prepare(torch.float16, DAILY_CASES[2])
    fn = torch.compile(lambda a: call(a), backend="eager", fullgraph=True)
    fn(c["args"])
    torch.npu.synchronize()
    for name in ("q", "qv", "k", "v", "page_table"):
        with storage_fault(c["args"][name]), pytest.raises(RuntimeError, match="MLA"):
            fn(c["args"])


def test_fake_accumulator_overflow():
    with FakeTensorMode():
        b, sq, h = 2**23, 65536, 128
        q = torch.empty_strided((b, sq, h, 64), (0, 0, 0, 1), dtype=torch.float16, device="npu")
        qv = torch.empty_strided((b, sq, h, 512), (0, 0, 0, 1), dtype=torch.float16, device="npu")
        k = torch.empty((1, 16, 1, 64), dtype=torch.float16, device="npu")
        v = torch.empty((1, 16, 1, 512), dtype=torch.float16, device="npu")
        table = torch.empty_strided((b, 1), (0, 1), dtype=torch.int32, device="npu")
        with pytest.raises(RuntimeError, match="MLA:.*overflow"):
            torch.ops.flash_attn_npu_3._mla_split_paged_bsnd_forward(q, k, v, qv, table, 256)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["broadcast", "wide-row"])
def test_layout_extremes(dtype, layout):
    c = prepare(dtype, spec(lengths=(224, 224), capacity=224))
    a = c["args"]
    if layout == "broadcast":
        for name in ("q", "qv", "k", "v"):
            a[name] = a[name][:1, :1, :1].expand_as(a[name])
        a["page_table"] = a["page_table"][:1].expand_as(a["page_table"])
    else:
        for name in ("q", "qv"):
            x = a[name]
            b, sq, h, d = x.shape
            row = 65536 + 16
            storage = torch.empty(b * sq * row + 16, dtype=dtype, device="npu")
            a[name] = storage.as_strided(x.shape, (sq * row, row, d, 1), 16)
            a[name].copy_(x)
    c["raw"] = {name: a[name].cpu().contiguous() for name in ("q", "qv", "k", "v")}
    c["table"] = a["page_table"].cpu().contiguous()
    refs = reference(c)
    md = metadata(c)
    for entry in ("public", "cpp"):
        compare(c, call(a, entry, md), refs)


@pytest.mark.parametrize("fake", [False, True])
def test_unopened_paged_split(fake):
    with FakeTensorMode() if fake else nullcontext():
        c = prepare(torch.float16, DAILY_CASES[0])
        a = dict(c["args"], num_splits=0)
        for entry in ("public", "eager", "cpp"):
            with pytest.raises(RuntimeError, match="MLA"):
                call(a, entry)
        with pytest.raises(RuntimeError, match="MLA"):
            metadata(dict(c, spec=dict(c["spec"], splits=0), args=a))
        a = dict(c["args"])
        a["q"] = a["q"].flatten(0, 1)
        a["qv"] = a["qv"].flatten(0, 1)
        a["cu_seqlens_q"] = torch.tensor([0, 3, 6], dtype=torch.int32, device="npu")
        a["max_seqlen_q"] = 3
        for entry in ("eager", "cpp"):
            with pytest.raises(RuntimeError, match="MLA"):
                call(a, entry)
