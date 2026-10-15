"""v25 production and public PA split; independent raw-page/math mapping."""

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
from tests.mla.support.mla_metadata_error_probe import forbid_host_readback
from tests.mla.support.mla_split_paged_tnd_cases import (
    DAILY_CASES,
    REPRESENTATIVES,
    BOUNDARIES,
    spec,
)
from tests.mla.support.public_cases.bsnd_stride_production import storage_fault as storage_fault

from tests.mla.support.public_cases.split_paged_tnd_public import (
    call as call,
    metadata as metadata,
    prepare as prepare,
    reference as reference,
)


def compare(c, result, refs=None):
    refs = reference(c) if refs is None else refs
    for i, name in enumerate(("out", "lse", "accum", "accum_lse")):
        got = result[i].cpu()
        assert got.shape == refs[name + "_high"].shape
        assert got.dtype == (c["raw"]["q"].dtype if i == 0 else torch.float32)
        assert_fa_close(
            got, refs[name + "_high"], refs[name + "_low"], name="tnd-paged-split-" + name
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
                (7, 3, 512),
                (3, 7),
                (8, 3, 7, 512),
                (8, 3, 7),
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
        assert words[1:3] == [25, 25] and words[5] != 0 and not any(words[8:])
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
                    25,
                    25,
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
        "prefix-dtype",
        "prefix-size",
        "prefix-stride",
        "max-q",
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
        elif fault == "prefix-dtype":
            a["cu_seqlens_q"] = a["cu_seqlens_q"].to(torch.int64)
        elif fault == "prefix-size":
            a["cu_seqlens_q"] = a["cu_seqlens_q"][:2]
        elif fault == "prefix-stride":
            a["cu_seqlens_q"] = torch.zeros(6, dtype=torch.int32, device="npu")[::2]
        elif fault == "max-q":
            a["max_seqlen_q"] = -1
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
        for name in ("q", "qv", "k", "v", "page_table", "cu_seqlens_q", "cache_seqlens"):
            tensor = c["args"][name]
            # The existing fixture truncates to the compact bytes of its view.
            # A one-element alias truncates a contiguous descriptor's shared
            # storage while the forwarded prefix/length keeps its full shape.
            fault_view = tensor[:1] if name in ("cu_seqlens_q", "cache_seqlens") else tensor
            for entry in ("public", "op") if fake else ("public", "op", "cpp"):
                with storage_fault(fault_view), pytest.raises(RuntimeError, match="MLA"):
                    call(c["args"], entry)


def test_compiled_storage(storage_fault):
    c = prepare(torch.float16, DAILY_CASES[2])
    fn = torch.compile(lambda a: call(a), backend="eager", fullgraph=True)
    fn(c["args"])
    torch.npu.synchronize()
    for name in ("q", "qv", "k", "v", "page_table", "cu_seqlens_q", "cache_seqlens"):
        tensor = c["args"][name]
        fault_view = tensor[:1] if name in ("cu_seqlens_q", "cache_seqlens") else tensor
        with storage_fault(fault_view), pytest.raises(RuntimeError, match="MLA"):
            fn(c["args"])


def test_fake_accumulator_overflow():
    # Static TND total, not B*maxQ, defines accumulator bytes.
    with pytest.raises(RuntimeError, match="overflow"):
        api._check_mla_paged_split_budget(1, 65536, 65536, 16, 16, 256, total_q=2**31 - 1)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["broadcast", "wide-row"])
def test_layout_extremes(dtype, layout):
    c = prepare(dtype, spec(lengths=(224, 224), capacity=224))
    a = c["args"]
    if layout == "broadcast":
        for name in ("q", "qv", "k", "v"):
            a[name] = a[name][tuple(slice(1) for _ in range(a[name].ndim - 1))].expand_as(a[name])
        a["page_table"] = a["page_table"][:1].expand_as(a["page_table"])
    else:
        for name in ("q", "qv"):
            x = a[name]
            sq, h, d = x.shape
            row = 65536 + 16
            storage = torch.empty(sq * row + 16, dtype=dtype, device="npu")
            a[name] = storage.as_strided(x.shape, (row, d, 1), 16)
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
        c["spec"] = dict(c["spec"], splits=0)
        for entry in ("public", "op") if fake else ("public", "op", "cpp"):
            with pytest.raises(RuntimeError, match="MLA"):
                metadata(c, entry)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_prefix_generation_and_empty_allocation(dtype):
    for queries, bound in [((0, 0), 17), ((3, 0, 7), 9), ((0, 3, 0), 5)]:
        c = prepare(dtype, spec(q=queries, max_q=bound, lengths=(17,) * len(queries)))
        md = metadata(c)
        compare(c, call(c["args"], record=md))
        for bad in ([1] + [sum(queries)] * len(queries), [0] + [-1] * len(queries)):
            c["args"]["cu_seqlens_q"].copy_(torch.tensor(bad, dtype=torch.int32, device="npu"))
            failed = metadata(c).cpu().tolist()
            assert failed[5] and not any(failed[8:])
