"""TND split entry, compact allocation addresses, real accumulators and lifecycle."""

import copy
import gc
import json
import os
import time
from pathlib import Path
import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.common.compare import assert_fa_close
from tests.mla.e2e.window_tnd.test_flash_attn_npu_v3_mla_window_tnd_public import (
    prepare as prepare_window,
    effective,
)
from tests.mla.e2e.dense.test_flash_attn_npu_v3_mla_contract import _forward
from tests.mla.support.mla_metadata_error_probe import forbid_host_readback
from tests.mla.support.mla_split_bsnd_reference import references
from tests.mla.support.mla_split_tnd_cases import spec, REPRESENTATIVES, DAILY_CASES


def prepare(dtype, s):
    case = prepare_window(dtype, copy.deepcopy(s))
    case["args"]["num_splits"] = s["splits"]
    return case


def op_args(args):
    options = dict(args)
    options["window_left"], options["window_right"] = options.pop("window_size")
    return options


def call(args, entry="public", record=None):
    if entry == "public":
        assert record is None
        return api.flash_attn_varlen_func(**args, return_attn_probs=True)
    if entry == "op":
        return torch.ops.flash_attn_npu_3._mla_split_tnd_forward(
            **op_args(args), scheduler_metadata=record
        )
    options = dict(args, scheduler_metadata=record)
    options["window_size_left"], options["window_size_right"] = options.pop("window_size")
    return _forward(entry, options)


def metadata(c):
    options = op_args(c["args"])
    options.pop("softmax_scale")
    return torch.ops.flash_attn_npu_3._mla_split_tnd_metadata(**options)


def reference(c, tile=None):
    a, s = c["args"], c["spec"]
    # Only tile choice comes from metadata; mask, allocation and used geometry
    # are independently reconstructed from original caller data.
    if tile is None:
        md = metadata(c).cpu()
        tile = (int(md[15]), int(md[16])) if sum(s["q"]) and sum(s["kv"]) else (256, 512)
    padded = {}
    for name, t in c["raw"].items():
        query = name in ("q", "qv")
        lengths = s["q"] if query else s["kv"]
        cap = a["max_seqlen_q"] if query else a["max_seqlen_k"]
        p = torch.zeros((len(lengths), cap, *t.shape[1:]), dtype=t.dtype)
        offset = 0
        for b, length in enumerate(lengths):
            p[b, :length] = t[offset : offset + length]
            offset += length
        padded[name] = p
    refs = references(
        padded,
        *effective(s),
        s["splits"],
        *tile,
        s.get("causal", False),
        s["window"],
        s.get("scale") if s.get("scale") is not None else 576**-0.5,
    )
    result = {}
    for name, t in refs.items():
        axis = 0 if name.startswith("out_") else 1 if name.startswith("lse_") else 2
        pieces = [
            t[b, :q] if axis == 0 else t[b, :, :q] if axis == 1 else t[:, b, :, :q]
            for b, q in enumerate(s["q"])
        ]
        shape = (
            [0, s["heads"], 512]
            if axis == 0
            else [s["heads"], 0]
            if axis == 1
            else [s["splits"], s["heads"], 0] + ([512] if not name.startswith("accum_lse") else [])
        )
        result[name] = torch.cat(pieces, dim=axis) if pieces else t.new_empty(shape)
        if sum(s["kv"]) == 0 and name.startswith("lse_"):
            result[name].fill_(torch.inf)
    return result


def compare(c, result, refs=None):
    refs = reference(c) if refs is None else refs
    qs, _ = effective(c["spec"])
    for i, (actual, name) in enumerate(zip(result, ("out", "lse", "accum", "accum_lse"))):
        actual = actual.cpu()
        assert actual.shape == refs[name + "_high"].shape
        assert actual.dtype == (c["raw"]["q"].dtype if i == 0 else torch.float32)
        start = 0
        for allocation, used in zip(c["spec"]["q"], qs):
            idx = (
                (slice(start, start + used),)
                if i == 0
                else (slice(None), slice(start, start + used))
                if i == 1
                else (slice(None), slice(None), slice(start, start + used))
            )
            assert_fa_close(
                actual[idx],
                refs[name + "_high"][idx],
                refs[name + "_low"][idx],
                name="tnd-split-" + name,
            )
            start += allocation
    for before, actual in c["guards"]:
        torch.testing.assert_close(actual.cpu(), before, rtol=0, atol=0, equal_nan=True)
    return refs


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("s", REPRESENTATIVES)
def test_representative(dtype, s):
    started = time.monotonic()
    c = prepare(dtype, s)
    result = call(c["args"], "op")
    compare(c, result)
    if os.environ.get("MLA_VALIDATION_RESULT"):
        path = (
            Path(os.environ["MLA_VALIDATION_RESULT"])
            / f"cost-{str(dtype).split('.')[-1]}-{s['splits']}-{sum(s['kv'])}.json"
        )
        path.write_text(
            json.dumps(
                dict(
                    spec=s,
                    seconds=time.monotonic() - started,
                    accumulator_bytes=sum(t.nbytes for t in result[2:]),
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
        result = call(c["args"])
    compare(c, result, refs)
    # The private path verifies returned accumulators and the immutable record.
    record = metadata(c)
    saved = record.clone()
    compare(c, call(c["args"], "op", record), refs)
    assert torch.equal(record.cpu(), saved.cpu())


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "s",
    [
        spec(q=(0, 0), k=(3, 7)),
        spec(q=(3, 7), k=(0, 0)),
        spec(q=(), k=()),
        spec(uq=[0, 0, 0]),
        spec(uk=[0, 0, 0]),
        spec(q=(65, 1), k=(513, 3), h=3, splits=193, window=(0, 0)),
        spec(q=(3, 2), k=(513, 17), splits=255, window=(-1, 1)),
        spec(q=(3, 7), k=(33, 513), uq=[1, 2], uk=[17, 257], sq=64, sk=1025, window=(5, -1)),
    ],
)
def test_boundaries(dtype, s):
    c = prepare(dtype, s)
    compare(c, call(c["args"], "op"))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("query_empty", [False, True])
def test_empty(dtype, query_empty):
    s = spec(q=(0, 0) if query_empty else (3, 1), k=(7, 3) if query_empty else (0, 0), h=2)
    c = prepare(dtype, s)
    compare(c, call(c["args"], "op"))


@pytest.mark.parametrize("entry", ["public", "eager", "cpp", "op"])
def test_entries(entry):
    c = prepare(torch.float16, DAILY_CASES[2])
    compare(c, call(c["args"], entry))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_reuse_updates_and_lifetime(dtype):
    c = prepare(dtype, DAILY_CASES[2])
    record = metadata(c)
    for scale in (0.0, -0.1, None):
        c["spec"]["scale"] = scale
        c["args"]["softmax_scale"] = scale
        compare(c, call(c["args"], "op", record))
    c["spec"]["uk"] = [3, 0, 0]
    c["args"]["seqused_k"].copy_(torch.tensor([3, 0, 0], dtype=torch.int32))
    compare(c, call(c["args"]))
    refs = reference(c)
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream), forbid_host_readback():
        record = metadata(c)
        result = call(c["args"], "op", record)
        del record
    # Captured tensors, including prefix/used storage, must outlive submission.
    args = c.pop("args")
    del args
    gc.collect()
    trash = [torch.empty(1024, dtype=torch.int32, device="npu") for _ in range(32)]
    stream.synchronize()
    compare(c, result, refs)
    del trash


def test_fake():
    with FakeTensorMode():
        c = prepare(torch.float16, DAILY_CASES[2])
        md = metadata(c)
        r = call(c["args"], "op", md)
        assert md.shape == (446,) and len(r) == 4
        assert r[0].shape == (10, 3, 512) and r[1].shape == (3, 10)
        assert r[2].shape == (8, 3, 10, 512) and r[3].shape == (8, 3, 10)
        for t in r[1:]:
            assert t.dtype == torch.float32


def test_compile():
    c = prepare(torch.bfloat16, DAILY_CASES[2])
    fn = torch.compile(lambda a: call(a), backend="eager", fullgraph=True)
    compare(c, fn(c["args"]))


@pytest.mark.parametrize("fake", [False, True])
@pytest.mark.parametrize(
    "fault",
    [
        "split-zero",
        "split-over",
        "split-negative",
        "scale",
        "used-size",
        "prefix-dtype",
        "metadata-size",
        "grad",
        "stride",
    ],
)
def test_reject(fake, fault):
    from contextlib import nullcontext

    with FakeTensorMode() if fake else nullcontext():
        c = prepare(torch.float16, DAILY_CASES[0])
        a = c["args"]
        record = None
        if fault.startswith("split"):
            a["num_splits"] = {"split-zero": 0, "split-over": 257, "split-negative": -1}[fault]
        elif fault == "scale":
            a["softmax_scale"] = float("nan")
        elif fault == "used-size":
            a["seqused_q"] = torch.zeros(1, dtype=torch.int32, device="npu")
        elif fault == "prefix-dtype":
            a["cu_seqlens_k"] = a["cu_seqlens_k"].to(torch.int64)
        elif fault == "metadata-size":
            record = torch.empty(1, dtype=torch.int32, device="npu")
        elif fault == "grad":
            a["q"].requires_grad_()
        elif fault == "stride":
            a["q"] = torch.empty((*a["q"].shape[:-1], 128), device="npu", dtype=a["q"].dtype)[
                ..., ::2
            ]
        with pytest.raises(RuntimeError, match="MLA"):
            call(a, "op", record)


def test_fake_accumulator_overflow():
    with FakeTensorMode():
        total, heads, batch = 2**31 - 1, 2**20, 2**23
        q = torch.empty_strided((total, heads, 64), (0, 0, 1), dtype=torch.float16, device="npu")
        qv = torch.empty_strided((total, heads, 512), (0, 0, 1), dtype=torch.float16, device="npu")
        k = torch.empty((1, 1, 64), dtype=torch.float16, device="npu")
        v = torch.empty((1, 1, 512), dtype=torch.float16, device="npu")
        cu = torch.empty(batch + 1, dtype=torch.int32, device="npu")
        with pytest.raises(RuntimeError, match="MLA: TND split.*overflow"):
            torch.ops.flash_attn_npu_3._mla_split_tnd_forward(q, k, v, qv, cu, cu, 256, 1, 256)
