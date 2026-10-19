"""Public INT8 fixtures; source NZ packing stays exclusively in test helpers."""

import json
import itertools
import os
from pathlib import Path

import torch

from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.common.compare import assert_fa_close
from tests.mla.support.mla_int8_cases import SOURCE_CASES, strided_inputs, NativeCall
from tests.mla.support.mla_int8_component import ComponentCall
from tests.mla.support.mla_int8_layout import preserve_view
from tests.mla.support.mla_int8_reference import references


BSND_CASES = tuple(spec for spec in SOURCE_CASES if spec[1] == "BSND")


def make_case(spec=SOURCE_CASES[8], view="contiguous", splits=0, uniform=False):
    raw = strided_inputs(spec, view)
    if view in ("query-gap", "scale-gap"):
        for name in ("q", "qv") if view == "query-gap" else ("qs",):
            t = raw[name]
            shape = list(t.shape)
            shape[-2 if name != "qs" else -1] *= 2
            backing = torch.zeros(shape, dtype=t.dtype)
            v = backing[..., ::2, :] if name != "qs" else backing[..., ::2]
            v.copy_(t)
            raw[name] = v
    if uniform:
        raw["q"].zero_()
        raw["qv"].zero_()
    tensors = {name: preserve_view(t) for name, t in raw.items()}
    lengths = torch.tensor(spec[4], dtype=torch.int32, device="npu")
    case = dict(spec=spec, raw=raw, tensors=tensors, lengths=lengths, splits=splits, view=view)
    if spec[1] == "TND":
        case["prefix"] = torch.tensor(
            [0, *itertools.accumulate(spec[2])], dtype=torch.int32, device="npu"
        )
    return case


def kwargs(c, record=None, lse=True):
    t = c["tensors"]
    return dict(
        cu_seqlens_q=c.get("prefix"),
        max_seqlen_q=max(c["spec"][2]) if "prefix" in c else None,
        q=t["q"],
        qv=t["qv"],
        k_cache=t["k"],
        v_cache=t["v"],
        page_table=t["table"],
        mla_q_scale=t["qs"],
        mla_k_scale=t["ks"],
        mla_v_scale=t["vs"],
        cache_seqlens=c["lengths"],
        scheduler_metadata=record,
        causal=bool(c["spec"][5]),
        num_splits=c["splits"],
        return_softmax_lse=lse,
    )


def metadata(c):
    t = c["tensors"]
    return api.get_scheduler_metadata(
        len(c["spec"][2]),
        max(c["spec"][2]),
        t["table"].shape[1] * 128,
        t["q"].shape[-2],
        1,
        64,
        c["lengths"],
        qkv_dtype=torch.int8,
        cu_seqlens_q=c.get("prefix"),
        headdim_v=512,
        page_size=128,
        causal=bool(c["spec"][5]),
        num_splits=c["splits"],
    )


def invoke(c, record=None, lse=True):
    return api.flash_attn_with_kvcache(**kwargs(c, record, lse))


def equal(left, right):
    for a, b in zip(left, right):
        assert a.shape == b.shape and a.dtype == b.dtype
        torch.testing.assert_close(a.cpu(), b.cpu(), rtol=0, atol=0)


def check_math(c, result, *, source_equal=False):
    refs = references(c["raw"], c["spec"])
    report = {}
    for i, name in enumerate(("out", "lse")):
        actual = result[i].cpu()
        high, low = refs[name + "_high"], refs[name + "_low"]
        if i:
            axis = 0 if "prefix" in c else 1
            high, low = high.transpose(axis, axis + 1), low.transpose(axis, axis + 1)
        else:
            high, low = high.to(torch.bfloat16), low.to(torch.bfloat16)
        try:
            assert_fa_close(actual, high, low, name="public-int8-" + name)
        except AssertionError as error:
            if not source_equal:
                raise
            report[name] = dict(status="source_limit", error=str(error))
        else:
            report[name] = dict(status="passed")
    return refs, report


def check_partials(c, result, source):
    out, lse, accum, logs = (t.cpu() for t in result)
    if c["splits"] == 1:
        assert accum.numel() == logs.numel() == 0
        return
    assert accum.dtype == logs.dtype == torch.float32
    tnd = "prefix" in c
    hq = out.shape[-2]
    assert accum.shape == (
        (26, hq, out.shape[0], 512) if tnd else (26, out.shape[0], hq, out.shape[1], 512)
    )
    p = source.params
    mode = source.fixture["mode"]
    if mode & 2:
        source_out = (
            source.workspace[
                source.partial_offset : source.partial_offset + p.splitKVParams.accumOutSize * 4
            ]
            .view(torch.float32)
            .cpu()
        )
        for b, sk in enumerate(c["spec"][4]):
            for q in range(c["spec"][2][b]):
                for h in range(hq):
                    index = (h, sum(c["spec"][2][:b]) + q) if tnd else (b, h, q)
                    if not sk:
                        continue
                    if mode & 4:
                        fd = p.tndSplitCoreParams
                        task = next(
                            (
                                i
                                for i in range(fd.tndFDCoreArrLen)
                                if fd.balanceFDCoreBArr[i] == b
                                and fd.balanceFDCoreS1Arr[i]
                                == q // p.increFlashAttentionSingleCoreParams.s1SplitSize
                            ),
                            None,
                        )
                        if task is None:
                            continue
                        parts = fd.balanceFDCoreKVSplitArr[task]
                        start = sum(fd.balanceFDCoreKVSplitArr[i] for i in range(task))
                        row = (q % p.increFlashAttentionSingleCoreParams.s1SplitSize) * hq + h

                        def offset(part):
                            return ((start + part) * 128 + row) * 512
                    else:
                        parts = (
                            sk + p.splitKVParams.sInnerLoopSize - 1
                        ) // p.splitKVParams.sInnerLoopSize

                        def offset(part):
                            return (b * hq * p.splitKVParams.s2 + part * hq + h) * 512

                    for part in range(parts):
                        torch.testing.assert_close(
                            accum[(part, *index)],
                            source_out[offset(part) : offset(part) + 512],
                            rtol=0,
                            atol=0,
                        )
                    assert torch.count_nonzero(accum[(slice(parts, None), *index)]) == 0
                    assert torch.isneginf(logs[(slice(parts, None), *index)]).all()
    weights = torch.softmax(logs, dim=0).nan_to_num()
    merged = (
        (weights.unsqueeze(-1) * accum)
        .sum(0)
        .transpose(0 if tnd else 1, 1 if tnd else 2)
        .to(torch.bfloat16)
    )
    torch.testing.assert_close(merged, out, rtol=0.008, atol=0.004)
    finite = torch.isfinite(lse)
    torch.testing.assert_close(
        torch.logsumexp(logs, dim=0)[finite], lse[finite], rtol=2e-5, atol=2e-5
    )
    assert torch.isneginf(logs[:, ~finite]).all()


def check_source(c, *, label="source"):
    spec, raw = c["spec"], c["raw"]
    source_spec = c.get("source_spec", spec)
    source_raw = dict(raw)
    if source_spec != spec:
        # B=1 fast FD has identical source matrices/partitions in either layout.
        # Use its frozen source component for LSE/partials, and independently
        # execute native TND for the public layout's output comparison.
        assert len(spec[2]) == 1
        for key in ("q", "qv", "qs"):
            source_raw[key] = raw[key].unsqueeze(0)
    oracle = ComponentCall(source_spec, source_raw, oracle=True, lse=False)
    expected = oracle().cpu().reshape(raw["qv"].shape)
    native = (NativeCall(spec, raw)() if source_spec != spec else oracle.native()).cpu()
    assert torch.equal(expected, native)
    record = metadata(c)
    result = invoke(c, record)
    actual = result[0].cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    lse_source = ComponentCall(
        source_spec,
        source_raw,
        oracle=True,
        lse=True,
        legacy_fd=bool(oracle.params.tndSplitCoreParams.FdBalanceFlag),
    )
    lse_source()
    if oracle.params.tndSplitCoreParams.FdBalanceFlag:
        torch.testing.assert_close(
            lse_source.workspace[lse_source.partial_offset : lse_source.workspace_size].cpu(),
            oracle.workspace[oracle.partial_offset : oracle.workspace_size].cpu(),
            rtol=0,
            atol=0,
        )
    source_lse = lse_source.lse.cpu()
    if "prefix" in c:
        source_lse = source_lse.reshape(actual.shape[:-1]).transpose(0, 1)
    elif oracle.fixture["mode"] & 2:
        source_lse = source_lse.reshape(actual.shape[:-1]).transpose(1, 2)
    else:
        source_lse = source_lse.reshape(result[1].shape)
    refs = references(raw, spec)
    valid = torch.isfinite(
        refs["lse_high"].transpose(0 if "prefix" in c else 1, 1 if "prefix" in c else 2)
    )
    assert torch.isfinite(source_lse[valid]).all()
    assert torch.isposinf(source_lse[~valid]).all()
    torch.testing.assert_close(result[1].cpu()[valid], source_lse[valid], rtol=0, atol=0)
    assert torch.isneginf(result[1].cpu()[~valid]).all()
    refs, math_report = check_math(c, result, source_equal=True)
    check_partials(c, result, oracle)
    root = Path(os.environ["MLA_VALIDATION_RESULT"])
    name = spec[0] + "-" + c["view"] + "-" + label
    (root / (name + ".json")).write_text(
        json.dumps(dict(spec=spec, source_elementwise=True, math=math_report), indent=2) + "\n"
    )
    torch.save(
        dict(
            actual=[t.cpu() for t in result],
            source=expected,
            native=native,
            source_lse=source_lse,
            refs=refs,
            metadata=record.cpu(),
            inputs={k: v.contiguous() for k, v in raw.items()},
        ),
        root / (name + ".pt"),
    )
    return result
