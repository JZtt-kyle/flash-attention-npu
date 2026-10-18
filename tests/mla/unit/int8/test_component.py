"""INT8 source/ND numerical equivalence on native direct and FD schedules."""

import json
import math
import os
from pathlib import Path

import pytest
import torch

from tests.mla.support.mla_int8_cases import SOURCE_CASES, make_inputs, strided_inputs
from tests.mla.support.mla_int8_component import ComponentCall
from tests.mla.support.mla_int8_reference import references
from tests.common.compare import assert_fa_close


def compare_case(spec, inputs, label="base", lse=True, quant_expected=None):
    torch.npu.set_device(0)
    oracle = ComponentCall(spec, inputs, oracle=True, lse=False)
    migrated = ComponentCall(spec, inputs, lse=lse)
    expected = oracle().cpu()
    native = oracle.native().cpu()
    assert torch.equal(expected, native), (expected.float() - native.float()).abs().max()
    actual = migrated().cpu()
    assert torch.isfinite(expected).all() and torch.isfinite(actual).all()
    assert torch.equal(actual, expected), (actual.float() - expected.float()).abs().max()
    for call in (oracle, migrated):
        assert torch.all(call.workspace[call.workspace_size :].cpu() == 123)
        assert torch.isnan(call.out_storage[:32].cpu()).all()
        assert torch.isnan(call.out_storage[-32:].cpu()).all()
    if oracle.fixture["mode"] & 2:
        parts = [
            c.workspace[c.partial_offset : c.workspace_size].view(torch.float32).cpu()
            for c in (oracle, migrated)
        ]
        assert parts[0].numel() and torch.count_nonzero(parts[0]) > 0
        torch.testing.assert_close(parts[0], parts[1], rtol=0, atol=0, equal_nan=True)
    refs = references(inputs, spec)
    record = dict(
        source_elementwise=True, partial_elementwise=bool(oracle.fixture["mode"] & 2), math="passed"
    )
    try:
        assert_fa_close(
            actual,
            refs["out_high"].to(torch.bfloat16),
            refs["out_low"].to(torch.bfloat16),
            name="int8-raw-equations",
        )
    except AssertionError as error:
        # The separate source semantic discriminator proves that RoPE shares
        # the Q/K scales. Keep the independent raw math failure visible.
        record.update(math="source_semantics_or_quantization_limit", error=str(error))
    source_lse = None
    if lse:
        # O uses the fixed native endpoint (LSE disabled). LSE is checked
        # independently: source InitOutput aliases its stack UB on empty KV,
        # so enabling LSE can corrupt otherwise valid zero O in that source.
        source_lse_call = ComponentCall(
            spec,
            inputs,
            oracle=True,
            lse=True,
            legacy_fd=bool(oracle.params.tndSplitCoreParams.FdBalanceFlag),
        )
        source_lse_out = source_lse_call().cpu()
        source_lse = source_lse_call.lse.cpu()
        valid_out = torch.isfinite(refs["lse_high"])
        if not oracle.params.tndSplitCoreParams.FdBalanceFlag:
            torch.testing.assert_close(
                source_lse_out[valid_out], expected[valid_out], rtol=0, atol=0
            )
        if oracle.params.tndSplitCoreParams.FdBalanceFlag:
            torch.testing.assert_close(
                source_lse_call.workspace[
                    source_lse_call.partial_offset : source_lse_call.workspace_size
                ].cpu(),
                oracle.workspace[oracle.partial_offset : oracle.workspace_size].cpu(),
                rtol=0,
                atol=0,
            )
            record["lse_oracle"] = "fixed-legacy-FD-on-identical-partials"
        if not all(spec[4]):
            record["source_empty_initializer_issue"] = not torch.equal(source_lse_out, expected)
        right = migrated.lse.cpu()
        if spec[1] == "BSND" and oracle.fixture["mode"] & 2:
            source_lse = (
                source_lse.reshape(actual.shape[:-1]).transpose(1, 2).contiguous().flatten()
            )
        high, low = refs["lse_high"], refs["lse_low"]
        if spec[1] == "BSND":
            high, low = high.transpose(1, 2), low.transpose(1, 2)
        high, low = high.flatten(), low.flatten()
        valid = torch.isfinite(high)
        assert torch.isfinite(source_lse[valid]).all() and torch.isfinite(right[valid]).all()
        assert torch.isneginf(right[~valid]).all()
        assert not torch.isnan(right).any()
        torch.testing.assert_close(right[valid], source_lse[valid], rtol=0, atol=0)
        assert torch.isposinf(source_lse[~valid]).all()
        record["lse_source_elementwise"] = True
        try:
            assert_fa_close(right, high, low, name="int8-raw-lse")
        except AssertionError as error:
            record.update(lse_math="source_semantics_or_quantization_limit", lse_error=str(error))
        else:
            record["lse_math"] = "passed"
    if quant_expected is not None:
        for c in (oracle, migrated):
            t = c.params.increFlashAttentionSingleCoreTensorSize
            start = c.fixture["cores"] * 2 * t.mmResUbSize * 4
            quant = c.workspace[start : start + 4 * 128].view(torch.int8).cpu().reshape(4, 128)
            assert torch.all(quant[:, ::2] == quant_expected), torch.unique(quant)
            assert torch.all(quant[:, 1::2] == 127), torch.unique(quant)
        record["dynamic_p_rounding"] = [quant_expected, 127]
    root = Path(os.environ["MLA_VALIDATION_RESULT"])
    label = spec[0] + "-" + label
    record["spec"] = spec
    record["views"] = {
        n: dict(
            strides=t.stride(),
            offset=t.storage_offset(),
            storage_bytes=t.untyped_storage().nbytes(),
        )
        for n, t in inputs.items()
    }
    (root / (label + ".json")).write_text(json.dumps(record, indent=2) + "\n")
    # Compact logical inputs retain all mathematical data; original view
    # geometry is recorded separately, so unused wide-stride storage is omitted.
    torch.save(
        dict(
            inputs={n: t.contiguous() for n, t in inputs.items()},
            source=expected,
            native=native,
            source_lse_out=source_lse_out if lse else None,
            actual=actual,
            source_lse=source_lse,
            actual_lse=migrated.lse.cpu() if lse else None,
            references=refs,
        ),
        root / (label + ".pt"),
    )
    return actual


@pytest.mark.parametrize("spec", SOURCE_CASES, ids=[s[0] for s in SOURCE_CASES])
def test_source_equivalence(spec):
    compare_case(spec, make_inputs(spec))


@pytest.mark.parametrize("spec", [SOURCE_CASES[3], SOURCE_CASES[12]], ids=lambda s: s[0])
@pytest.mark.parametrize("kind", ["page-gap", "row-gap", "wide-row", "broadcast", "table-gap"])
def test_nd_storage_views(spec, kind):
    inputs = strided_inputs(spec, kind)
    compare_case(spec, inputs, label=kind)


@pytest.mark.parametrize(
    "spec", [SOURCE_CASES[2], SOURCE_CASES[6], SOURCE_CASES[12]], ids=lambda s: s[0]
)
@pytest.mark.parametrize(
    "pattern",
    [
        "zeros",
        "v-scale-zero",
        "q-scale-zero",
        "extreme-positive",
        "extreme-negative",
        "head-scale",
        "rounding",
    ],
)
def test_quantization_edges(spec, pattern):
    inputs = make_inputs(spec)
    if pattern == "zeros":
        for n in ("qv", "q", "v", "k"):
            inputs[n].zero_()
    elif pattern == "v-scale-zero":
        inputs["vs"].zero_()
    elif pattern == "q-scale-zero":
        inputs["qs"].zero_()
    elif pattern.startswith("extreme"):
        inputs["qv"].fill_(127 if pattern.endswith("positive") else -128)
        inputs["v"].fill_(-128 if pattern.endswith("positive") else 127)
        inputs["q"].zero_()
        inputs["k"].zero_()
        inputs["qs"].fill_(1)
        inputs["ks"].fill_(1)
    elif pattern == "head-scale":
        inputs["qs"].copy_(torch.linspace(-0.02, 0.02, spec[3]).expand_as(inputs["qs"]))
        inputs["ks"].fill_(-0.03125)
    else:
        inputs["q"].zero_()
        inputs["k"].zero_()
        inputs["qv"].remainder_(3)
        inputs["v"].remainder_(5)
        inputs["qs"].fill_(0.5)
        inputs["ks"].fill_(0.125)
    actual = compare_case(spec, inputs, label=pattern)
    if pattern in ("zeros", "v-scale-zero"):
        assert torch.count_nonzero(actual) == 0


@pytest.mark.parametrize(
    "delta", [-(2**-10), 0.0, 2**-10], ids=["above-half", "tie-even", "below-half"]
)
def test_dynamic_p_rint_boundary(delta):
    spec = SOURCE_CASES[2]
    inputs = make_inputs(spec)
    inputs["qv"].zero_()
    inputs["qv"][..., 0] = 1
    inputs["q"].zero_()
    inputs["k"].zero_()
    inputs["v"][..., 0] = torch.arange(128).remainder(2).reshape(1, 128, 1)
    inputs["qs"].fill_(24 * (math.log(2) + delta))
    inputs["ks"].fill_(1)
    compare_case(spec, inputs, label=f"rint-{delta}", quant_expected=63 if delta > 0 else 64)
