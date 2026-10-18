"""Run the pinned INT8 native package before migrating its numerical pipeline."""

import json
import os
from pathlib import Path

import pytest
import torch

from tests.mla.support.mla_int8_cases import NativeCall, SOURCE_CASES, make_inputs, case_identity
from tests.mla.support.mla_int8_native import native_vendor
from tests.mla.support.mla_int8_reference import references
from tests.common.compare import assert_fa_close


@pytest.fixture(scope="module", autouse=True)
def fixed_source():
    vendor = native_vendor()
    assert os.environ.get("ASCEND_CUSTOM_OPP_PATH") == str(vendor)
    torch.npu.set_device(0)
    assert "910B" in torch.npu.get_device_name(0)


@pytest.mark.parametrize("spec", SOURCE_CASES, ids=[s[0] for s in SOURCE_CASES])
def test_source_shape(spec):
    inputs = make_inputs(spec)
    native = NativeCall(spec, inputs)
    index = SOURCE_CASES.index(spec)
    print(f"MLA_STRIDE_NATIVE_BEGIN int8 {index}", flush=True)
    print(f"MLA_INT8_CASE_SIGNATURE {case_identity(spec)}", flush=True)
    result = native()
    torch.npu.synchronize()
    print(f"MLA_STRIDE_NATIVE_END int8 {index}", flush=True)
    value = result.cpu()
    assert value.dtype == torch.bfloat16 and value.shape == inputs["qv"].shape
    assert torch.isfinite(value).all()
    refs = references(inputs, spec)
    precision = dict(math="passed")
    try:
        assert_fa_close(
            value,
            refs["out_high"].to(torch.bfloat16),
            refs["out_low"].to(torch.bfloat16),
            name="native-int8",
        )
    except AssertionError as error:
        precision = dict(math="source_semantics_or_quantization_limit", error=str(error))
    # This is source characterization, not evidence of a migrated component.
    root = Path(os.environ["MLA_VALIDATION_RESULT"])
    torch.save(value, root / (spec[0] + ".pt"))
    (root / (spec[0] + ".json")).write_text(
        json.dumps(dict(spec=spec, source_finite=True, **precision)) + "\n"
    )


def test_source_rope_scale_order():
    """Distinguish combined-dot scaling from independent RoPE dequantization."""
    spec = ("rope-scale-order", "BSND", (1,), 1, (2,), 0)
    inputs = make_inputs(spec)
    inputs["qv"].zero_()
    inputs["q"].zero_()
    inputs["q"][..., 0] = 32
    inputs["k"].zero_()
    inputs["k"][0, 1, 0, 0] = 1
    inputs["v"].zero_()
    inputs["v"][0, 1] = 100
    inputs["vs"].fill_(0.01)
    rows = []
    for scale in (1.0, 0.01, 0.0):
        inputs["qs"].fill_(scale)
        inputs["ks"].fill_(scale)
        result = NativeCall(spec, inputs)().cpu()
        assert torch.isfinite(result).all()
        combined = torch.sigmoid(torch.tensor(32 * scale * scale / 24))
        independent = torch.sigmoid(torch.tensor(32 / 24))
        rows.append(
            dict(
                scale=scale,
                actual=float(result.flatten()[0]),
                combined=float(combined),
                independent=float(independent),
            )
        )
        # A coarse semantic discriminator, not a numerical acceptance threshold.
        assert abs(rows[-1]["actual"] - rows[-1]["combined"]) < 0.02
    root = Path(os.environ["MLA_VALIDATION_RESULT"])
    (root / "rope-scale-order.json").write_text(json.dumps(rows, indent=2) + "\n")
