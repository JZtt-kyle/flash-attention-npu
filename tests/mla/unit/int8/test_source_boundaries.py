"""Pinned host boundary characterization, independent of the ND adapter."""

import json
import os
from pathlib import Path

import pytest
import torch

from tests.mla.support.mla_int8_cases import NativeCall, make_inputs
from tests.mla.support.mla_int8_native import native_vendor


@pytest.fixture(scope="module", autouse=True)
def source_environment():
    assert os.environ.get("ASCEND_CUSTOM_OPP_PATH") == str(native_vendor())
    torch.npu.set_device(0)


@pytest.mark.parametrize(
    "fault",
    [
        "heads",
        "bsnd-q33",
        "tnd-q33",
        "decode-causal",
        "mtp-no-mask",
        "missing-mask",
        "nd-cache",
        "q-scale-shape",
        "q-scale-dtype",
        "k-scale-shape",
        "v-scale-dtype",
        "missing-scale",
        "quant-mode",
    ],
)
def test_source_rejection(fault):
    spec = ("reject-" + fault, "BSND", (1,), 8, (129,), 0)
    options, nz = {}, True
    if fault == "heads":
        spec = (*spec[:3], 3, *spec[4:])
    elif fault in ("bsnd-q33", "tnd-q33"):
        spec = (spec[0], "TND" if fault.startswith("tnd") else "BSND", (33,), 8, (129,), 3)
    elif fault == "decode-causal":
        spec = (*spec[:-1], 3)
    elif fault in ("mtp-no-mask", "missing-mask"):
        spec = (spec[0], "BSND", (2,), 8, (129,), 0 if fault == "mtp-no-mask" else 3)
        options["atten_mask"] = None
    elif fault == "nd-cache":
        nz = False
    elif fault == "missing-scale":
        options["dequant_scale_query"] = None
    elif fault == "quant-mode":
        options["query_quant_mode"] = 0
    inputs = make_inputs(spec)
    if fault == "q-scale-shape":
        inputs["qs"] = inputs["qs"][..., :1]
    elif fault == "q-scale-dtype":
        inputs["qs"] = inputs["qs"].half()
    elif fault == "k-scale-shape":
        inputs["ks"] = inputs["ks"].repeat(2)
    elif fault == "v-scale-dtype":
        inputs["vs"] = inputs["vs"].half()
    call = NativeCall(spec, inputs, nz=nz, overrides=options)
    with pytest.raises(RuntimeError) as captured:
        call()
        torch.npu.synchronize()
    error = str(captured.value)
    assert "kernel not found" not in error.lower()
    root = Path(os.environ["MLA_VALIDATION_RESULT"])
    (root / (spec[0] + ".json")).write_text(
        json.dumps(dict(spec=spec, source="rejected", error=error)) + "\n"
    )
