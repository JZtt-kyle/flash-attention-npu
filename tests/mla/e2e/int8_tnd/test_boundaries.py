"""Independent public host/fake rejections and device value/record checks."""

from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.mla.support.public_cases.int8_tnd import make_case, kwargs, metadata
from tests.mla.support.mla_trap_process import run_isolated
from tests.mla.support.mla_validation_build import sha256


def test_cold_mask_lifetime():
    root = Path(os.environ["MLA_VALIDATION_RESULT"])
    evidence = root / "cold-mask.json"
    result = run_isolated(
        [sys.executable, "-m", "tests.mla.support.mla_int8_tnd_mask", str(evidence)],
        root / "cold-mask.log",
    )
    assert result["returncode"] == 0 and not result["timed_out"], result
    proof = json.loads(evidence.read_text())
    extension = Path(api.flash_attn_npu_3.__file__)
    assert proof == dict(
        cold_graphs=2, thread_exit=True, extension=str(extension), sha256=sha256(extension)
    )


@pytest.mark.parametrize(
    "fault",
    [
        "missing-scale",
        "rope-dtype",
        "content-dtype",
        "scale-dtype",
        "scale-shape",
        "key-scale-shape",
        "page-size",
        "heads",
        "query-limit",
        "noncausal",
        "split",
        "softcap",
        "window",
        "pack",
        "margin",
        "descale",
        "append",
        "rotary",
        "grad",
        "metadata-size",
        "length-dtype",
        "q-last-stride",
        "missing-prefix",
        "missing-max",
        "prefix-dtype",
        "prefix-shape",
        "prefix-stride",
        "bsnd-prefix",
    ],
)
@pytest.mark.parametrize("fake", [False, True])
def test_rejections(fault, fake):
    c = make_case()
    args = kwargs(c)
    if fault == "missing-scale":
        args["mla_q_scale"] = None
    elif fault == "rope-dtype":
        args["q"] = args["q"].half()
    elif fault == "content-dtype":
        args["qv"] = args["qv"].bfloat16()
    elif fault == "scale-dtype":
        args["mla_q_scale"] = args["mla_q_scale"].bfloat16()
    elif fault == "scale-shape":
        args["mla_q_scale"] = args["mla_q_scale"][..., :1]
    elif fault == "key-scale-shape":
        args["mla_k_scale"] = args["mla_k_scale"].expand(2)
    elif fault == "page-size":
        args["k_cache"] = args["k_cache"][:, :64]
        args["v_cache"] = args["v_cache"][:, :64]
    elif fault == "heads":
        for key in ("q", "qv"):
            args[key] = args[key][:, :3]
        args["mla_q_scale"] = args["mla_q_scale"][:, :3]
    elif fault == "query-limit":
        args["max_seqlen_q"] = 33
    elif fault == "noncausal":
        args["causal"] = False
    elif fault == "split":
        args["num_splits"] = 2
    elif fault == "softcap":
        args["softcap"] = 2
    elif fault == "window":
        args["window_size"] = (64, 0)
    elif fault == "pack":
        args["pack_gqa"] = False
    elif fault == "margin":
        args["sm_margin"] = 1
    elif fault == "descale":
        args["q_descale"] = args["mla_q_scale"]
    elif fault == "append":
        args["k"] = args["k_cache"]
    elif fault == "rotary":
        args["rotary_cos"] = args["q"]
    elif fault == "grad":
        args["q"].requires_grad_(True)
    elif fault == "metadata-size":
        args["scheduler_metadata"] = metadata(c)[:-1]
    elif fault == "length-dtype":
        args["cache_seqlens"] = args["cache_seqlens"].long()
    elif fault == "q-last-stride":
        args["q"] = args["q"][..., ::2]
    elif fault == "missing-prefix":
        args["cu_seqlens_q"] = None
    elif fault == "missing-max":
        args["max_seqlen_q"] = None
    elif fault == "prefix-dtype":
        args["cu_seqlens_q"] = c["prefix"].long()
    elif fault == "prefix-shape":
        args["cu_seqlens_q"] = c["prefix"][:-1]
    elif fault == "prefix-stride":
        args["cu_seqlens_q"] = torch.zeros(8, dtype=torch.int32, device="npu")[::2]
    elif fault == "bsnd-prefix":
        for key in ("q", "qv", "mla_q_scale"):
            args[key] = args[key].unsqueeze(0)
    with FakeTensorMode() if fake else nullcontext() as mode:
        if fake:
            args = {
                key: mode.from_tensor(t) if isinstance(t, torch.Tensor) else t
                for key, t in args.items()
            }
        with pytest.raises(RuntimeError, match="MLA"):
            api.flash_attn_with_kvcache(**args)


@pytest.mark.parametrize(
    "fault",
    [
        "length-negative",
        "length-capacity",
        "page-negative",
        "page-pool",
        "page-late",
        "page-empty-query",
        "qs",
        "ks",
        "vs",
        "metadata-version",
        "metadata-tiling",
        "metadata-tail",
        "metadata-length",
        "graph-stale",
        "prefix-start",
        "prefix-negative",
        "prefix-order",
        "prefix-tail",
        "prefix-max",
        "metadata-prefix",
        "metadata-total",
        "bsnd-record",
        "graph-prefix",
    ],
)
def test_device_trap(fault):
    root = Path(os.environ["MLA_VALIDATION_RESULT"])
    evidence = root / (fault + ".json")
    result = run_isolated(
        [sys.executable, "-m", "tests.mla.support.mla_int8_tnd_trap", fault, str(evidence)],
        root / (fault + ".log"),
    )
    assert result["returncode"] == 0 and not result["timed_out"], result
    proof = json.loads(evidence.read_text())
    assert proof["fault"] == fault and proof["rejected"]
