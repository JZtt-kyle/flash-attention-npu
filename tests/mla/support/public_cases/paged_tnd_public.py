"""Shared helpers extracted from tests/mla/e2e/paged_tnd/test_flash_attn_npu_v3_mla_paged_tnd_public.py.

TND read-only PA kvcache: public/raw/fake contract and source/math checks.
"""

from contextlib import nullcontext
import hashlib
import json
from pathlib import Path as Path
from unittest.mock import patch
import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.mla.support.mla_metadata_error_probe import forbid_host_readback
from tests.mla.support.public_cases.contract import _forward
from tests.mla.support.public_cases import paged_tnd_production as production
from tests.mla.support.public_cases.paged_tnd_production import (
    component,
    generate,
    make_case as make_case,
    tnd as tnd,
    paged_attention as paged_attention,
    paged_probe as paged_probe,
    paged_artifacts as paged_artifacts,
    paged_oracles as paged_oracles,
    storage_fault as storage_fault,
)

# Same page/length matrix as production, using full TND Q allocations.
# Keep native FD cases and fix the matrix before collecting any timings.
PERFORMANCE_CASES = [dict(c, uq=None) for c in production.PERFORMANCE_CASES]


def public(args, lse=True):
    args = dict(args)
    args["k_cache"], args["v_cache"] = args.pop("k"), args.pop("v")
    args.setdefault("num_splits", 1)
    return api.flash_attn_with_kvcache(**args, return_softmax_lse=lse)


def invoke(entry, args):
    entry = entry.removesuffix("_fake")
    if entry == "public":
        return public(args)
    if entry == "op":
        return torch.ops.flash_attn_npu_3._mla_paged_tnd_forward(**args)
    args = dict(args)
    args["seqused_k"] = args.pop("cache_seqlens", None)
    return _forward(entry, args)


@pytest.fixture(params=("public", "op", "eager", "cpp", "public_fake", "op_fake", "eager_fake"))
def entry(request):
    with FakeTensorMode() if request.param.endswith("fake") else nullcontext():
        yield request.param


@pytest.fixture(params=(torch.float16, torch.bfloat16), ids=("fp16", "bf16"))
def dtype(request):
    return request.param


def small_inputs(dtype=torch.float16, **kwargs):
    args = production.small_inputs(dtype, **kwargs)
    args["v"].fill_(3)
    return args


def check_constant(result, lengths, value=3, allocations=(5, 0, 2)):
    production.check_constant(result[:2], result[0].dtype, allocations, allocations, lengths, value)


def check_case(probes, case, entry="public", metadata=None):
    args = production.forward_args(case)
    assert args.pop("seqused_q") is None
    with (
        forbid_host_readback(),
        patch.object(torch.Tensor, "contiguous", side_effect=AssertionError("input copy")),
    ):
        if production.source_available(case):
            assert component.launch(probes, case, "oracle") == 0
        record = generate(args) if metadata is None else metadata
        implicit = invoke(entry, args)
        actual = invoke(entry, dict(args, scheduler_metadata=record))
    report = production.check_result(case, actual[:2])
    for a, b in zip(implicit[:2], actual[:2]):
        torch.testing.assert_close(a.cpu(), b.cpu(), rtol=0, atol=0)
    assert [x & 0xFFFFFFFF for x in record.cpu().tolist()] == case["expected"]
    for before, after in case["input_checks"]:
        torch.testing.assert_close(after.cpu(), before, rtol=0, atol=0, equal_nan=True)
    report.update(
        config=case["config"], entry=entry, scale=case["scale"], expected_metadata=case["expected"]
    )
    target = probes[1] / "tnd-public-accuracy"
    target.mkdir(exist_ok=True)
    digest = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()[:20]
    torch.save(
        dict(
            host=case["host"],
            pages=case["table_host"],
            actual=[t.cpu() for t in actual[:2]],
            source=[t.cpu() for t in case["outputs"]["oracle"]],
            report=report,
        ),
        target / (digest + ".pt"),
    )
    with (target / "results.jsonl").open("a") as f:
        f.write(json.dumps(report) + "\n")
    return report
