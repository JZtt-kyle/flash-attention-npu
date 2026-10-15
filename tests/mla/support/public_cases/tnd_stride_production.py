"""Shared helpers extracted from tests/mla/e2e/tnd/test_flash_attn_npu_v3_mla_tnd_stride_production.py.

Production TND Q/KV strides: public v3/v5, storage, updates and lifetime.
"""

from contextlib import nullcontext as nullcontext
import itertools
from pathlib import Path as Path
from unittest.mock import patch
import torch
from torch._subclasses.fake_tensor import FakeTensorMode as FakeTensorMode
from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.mla.support.mla_metadata_error_probe import forbid_host_readback
from tests.mla.support.components.bsnd_stride import NAMES, strides_for
from tests.mla.support.public_cases.bsnd_stride_production import storage_fault as storage_fault
from tests.mla.support.public_cases.contract import _forward
from tests.mla.support.components.tnd_stride import (
    BOUNDARY_CASES,
    EMPTY_CASES,
    LAYOUT_CASES,
    PERFORMANCE_CASES as COMPONENT_CASES,
    WIDE_CASES as WIDE_CASES,
    launch,
    make_case,
    spec,
    stride_attention as stride_attention,
    stride_producer as stride_producer,
)
from tests.mla.support.components.used_attention import effective
from tests.mla.support.public_cases.varlen_used import check_result, forward_args

# Fix both matrices before sampling. Public native FIA accepts Hq<=128;
# larger heads and original geometry remain in the component/accuracy matrix.
PERFORMANCE_CASES = [dict(c, aic=None) for c in COMPONENT_CASES if c["item"]["layout"] == 1]

PUBLIC_CASES = [
    dict(c, item=dict(c["item"], heads=min(c["item"]["heads"], 128))) for c in PERFORMANCE_CASES
] + [
    spec(3, profile="broadcast", heads=8),
    spec(5, profile="ntd", uq=(2, 0, 1)),
    spec(5, uk=(7, 0, 17)),
    spec(5, uq=(3, 0, 1), uk=(17, 0, 33)),
    spec(5, uq=(2, 0, 1), uk=(0, 0, 0), heads=128),
]

ACCURACY_CASES = (
    [c for c in LAYOUT_CASES + BOUNDARY_CASES + EMPTY_CASES if c["item"]["layout"] == 1]
    + PERFORMANCE_CASES
    + PUBLIC_CASES
)


def production_case(probes, producer, config, **kwargs):
    assert config["item"]["layout"] == 1
    return make_case(probes, producer, config, full_launch=True, **kwargs)


def invoke(args, entry="public"):
    if entry == "public":
        return api.flash_attn_varlen_func(**args, num_splits=1, return_attn_probs=True)
    if entry == "op":
        return torch.ops.flash_attn_npu_3._mla_tnd_forward(**args)
    return _forward(entry, args)


def check_production(probes, case, entry="public"):
    with forbid_host_readback():
        if sum(effective(case["item"])[0]) and sum(case["item"]["kv"]):
            assert launch(probes, case, "oracle") == 0
        with patch.object(torch.Tensor, "contiguous", side_effect=AssertionError("input copy")):
            results = [invoke(forward_args(case), entry) for _ in range(2)]
    for result in results:
        errors = check_result(case, result, source=True)
    for before, after in case["input_checks"]:
        torch.testing.assert_close(after.cpu(), before, rtol=0, atol=0, equal_nan=True)
    return errors


def inputs(dtype, profile="independent", qs=(7, 3), ks=(33, 17), used=False):
    args = {}
    for i, name in enumerate(NAMES):
        shape = (1, sum(qs if i < 2 else ks), 8 if i < 2 else 1, 64 if i % 2 == 0 else 512)
        strides = strides_for(shape, "bnsd" if profile == "ntd" else profile, name)
        span = 0 if shape[1] == 0 else 1 + sum((n - 1) * s for n, s in zip(shape, strides))
        base = torch.zeros(span + 32, dtype=dtype, device="npu")
        args[name] = base.as_strided(shape, strides, i + 3)[0]
    args.update(
        cu_seqlens_q=torch.tensor(
            list(itertools.accumulate(qs, initial=0)), dtype=torch.int32, device="npu"
        ),
        cu_seqlens_k=torch.tensor(
            list(itertools.accumulate(ks, initial=0)), dtype=torch.int32, device="npu"
        ),
        max_seqlen_q=max(qs, default=0),
        max_seqlen_k=max(ks, default=0),
    )
    if used:
        args.update(
            seqused_q=torch.tensor(qs, dtype=torch.int32, device="npu"),
            seqused_k=torch.tensor(ks, dtype=torch.int32, device="npu"),
        )
    return args
