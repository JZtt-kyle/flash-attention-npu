"""Shared helpers extracted from tests/mla/e2e/tnd/test_flash_attn_npu_v3_mla_varlen_used.py.

Public TND used: allocation offsets, v5 validation, accuracy and lifetime.
"""

from contextlib import nullcontext
import itertools
from pathlib import Path as Path
import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.mla.support.mla_metadata_error_probe import forbid_host_readback as forbid_host_readback
from tests.mla.support.public_cases.contract import _forward
from tests.mla.support.components.used_attention import (
    used_attention as used_attention,
    used_probe as used_probe,
    CASES,
    EMPTY_CASES,
    check_math,
    effective,
    launch as launch,
    make_case,
    references,
    used_case,
    valid_outputs,
)
from tests.mla.support.components.used_metadata_consumer import (
    used_artifacts as used_artifacts,
    used_oracle as used_oracle,
    check_records as check_records,
)

# Fixed before sampling. Native host accepts Hq<=128; accuracy also checks
# larger heads. Include both optional used axes, unequal/zero lengths and tails.
PERFORMANCE_CASES = [
    dict(c, heads=min(c["heads"], 128)) for c in CASES[:14] if c["layout"] == 1
] + [
    used_case(1, [7, 3], [33, 513], [3, 1], None, heads=16),
    used_case(1, [7, 3], [33, 513], None, [17, 257], heads=16),
    used_case(1, [3, 0, 1], [33, 17, 65], [2, 0, 1], [0, 0, 0], heads=128),
    used_case(1, [3] * 10, [511] * 10, [3] * 10, [511] * 10, heads=128),
    used_case(1, [3, 0] * 21, [17, 513] * 21, [2, 0] * 21, [0, 513] * 21, heads=128),
]

ACCURACY_CASES = (
    PERFORMANCE_CASES
    + [c for c in EMPTY_CASES if c["layout"] == 1]
    + [c for c in CASES[14:] if c["layout"] == 1]
    + [
        used_case(1, [3, 2], [513, 33], [2, 1], [513, 17], heads=h)
        for h in (1, 15, 16, 17, 127, 129, 255, 256, 257)
    ]
)


def production_case(probes, producer, spec, **kwargs):
    spec = dict(spec, aic=producer["platform"].aic)
    spec["block_dim"] = spec["aic"] if sum(spec["q"]) and sum(spec["kv"]) else 0
    assert spec["layout"] == 1 and (spec["uq"] is not None or spec["uk"] is not None)
    return make_case(probes, producer, spec, **kwargs)


def forward_args(case):
    c = case["context"]
    return dict(
        **case["devices"]["migrated"],
        cu_seqlens_q=case["inputs"][0],
        cu_seqlens_k=case["inputs"][1],
        seqused_q=case["inputs"][2],
        seqused_k=case["inputs"][3],
        max_seqlen_q=c.query.maximum,
        max_seqlen_k=c.kv.maximum,
        softmax_scale=case["scale"],
    )


def invoke(entry, args):
    if entry.endswith("public"):
        return api.flash_attn_varlen_func(**args, return_attn_probs=True)
    if entry.endswith("op"):
        return torch.ops.flash_attn_npu_3._mla_tnd_forward(**args)
    return _forward(entry, args)


def check_result(case, result, *, source=False):
    item, heads = case["item"], case["context"].heads
    assert result[0].shape == (sum(item["q"]), heads, 512)
    assert result[1].shape == (heads, sum(item["q"]))
    assert result[0].dtype == case["host"]["v"].dtype and result[1].dtype == torch.float32
    host = [t.cpu() for t in result[:2]]
    actual = valid_outputs(case, host)
    errors = check_math(actual, references(case))
    if not sum(item["kv"]):
        assert (host[0] == 0).all() and host[1].isposinf().all()
    uq, uk = effective(item)
    if source and sum(uq) and sum(item["kv"]):
        expected = valid_outputs(case, [t.cpu() for t in case["outputs"]["oracle"]], source=True)
        nonempty = torch.tensor([k > 0 for q, k in zip(uq, uk) for _ in range(q)], dtype=torch.bool)
        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[1][:, nonempty], expected[1][:, nonempty], rtol=0, atol=0)
        assert expected[1][:, ~nonempty].isposinf().all()
    return errors


def inputs(dtype=torch.float16, qs=(7, 3), ks=(33, 17)):
    return dict(
        q=torch.zeros((sum(qs), 8, 64), dtype=dtype, device="npu"),
        qv=torch.zeros((sum(qs), 8, 512), dtype=dtype, device="npu"),
        k=torch.zeros((sum(ks), 1, 64), dtype=dtype, device="npu"),
        v=torch.zeros((sum(ks), 1, 512), dtype=dtype, device="npu"),
        cu_seqlens_q=torch.tensor(
            list(itertools.accumulate(qs, initial=0)), dtype=torch.int32, device="npu"
        ),
        cu_seqlens_k=torch.tensor(
            list(itertools.accumulate(ks, initial=0)), dtype=torch.int32, device="npu"
        ),
        seqused_q=torch.tensor([min(q, 2) for q in qs], dtype=torch.int32, device="npu"),
        seqused_k=torch.tensor([min(k, 9) for k in ks], dtype=torch.int32, device="npu"),
        max_seqlen_q=max(qs, default=0) + 3,
        max_seqlen_k=max(ks, default=0) + 7,
    )


@pytest.fixture(params=[torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def dtype(request):
    return request.param


@pytest.fixture(params=["public", "op", "eager", "cpp", "fake", "fake-public", "fake-op"])
def entry(request):
    with FakeTensorMode() if request.param.startswith("fake") else nullcontext():
        yield request.param
