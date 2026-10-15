"""Shared helpers extracted from tests/mla/e2e/bsnd/test_flash_attn_npu_v3_mla_bsnd_used.py.

Internal BSND Q-used production entry: source/math, validation and lifetime.
"""

import pytest
import torch
from tests.mla.support.components.used_attention import (
    CASES,
    check_math,
    effective,
    make_case,
    references,
    used_case,
    valid_outputs,
)

# Fixed before profiling: U01, unequal/zero lengths, M/S2 tails and long KV.
# Production launches all platform cores, independently of the device schedule.
PERFORMANCE_CASES = [dict(c) for c in CASES[:14] if c["layout"] == 0] + [
    used_case(q=[3] * 43, kv=[513] * 43, uq=[2, 0] * 21 + [1], uk=[0, 513] * 21 + [0], heads=128),
    used_case(uq=[3, 1], heads=128),
    used_case(uq=[3, 1], uk=[0, 0], heads=128),
]


def production_case(probes, producer, spec, **kwargs):
    spec = dict(spec, aic=producer["platform"].aic)
    spec["block_dim"] = spec["aic"] if sum(spec["q"]) and sum(spec["kv"]) else 0
    assert spec["layout"] == 0 and spec["uq"] is not None
    return make_case(probes, producer, spec, **kwargs)


def forward_args(case):
    c = case["context"]
    args = {
        name: tensor.view(
            c.batch,
            c.query.maximum if name in ("q", "qv") else c.kv.maximum,
            c.heads if name in ("q", "qv") else 1,
            tensor.shape[-1],
        )
        for name, tensor in case["devices"]["migrated"].items()
    }
    args.update(
        seqused_q=case["inputs"][2], seqused_k=case["inputs"][3], softmax_scale=case["scale"]
    )
    return args


def check_result(case, result, *, source=False):
    c = case["context"]
    assert result[0].shape == (c.batch, c.query.maximum, c.heads, 512)
    assert result[1].shape == (c.batch, c.heads, c.query.maximum)
    assert result[0].dtype == case["host"]["v"].dtype and result[1].dtype == torch.float32
    host = [t.cpu() for t in result[:2]]
    actual = valid_outputs(case, [host[0].reshape(-1, c.heads, 512), host[1]])
    errors = check_math(actual, references(case))
    if not sum(case["item"]["kv"]):
        assert (host[0] == 0).all() and host[1].isposinf().all()
    uq, uk = effective(case["item"])
    if source and sum(uq) and sum(case["item"]["kv"]):
        expected = valid_outputs(case, [t.cpu() for t in case["outputs"]["oracle"]], source=True)
        nonempty = torch.tensor([k > 0 for q, k in zip(uq, uk) for _ in range(q)], dtype=torch.bool)
        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[1][:, nonempty], expected[1][:, nonempty], rtol=0, atol=0)
        assert expected[1][:, ~nonempty].isposinf().all()
    return errors


def inputs(dtype=torch.float16, batch=2, sq=7, sk=33):
    return dict(
        q=torch.zeros((batch, sq, 8, 64), dtype=dtype, device="npu"),
        qv=torch.zeros((batch, sq, 8, 512), dtype=dtype, device="npu"),
        k=torch.zeros((batch, sk, 1, 64), dtype=dtype, device="npu"),
        v=torch.zeros((batch, sk, 1, 512), dtype=dtype, device="npu"),
        seqused_q=torch.full((batch,), min(sq, 3), dtype=torch.int32, device="npu"),
        seqused_k=torch.full((batch,), min(sk, 17), dtype=torch.int32, device="npu"),
    )


@pytest.fixture(params=[torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def dtype(request):
    return request.param
