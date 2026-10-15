"""Shared helpers extracted from tests/mla/e2e/mixed/test_flash_attn_npu_v3_mla_mixed_used.py.

Internal mixed Q-used: independent Q allocation and physical BSND cache slots.
"""

import torch
from tests.mla.support.public_cases.contract import dtype as dtype
from tests.mla.support.public_cases.mixed_public import inputs as mixed_inputs
from tests.mla.support.components.used_attention import CASES, make_case, used_case
from tests.mla.support.public_cases.varlen_used import check_result as check_result

# Fixed before profiling: mixed T01, unequal/zero lengths, tails, long KV,
# more batches than cores, optional/full KV-used and a multi-block query.
PERFORMANCE_CASES = [dict(c) for c in CASES[:14] if c["layout"] == 2] + [
    used_case(2, [7, 3], [513] * 2, [3, 1], None, heads=16),
    used_case(2, [3, 0, 1], [33] * 3, [2, 0, 1], [0, 0, 0], heads=128),
    used_case(2, [260, 3, 1], [513] * 3, [257, 1, 0], [33, 513, 0], heads=1),
    used_case(2, [7, 3], [513] * 2, [7, 3], [513] * 2, heads=16),
]


def production_case(probes, producer, spec, **kwargs):
    spec = dict(spec, aic=producer["platform"].aic)
    spec["block_dim"] = spec["aic"] if sum(spec["q"]) and sum(spec["kv"]) else 0
    assert spec["layout"] == 2 and spec["uq"] is not None
    return make_case(probes, producer, spec, **kwargs)


def forward_args(case):
    c = case["context"]
    args = dict(case["devices"]["migrated"])
    for name in ("k", "v"):
        args[name] = args[name].view(c.batch, c.kv.maximum, 1, args[name].shape[-1])
    return dict(
        **args,
        cu_seqlens_q=case["inputs"][0],
        seqused_q=case["inputs"][2],
        seqused_k=case["inputs"][3],
        max_seqlen_q=c.query.maximum,
        softmax_scale=case["scale"],
    )


def inputs(dtype=torch.float16, qs=(7, 3), capacity=33):
    args = mixed_inputs(dtype, qs, [min(capacity, 9)] * len(qs), capacity, heads=8)
    args["seqused_k"] = args.pop("cache_seqlens")
    args["seqused_q"] = torch.tensor(
        [123, *[min(q, 2) for q in qs], 456], dtype=torch.int32, device="npu"
    )[1:-1]
    return args
