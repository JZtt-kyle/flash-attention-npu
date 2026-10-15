"""Shared helpers extracted from tests/mla/e2e/mixed/test_flash_attn_npu_v3_mla_mixed_stride_production.py.

Production mixed strides: public v4 reuse and internal Q-used v5.
"""

from tests.mla.support.components.tnd_stride import (
    PERFORMANCE_CASES as COMPONENT_CASES,
    launch as launch,
    make_case,
    spec,
)
from tests.mla.support.public_cases.varlen_used import check_result as check_result

# Fix matrices before sampling. Public kvcache uses all allocated Q tokens;
# Q-used remains internal. Native FIA limits Hq to 128, unlike the component.
PERFORMANCE_CASES = [dict(c, aic=None) for c in COMPONENT_CASES if c["item"]["layout"] == 2]

PUBLIC_CASES = [
    dict(c, version=4, item=dict(c["item"], uq=None, heads=min(c["item"]["heads"], 128)))
    for c in PERFORMANCE_CASES
] + [
    spec(4, 2, "contiguous", uk=(17, 0, 33)),
    spec(4, 2, "matrix_65536", uk=(17, 0, 33), heads=8),
    spec(4, 2, "head_65536", uk=(17, 0, 33), heads=8),
    spec(4, 2, "overlap", uk=(17, 0, 33), heads=8),
    spec(4, 2, uk=(0, 0, 0), heads=128),
]


def production_case(probes, producer, config, **kwargs):
    assert config["item"]["layout"] == 2
    if config["version"] == 5 and config["item"]["uq"] is None:
        config = dict(config, item=dict(config["item"], uq=config["item"]["q"]))
    return make_case(probes, producer, config, full_launch=True, **kwargs)


def forward_args(case):
    return dict(
        **case["devices"]["migrated"],
        cu_seqlens_q=case["inputs"][0],
        seqused_q=case["inputs"][2],
        seqused_k=case["inputs"][3],
        max_seqlen_q=case["context"].query.maximum,
        softmax_scale=case["scale"],
    )


def public_args(args):
    args = dict(args)
    assert args.pop("seqused_q", None) is None
    args["cache_seqlens"] = args.pop("seqused_k", None)
    return args
