"""Shared helpers extracted from tests/mla/e2e/paged_bsnd/test_flash_attn_npu_v3_mla_paged_bsnd_production.py.

Internal production BSND PA: real storage, v6 reuse, source/math and lifetime.
"""

import torch
from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.mla.support.components import paged_attention as component
from tests.mla.support.components.paged_attention import (
    paged_attention as paged_attention,
    paged_probe as paged_probe,
    paged_artifacts as paged_artifacts,
    paged_oracles as paged_oracles,
)
from tests.mla.support.public_cases.bsnd_stride_production import storage_fault as storage_fault

PERFORMANCE_CASES = [dict(c, aic=None) for c in component.PERFORMANCE_CASES]


def make_case(probes, producer, spec, **kwargs):
    spec = dict(spec, aic=None)
    if spec["capacity"] is None:
        p = spec["page"]
        spec["capacity"] = ((max(spec["lengths"], default=0) + p - 1) // p + 1) * p
    return component.make_case(probes, producer, spec, full_launch=True, **kwargs)


def forward_args(case):
    return dict(
        case["devices"]["migrated"],
        page_table=case["tables"]["migrated"],
        softmax_scale=case["scale"],
        cache_seqlens=case["inputs"][2],
        seqused_q=case["inputs"][1],
    )


def forward(args, metadata=None):
    return torch.ops.flash_attn_npu_3._mla_paged_bsnd_forward(**args, scheduler_metadata=metadata)


def generate(args):
    q, k, table = (args[n] for n in ("q", "k", "page_table"))
    return api.get_scheduler_metadata(
        q.shape[0],
        q.shape[1],
        table.shape[1] * k.shape[1],
        q.shape[2],
        1,
        64,
        args["cache_seqlens"],
        qkv_dtype=q.dtype,
        headdim_v=512,
        page_size=k.shape[1],
        num_splits=1,
    )


def source_available(case):
    return bool(
        case["context"].batch and case["context"].query.maximum and case["schedule"].tiling.sk
    )


def check_result(case, result, *, source=True):
    q = case["devices"]["migrated"]["q"]
    assert result[0].shape == (*q.shape[:-1], 512) and result[0].dtype == q.dtype
    assert result[1].shape == (q.shape[0], q.shape[2], q.shape[1])
    assert result[1].dtype == torch.float32
    actual = component.valid_outputs(case, [t.cpu() for t in result])
    used = case["item"]["q"] if case["item"]["uq"] is None else case["item"]["uq"]
    if source and source_available(case):
        expected = component.valid_outputs(
            case, [t.cpu() for t in case["outputs"]["oracle"]], source=True
        )
        empty = torch.tensor(
            [k == 0 for q, k in zip(used, case["config"]["lengths"]) for _ in range(q)],
            dtype=torch.bool,
        )
        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[1][:, ~empty], expected[1][:, ~empty], rtol=0, atol=0)
        assert expected[1][:, empty].isposinf().all()
    refs = component.references(case)
    errors = component.check_math(actual, refs)
    return dict(errors=errors, source_compared=source and source_available(case))


def small_inputs(dtype=torch.float16, *, sq=3, pool=7, width=4, used=False):
    args = {}
    for name, dim in (("q", 64), ("qv", 512), ("k", 64), ("v", 512)):
        b, s, h = (2, sq, 8) if name in ("q", "qv") else (pool, 112, 1)
        base = torch.zeros((b, 2 * s + 1, h, dim + 16), dtype=dtype, device="npu")
        args[name] = base[:, 1::2, :, :dim]
    table = torch.full((2, width + 3), -1, dtype=torch.int32, device="npu")
    args["page_table"] = table[:, 1 : 1 + width]
    if width and pool:
        args["page_table"].copy_(
            torch.arange(width, dtype=torch.int32, device="npu").remainder(pool)
        )
    args["cache_seqlens"] = torch.tensor(
        [min(113, width * 112), 0], dtype=torch.int32, device="npu"
    )
    if used:
        args["seqused_q"] = torch.tensor([min(2, sq), 0], dtype=torch.int32, device="npu")
    return args
