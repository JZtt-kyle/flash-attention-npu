"""Shared helpers extracted from tests/mla/e2e/paged_tnd/test_flash_attn_npu_v3_mla_paged_tnd_production.py.

Internal production TND PA: real storage, allocation prefixes, v6 reuse and lifetime.
"""

import torch
from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.mla.support.components import paged_attention as component
from tests.mla.support.components import tnd_paged_attention as tnd
from tests.mla.support.components.paged_attention import (
    paged_attention as paged_attention,
    paged_probe as paged_probe,
    paged_artifacts as paged_artifacts,
    paged_oracles as paged_oracles,
)
from tests.mla.support.public_cases.bsnd_stride_production import storage_fault as storage_fault

PERFORMANCE_CASES = [dict(c, aic=None) for c in tnd.PERFORMANCE_CASES]


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
        cu_seqlens_q=case["inputs"][0],
        max_seqlen_q=case["context"].query.maximum,
    )


def forward(args, metadata=None):
    return torch.ops.flash_attn_npu_3._mla_paged_tnd_forward(**args, scheduler_metadata=metadata)


def generate(args):
    q, k, table = (args[n] for n in ("q", "k", "page_table"))
    return api.get_scheduler_metadata(
        table.shape[0],
        args["max_seqlen_q"],
        table.shape[1] * k.shape[1],
        q.shape[1],
        1,
        64,
        args["cache_seqlens"],
        qkv_dtype=q.dtype,
        headdim_v=512,
        page_size=k.shape[1],
        cu_seqlens_q=args["cu_seqlens_q"],
        num_splits=1,
    )


def source_available(case):
    return bool(
        case["context"].batch
        and case["context"].query.maximum
        and case["schedule"].tiling.sk
        and sum(case["item"]["q"] if case["item"]["uq"] is None else case["item"]["uq"])
    )


def check_result(case, result, *, source=True):
    q = case["devices"]["migrated"]["q"]
    assert result[0].shape == (*q.shape[:-1], 512) and result[0].dtype == q.dtype
    assert result[1].shape == (q.shape[1], q.shape[0])
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


def small_inputs(dtype=torch.float16, *, sq=5, pool=7, width=4, used=False):
    args = {}
    allocations = (sq, 0, min(2, sq))
    for name, dim in (("q", 64), ("qv", 512)):
        base = torch.zeros((2 * sum(allocations) + 1, 8, dim + 16), dtype=dtype, device="npu")
        args[name] = base[1::2, :, :dim]
    for name, dim in (("k", 64), ("v", 512)):
        base = torch.zeros((pool, 225, 1, dim + 16), dtype=dtype, device="npu")
        args[name] = base[:, 1::2, :, :dim]
    table = torch.full((3, width + 3), -1, dtype=torch.int32, device="npu")
    args["page_table"] = table[:, 1 : 1 + width]
    if width and pool:
        args["page_table"].copy_(
            torch.arange(width, dtype=torch.int32, device="npu").remainder(pool)
        )
    args["cache_seqlens"] = torch.tensor(
        [min(113, width * 112), min(17, width * 112), 0], dtype=torch.int32, device="npu"
    )
    args["cu_seqlens_q"] = torch.tensor(
        [0, sq, sq, sum(allocations)], dtype=torch.int32, device="npu"
    )
    args["max_seqlen_q"] = sq
    if used:
        args["seqused_q"] = torch.tensor(
            [min(3, sq), 0, min(1, sq)], dtype=torch.int32, device="npu"
        )
    return args


def check_constant(result, dtype, allocations, used, kv, value):
    out, lse = (t.cpu() for t in result)
    assert out.shape == (sum(allocations), 8, 512) and lse.shape == (8, sum(allocations))
    pos = 0
    for allocation, length, n in zip(allocations, used, kv):
        torch.testing.assert_close(
            out[pos : pos + length], torch.full((length, 8, 512), value if n else 0, dtype=dtype)
        )
        torch.testing.assert_close(
            lse[:, pos : pos + length], torch.full((8, length), float(n)).log()
        )
        pos += allocation
