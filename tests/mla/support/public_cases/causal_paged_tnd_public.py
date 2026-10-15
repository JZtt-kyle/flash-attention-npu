"""Shared helpers extracted from tests/mla/e2e/causal_paged_tnd/test_flash_attn_npu_v3_mla_causal_paged_tnd_public.py.

TND PA causal: v11 prefixes/used, source math and complete public dispatch.
"""

from contextlib import nullcontext as nullcontext
import json
import os
from pathlib import Path
import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode as FakeTensorMode
from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.mla.support.mla_dense_build import build_probes
from tests.mla.support.mla_metadata_error_probe import forbid_host_readback
from tests.mla.support.mla_validation_policy import public_cases
from tests.mla.support.public_cases import paged_tnd_production as production
from tests.mla.support.public_cases import paged_tnd_public as public_base
from tests.mla.support.components.paged_attention import paged_probe as paged_probe
from tests.mla.support.public_cases.bsnd_stride_production import storage_fault as storage_fault
from tests.mla.support.public_cases.causal_paged_public import compare, masked_rows as masked_rows

component = production.component

REPRESENTATIVES = [
    production.tnd.config(112, q=(7, 0, 3), lengths=(3, 17, 225), uq=None, heads=8),
    production.tnd.config(528, q=(257, 3), lengths=(257, 513), uq=None, heads=1),
    production.tnd.config(16, q=(7, 0, 3), lengths=(3, 17, 33), uq=(5, 0, 1), repeat=True),
]

PERFORMANCE_CASES = public_cases(
    [
        *[dict(c, uq=None) for c in production.PERFORMANCE_CASES],
        *REPRESENTATIVES[:2],
        production.tnd.config(112, "independent", q=(7, 7), lengths=(3, 7), uq=None),
    ],
    group="paged-tnd",
)

ACCURACY_CASES = [s for s in PERFORMANCE_CASES if s not in REPRESENTATIVES] + [
    production.tnd.config(112, q=(513, 259), lengths=(257, 129), uq=(513, 257), heads=3),
    production.tnd.config(48, q=(7, 3), lengths=(3, 129), uq=(0, 0)),
]


@pytest.fixture(scope="module", params=("fp16", "bf16"))
def causal_paged_tnd_attention(tmp_path_factory, request):
    root = Path(
        os.environ.get("MLA_VALIDATION_RESULT", tmp_path_factory.mktemp("causal-paged-tnd"))
    )
    directory = root / "causal-paged-tnd-probes" / request.param
    build_probes(directory, request.param, paged=True, paged_causal=True)
    assert json.loads((directory / "build-manifest.json").read_text())["paged_causal"]
    return component.load_attention(directory)


@pytest.fixture(params=(torch.float16, torch.bfloat16), ids=("fp16", "bf16"))
def dtype(request):
    return request.param


def make_case(probes, producer, spec, **kwargs):
    return production.make_case(probes, producer, dict(spec, causal=True), **kwargs)


def forward_args(case):
    return dict(production.forward_args(case), causal=True)


def call(args, entry="public", *, lse=True):
    args = dict(args, causal=True)
    if entry == "public":
        assert args.pop("seqused_q", None) is None
        return public_base.public(args, lse)
    return public_base.invoke(entry, args)


def metadata(args):
    q, k, table = (args[n] for n in ("q", "k", "page_table"))
    lengths = args.get("cache_seqlens")
    if lengths is None or isinstance(lengths, int):
        lengths = torch.full(
            (table.shape[0],),
            table.shape[1] * k.shape[1] if lengths is None else lengths,
            dtype=torch.int32,
            device=q.device,
        )
    return api.get_scheduler_metadata(
        table.shape[0],
        args["max_seqlen_q"],
        table.shape[1] * k.shape[1],
        q.shape[1],
        1,
        64,
        lengths,
        qkv_dtype=q.dtype,
        headdim_v=512,
        page_size=k.shape[1],
        cu_seqlens_q=args["cu_seqlens_q"],
        causal=True,
        num_splits=1,
    )


def check_spec(probes, producer, spec, **kwargs):
    case = make_case(probes, producer, spec, **kwargs)
    args = forward_args(case)
    entries = ("op", "eager", "cpp") if spec["uq"] is not None else ("public", "op", "eager", "cpp")
    with forbid_host_readback():
        results = [call(args, e) for e in entries]
        if spec["uq"] is None:
            record = metadata(args)
            results += [call(dict(args, scheduler_metadata=record), e) for e in entries]
    if spec["uq"] is None:
        assert [x & 0xFFFFFFFF for x in record.cpu().tolist()] == case["expected"]
    compare(probes, case, results)
    return case


def check_constant(result, lengths, value=3, used=None, allocations=(7, 0, 2)):
    out, lse = result[:2]
    assert out.shape[0] == sum(allocations)
    used = allocations if used is None else used
    start = 0
    for allocation, q, k in zip(allocations, used, lengths):
        count = (torch.arange(q) + k - q + 1).clamp(0, k)
        expected = torch.where(count > 0, value, 0).to(out.dtype)
        torch.testing.assert_close(
            out[start : start + q].cpu(), expected[:, None, None].expand(q, out.shape[1], 512)
        )
        torch.testing.assert_close(
            lse[:, start : start + q].cpu(), count.float().log()[None].expand(out.shape[1], q)
        )
        start += allocation


TRAPS = (
    "v6",
    "v11-unmasked",
    "v10",
    "stale-prefix",
    "negative-prefix",
    "nonzero-prefix",
    "overflow-prefix",
    "decreasing-prefix",
    "zero-middle-page",
    "header",
    "endpoint",
    "stale-kv",
    "negative-kv",
    "overflow-kv",
    "negative-page",
    "overflow-page",
    "zero-query-page",
    "zero-used-page",
    "negative-used",
    "overflow-used",
)
