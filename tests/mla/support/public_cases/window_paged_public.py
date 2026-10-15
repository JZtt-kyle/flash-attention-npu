"""Shared helpers extracted from tests/mla/e2e/window_paged/test_flash_attn_npu_v3_mla_window_paged_public.py.

BSND PA window: v15, independent page/mask math and public/internal dispatch.
"""

from contextlib import nullcontext as nullcontext
import hashlib
import json
import os
from pathlib import Path
import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode as FakeTensorMode
from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.mla.support.mla_dense_build import build_probes
from tests.mla.support.mla_metadata_error_probe import forbid_host_readback
from tests.mla.support.mla_source_compare import assert_mla_source_close
from tests.mla.support.public_cases import paged_bsnd_production as production
from tests.mla.support.public_cases import paged_public as public_base
from tests.mla.support.components.paged_attention import paged_probe as paged_probe
from tests.mla.support.public_cases.bsnd_stride_production import storage_fault as storage_fault
from tests.mla.support import mla_window_paged_helpers as component
from tests.mla.support.mla_window_paged_reference import window_for
from tests.mla.support.mla_window_paged_cases import (
    REPRESENTATIVES as REPRESENTATIVES,
    ACCURACY_CASES as ACCURACY_CASES,
    COST_CASES as COST_CASES,
)
from tests.mla.support.mla_window_paged_cases import PERFORMANCE_CASES as PERFORMANCE_CASES


@pytest.fixture(scope="module", params=("fp16", "bf16"))
def window_paged_attention(tmp_path_factory, request):
    root = Path(os.environ.get("MLA_VALIDATION_RESULT", tmp_path_factory.mktemp("window-paged")))
    directory = root / "window-paged-probes" / request.param
    build_probes(directory, request.param, paged=True, paged_window=True)
    assert json.loads((directory / "build-manifest.json").read_text())["paged_window"]
    return component.load_attention(directory)


@pytest.fixture(params=(torch.float16, torch.bfloat16), ids=("fp16", "bf16"))
def dtype(request):
    return request.param


def make_case(probes, producer, spec, **kwargs):
    return component.make_case(probes, producer, spec, **kwargs)


def forward_args(case):
    return dict(
        production.forward_args(case),
        causal=case["config"].get("causal", False),
        window_size=case["config"].get("window", (2, 1)),
    )


def call(args, entry="public", *, lse=True):
    args = dict(args)
    window = args.pop("window_size", (2, 1))
    args.setdefault("causal", False)
    if entry == "public":
        assert args.pop("seqused_q", None) is None
        return public_base.public(dict(args, window_size=window), lse)
    if entry == "op":
        return public_base.invoke(entry, dict(args, window_left=window[0], window_right=window[1]))
    return public_base.invoke(
        entry, dict(args, window_size_left=window[0], window_size_right=window[1])
    )


def metadata(args):
    q, k, table = (args[n] for n in ("q", "k", "page_table"))
    lengths = args.get("cache_seqlens")
    if lengths is None or isinstance(lengths, int):
        lengths = torch.full(
            (q.shape[0],),
            table.shape[1] * k.shape[1] if lengths is None else lengths,
            dtype=torch.int32,
            device=q.device,
        )
    return api.get_scheduler_metadata(
        q.shape[0],
        q.shape[1],
        table.shape[1] * k.shape[1],
        q.shape[2],
        1,
        64,
        lengths,
        qkv_dtype=q.dtype,
        headdim_v=512,
        page_size=k.shape[1],
        causal=args.get("causal", False),
        window_size=args.get("window_size", (2, 1)),
        num_splits=1,
    )


def masked_rows(case):
    used = case["item"]["q"] if case["item"]["uq"] is None else case["item"]["uq"]
    kind, left, right = window_for(case)
    return torch.tensor(
        [
            not any(
                kind == 0
                or (
                    j <= i + k - q + (0 if kind == 1 else right)
                    and (kind == 1 or j >= i + k - q - left)
                )
                for j in range(k)
            )
            for q, k in zip(used, case["config"]["lengths"])
            for i in range(q)
        ],
        dtype=torch.bool,
    )


def compare(probes, case, results):
    actual = [component.valid_outputs(case, [t.cpu() for t in result[:2]]) for result in results]
    refs = component.references(case)
    source = None
    reports = []
    if production.source_available(dict(case, schedule=case["schedule"].schedule)):
        assert component.launch(probes, case, "oracle") == 0
        source = component.valid_outputs(
            case, [t.cpu() for t in case["outputs"]["oracle"]], source=True
        )
        masked = masked_rows(case)
        assert (source[0][masked] == 0).all() and source[1][:, masked].isposinf().all()
        source[1][:, masked] = -torch.inf
    for result in actual:
        if source is not None:
            reports.append(
                {
                    name: assert_mla_source_close(
                        result[i], source[i], refs[i], refs[i + 2], name=name
                    )
                    for i, name in enumerate(("out", "lse"))
                }
            )
        else:
            reports.append(dict(empty_math=component.check_math(result, refs)))
    for before, after in case["input_checks"]:
        torch.testing.assert_close(after.cpu(), before, rtol=0, atol=0, equal_nan=True)
    target = probes[1] / "window-paged-accuracy"
    target.mkdir(exist_ok=True)
    identity = hashlib.sha256(
        json.dumps([case["config"], case["scale"]], sort_keys=True).encode()
    ).hexdigest()[:20]
    torch.save(
        dict(
            inputs=case["host"],
            pages=case["table_host"],
            actual=actual,
            source=source,
            references=refs,
            spec=case["config"],
            scale=case["scale"],
            reports=reports,
        ),
        target / f"{identity}.pt",
    )
    with (target / "results.jsonl").open("a") as f:
        f.write(json.dumps(dict(identity=identity, spec=case["config"], reports=reports)) + "\n")
    return reports


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


def check_constant(result, lengths, value=3, used=None, window=(2, 1), causal=False, capacity=448):
    out, lse = result[:2]
    used = [out.shape[1]] * out.shape[0] if used is None else used
    kind, left, right = window_for(
        dict(
            config=dict(
                q=[out.shape[1]] * out.shape[0], capacity=capacity, window=window, causal=causal
            )
        )
    )
    for b, (q, k) in enumerate(zip(used, lengths)):
        count = torch.tensor(
            [
                sum(
                    kind == 0
                    or (
                        j <= i + k - q + (0 if kind == 1 else right)
                        and (kind == 1 or j >= i + k - q - left)
                    )
                    for j in range(k)
                )
                for i in range(q)
            ]
        )
        expected = torch.where(count > 0, value, 0).to(out.dtype)
        torch.testing.assert_close(
            out[b, :q].cpu(), expected[:, None, None].expand(q, out.shape[2], 512)
        )
        torch.testing.assert_close(
            lse[b, :, :q].cpu(), count.float().log()[None].expand(out.shape[2], q)
        )


TRAPS = (
    "v6",
    "v10",
    "v11",
    "window-kind",
    "window-left",
    "window-right",
    "stale-window",
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
