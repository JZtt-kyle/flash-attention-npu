"""Small public E2E regression: real data, independent math, no component builds."""

from itertools import accumulate

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.common.compare import assert_fa_close
from tests.mla.regression_cases import BEHAVIORS, COMPILE_BEHAVIORS, daily_cases
from tests.mla.support.mla_ref import mla_reference_pair


def prepare(capability, dtype_name, scenario):
    layout, mask = BEHAVIORS[capability]
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
    paged = layout.startswith("paged")
    packed = layout in {"tnd", "mixed", "paged_tnd"}
    strided = scenario == "strided_reuse"
    page = 112 if strided else 16
    capacity = 2 * page if paged else 17
    qlens = (3, 7) if packed else (7, 7)
    klens = (capacity, capacity) if layout == "func" else (page + 1 if paged else 17, 2)
    if scenario == "empty":
        klens = (0, 0)
        if layout == "func":
            capacity = 0
    generator = torch.Generator().manual_seed(20261014)
    raw = {
        name: torch.randn(
            (2, 7 if name in {"q", "qv"} else capacity, 2 if name in {"q", "qv"} else 1, dim),
            generator=generator,
        ).to(dtype)
        for name, dim in (("q", 64), ("qv", 512), ("k", 64), ("v", 512))
    }

    def device(host):
        if not strided:
            return host.npu()
        shape = list(host.shape)
        shape[-2] *= 2
        base = torch.empty(shape, dtype=dtype, device="npu")
        view = base[..., ::2, :]
        view.copy_(host)
        return view

    pages = [[2, 0], [3, 1]]
    table_cpu = torch.tensor(pages, dtype=torch.int32)
    tensors = {}
    for name, host in raw.items():
        if name in {"q", "qv"} and packed:
            host = torch.cat([host[b, :n] for b, n in enumerate(qlens)])
        elif name in {"k", "v"} and layout == "tnd":
            host = host.flatten(0, 1)
        elif name in {"k", "v"} and paged:
            pool = torch.empty((4, page, 1, host.shape[-1]), dtype=dtype)
            for b in range(2):
                for i in range(2):
                    pool[pages[b][i]] = host[b, i * page : (i + 1) * page]
            host = pool
        tensors[name] = device(host)
    options = dict(
        softmax_scale=0.125,
        causal=mask == "causal",
        window_size=(2, 1) if mask == "window" else (-1, -1),
        num_splits=1,
    )
    prefix = torch.tensor(list(accumulate(qlens, initial=0)), dtype=torch.int32, device="npu")
    lengths = torch.tensor(klens, dtype=torch.int32, device="npu")
    return dict(
        layout=layout,
        raw=raw,
        tensors=tensors,
        options=options,
        qlens=qlens,
        klens=klens,
        capacity=capacity,
        page=page,
        prefix=prefix,
        lengths=lengths,
        table=table_cpu.npu(),
    )


def metadata(case):
    return api.get_scheduler_metadata(
        2,
        7,
        case["capacity"],
        2,
        1,
        64,
        case["lengths"],
        qkv_dtype=case["tensors"]["q"].dtype,
        headdim_v=512,
        cu_seqlens_q=case["prefix"] if case["layout"] in {"mixed", "paged_tnd"} else None,
        page_size=case["page"] if case["layout"].startswith("paged") else None,
        causal=case["options"]["causal"],
        window_size=case["options"]["window_size"],
        num_splits=1,
    )


def invoke(case, q=None, record=None):
    tensors = dict(case["tensors"])
    if q is not None:
        tensors["q"] = q
    if case["layout"] == "func":
        return api.flash_attn_func(**tensors, **case["options"], return_attn_probs=True)[:2]
    if case["layout"] == "tnd":
        prefix_k = torch.arange(3, dtype=torch.int32, device=tensors["q"].device) * case["capacity"]
        return api.flash_attn_varlen_func(
            **tensors,
            **case["options"],
            cu_seqlens_q=case["prefix"],
            cu_seqlens_k=prefix_k,
            max_seqlen_q=7,
            max_seqlen_k=case["capacity"],
            seqused_k=case["lengths"],
            return_attn_probs=True,
        )[:2]
    tensors["k_cache"], tensors["v_cache"] = tensors.pop("k"), tensors.pop("v")
    return api.flash_attn_with_kvcache(
        **tensors,
        **case["options"],
        cache_seqlens=case["lengths"],
        scheduler_metadata=record,
        cu_seqlens_q=case["prefix"] if case["layout"] in {"mixed", "paged_tnd"} else None,
        max_seqlen_q=7 if case["layout"] in {"mixed", "paged_tnd"} else None,
        page_table=case["table"] if case["layout"].startswith("paged") else None,
        return_softmax_lse=True,
    )[:2]


def compare(case, result):
    out, lse = [t.cpu() for t in result]
    offset = 0
    for b, (nq, nk) in enumerate(zip(case["qlens"], case["klens"])):
        if case["layout"] in {"tnd", "mixed", "paged_tnd"}:
            actual, actual_lse = (
                out[offset : offset + nq].unsqueeze(0),
                lse[:, offset : offset + nq].unsqueeze(0),
            )
        else:
            actual, actual_lse = out[b : b + 1], lse[b : b + 1]
        offset += nq
        if nk == 0:
            assert torch.count_nonzero(actual) == 0
            expected = float("inf") if case["layout"] == "func" else -float("inf")
            assert torch.all(actual_lse == expected)
            continue
        raw = {
            name: host[b : b + 1, : nq if name in {"q", "qv"} else nk]
            for name, host in case["raw"].items()
        }
        ref, lse_ref, pt, lse_pt = mla_reference_pair(
            **raw,
            softmax_scale=case["options"]["softmax_scale"],
            causal=case["options"]["causal"],
            window_size=case["options"]["window_size"],
        )
        assert_fa_close(actual, ref, pt, name="public O")
        assert_fa_close(actual_lse, lse_ref, lse_pt, name="public LSE")


@pytest.mark.parametrize(
    "capability,dtype,scenario", daily_cases(), ids=["-".join(c) for c in daily_cases()]
)
def test_public_behavior(capability, dtype, scenario):
    case = prepare(capability, dtype, scenario)
    compare(case, invoke(case))
    if scenario == "strided_reuse":
        record = metadata(case) if case["layout"] not in {"func", "tnd"} else None
        compare(case, invoke(case, record=record))
        # New content with identical shape/lengths must not return a cached output.
        case["raw"]["q"].neg_()
        case["tensors"]["q"].neg_()
        compare(case, invoke(case, record=record))


@pytest.mark.parametrize("capability", BEHAVIORS)
def test_public_fake(capability):
    with FakeTensorMode():
        case = prepare(capability, "fp16", "lengths")
        out, lse = invoke(case)
        q = case["tensors"]["q"]
        assert out.shape == (*q.shape[:-1], 512)
        assert lse.dtype == torch.float32


@pytest.mark.parametrize("capability", COMPILE_BEHAVIORS)
def test_public_compile(capability):
    case = prepare(capability, "bf16", "lengths")
    compiled = torch.compile(lambda q: invoke(case, q=q), backend="eager", fullgraph=True)
    compare(case, compiled(case["tensors"]["q"]))


@pytest.mark.parametrize("option", ["num_splits", "softcap"])
def test_unopened_options(option):
    case = prepare("dense", "fp16", "lengths")
    case["options"][option] = 0 if option == "num_splits" else 2
    with pytest.raises((ValueError, RuntimeError), match="not implemented"):
        invoke(case)
