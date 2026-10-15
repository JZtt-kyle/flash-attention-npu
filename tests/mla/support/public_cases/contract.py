"""Shared helpers extracted from tests/mla/e2e/dense/test_flash_attn_npu_v3_mla_contract.py.

MLA supported subset and rejections through eager, fake and raw C++ entries.
"""

from contextlib import nullcontext
import inspect
import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from flash_attn_npu_3 import flash_attn_npu_interface as api


@pytest.fixture(params=["eager", "fake", "cpp"])
def entry(request):
    with FakeTensorMode() if request.param == "fake" else nullcontext():
        yield request.param


@pytest.fixture(params=[torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
def dtype(request):
    return request.param


def _tensor(shape, dtype, strided=False):
    if not strided:
        return torch.zeros(shape, dtype=dtype, device="npu")
    storage = torch.zeros(
        (*(2 * size + 1 for size in shape[:-1]), shape[-1]), dtype=dtype, device="npu"
    )
    return storage[(*(slice(1, None, 2) for _ in shape[:-1]), slice(None))]


def _inputs(representation="compressed", layout="dense", dtype=torch.float16, strided=False):
    dq, dv, hkv = (64, 512, 1) if representation == "compressed" else (192, 128, 2)
    varlen_q = layout in ("tnd", "paged_tnd")
    q_shape = (4, 4, dq) if varlen_q else (2, 3, 4, dq)
    k_shape = (17, hkv, dq) if layout == "tnd" else (2, 17, hkv, dq)
    if layout.startswith("paged"):
        k_shape = (4, 16, hkv, dq)
    args = dict(
        q=_tensor(q_shape, dtype, strided),
        k=_tensor(k_shape, dtype, strided),
        v=_tensor((*k_shape[:-1], dv), dtype, strided),
        qv=_tensor((*q_shape[:-1], dv), dtype, strided) if representation == "compressed" else None,
        softmax_scale=0.23,
    )
    if varlen_q:
        args.update(
            cu_seqlens_q=torch.tensor([0, 3, 4], dtype=torch.int32, device="npu"), max_seqlen_q=3
        )
    if layout == "tnd":
        args.update(
            cu_seqlens_k=torch.tensor([0, 9, 17], dtype=torch.int32, device="npu"), max_seqlen_k=9
        )
    if layout.startswith("paged"):
        args["page_table"] = torch.tensor([[2, 0], [3, 1]], dtype=torch.int32, device="npu")
    return args


def _forward(entry, args):
    if entry == "cpp":
        bound = inspect.signature(api._flash_attn_forward_fake).bind(**args)
        bound.apply_defaults()
        return api.flash_attn_npu_3.fwd(*bound.arguments.values())
    return api._flash_attn_forward(**args)
