"""M18 components must not silently enable mixed INT8 public execution."""

from contextlib import nullcontext

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from flash_attn_npu_3 import flash_attn_npu_interface as api


def inputs(tnd):
    qshape = (1, 2) if tnd else (1, 1, 2)
    return dict(
        q=torch.empty((*qshape, 64), dtype=torch.bfloat16, device="npu"),
        qv=torch.empty((*qshape, 512), dtype=torch.int8, device="npu"),
        k_cache=torch.empty((1, 128, 1, 64), dtype=torch.bfloat16, device="npu"),
        v_cache=torch.empty((1, 128, 1, 512), dtype=torch.int8, device="npu"),
        cache_seqlens=torch.ones(1, dtype=torch.int32, device="npu"),
        page_table=torch.zeros((1, 1), dtype=torch.int32, device="npu"),
        cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32, device="npu") if tnd else None,
        max_seqlen_q=1 if tnd else None,
        num_splits=1,
    )


@pytest.mark.parametrize("tnd", [False, True], ids=["bsnd", "tnd"])
@pytest.mark.parametrize("entry", ["forward", "metadata"])
@pytest.mark.parametrize("fake", [False, True], ids=["real", "fake"])
def test_public_int8_stays_rejected(tnd, entry, fake):
    with FakeTensorMode() if fake else nullcontext():
        args = inputs(tnd)
        with pytest.raises(RuntimeError, match="dtype|FP16|BF16"):
            if entry == "forward":
                api.flash_attn_with_kvcache(**args)
            else:
                api.get_scheduler_metadata(
                    1,
                    1,
                    128,
                    2,
                    1,
                    64,
                    args["cache_seqlens"],
                    qkv_dtype=torch.int8,
                    headdim_v=512,
                    page_size=128,
                    cu_seqlens_q=args["cu_seqlens_q"],
                    num_splits=1,
                )


def test_compile_int8_stays_rejected():
    args = inputs(False)

    def forward(qv):
        return api.flash_attn_with_kvcache(**dict(args, qv=qv))

    with pytest.raises(RuntimeError, match="dtype|FP16|BF16"):
        torch.compile(forward, backend="eager", fullgraph=True)(args["qv"])
