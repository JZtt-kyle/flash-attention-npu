"""Keep missing-scale BSND and the unconnected TND INT8 path explicit."""

from contextlib import nullcontext

import pytest
import torch
from torch._dynamo.exc import Unsupported
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


@pytest.mark.parametrize("tnd,entry", [(False, "forward"), (True, "forward"), (True, "metadata")])
@pytest.mark.parametrize("fake", [False, True], ids=["real", "fake"])
def test_public_int8_stays_rejected(tnd, entry, fake):
    with FakeTensorMode() if fake else nullcontext():
        args = inputs(tnd)
        with pytest.raises(RuntimeError, match="INT8|dtype|FP16|BF16"):
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

    with pytest.raises(RuntimeError, match="MLA INT8: qv, ND page_table and all three"):
        forward(args["qv"])
    # Dynamo wraps a Python precondition failure instead of preserving its text.
    with pytest.raises(Unsupported, match="Observed exception") as error:
        torch.compile(forward, backend="eager", fullgraph=True)(args["qv"])
    assert "raised exception ExceptionVariable(<class 'RuntimeError'>)" in str(error.value)
