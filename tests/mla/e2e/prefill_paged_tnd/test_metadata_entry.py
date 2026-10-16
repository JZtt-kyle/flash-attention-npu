"""The C++ metadata binding and registered fake share the public v31 contract."""

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.mla.support.public_cases.prefill_paged_tnd import make_case, metadata, invoke, compare
from tests.mla.support.public_cases.contract import metadata_raw_args


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("splits,mask", [(0, "none"), (1, "causal"), (3, "window")])
def test_cpp_metadata(dtype, splits, mask):
    c = make_case(dtype, (3, 7, 4, 65, 2), splits, mask, True, [65, 31, 0], page=16, q=[3, 0, 7])
    storage = torch.full((6,), 123, dtype=torch.int32, device="npu")
    storage[1:5].copy_(c["lengths"]["cu_seqlens_q"])
    c["lengths"]["cu_seqlens_q"] = storage[1:5]
    args = dict(
        batch_size=3,
        max_seqlen_q=7,
        max_seqlen_k=80,
        num_heads_q=4,
        num_heads_kv=2,
        headdim=192,
        headdim_v=128,
        qkv_dtype=c["t"]["q"].dtype,
        cache_seqlens=c["lengths"]["seqused_k"],
        cu_seqlens_q=c["lengths"]["cu_seqlens_q"],
        page_size=16,
        num_splits=splits,
        causal=c["options"]["causal"],
        window_size_left=c["options"]["window_size"][0],
        window_size_right=c["options"]["window_size"][1],
    )
    args = metadata_raw_args(args)
    record = api.flash_attn_npu_3.get_scheduler_metadata(*args.values())
    torch.testing.assert_close(record, metadata(c), rtol=0, atol=0)
    compare(c, invoke(c, record), record)
    with FakeTensorMode() as mode:
        fake = {
            k: mode.from_tensor(v) if isinstance(v, torch.Tensor) else v for k, v in args.items()
        }
        result = api._get_scheduler_metadata_fake(**fake)
        assert result.shape == record.shape and result.dtype == record.dtype
