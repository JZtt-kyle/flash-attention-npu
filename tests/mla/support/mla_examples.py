"""Runnable public MLA examples: python -m tests.mla.support.mla_examples --device 0."""

import argparse

import torch
import torch_npu  # noqa: F401

from flash_attn_npu_3 import (
    flash_attn_func,
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
    get_scheduler_metadata,
)


@torch.no_grad()
def run_examples(dtype=torch.float16):
    def randn(*shape):
        return torch.randn(*shape, device="npu", dtype=dtype)

    # Shared 512-dimensional content K/V and separate 64-dimensional RoPE.
    q, k = randn(1, 2, 4, 64), randn(1, 16, 1, 64)
    qv, v = randn(1, 2, 4, 512), randn(1, 16, 1, 512)
    dense = flash_attn_func(q, k, v, qv=qv, return_attn_probs=True)
    assert dense[0].shape == (1, 2, 4, 512) and dense[1].shape == (1, 4, 2)

    # Independent 192-dimensional Q/K and 128-dimensional V, with TND lengths.
    q, k, v = randn(3, 4, 192), randn(9, 2, 192), randn(9, 2, 128)
    cu_q = torch.tensor([0, 2, 3], device="npu", dtype=torch.int32)
    cu_k = torch.tensor([0, 4, 9], device="npu", dtype=torch.int32)
    varlen = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_q,
        cu_k,
        2,
        5,
        causal=True,
        softcap=15,
        num_splits=0,
        pack_gqa=False,
        sm_margin=1,
        return_attn_probs=True,
    )
    assert varlen[0].shape == (3, 4, 128) and varlen[1].shape == (4, 3)

    # ND paged prefill cache. The table maps logical pages to physical pages.
    q, k, v = randn(2, 2, 4, 192), randn(4, 16, 2, 192), randn(4, 16, 2, 128)
    page_table = torch.tensor([[2, 0], [3, 1]], device="npu", dtype=torch.int32)
    lengths = torch.tensor([23, 17], device="npu", dtype=torch.int32)
    options = dict(causal=True, num_splits=0, pack_gqa=False, sm_margin=1)

    def metadata():
        return get_scheduler_metadata(
            2,
            2,
            32,
            4,
            2,
            192,
            lengths,
            qkv_dtype=dtype,
            headdim_v=128,
            page_size=16,
            has_softcap=True,
            **options,
        )

    def forward(record):
        return flash_attn_with_kvcache(
            q,
            k,
            v,
            cache_seqlens=lengths,
            page_table=page_table,
            scheduler_metadata=record,
            softcap=15,
            return_softmax_lse=True,
            **options,
        )

    # One record can serve multiple layers with the same bound geometry/lengths.
    record = metadata()
    paged = forward(record)
    repeated = forward(record)
    for actual, expected in zip(repeated, paged):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert paged[0].shape == (2, 2, 4, 128)
    assert paged[2].shape == (26, 2, 4, 2, 128)

    # Warm up before capture. Regenerate metadata inside the graph so replay
    # reads the current lengths. Shapes, strides, addresses and options stay fixed.
    forward(metadata())
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        replayed = forward(metadata())
    lengths.copy_(lengths.new_tensor([19, 0]))
    graph.replay()
    torch.npu.synchronize()
    expected = forward(metadata())
    for actual, reference in zip(replayed, expected):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    assert torch.count_nonzero(replayed[0][1]).item() == 0
    assert torch.isneginf(replayed[1][1]).all().item()
    return dense, varlen, paged, replayed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    torch.manual_seed(20261018)
    run_examples({"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype])
    torch.npu.synchronize()
    print(
        f"MLA dense, varlen, paged metadata reuse and dynamic graph examples passed ({args.dtype})"
    )


if __name__ == "__main__":
    main()
