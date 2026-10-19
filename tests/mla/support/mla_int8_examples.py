"""Run INT8 MLA: python3 -m tests.mla.support.mla_int8_examples --device 0."""

import argparse
import math

import torch
import torch_npu  # noqa: F401

from flash_attn_npu_3 import flash_attn_with_kvcache, get_scheduler_metadata


@torch.no_grad()
def run_example(layout="bsnd"):
    """Read ND pages with BSND/TND queries, reuse metadata, then replay a graph."""
    if layout not in ("bsnd", "tnd"):
        raise ValueError("layout must be bsnd or tnd")
    batch, max_query, heads, page_size = 3, 2, 4, 128
    # TND allocates complete query segments of lengths [2, 0, 1].
    prefix = (
        torch.tensor([0, 2, 2, 3], dtype=torch.int32, device="npu") if layout == "tnd" else None
    )
    query_shape = (3, heads) if layout == "tnd" else (batch, max_query, heads)
    q = torch.randn(*query_shape, 64, dtype=torch.bfloat16, device="npu")
    qv = torch.randint(-8, 9, (*query_shape, 512), dtype=torch.int8).to("npu")
    k = torch.randn(6, page_size, 1, 64, dtype=torch.bfloat16, device="npu")
    # One INT8 content cache supplies both K and V; no external NZ packing.
    v = torch.randint(-8, 9, (6, page_size, 1, 512), dtype=torch.int8).to("npu")
    table = torch.tensor([[4, 0], [5, 1], [3, 2]], dtype=torch.int32, device="npu")
    lengths = torch.tensor([129, 0, 65], dtype=torch.int32, device="npu")
    q_scale = (
        torch.linspace(1 / 64, 1 / 32, math.prod(query_shape), dtype=torch.float32)
        .reshape(query_shape)
        .to("npu")
    )
    k_scale = torch.tensor([1 / 32], dtype=torch.float32, device="npu")
    v_scale = torch.tensor([1 / 16], dtype=torch.float32, device="npu")
    options = dict(causal=True, num_splits=0)

    def metadata():
        # qkv_dtype describes INT8 content; RoPE and output remain BF16.
        return get_scheduler_metadata(
            batch,
            max_query,
            table.shape[1] * page_size,
            heads,
            1,
            64,
            lengths,
            qkv_dtype=torch.int8,
            headdim_v=512,
            page_size=page_size,
            cu_seqlens_q=prefix,
            **options,
        )

    def forward(record=None):
        return flash_attn_with_kvcache(
            q,
            k,
            v,
            qv=qv,
            cache_seqlens=lengths,
            page_table=table,
            cu_seqlens_q=prefix,
            max_seqlen_q=max_query if prefix is not None else None,
            mla_q_scale=q_scale,
            mla_k_scale=k_scale,
            mla_v_scale=v_scale,
            scheduler_metadata=record,
            return_softmax_lse=True,
            **options,
        )

    def equal(left, right):
        for actual, expected in zip(left, right):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    record = metadata()
    initial = forward(record)
    equal(initial, forward())
    # Records bind geometry/lengths, not scale values or user tensor addresses.
    # Updating V scale alone scales O and leaves attention probabilities intact.
    v_scale.mul_(2)
    scaled = forward(record)
    equal(scaled, forward())

    # Warm up, then regenerate metadata inside capture so new KV lengths are
    # read on replay. Shapes, strides, addresses and scalar options stay fixed.
    forward(metadata())
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        replayed = forward(metadata())
    lengths.copy_(lengths.new_tensor([65, 0, 0]))
    q_scale.mul_(0.5)
    k_scale.mul_(0.75)
    v_scale.mul_(1.5)
    graph.replay()
    torch.npu.synchronize()
    equal(replayed, forward(metadata()))
    graph.reset()
    return initial, scaled, replayed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--layout", choices=("bsnd", "tnd", "all"), default="all")
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    torch.manual_seed(20261019)
    for layout in ("bsnd", "tnd") if args.layout == "all" else (args.layout,):
        initial, scaled, replayed = run_example(layout)
        torch.npu.synchronize()
        print(
            f"INT8 MLA {layout.upper()}: O {tuple(initial[0].shape)}, "
            f"LSE {tuple(initial[1].shape)}; metadata reuse, scale update and graph replay passed"
        )


if __name__ == "__main__":
    main()
