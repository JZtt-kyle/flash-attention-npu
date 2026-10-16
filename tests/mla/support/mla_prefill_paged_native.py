"""Pinned FIA PA adapter. All required view packing is inside each call."""

import torch
from tests.mla.support.mla_ref import normalize_fa3_window


class PrefillPagedCall:
    def __init__(self, c):
        self.c = c
        q = c["t"]["q"]
        b, sq, h, cap, hk = c["shape"]
        o = c["options"]
        left, right = normalize_fa3_window(
            sq, cap, causal=o["causal"], window_size=o["window_size"]
        )
        kind = 1 if left < 0 and right == 0 else 2 if left >= 0 or right >= 0 else 0
        self.kwargs = dict(
            input_layout="BSND",
            num_query_heads=h,
            num_key_value_heads=hk,
            block_size=c["page"],
            actual_seq_kvlen=list(c["klens"]),
            softmax_scale=o["softmax_scale"],
            sparse_mode=3 if kind == 1 else 4 if kind == 2 else 0,
            return_softmax_lse=True,
            pre_tokens=2**31 - 1 if kind == 1 else cap - 1 if left < 0 else left,
            next_tokens=0 if kind == 1 else sq - 1 if right < 0 else right,
        )
        if kind:
            self.kwargs["atten_mask"] = torch.ones(
                (2048, 2048), dtype=torch.bool, device=q.device
            ).triu(1)

    def __call__(self):
        c = self.c
        q = c["t"]["q"].contiguous()
        # Native BnBsH holds heads within each page row. FA3 ND PA has the
        # same logical order, but arbitrary outer strides require packing.
        k = c["t"]["k"].contiguous().flatten(2)
        v = c["t"]["v"].contiguous().flatten(2)
        table = c["page_table"].contiguous()
        out = torch.empty((*q.shape[:-1], 128), dtype=q.dtype, device=q.device)
        lse = torch.empty(
            (q.shape[0], q.shape[2], q.shape[1], 1), dtype=torch.float32, device=q.device
        )
        result = torch.ops.npu.npu_fused_infer_attention_score_v2.out(
            q, k, v, block_table=table, **self.kwargs, out=[out, lse]
        )
        return result[0], result[1].squeeze(-1)

    def description(self):
        c = self.c
        return dict(
            page_size=c["page"],
            physical_pages=c["t"]["k"].shape[0],
            table_shape=list(c["page_table"].shape),
            table_stride=list(c["page_table"].stride()),
            tensors={
                n: dict(shape=list(t.shape), stride=list(t.stride()), offset=t.storage_offset())
                for n, t in c["t"].items()
            },
            lengths=c["klens"],
            sparse_mode=self.kwargs["sparse_mode"],
            layout="BSND/BnBsH",
            packing="inside every call",
            representation="combined 192/128",
            split="native automatic",
        )
