"""Raw INT8 MLA fixtures shared by source and migrated component checks."""

import itertools
import hashlib
import json
from pathlib import Path
import time

import torch


# Named tuples retain corners that depend on more than two protocol options.
# (name, layout, query lengths, heads, KV lengths, sparse mode)
SOURCE_CASES = (
    ("decode-min", "BSND", (1,), 1, (1,), 0),
    ("decode-h2-tail", "BSND", (1,), 2, (127,), 0),
    ("decode-h4-page", "BSND", (1,), 4, (128,), 0),
    ("decode-h8-page-tail", "BSND", (1,), 8, (129,), 0),
    ("decode-h16", "BSND", (1,), 16, (512,), 0),
    ("decode-h32", "BSND", (1,), 32, (512,), 0),
    ("decode-h64-fd", "BSND", (1,), 64, (4097,), 0),
    ("decode-h128", "BSND", (1,), 128, (129,), 0),
    ("bsnd-two-query", "BSND", (2, 2), 8, (129, 127), 3),
    ("bsnd-four-query", "BSND", (4,), 32, (512,), 3),
    ("bsnd-sixteen-query", "BSND", (16,), 128, (512,), 3),
    ("tnd-single-query", "TND", (1, 1), 16, (127, 129), 0),
    ("tnd-ragged", "TND", (2, 4, 1), 8, (129, 512, 127), 3),
    ("tnd-sixteen-query", "TND", (16, 2), 64, (4097, 129), 3),
    ("bsnd-q32", "BSND", (32,), 128, (512,), 3),
    ("tnd-q32", "TND", (32,), 8, (512,), 3),
    ("bsnd-fd-balance", "BSND", (4,), 8, (4097,), 3),
    ("decode-empty-kv", "BSND", (1,), 4, (0,), 0),
    ("tnd-empty-q", "TND", (0, 2), 8, (127, 129), 3),
    ("bsnd-causal-empty-rows", "BSND", (4,), 8, (2,), 3),
    ("tnd-empty-kv", "TND", (2, 1), 8, (0, 129), 3),
    ("bsnd-mtp-fd", "BSND", (2,), 128, (4097,), 3),
)


def case_identity(spec):
    return hashlib.sha256(json.dumps(spec).encode() + Path(__file__).read_bytes()).hexdigest()


def pack_nz(value):
    """Source physical [Np,H,D/C0,P,C0], independently checked by coordinates."""
    if value.ndim != 4 or value.shape[1] != 128 or value.shape[2] != 1:
        raise ValueError("Expected ND pages [Np,128,1,D]")
    if value.dtype not in (torch.int8, torch.bfloat16):
        raise ValueError("Expected INT8 content or BF16 RoPE")
    c0 = 32 // value.element_size()
    npages, page, heads, dim = value.shape
    if dim % c0:
        raise ValueError("NZ dimension must be aligned to C0")
    return value.reshape(npages, page, heads, dim // c0, c0).permute(0, 2, 3, 1, 4).contiguous()


def unpack_nz(value):
    if value.ndim != 5 or value.shape[1] != 1 or value.shape[3] != 128:
        raise ValueError("Expected source NZ pages [Np,1,D/C0,128,C0]")
    return value.permute(0, 3, 1, 2, 4).flatten(3).contiguous()


def make_inputs(spec):
    _, layout, qlens, heads, klens, _ = spec
    batch, sq = len(qlens), max(qlens)
    capacity = max(1, (max(klens) + 127) // 128)
    pages = batch * capacity
    generator = torch.Generator().manual_seed(18001)
    qshape = (sum(qlens), heads) if layout == "TND" else (batch, sq, heads)
    # All references start from these original integers, scales and BF16s.
    return dict(
        qv=torch.randint(-127, 128, (*qshape, 512), dtype=torch.int8, generator=generator),
        q=torch.randn(*qshape, 64, generator=generator).to(torch.bfloat16),
        v=torch.randint(-127, 128, (pages, 128, 1, 512), dtype=torch.int8, generator=generator),
        k=torch.randn(pages, 128, 1, 64, generator=generator).to(torch.bfloat16),
        qs=torch.rand(*qshape, generator=generator) * 0.015 + 0.005,
        ks=torch.tensor([0.0125]),
        vs=torch.tensor([0.0175]),
        table=torch.arange(pages - 1, -1, -1, dtype=torch.int32).reshape(batch, capacity),
    )


class NativeCall:
    """Pack source-only cache once; never called from the production path."""

    def __init__(self, spec, inputs, *, device="npu:0", nz=True, overrides=None):
        _, layout, qlens, heads, klens, sparse = spec
        self.inputs = {k: v.to(device) for k, v in inputs.items()}
        torch.npu.synchronize()
        start = time.perf_counter()
        self.key = pack_nz(self.inputs["v"]) if nz else self.inputs["v"].flatten(2)
        self.rope = pack_nz(self.inputs["k"]) if nz else self.inputs["k"].flatten(2)
        torch.npu.synchronize()
        self.pack_seconds = time.perf_counter() - start if nz else 0.0
        self.kwargs = dict(
            query_rope=self.inputs["q"],
            key_rope=self.rope,
            block_table=self.inputs["table"],
            dequant_scale_query=self.inputs["qs"],
            dequant_scale_key=self.inputs["ks"],
            dequant_scale_value=self.inputs["vs"],
            query_quant_mode=3,
            num_query_heads=heads,
            num_key_value_heads=1,
            input_layout=layout,
            actual_seq_kvlen=list(klens),
            block_size=128,
            sparse_mode=sparse,
            softmax_scale=576**-0.5,
            return_softmax_lse=False,
        )
        if layout == "TND":
            self.kwargs["actual_seq_qlen"] = list(itertools.accumulate(qlens))
        if sparse:
            self.kwargs["atten_mask"] = torch.ones(
                2048, 2048, dtype=torch.bool, device=device
            ).triu(1)
        self.kwargs.update(overrides or {})
        self.out = torch.empty_like(self.inputs["qv"], dtype=torch.bfloat16)
        self.lse = torch.empty(1, dtype=torch.float32, device=device)

    def __call__(self):
        result = torch.ops.npu.npu_fused_infer_attention_score_v2.out(
            self.inputs["qv"], self.key, self.key, **self.kwargs, out=[self.out, self.lse]
        )
        return result[0]


def strided_inputs(spec, kind):
    inputs = make_inputs(spec)
    for name in ("k", "v"):
        raw = inputs[name]
        pages, rows, heads, dim = raw.shape
        if kind == "page-gap":
            backing = torch.empty((pages * 2 + 1, rows, heads, dim), dtype=raw.dtype)
            viewed = backing[1::2]
        elif kind in ("row-gap", "wide-row"):
            row_stride = dim + 32 if kind == "row-gap" else 65536
            stride = (row_stride * 128 + 32, row_stride, dim, 1)
            size = (pages - 1) * stride[0] + 127 * stride[1] + dim + 32
            backing = torch.empty(size, dtype=raw.dtype)
            viewed = torch.as_strided(backing, raw.shape, stride, 32)
        elif kind == "broadcast":
            inputs[name] = raw[:, :1].expand_as(raw)
            continue
        else:
            continue
        viewed.copy_(raw)
        inputs[name] = viewed
    if kind == "table-gap":
        raw = inputs["table"]
        backing = torch.full((raw.shape[0] + 1, raw.shape[1] + 3), -1, dtype=torch.int32)
        inputs["table"] = backing[1:, 1 : raw.shape[1] + 1]
        inputs["table"].copy_(raw)
        inputs["table"][:, -1] = inputs["table"][:, 0]
    return inputs
