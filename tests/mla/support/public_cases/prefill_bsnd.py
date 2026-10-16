"""Public BSND prefill inputs and reuse of independent fragment equations."""

import torch
from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.mla.support.mla_auto_split_bsnd_reference import references
from tests.common.compare import assert_fa_close


def make_case(
    dtype="fp16", shape=(2, 17, 8, 1057, 2), splits=1, mask="none", strided=False, lengths=None
):
    b, sq, h, sk, hk = shape
    dt = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
    gen = torch.Generator().manual_seed(1729)
    raw = {
        name: torch.randn(dim, generator=gen).to(dt)
        for name, dim in {
            "q": (b, sq, h, 192),
            "k": (b, sk, hk, 192),
            "v": (b, sk, hk, 128),
        }.items()
    }
    tensors = {name: value.npu() for name, value in raw.items()}
    if strided:
        for name, value in tensors.items():
            bb, ss, hh, dd = value.shape
            storage = torch.empty(
                (bb * 2 + 1, ss * 2 + 1, hh * 2 + 1, dd + 16), dtype=dt, device="npu"
            )
            view = storage[1 : 1 + 2 * bb : 2, 1 : 1 + 2 * ss : 2, 1 : 1 + 2 * hh : 2, 8 : 8 + dd]
            view.copy_(value)
            tensors[name] = view
    klens = [sk] * b if lengths is None else lengths
    return dict(
        shape=shape,
        raw=raw,
        t=tensors,
        klens=klens,
        qlens=[sq] * b,
        lengths=torch.tensor(klens, dtype=torch.int32, device="npu"),
        options=dict(
            num_splits=splits,
            causal=mask == "causal",
            window_size=(7, 2) if mask == "window" else (-1, -1),
            softmax_scale=192**-0.5,
        ),
    )


def metadata(c):
    b, sq, h, sk, hk = c["shape"]
    return api.get_scheduler_metadata(
        b,
        sq,
        sk,
        h,
        hk,
        192,
        c["lengths"],
        headdim_v=128,
        qkv_dtype=c["t"]["q"].dtype,
        **{k: v for k, v in c["options"].items() if k != "softmax_scale"},
    )


def invoke(c, record=None):
    return api.flash_attn_with_kvcache(
        c["t"]["q"],
        c["t"]["k"],
        c["t"]["v"],
        cache_seqlens=c["lengths"],
        scheduler_metadata=record,
        return_softmax_lse=True,
        **c["options"],
    )


def compare(c, result, record):
    b, sq, h, sk, hk = c["shape"]
    g = h // hk
    r = record.cpu().tolist()
    base = c.get("metadata_prefix", 20) + 2 * b
    ends = list(
        zip(
            r[base + 4 : base + 4 + r[base + 3]],
            r[base + 30 : base + 30 + r[base + 3]],
            r[base + 56 : base + 56 + r[base + 3]],
        )
    )
    raw = {
        name: value.reshape(b, value.shape[1], hk, g if name == "q" else 1, value.shape[-1])
        .permute(0, 2, 1, 3, 4)
        .reshape(b * hk, value.shape[1], g if name == "q" else 1, value.shape[-1])
        for name, value in c["raw"].items()
    }
    o = c["options"]
    refs = references(
        raw,
        [x for x in c["qlens"] for _ in range(hk)],
        [x for x in c["klens"] for _ in range(hk)],
        ends,
        r[13],
        r[14],
        o["causal"],
        o["window_size"],
        o["softmax_scale"],
        num_splits=o["num_splits"],
    )
    for precision in ("high", "low"):
        refs["out_" + precision] = (
            refs["out_" + precision]
            .reshape(b, hk, sq, g, 128)
            .permute(0, 2, 1, 3, 4)
            .reshape(b, sq, h, 128)
        )
        refs["lse_" + precision] = refs["lse_" + precision].reshape(b, h, sq)
        refs["accum_" + precision] = refs["accum_" + precision].reshape(
            o["num_splits"] or 26, b, h, sq, 128
        )
        refs["accum_lse_" + precision] = refs["accum_lse_" + precision].reshape(
            o["num_splits"] or 26, b, h, sq
        )
    for i, name in enumerate(("out", "lse", "accum", "accum_lse")):
        actual = result[i].cpu()
        if i >= 2 and o["num_splits"] == 1:
            assert actual.numel() == 0
        else:
            assert actual.shape == refs[name + "_high"].shape
            assert_fa_close(
                actual, refs[name + "_high"], refs[name + "_low"], name="prefill-" + name
            )
    return refs
