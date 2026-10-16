"""BSND softcap invocation adapter; reuse inputs and independent interval math."""

import hashlib
import os
from pathlib import Path
import torch
from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.common.compare import assert_fa_close
from tests.mla.support.public_cases.prefill_bsnd import make_case as prefill_case
from tests.mla.support.public_cases.split_kv_bsnd_public import make_case as compressed_case
from tests.mla.support.components.softcap import pattern
from tests.mla.support.mla_auto_split_bsnd_reference import references
from tests.mla.support.public_cases.contract import _forward


# Named tuples cover independent crossings without expanding all axes.
CASES = [
    dict(id="none-single", splits=1),
    dict(id="causal-single-empty-rows", splits=1, mask="causal", shape=(2, 17, 8, 3)),
    dict(id="window-single-negative", splits=1, mask="window", scale=-0.23, cap=30),
    dict(id="none-explicit", splits=3, shape=(2, 7, 8, 1057)),
    dict(id="causal-empty-splits", splits=8, mask="causal", shape=(1, 33, 8, 3)),
    dict(
        id="window-explicit-used-stride",
        splits=4,
        mask="window",
        shape=(2, 17, 8, 1057),
        strided=True,
        lengths=[1057, 7],
    ),
    dict(id="none-auto", splits=0, shape=(2, 3, 8, 1057)),
    dict(id="causal-auto", splits=0, mask="causal", shape=(2, 65, 8, 1057), lengths=[1057, 0]),
    dict(id="window-auto", splits=0, mask="window", window=(256, 2), shape=(1, 17, 8, 513)),
    dict(
        id="zero-scale-single",
        splits=1,
        scale=0,
        shape=(1, 1, 8, 33),
        window=(1, 0),
        pattern="rounding-ties",
    ),
    dict(
        id="zero-scale-multi",
        splits=2,
        scale=0,
        shape=(1, 1, 8, 33),
        window=(1, 0),
        pattern="rounding-ties",
    ),
    dict(id="saturation", splits=2, scale=0.23, pattern="saturation"),
    dict(id="component-cancellation", splits=2, scale=0.23, pattern="cancellation"),
    dict(id="near-zero", splits=1, cap=1e8),
    dict(id="small-cap", splits=0, cap=1e-4),
]
DAILY = {"none-single", "window-explicit-used-stride", "causal-auto"}
REPRESENTATIVE = DAILY | {"zero-scale-multi", "component-cancellation"}


def make_case(representation, dtype="fp16", **spec):
    b, q, h, k = spec.get("shape", (2, 7, 8, 65))
    splits, mask = spec.get("splits", 1), spec.get("mask", "none")
    common = dict(splits=splits, mask=mask, strided=spec.get("strided", False))
    if representation == "prefill":
        c = prefill_case(dtype, shape=(b, q, h, k, 2), **common)
    else:
        c = compressed_case(dtype, shape=(b, q, h, k), **common)
    c["representation"] = representation
    c["klens"] = spec.get("lengths", [k] * b)
    c["lengths"] = torch.tensor(c["klens"], dtype=torch.int32, device="npu")
    c["options"].update(softcap=spec.get("cap", 15), softmax_scale=spec.get("scale"))
    if "window" in spec:
        c["options"]["window_size"] = spec["window"]
    pattern(c["raw"], spec.get("pattern"), representation)
    for name, value in c["raw"].items():
        c["t"][name].copy_(value)
    return c


def metadata(c, *, enabled=None):
    b, q, h, k = c["shape"][:4]
    prefill = c["representation"] == "prefill"
    return api.get_scheduler_metadata(
        b,
        q,
        k,
        h,
        c["shape"][4] if prefill else 1,
        192 if prefill else 64,
        c["lengths"],
        headdim_v=128 if prefill else 512,
        qkv_dtype=c["t"]["q"].dtype,
        has_softcap=c["options"]["softcap"] > 0 if enabled is None else enabled,
        **{name: c["options"][name] for name in ("num_splits", "causal", "window_size")},
    )


def invoke(c, record=None, *, entry="cache"):
    if entry in ("cpp", "raw", "fake"):
        args = dict(**c["t"], seqused_k=c["lengths"], scheduler_metadata=record, **c["options"])
        args["window_size_left"], args["window_size_right"] = args.pop("window_size")
        if c.get("usedq") is not None:
            args["seqused_q"] = c["usedq"]
        return _forward(entry, args)
    if entry == "func":
        return api.flash_attn_func(**c["t"], return_attn_probs=True, **c["options"])
    return api.flash_attn_with_kvcache(
        c["t"]["q"],
        c["t"]["k"],
        c["t"]["v"],
        qv=c["t"].get("qv"),
        cache_seqlens=c["lengths"],
        scheduler_metadata=record,
        return_softmax_lse=True,
        **c["options"],
    )


def compare(c, result, record):
    b, sq, h, sk = c["shape"][:4]
    hk = c["shape"][4] if c["representation"] == "prefill" else 1
    g = h // hk
    r = record.cpu().tolist()
    prefill = c["representation"] == "prefill"
    m, n = r[13:15] if prefill else r[15:17]
    base = 20 + 2 * b if prefill else 410 + 8 * b
    ends = []
    if prefill or c["options"]["num_splits"] == 0:
        count = r[base + 3]
        ends = list(
            zip(
                r[base + 4 : base + 4 + count],
                r[base + 30 : base + 30 + count],
                r[base + 56 : base + 56 + count],
            )
        )
    raw = {
        name: x.reshape(b, x.shape[1], hk, g if name in ("q", "qv") else 1, x.shape[-1])
        .permute(0, 2, 1, 3, 4)
        .reshape(b * hk, x.shape[1], g if name in ("q", "qv") else 1, x.shape[-1])
        for name, x in c["raw"].items()
    }
    o = c["options"]
    scale = o["softmax_scale"]
    if scale is None:
        scale = (192 if prefill else 576) ** -0.5
    refs = references(
        raw,
        [q for q in c["qlens"] for _ in range(hk)],
        [k for k in c["klens"] for _ in range(hk)],
        ends,
        m or 256,
        n or 512,
        o["causal"],
        o["window_size"],
        scale,
        num_splits=o["num_splits"],
        softcap=o["softcap"],
    )
    dv = 128 if prefill else 512
    for precision in ("high", "low"):
        refs["out_" + precision] = (
            refs["out_" + precision]
            .reshape(b, hk, sq, g, dv)
            .permute(0, 2, 1, 3, 4)
            .reshape(b, sq, h, dv)
        )
        refs["lse_" + precision] = refs["lse_" + precision].reshape(b, h, sq)
        refs["accum_" + precision] = refs["accum_" + precision].reshape(
            o["num_splits"] or 26, b, h, sq, dv
        )
        refs["accum_lse_" + precision] = refs["accum_lse_" + precision].reshape(
            o["num_splits"] or 26, b, h, sq
        )
    actual = [x.cpu() for x in result]
    directory = os.environ.get("MLA_VALIDATION_RESULT")
    if directory:
        key = hashlib.sha256(
            (os.environ.get("PYTEST_CURRENT_TEST", "") + str(o)).encode()
        ).hexdigest()[:20]
        torch.save(
            dict(
                raw=c["raw"],
                record=r,
                options=o,
                qlens=c["qlens"],
                klens=c["klens"],
                actual=actual,
                refs=refs,
            ),
            Path(directory) / ("softcap-" + key + ".pt"),
        )
    for i, name in enumerate(("out", "lse", "accum", "accum_lse")[: len(actual)]):
        if i >= 2 and o["num_splits"] == 1:
            assert actual[i].numel() == 0 and actual[i].dtype == torch.get_default_dtype()
        else:
            assert actual[i].shape == refs[name + "_high"].shape
            # Compressed BSND has no output promise outside Q-used; retain
            # its existing allocation contract rather than checking padding.
            selections = (
                [()]
                if prefill
                else [
                    (batch, slice(used))
                    if i == 0
                    else (batch, slice(None), slice(used))
                    if i == 1
                    else (slice(None), batch, slice(None), slice(used))
                    for batch, used in enumerate(c["qlens"])
                ]
            )
            for index in selections:
                assert_fa_close(
                    actual[i][index],
                    refs[name + "_high"][index],
                    refs[name + "_low"][index],
                    softcap=o["softcap"],
                    name="softcap-bsnd-" + name,
                )
    return refs
