"""Shared helpers extracted from tests/mla/e2e/causal_bsnd/test_flash_attn_npu_v3_mla_causal_public.py.

BSND causal production/public paths, fixed source, independent math and contracts.
"""

import hashlib
import json
import torch
from flash_attn_npu_3 import flash_attn_npu_interface as api
from tests.mla.support.mla_profile_cases import (
    CASES,
    S1G_CASES,
    KV_CASES,
    LONG_KV_CASES,
    BSND_CASES,
    STRIDE_CASES,
)
from tests.mla.support.components.causal_attention import make_case
from tests.mla.support.components.bsnd_stride import make_view, strides_for
from tests.mla.support.components.used_schedule import used_case, execute
from tests.mla.support.mla_validation_policy import public_cases

PERFORMANCE_CASES = [
    dict(shape=s, scale=scale) for s, scale in CASES + S1G_CASES + KV_CASES + LONG_KV_CASES
]

PERFORMANCE_CASES += [dict(shape=s, scale=scale, lengths=k) for s, scale, k in BSND_CASES]

PERFORMANCE_CASES += [
    dict(shape=s, scale=scale, lengths=k, stride=p) for s, scale, k, p in STRIDE_CASES
]

PERFORMANCE_CASES += [dict(shape=(2, 513, 3, 257), lengths=[257, 129])]

FULL_PERFORMANCE_CASES = public_cases(PERFORMANCE_CASES)

PERFORMANCE_CASES = public_cases(FULL_PERFORMANCE_CASES, group="bsnd")

NATIVE_PREFLIGHT_CASES = [
    dict(shape=(2, 7, 16, 33), lengths=[3, 0]),
    dict(shape=(1, 7, 16, 8193)),
    FULL_PERFORMANCE_CASES[4],
    FULL_PERFORMANCE_CASES[24],
]

USED_CASES = [
    dict(shape=(2, 7, 16, 33), used=[5, 3], lengths=[3, 0]),
    dict(shape=(2, 129, 3, 513), used=[65, 129], lengths=[17, 513]),
    dict(shape=(2, 7, 129, 16384), used=[7, 1], lengths=[8193, 16384]),
    dict(shape=(2, 7, 16, 33), used=[0, 0], lengths=[0, 33]),
    dict(shape=(2, 7, 16, 33), used=[5, 3], lengths=[3, 0], stride="independent"),
    dict(shape=(2, 7, 16, 33), used=[5, 3], implicit_kv=True),
]


def generate(device, lengths, causal=True):
    q, k = device["q"], device["k"]
    return api.get_scheduler_metadata(
        q.shape[0],
        q.shape[1],
        k.shape[1],
        q.shape[2],
        1,
        64,
        lengths,
        qkv_dtype=q.dtype,
        headdim_v=512,
        causal=causal,
        num_splits=1,
    )


def cached(device, lengths=None, record=None, scale=None, causal=True):
    return api.flash_attn_with_kvcache(
        device["q"],
        device["k"],
        device["v"],
        qv=device["qv"],
        cache_seqlens=lengths,
        scheduler_metadata=record,
        softmax_scale=scale,
        causal=causal,
        num_splits=1,
        return_softmax_lse=True,
    )


def prepare(probes, spec):
    b, sq, h, sk = spec["shape"]
    case = make_case(probes, dict(spec, lengths=spec.get("lengths") or [sk] * b))
    raw = case["inputs"]
    used = spec.get("used", [sq] * b)
    for sample, length in enumerate(used):
        for name in ("q", "qv"):
            raw[name][sample, length:] = torch.nan
    device = {name: t.npu() for name, t in raw.items()}
    if spec.get("stride"):
        for name, offset in zip(raw, (5, 11, 19, 29)):
            device[name], raw[name], _ = make_view(
                raw[name],
                strides_for(raw[name].shape, spec["stride"], name),
                offset,
                overlap=spec["stride"] in ("broadcast", "overlap"),
            )
    lengths = torch.tensor(case["lengths"], dtype=torch.int32, device="npu")
    uq = torch.tensor(used, dtype=torch.int32, device="npu") if "used" in spec else None
    return raw, device, lengths, uq, case["details"]


def source_tiles(probes, spec, oracle, details):
    """Preserve the original allocation's independently selected source tile."""
    b, sq, heads, sk = spec["shape"]
    item = used_case(
        q=[sq] * b,
        kv=[sk] * b,
        uq=spec.get("used"),
        uk=None if spec.get("implicit_kv") else spec.get("lengths") or [sk] * b,
        heads=heads,
        aic=int(details[2]),
        ratio=2,
    )
    result = execute(oracle, [item])[0]
    assert result[0] == 0
    identity = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]
    (probes[1] / f"public-source-schedule-{identity}.json").write_text(
        json.dumps(dict(input=item, result=result), indent=2) + "\n"
    )
    return result[1:3]
