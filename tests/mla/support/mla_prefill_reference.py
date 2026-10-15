"""Independent raw-input equations for 192/128 fragments and their full rows."""

from pathlib import Path
import torch
from tests.common import golden_cache
from tests.mla.support.mla_ref import normalize_fa3_window
from tests.mla.support.mla_split_kv_attention_reference import fragments


def fragment_rows(spec):
    batch, sq, heads, sk, kv_heads = spec["shape"]
    # Existing independent visibility/ceil-div enumeration, treating each KV
    # head as one virtual batch with its own G query heads and independent V.
    expanded = dict(
        spec,
        shape=(batch * kv_heads, sq, heads // kv_heads, sk),
        lengths=[v for v in spec.get("lengths", [sk] * batch) for _ in range(kv_heads)],
    )
    return list(fragments(expanded, spec["m"], spec["n"]))


def equations(inputs, spec, upcast):
    batch, sq, heads, sk, kv_heads = spec["shape"]
    g, m, n = heads // kv_heads, spec["m"], spec["n"]
    tasks = fragment_rows(spec)
    part = torch.zeros(len(tasks), m, 128)
    lse = torch.full((len(tasks), m), -torch.inf)
    maximum, total = lse.clone(), torch.zeros_like(lse)
    full_out, full_lse = part.clone(), lse.clone()
    dtype = torch.float64 if upcast else inputs["q"].dtype
    scale = spec.get("scale", 192**-0.5)
    left, right = normalize_fa3_window(
        sq, sk, causal=spec.get("causal", False), window_size=spec.get("window", (-1, -1))
    )
    cached = {}
    for index, ((virtual, block, split, begin, end), row, visible) in enumerate(tasks):
        b, h = divmod(virtual, kv_heads)
        length = spec.get("lengths", [sk] * batch)[b]
        count = visible.shape[0]
        key = (virtual, block)
        if key not in cached:
            q = (
                inputs["q"][b, :, h * g : (h + 1) * g]
                .reshape(sq * g, 192)[row : row + count]
                .to(dtype)
            )
            k = inputs["k"][b, :length, h].to(dtype)
            v = inputs["v"][b, :length, h].to(dtype)
            scores = (q if upcast else q * scale) @ k.T
            scores = scores * scale if upcast else scores.float()
            queries = torch.arange(row, row + count) // g
            center = queries[:, None] + length - sq
            keys = torch.arange(length)
            allowed = torch.ones(count, length, dtype=torch.bool)
            if left >= 0:
                allowed &= keys >= center - left
            if right >= 0:
                allowed &= keys <= center + right
            scores = scores.masked_fill(~allowed, -torch.inf)
            cached[key] = scores, v, allowed
        scores, v, allowed = cached[key]

        def compute(scores, values, valid):
            result = torch.zeros(count, 128)
            logs = torch.full((count,), -torch.inf)
            mx, sums = logs.clone(), torch.zeros_like(logs)
            if valid.any():
                current = scores[valid]
                maxima = current.amax(-1)
                weights = torch.exp(current - maxima[:, None])
                denom = weights.sum(-1)
                out = (
                    (weights @ values) / denom[:, None]
                    if upcast
                    else torch.softmax(current, -1).to(dtype) @ values
                )
                result[valid] = out.float()
                logs[valid] = torch.logsumexp(current, -1).float()
                mx[valid] = maxima.float()
                sums[valid] = denom.float()
            return result, logs, mx, sums

        lo, hi = begin * n, min(end * n, length)
        values = compute(scores[:, lo:hi], v[lo:hi], visible[:, lo:hi].any(-1))
        for dest, value in zip((part, lse, maximum, total), values):
            dest[index, :count] = value
        if split == 0:
            full_values = compute(scores, v, allowed.any(-1))
            full_out[index, :count], full_lse[index, :count] = full_values[:2]
    return part, lse, maximum, total, full_out, full_lse


def references(inputs, spec):
    names = tuple(
        f"{kind}_{precision}"
        for precision in ("high", "low")
        for kind in ("out", "lse", "max", "sum", "full_out", "full_lse")
    )
    values = golden_cache.get_or_compute_golden(
        nodeid="mla-prefill-component-reference",
        metadata=dict(
            spec=spec,
            threads=torch.get_num_threads(),
            interop_threads=torch.get_num_interop_threads(),
            torch_config=torch.__config__.show(),
        ),
        inputs=inputs,
        compute_fn=lambda: dict(
            zip(names, (*equations(inputs, spec, True), *equations(inputs, spec, False)))
        ),
        expected_keys=names,
        source_files=[
            __file__,
            str(Path(__file__).with_name("mla_split_kv_attention_reference.py")),
            str(Path(__file__).with_name("mla_ref.py")),
            golden_cache.__file__,
        ],
    )
    return values
