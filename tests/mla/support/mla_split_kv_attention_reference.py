"""Raw-input partial math; neither device tasks nor migrated bounds are inputs."""

import math
from pathlib import Path
import torch

from tests.common import golden_cache
from tests.mla.support.mla_ref import normalize_fa3_window


def fragments(spec, m, n):
    batch, sq, heads, sk = spec["shape"]
    left, right = normalize_fa3_window(
        sq, sk, causal=spec.get("causal", False), window_size=spec.get("window", (-1, -1))
    )
    for b, length in enumerate(spec.get("lengths", [sk] * batch)):
        for block, row in enumerate(range(0, sq * heads, m)):
            count = min(m, sq * heads - row)
            queries = torch.arange(row, row + count) // heads
            keys = torch.arange(length)
            center = queries[:, None] + length - sq
            visible = torch.ones((count, length), dtype=torch.bool)
            if left >= 0:
                visible &= keys >= center - left
            if right >= 0:
                visible &= keys <= center + right
            union = visible.any(0).nonzero().flatten()
            begin, end = (int(union[0]) // n, int(union[-1]) // n + 1) if len(union) else (0, 0)
            per_split = math.ceil((end - begin) / spec["splits"])
            for split in range(spec["splits"]):
                start = min(end, begin + split * per_split)
                stop = min(end, start + per_split)
                mask = visible.clone()
                mask[:, : start * n] = False
                mask[:, stop * n :] = False
                yield (b, block, split, start, stop), row, mask


def partial_math(inputs, spec, m, n, upcast):
    batch, sq, heads, _ = spec["shape"]
    tasks = list(fragments(spec, m, n))
    out = torch.zeros((len(tasks), m, 512), dtype=torch.float32)
    lse = torch.full((len(tasks), m), -torch.inf)
    maxima, sums = lse.clone(), torch.zeros_like(lse)
    scale = spec.get("scale", 576**-0.5)
    dtype = torch.float64 if upcast else inputs["q"].dtype
    for i, ((b, _, _, begin, end), row, visible) in enumerate(tasks):
        lo, hi = begin * n, min(end * n, visible.shape[1])
        if lo >= hi:
            continue
        mask = visible[:, lo:hi]
        valid = mask.any(-1)
        if not valid.any():
            continue
        count = mask.shape[0]
        q = inputs["q"][b].reshape(sq * heads, 64)[row : row + count].to(dtype)
        qc = inputs["qv"][b].reshape(sq * heads, 512)[row : row + count].to(dtype)
        k = inputs["k"][b, lo:hi, 0].to(dtype)
        v = inputs["v"][b, lo:hi, 0].to(dtype)
        scores = (q if upcast else q * scale) @ k.T
        scores += (qc if upcast else qc * scale) @ v.T
        scores = scores * scale if upcast else scores.float()
        scores = scores.masked_fill(~mask, -torch.inf)
        maximum = scores[valid].amax(-1)
        weights = torch.exp(scores[valid] - maximum[:, None])
        total = weights.sum(-1)
        if upcast:
            result = (weights @ v) / total[:, None]
        else:
            result = torch.softmax(scores[valid], -1).to(dtype) @ v
        out[i, :count][valid] = result.float()
        lse[i, :count][valid] = torch.logsumexp(scores[valid], -1).float()
        maxima[i, :count][valid] = maximum.float()
        sums[i, :count][valid] = total.float()
    return out, lse, maxima, sums


def references(inputs, spec, m, n):
    # The content cache covers both equations and their mapping, and survives
    # representative, acceptance and profile runs. Device work is never cached.
    names = tuple(
        f"{kind}_{precision}"
        for precision in ("high", "low")
        for kind in ("out", "lse", "max", "sum")
    )
    values = golden_cache.get_or_compute_golden(
        nodeid="mla-split-kv-partial-reference",
        metadata=dict(
            spec=spec,
            m=m,
            n=n,
            threads=torch.get_num_threads(),
            interop_threads=torch.get_num_interop_threads(),
            torch_config=torch.__config__.show(),
        ),
        inputs=inputs,
        compute_fn=lambda: dict(
            zip(
                names,
                (*partial_math(inputs, spec, m, n, True), *partial_math(inputs, spec, m, n, False)),
            )
        ),
        expected_keys=names,
        source_files=[__file__, str(Path(__file__).with_name("mla_ref.py")), golden_cache.__file__],
    )
    return tuple(values[name] for name in names)
