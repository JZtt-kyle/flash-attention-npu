"""Content-cached independent BSND split math, from raw allocation/used inputs."""

import torch
from tests.common import golden_cache
from tests.mla.support import mla_ref
from tests.mla.support.mla_ref import normalize_fa3_window


def references(raw, qlens, klens, splits, m, n, causal, window, scale):
    batch, sq, heads, _ = raw["q"].shape
    sk = raw["k"].shape[1]
    names = [
        f"{name}_{precision}"
        for precision in ("high", "low")
        for name in ("out", "lse", "accum", "accum_lse")
    ]
    metadata = dict(
        qlens=list(qlens),
        klens=list(klens),
        splits=splits,
        m=m,
        n=n,
        causal=causal,
        window=window,
        scale=scale,
    )

    def compute():
        result = {}
        left, right = normalize_fa3_window(sq, sk, causal=causal, window_size=window)
        for precision in ("high", "low"):
            high = precision == "high"
            dtype = torch.float64 if high else raw["q"].dtype
            out = torch.zeros((batch, sq, heads, 512), dtype=raw["q"].dtype)
            lse = torch.full((batch, heads, sq), -torch.inf)
            accum = torch.zeros((splits, batch, heads, sq, 512), dtype=torch.float32)
            accum_lse = torch.full((splits, batch, heads, sq), -torch.inf)
            if sk == 0:
                lse.fill_(torch.inf)
            for b, (q, k) in enumerate(zip(qlens, klens)):
                if not q or not k:
                    continue
                qr = raw["q"][b, :q].transpose(0, 1).to(dtype)
                qc = raw["qv"][b, :q].transpose(0, 1).to(dtype)
                kr = raw["k"][b, :k, 0].to(dtype)
                kv = raw["v"][b, :k, 0].to(dtype)
                scores = (qr if high else qr * scale) @ kr.T + (qc if high else qc * scale) @ kv.T
                scores = scores * scale if high else scores.float()
                center = torch.arange(q)[:, None] + k - q
                keys = torch.arange(k)[None, :]
                visible = torch.ones((q, k), dtype=torch.bool)
                if left >= 0:
                    visible &= keys >= center - left
                if right >= 0:
                    visible &= keys <= center + right

                def attention(mask):
                    logits = scores.masked_fill(~mask, -torch.inf)
                    sums = torch.logsumexp(logits, -1).float()
                    nonempty = mask.any(-1)
                    safe = logits.masked_fill(~nonempty[None, :, None], 0)
                    probs = torch.softmax(safe, -1).masked_fill(~nonempty[None, :, None], 0)
                    values = probs.to(dtype) @ kv
                    return values, sums

                values, sums = attention(visible)
                out[b, :q] = values.transpose(0, 1).to(out.dtype)
                lse[b, :, :q] = sums
                for block in range((q * heads + m - 1) // m):
                    begin = block * m
                    stop = min((block + 1) * m, q * heads)
                    query_ids = torch.arange(begin, stop) // heads
                    union = visible[query_ids].any(0).nonzero().flatten()
                    first = int(union[0]) // n if union.numel() else 0
                    last = (int(union[-1]) + n) // n if union.numel() else 0
                    width = (last - first + splits - 1) // splits
                    for split in range(splits):
                        a = min(last, first + split * width) * n
                        z = min(last, first + (split + 1) * width) * n
                        if a >= z:
                            continue
                        mask = visible & (keys >= a) & (keys < z)
                        values, sums = attention(mask)
                        for row in range(begin, stop):
                            qi, hi = divmod(row, heads)
                            accum[split, b, hi, qi] = values[hi, qi].float()
                            accum_lse[split, b, hi, qi] = sums[hi, qi]
            result.update(
                {
                    f"{name}_{precision}": value
                    for name, value in zip(
                        ("out", "lse", "accum", "accum_lse"), (out, lse, accum, accum_lse)
                    )
                }
            )
        return result

    return golden_cache.get_or_compute_golden(
        nodeid="mla-split-bsnd-reference",
        metadata=metadata,
        inputs=raw,
        compute_fn=compute,
        expected_keys=names,
        source_files=[__file__, golden_cache.__file__, mla_ref.__file__],
    )
