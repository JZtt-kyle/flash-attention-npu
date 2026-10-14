"""Independent merge equations and raw-input visibility; content-cached math."""

import torch
from tests.common import golden_cache


def merge_math(out, total, maximum, valid, dtype, high):
    precision = torch.float64 if high else torch.float32
    o, s, m = out.to(precision), total[..., 0].to(precision), maximum[..., 0].to(precision)
    logits = m + s.log()
    logits = logits.masked_fill(~valid, -torch.inf)
    any_valid = valid.any(0)
    lse = torch.full((out.shape[1],), -torch.inf, dtype=precision)
    result = torch.zeros(out.shape[1:], dtype=precision)
    if any_valid.any():
        lse[any_valid] = torch.logsumexp(logits[:, any_valid], 0)
        weights = torch.softmax(logits[:, any_valid], 0)
        values = o[:, any_valid].masked_fill(~valid[:, any_valid, None], 0)
        result[any_valid] = (weights[..., None] * values).sum(0)
    return result.to(dtype), lse.float()


def merge_references(out, total, maximum, valid, dtype):
    names = ("out_high", "lse_high", "out_low", "lse_low", "partial_lse_high", "partial_lse_low")
    tensors = dict(out=out, total=total, maximum=maximum, valid=valid)

    def compute():
        result = [
            *merge_math(**tensors, dtype=dtype, high=True),
            *merge_math(**tensors, dtype=dtype, high=False),
        ]
        for precision in (torch.float64, torch.float32):
            lse = (total[..., 0].to(precision).log() + maximum[..., 0].to(precision)).float()
            result.append(lse.masked_fill(~valid, -torch.inf))
        return dict(zip(names, result))

    cached = golden_cache.get_or_compute_golden(
        nodeid="mla-split-kv-merge-reference",
        metadata={"dtype": str(dtype)},
        inputs=tensors,
        compute_fn=compute,
        expected_keys=names,
        source_files=[__file__, golden_cache.__file__],
    )
    return tuple(cached[name] for name in names)
