"""Independent FP32/FP64 attention equations from raw integers and ND pages."""

import torch

from tests.common import golden_cache


def equations(inputs, spec, dtype):
    _, layout, qlens, heads, klens, sparse = spec
    outputs, logs = [], []
    offset = 0
    for batch, (sq, sk) in enumerate(zip(qlens, klens)):
        selection = slice(offset, offset + sq) if layout == "TND" else batch
        qv = inputs["qv"][selection].to(dtype)
        qr = inputs["q"][selection].to(dtype)
        qs = inputs["qs"][selection].to(dtype)
        pages = inputs["table"][batch].long()
        kv = inputs["v"][pages].reshape(-1, 512)[:sk].to(dtype)
        kr = inputs["k"][pages].reshape(-1, 64)[:sk].to(dtype)
        qcontent = qv * qs.unsqueeze(-1)
        key = kv * inputs["ks"].to(dtype)
        value = kv * inputs["vs"].to(dtype)
        score = (qcontent.reshape(-1, 512) @ key.T + qr.reshape(-1, 64) @ kr.T) * 576**-0.5
        if sparse == 3:
            allowed = (
                torch.arange(sk)[None, :]
                <= torch.arange(sq).repeat_interleave(heads)[:, None] + sk - sq
            )
            score.masked_fill_(~allowed, -torch.inf)
        valid = torch.isfinite(score).any(-1)
        out = torch.zeros(sq * heads, 512, dtype=dtype)
        lse = torch.full((sq * heads,), -torch.inf, dtype=dtype)
        if valid.any():
            out[valid] = torch.softmax(score[valid], dim=-1) @ value
            lse[valid] = torch.logsumexp(score[valid], dim=-1)
        outputs.append(out.reshape(sq, heads, 512).float())
        logs.append(lse.reshape(sq, heads).float())
        offset += sq
    combine = torch.cat if layout == "TND" else torch.stack
    return combine(outputs), combine(logs)


def references(inputs, spec):
    names = ("out_high", "lse_high", "out_low", "lse_low")
    return golden_cache.get_or_compute_golden(
        nodeid="mla-int8-raw-equations",
        metadata=dict(
            spec=spec, threads=torch.get_num_threads(), torch_config=torch.__config__.show()
        ),
        inputs=inputs,
        compute_fn=lambda: dict(
            zip(
                names,
                (*equations(inputs, spec, torch.float64), *equations(inputs, spec, torch.float32)),
            )
        ),
        expected_keys=names,
        source_files=[__file__, golden_cache.__file__],
    )
