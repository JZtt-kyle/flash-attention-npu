# Window normalization adapts hopper/flash_api.cpp at the revision below.
# Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar,
# Pradeep Ramani, Tri Dao. BSD-3-Clause; see LICENSE in the repository root.

"""Independent CPU references and canonical BSND input mappings for MLA.

The formulas are implemented here without calling FA3, CANN, or the existing
attention golden. Contracts come from docs/mla/design.md: CANN ops-transformer
ca18aee610729ca5270a44e2d84c751bc687fbf0 and Dao-AILab/flash-attention
d15f1531a460ba456f41b01a774f33ab2db8febf (hopper/). No kernel code is copied.

Only the two migration representations are accepted. These helpers do not
parse packed sequences, page tables, metadata, or split accumulators. Mappings
describe logical tensors, not a torch_npu operator invocation or device support.
"""

import math

import torch


def _check_tensors(*tensors):
    first = tensors[0]
    for tensor in tensors:
        if tensor.ndim != 4:
            raise ValueError("MLA reference inputs must use canonical BSND layout")
        if tensor.stride(-1) != 1:
            raise ValueError("MLA inputs must have a contiguous last dimension")
        if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            raise ValueError("MLA references require floating point inputs")
        if tensor.dtype != first.dtype or tensor.device != first.device:
            raise ValueError("All MLA inputs must have the same dtype and device")


def _check_fa3(q, k, v, qv):
    _check_tensors(q, k, v, *(() if qv is None else (qv,)))
    if q.shape[0] != k.shape[0] or k.shape[:3] != v.shape[:3]:
        raise ValueError("Q/K/V batch, KV sequence, and KV head counts must agree")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("Q and K must have the same head dimension")
    if k.shape[2] == 0 or q.shape[2] == 0 or q.shape[2] % k.shape[2]:
        raise ValueError("The positive KV head count must divide the query head count")
    if qv is None:
        if (q.shape[-1], v.shape[-1]) != (192, 128):
            raise ValueError("MLA prefill requires Q/K dim 192 and V dim 128 with qv=None")
    else:
        if (q.shape[-1], v.shape[-1], k.shape[2]) != (64, 512, 1):
            raise ValueError("Compressed MLA requires Q/K dim 64, V dim 512, and one KV head")
        if qv.shape != (*q.shape[:3], v.shape[-1]):
            raise ValueError("qv must have Q's batch/sequence/heads and V's head dimension")


def _scale(q, qv, softmax_scale):
    if softmax_scale is None:
        return (q.shape[-1] + (0 if qv is None else qv.shape[-1])) ** -0.5
    if not math.isfinite(softmax_scale):
        raise ValueError("The reference requires a finite softmax_scale")
    return softmax_scale


def fa3_to_cann_inputs(q, k, v, qv=None, *, softmax_scale=None, split_rope=True):
    """Map logical FA3 tensors to CANN components, including an explicit scale.

    Compressed MLA passes the SAME tensor as content key and value. Prefill
    can use split RoPE views (which retain the 192-wide row stride) or combined
    Q/K. This function does not rotate the already formed positional features.
    """
    _check_fa3(q, k, v, qv)
    if qv is not None:
        if not split_rope:
            raise ValueError("Compressed MLA requires split RoPE in CANN")
        query, key, query_rope, key_rope = qv, v, q, k
    elif split_rope:
        query, key = q[..., :128], k[..., :128]
        query_rope, key_rope = q[..., 128:], k[..., 128:]
    else:
        query, key, query_rope, key_rope = q, k, None, None
    return dict(
        query=query,
        key=key,
        value=v,
        query_rope=query_rope,
        key_rope=key_rope,
        scale=_scale(q, qv, softmax_scale),
    )


def cann_to_fa3_inputs(query, key, value, *, query_rope=None, key_rope=None, scale):
    """Map canonical CANN components to FA3, preserving the supplied scale.

    Independent 512-dimensional content K/V cannot be represented by FA3 qv.
    Require the same storage view, including offset/strides, even when separate
    tensors happen to contain equal values. Split 128+64 prefill concatenates
    Q/K for FA3; its value tensor remains independent and is never replaced.
    """
    if scale is None:
        raise ValueError("CANN scale must be explicit; its default is not the FA3 default")
    if (query_rope is None) != (key_rope is None):
        raise ValueError("query_rope and key_rope must both be present or both absent")
    if query_rope is None:
        result = dict(q=query, k=key, v=value, qv=None)
    else:
        _check_tensors(query, key, value, query_rope, key_rope)
        if query_rope.shape != (*query.shape[:3], 64) or key_rope.shape != (
            *key.shape[:3],
            64,
        ):
            raise ValueError("RoPE must match its content tensor's BSN axes and have dim 64")
        if query.shape[-1] != key.shape[-1] or key.shape != value.shape:
            raise ValueError("Split CANN content Q/K/V must have matching head dimensions")
        if query.shape[-1] == 512:
            if not key.is_set_to(value):
                raise ValueError("Independent 512-dimensional content K/V cannot map to FA3 qv")
            result = dict(q=query_rope, k=key_rope, v=value, qv=query)
        elif query.shape[-1] == 128:
            result = dict(
                q=torch.cat((query, query_rope), dim=-1),
                k=torch.cat((key, key_rope), dim=-1),
                v=value,
                qv=None,
            )
        else:
            raise ValueError("Split MLA content dimension must be 128 or 512")
    _check_fa3(**result)
    result["softmax_scale"] = _scale(result["q"], result["qv"], scale)
    return result


def normalize_fa3_window(seqlen_q, seqlen_k, *, causal=False, window_size=(-1, -1)):
    """Mirror pinned hopper/flash_api.cpp::mha_fwd for the two MLA dimensions.

    Sizes here are dense tensor capacities, not a batch maximum inferred from
    actual lengths. Packed/paged callers will need separate capacity handling.
    Only the standard -1 (unbounded) and nonnegative windows are modeled.
    """
    left, right = window_size
    if seqlen_q < 0 or seqlen_k < 0 or left < -1 or right < -1:
        raise ValueError("Lengths must be nonnegative and window bounds at least -1")
    if left >= seqlen_k - 1:
        left = -1
    if right >= seqlen_q - 1:
        right = -1
    # Both migration representations satisfy the upstream head-dimension
    # condition (Dq <= 64 or Dq > 128), including paged KV.
    if seqlen_q == 1 and left == -1 and right == -1:
        causal = False
    if causal:
        right = 0
    return left, right


def mla_mask(seqlen_q, seqlen_k, *, causal=False, window_size=(-1, -1)):
    """Return a CPU [Sq, Sk] boolean mask; True means excluded from attention."""
    left, right = normalize_fa3_window(seqlen_q, seqlen_k, causal=causal, window_size=window_size)
    center = torch.arange(seqlen_q)[:, None] + seqlen_k - seqlen_q
    key_pos = torch.arange(seqlen_k)[None, :]
    mask = torch.zeros((seqlen_q, seqlen_k), dtype=torch.bool)
    if left >= 0:
        mask |= key_pos < center - left
    if right >= 0:
        mask |= key_pos > center + right
    return mask


@torch.no_grad()
def mla_reference(
    q,
    k,
    v,
    qv=None,
    *,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    upcast=True,
):
    """Return dense O [B,Sq,Hq,Dv] and natural-log LSE [B,Hq,Sq] on CPU.

    upcast=True uses FP64 arithmetic before rounding O to the input dtype and
    LSE to FP32, matching the existing dual-reference comparator's convention.
    upcast=False is the low-precision PyTorch comparison: scale Q first, use
    input-dtype dot products, FP32 softmax, and input-dtype probabilities/PV.

    Pinned FA3 returns +inf LSE for a globally empty K tensor (host fast path),
    but -inf for a fully masked row with nonempty K (softmax/epilogue path).
    Neither case may produce a NaN or a nonzero output.
    """
    _check_fa3(q, k, v, qv)
    if q.device.type != "cpu":
        raise ValueError("mla_reference is a CPU oracle; move logical inputs to CPU explicitly")
    if not math.isfinite(softcap) or softcap < 0:
        raise ValueError("softcap must be finite and nonnegative")
    scale = _scale(q, qv, softmax_scale)
    batch, seqlen_q, heads, _ = q.shape
    seqlen_k = k.shape[1]
    mask = mla_mask(seqlen_q, seqlen_k, causal=causal, window_size=window_size)
    if seqlen_k == 0:
        return (
            torch.zeros((batch, seqlen_q, heads, v.shape[-1]), dtype=q.dtype),
            torch.full((batch, heads, seqlen_q), torch.inf, dtype=torch.float32),
        )

    work_dtype = torch.float64 if upcast else q.dtype
    group_size = heads // k.shape[2]
    query = q.to(work_dtype).transpose(1, 2)
    key = k.to(work_dtype).repeat_interleave(group_size, dim=2).transpose(1, 2)
    value = v.to(work_dtype).repeat_interleave(group_size, dim=2).transpose(1, 2)
    scores = (query if upcast else query * scale) @ key.transpose(-1, -2)
    if qv is not None:
        query_content = qv.to(work_dtype).transpose(1, 2)
        scores = scores + (query_content if upcast else query_content * scale) @ value.transpose(
            -1, -2
        )
    if upcast:
        scores = scores * scale
    else:
        scores = scores.float()
    if softcap > 0:
        scores = softcap * torch.tanh(scores / softcap)
    scores = scores.masked_fill(mask, -torch.inf)
    lse = torch.logsumexp(scores, dim=-1).float()
    empty_rows = mask.all(dim=-1)
    # Avoid softmax(-inf, ..., -inf) without hiding NaNs in valid input rows.
    safe_scores = scores.masked_fill(empty_rows[:, None], 0)
    if upcast:
        weights = torch.exp(safe_scores - safe_scores.amax(dim=-1, keepdim=True))
        weights = weights.masked_fill(empty_rows[:, None], 0)
        # Divide after PV: for uniform scores this preserves exact sums of
        # FP16/BF16 values until the final division and output rounding.
        out = (weights @ value) / weights.sum(dim=-1, keepdim=True).clamp_min(1)
    else:
        probabilities = torch.softmax(safe_scores, dim=-1)
        probabilities = probabilities.masked_fill(empty_rows[:, None], 0)
        out = probabilities.to(work_dtype) @ value
    out = out.transpose(1, 2).to(q.dtype)
    return out, lse


def mla_reference_pair(q, k, v, qv=None, **kwargs):
    """Return (out_ref, lse_ref, out_pt, lse_pt) for assert_fa_close."""
    out_ref, lse_ref = mla_reference(q, k, v, qv, upcast=True, **kwargs)
    out_pt, lse_pt = mla_reference(q, k, v, qv, upcast=False, **kwargs)
    return out_ref, lse_ref, out_pt, lse_pt
