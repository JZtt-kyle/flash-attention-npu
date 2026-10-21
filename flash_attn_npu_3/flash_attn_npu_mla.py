# Copyright (c) 2023, Tri Dao.
# Modified by Minghua Shen, 2026

"""MLA adaptation and fake implementations for the Ascend 910B/C backend."""

from typing import Optional, Tuple

import math
import torch


if torch.__version__ >= "2.4.0":
    register_fake = torch.library.register_fake
else:

    def register_fake(op, fn=None, /, *, lib=None, _stacklevel=1):
        def wrap(func):
            return func

        return wrap if fn is None else fn


def _mla_fake_outputs(q, v, num_splits=None, partials=True, outputs=None):
    lse_shape = (q.shape[0], q.shape[2], q.shape[1]) if q.ndim == 4 else (q.shape[1], q.shape[0])
    if outputs is None:
        out = q.new_empty((*q.shape[:-1], v.shape[-1]))
        lse = q.new_empty(lse_shape, dtype=torch.float32)
    else:
        out, lse = outputs
    if num_splits is None:
        return out, lse
    if num_splits == 1 or not partials:
        return out, lse, torch.empty(0, device=q.device), torch.empty(0, device=q.device)
    capacity = 26 if num_splits == 0 else num_splits
    return (
        out,
        lse,
        q.new_empty((capacity, *lse_shape, v.shape[-1]), dtype=torch.float32),
        q.new_empty((capacity, *lse_shape), dtype=torch.float32),
    )


def _check_mla_layout(tensor):
    # Match BuildBsnd/TndTensorLayout, including the entire allocation when used
    # lengths are shorter. Untyped storage size is bytes and works in fake/compile.
    sizes, strides = tensor.shape, tensor.stride()
    layout = "TND" if tensor.ndim == 3 else "BSND"
    message = f"MLA: {layout} input view exceeds stride, storage or signed byte bounds"
    if any(stride < 0 for stride in strides[:-1]) or strides[-1] != 1:
        raise RuntimeError(message)
    offset = tensor.storage_offset()
    if offset < 0:
        raise RuntimeError(message)
    if any(size == 0 for size in sizes[:-2]):
        return
    span = sizes[-1] + sum((n - 1) * s for n, s in zip(sizes[:-1], strides[:-1]))
    end = (offset + span) * 2
    if end > (1 << 63) - 1 or end > tensor.untyped_storage().size():
        raise RuntimeError(message)


def check_inputs(q, k, v, qv=None, *, varlen_q=False, varlen_k=False, paged_kv=False):
    """Recognize validated MLA before copies or ordinary FA dispatch.

    Uses only tensor metadata, including for fake tensors. Keep in sync with
    ascend910/flash_attn_npu_3/mla/mla_validation.h. Outer strides and storage
    offsets are legal for BSND, TND and mixed views; no view is materialized here.
    """
    # Keep ordinary layout and malformed-query validation in the upstream path.
    if qv is None and (
        (varlen_q and q.ndim != 3) or (q.ndim > 0 and v.ndim > 0 and q.shape[-1] == v.shape[-1])
    ):
        return False

    def check(condition, message):
        if not condition:
            raise RuntimeError(message)

    check(q.ndim == (3 if varlen_q else 4), "FlashAttention: q rank must match BSND/TND layout")
    check(
        k.ndim == (3 if varlen_k else 4) and v.ndim == k.ndim,
        "FlashAttention: k/v rank must match dense, TND or paged KV layout",
    )
    check(q.shape[-1] == k.shape[-1], "FlashAttention: q and k head dimensions must match")
    check(not paged_kv or not varlen_k, "MLA: page_table and cu_seqlens_k cannot be combined")
    check(q.dtype in (torch.float16, torch.bfloat16), "MLA: inputs must have FP16 or BF16 dtype")
    check(q.device.type == "npu", "MLA: inputs must be NPU tensors")
    for tensor in (q, k, v) + (() if qv is None else (qv,)):
        check(tensor.dtype == q.dtype, "MLA: all inputs must have the same dtype")
        check(tensor.device == q.device, "MLA: all inputs must be on the same device")
        check(
            tensor.ndim > 0 and tensor.stride(-1) == 1,
            "MLA: inputs must have contiguous last dimension",
        )
    check(k.shape[:-1] == v.shape[:-1], "MLA: k/v shapes must match except for head dimension")
    heads_q, heads_kv = q.shape[-2], k.shape[-2]
    check(
        heads_q > 0 and heads_kv > 0 and heads_q % heads_kv == 0,
        "MLA: positive KV head count must divide query head count",
    )
    if qv is not None:
        check(
            qv.shape == (*q.shape[:-1], v.shape[-1]),
            "MLA: qv shape must match q with the value head dimension",
        )
        check(
            q.shape[-1] == 64 and v.shape[-1] == 512,
            "MLA: supported qv representation requires Dq=64 and Dv=512",
        )
        check(heads_kv == 1, "MLA: compressed representation requires one KV head")
        check(
            varlen_q or not varlen_k,
            "MLA compressed forward is not implemented yet for BSND Q with TND KV",
        )
        check(paged_kv or varlen_q or q.shape[0] == k.shape[0], "MLA: dense batch sizes must match")
        check(
            varlen_q
            or _fits_mla_dense_shape(
                q.shape[0], q.shape[1], heads_q, 0 if paged_kv else k.shape[1]
            ),
            "MLA compressed forward is not implemented yet outside B>=0, nonnegative S1G within "
            "address/field bounds, 0<=Sk<=16384",
        )
        for tensor in (q, k, v, qv):
            # Dynamo cannot trace storage_offset() in the public wrapper.
            # Registered fake kernels and the real C++ op validate storage
            # inside the operator boundary, including compiled execution.
            if not torch.compiler.is_compiling():
                _check_mla_layout(tensor)
            check(
                not tensor.requires_grad,
                "MLA compressed forward is not implemented yet for gradients",
            )
        return True
    check(
        q.shape[-1] == 192 and v.shape[-1] == 128,
        "MLA: supported representation without qv requires Dq=192 and Dv=128",
    )
    check(
        varlen_q or not varlen_k,
        "MLA prefill forward is not implemented yet for BSND Q with TND KV",
    )
    check(paged_kv or varlen_q or q.shape[0] == k.shape[0], "MLA: dense batch sizes must match")
    check(
        varlen_q
        or _fits_mla_dense_shape(q.shape[0], q.shape[1], heads_q, 0 if paged_kv else k.shape[1]),
        "MLA: prefill BSND shape bounds exceeded",
    )
    for tensor in (q, k, v):
        if not torch.compiler.is_compiling():
            _check_mla_layout(tensor)
        check(not tensor.requires_grad, "MLA prefill forward is not implemented yet for gradients")
    return True


def _check_mla_tnd_inputs(q, k, cu_q, cu_k, max_q, max_k):
    if max_q is None or max_k is None:
        raise RuntimeError("MLA: compact TND requires max_seqlen_q/k")
    if (
        cu_q is None
        or cu_k is None
        or cu_q.ndim != 1
        or cu_q.numel() < 1
        or cu_k.ndim != 1
        or cu_k.numel() != cu_q.numel()
    ):
        raise RuntimeError("MLA: cu_seqlens_q/k must have matching shape [batch_size + 1]")
    for tensor in (cu_q, cu_k):
        if tensor.dtype != torch.int32:
            raise RuntimeError("MLA: cu_seqlens_q/k must have dtype int32")
        if tensor.device != q.device:
            raise RuntimeError("MLA: cu_seqlens_q/k must be on the same NPU device")
        if not tensor.is_contiguous():
            raise RuntimeError("MLA: cu_seqlens_q/k must be contiguous")
    if cu_q.numel() - 1 > ((1 << 32) - 1 - 405) // 4:
        raise RuntimeError("MLA: TND metadata word count overflow")
    if not (
        q.shape[0] <= (1 << 31) - 1
        and k.shape[0] <= (1 << 31) - 1
        and q.shape[1] <= ((1 << 32) - 1) // 512
        and 0 <= max_q <= (1 << 30)
        and 0 <= max_k <= 16384
        and max_q * q.shape[1] <= (1 << 32) - 256
        and q.shape[0] * q.shape[1] <= ((1 << 63) - 1) // 1024
    ):
        raise RuntimeError("MLA: compact TND shape or max_seqlen bounds exceeded")


def _check_mla_dense_options(
    *,
    softmax_scale=None,
    causal=False,
    allow_causal=False,
    allow_window=False,
    window_size=(-1, -1),
    attention_chunk=0,
    softcap=0.0,
    num_splits=1,
    pack_gqa=None,
    sm_margin=0,
    rotary_interleaved=True,
    optional_tensors=(),
    allow_split=False,
    allow_softcap=False,
):
    _check_mla_gqa_options(pack_gqa, sm_margin)
    if any(t is not None for t in optional_tensors):
        raise RuntimeError(
            "MLA compressed forward is not implemented yet for optional tensors or length/rotary options"
        )
    for name, unsupported in (
        ("causal", causal and not allow_causal),
        ("window_size", window_size != (-1, -1) and not allow_window),
        ("attention_chunk", attention_chunk != 0),
        ("softcap", softcap != 0.0 and not allow_softcap),
        ("num_splits", num_splits != 1 and not (allow_split and 0 <= num_splits <= 256)),
        ("rotary_interleaved", not rotary_interleaved),
    ):
        if unsupported:
            raise RuntimeError(f"MLA compressed forward is not implemented yet for {name}")
    if not torch.compiler.is_compiling() and (
        not math.isfinite(softcap)
        or softcap < 0
        or softcap > 3.4028234663852886e38
        or (0 < softcap < 2**-149)
    ):
        raise RuntimeError("MLA: softcap must be finite and non-negative in FP32")
    # The fake/real op validates scale inside the dispatcher. A symbolic Python
    # float cannot be passed to math.isfinite while tracing the public wrapper.
    if (
        not torch.compiler.is_compiling()
        and softmax_scale is not None
        and (not math.isfinite(softmax_scale) or abs(softmax_scale) > 3.4028234663852886e38)
    ):
        raise RuntimeError("MLA: softmax_scale must be finite in FP32")


def _check_mla_mixed_inputs(q, k, cu_q, max_q, cache_seqlens, metadata, seqused_q=None):
    if max_q is None:
        raise RuntimeError("MLA: mixed TND/BSND requires max_seqlen_q")
    if seqused_q is not None and metadata is not None:
        raise RuntimeError(
            "MLA compressed forward is not implemented yet for Q-used with explicit metadata"
        )
    batch = k.shape[0]
    if batch > ((1 << 32) - 1 - 406) // 5:
        raise RuntimeError("MLA: mixed metadata word count overflow")
    if not (
        _fits_mla_dense_shape(0, max_q, q.shape[1], k.shape[1])
        and q.shape[0] <= (1 << 31) - 1
        and q.shape[0] * q.shape[1] <= ((1 << 63) - 1) // 1024
    ):
        raise RuntimeError("MLA: mixed shape or max_seqlen bounds exceeded")
    if cu_q.dtype != torch.int32:
        raise RuntimeError("MLA: cu_seqlens_q must have dtype int32")
    if cu_q.device != q.device:
        raise RuntimeError("MLA: cu_seqlens_q must be on the same NPU device")
    if cu_q.ndim != 1 or cu_q.numel() != batch + 1:
        raise RuntimeError("MLA: cu_seqlens_q must have shape [batch_size + 1]")
    if not cu_q.is_contiguous():
        raise RuntimeError("MLA: cu_seqlens_q must be contiguous")
    if cache_seqlens is not None:
        _check_mla_cache_lengths(cache_seqlens, batch, q.device)
    if seqused_q is not None:
        _check_mla_used(q, batch, seqused_q, cache_seqlens)
    if metadata is not None:
        if metadata.dtype != torch.int32:
            raise RuntimeError("MLA: scheduler_metadata must have dtype int32")
        if metadata.device != q.device:
            raise RuntimeError("MLA: scheduler_metadata must be on the same NPU device")
        if metadata.ndim != 1 or metadata.numel() != 406 + 5 * batch:
            raise RuntimeError("MLA: scheduler_metadata must have shape [406 + 5 * batch_size]")
        if not metadata.is_contiguous():
            raise RuntimeError("MLA: scheduler_metadata must be contiguous")


def _check_mla_causal_mixed_inputs(q, k, cu_q, max_q, lengths, metadata, seqused_q=None):
    _check_mla_mixed_inputs(q, k, cu_q, max_q, lengths, None, seqused_q)
    _check_mla_used(q, k.shape[0], seqused_q, lengths)
    if seqused_q is not None and metadata is not None:
        raise RuntimeError(
            "MLA compressed forward is not implemented yet for Q-used with explicit metadata"
        )
    if metadata is not None and (
        metadata.dtype != torch.int32
        or metadata.device != q.device
        or metadata.ndim != 1
        or metadata.numel() != 410 + 8 * k.shape[0]
        or not metadata.is_contiguous()
    ):
        raise RuntimeError(
            "MLA: causal scheduler_metadata must be same-device contiguous int32[410 + 8 * batch_size]"
        )


def _check_mla_window_mixed_inputs(
    q, k, cu_q, max_q, lengths, metadata, seqused_q=None, *, window_left, window_right, causal
):
    _check_mla_mixed_inputs(q, k, cu_q, max_q, lengths, None, seqused_q)
    _check_mla_used(q, k.shape[0], seqused_q, lengths)
    if seqused_q is not None and metadata is not None:
        raise RuntimeError(
            "MLA compressed forward is not implemented yet for Q-used with explicit metadata"
        )
    if metadata is not None and (
        metadata.dtype != torch.int32
        or metadata.device != q.device
        or metadata.ndim != 1
        or metadata.numel() != _mla_window_bsnd_words(k.shape[0])
        or not metadata.is_contiguous()
    ):
        raise RuntimeError(
            "MLA: window scheduler_metadata must be same-device contiguous int32[413 + 8 * batch_size]"
        )

    _mla_window_bsnd_words(k.shape[0])
    _normalize_mla_window(max_q, k.shape[1], window_left, window_right, causal)


def check_metadata(headdim, headdim_v):
    if headdim == headdim_v:
        return False
    # Select the schedule family; forward validates the representation.
    if (headdim, headdim_v) not in ((64, 512), (192, 128)):
        raise RuntimeError(
            "NPU FlashAttention MLA scheduler metadata is not implemented yet for these head dimensions"
        )
    return True


def _check_mla_cache_lengths(lengths, batch, device):
    if lengths.dtype != torch.int32:
        raise RuntimeError("MLA: cache_seqlens must have dtype int32")
    if lengths.device.type != "npu" or lengths.device != device:
        raise RuntimeError("MLA: cache_seqlens must be on the same NPU device")
    if lengths.ndim != 1 or lengths.shape[0] != batch:
        raise RuntimeError("MLA: cache_seqlens must have shape [batch_size]")
    if not lengths.is_contiguous():
        raise RuntimeError("MLA: cache_seqlens must be contiguous")


def _check_mla_bsnd_used(q, seqused_q, seqused_k, metadata):
    if metadata is not None:
        raise RuntimeError(
            "MLA compressed forward is not implemented yet for Q-used with explicit metadata"
        )
    _check_mla_used(q, q.shape[0], seqused_q, seqused_k)


def _check_mla_used(q, batch, seqused_q, seqused_k):
    if batch > ((1 << 32) - 1 - 410) // 8:
        raise RuntimeError("MLA: used metadata word count overflow")
    for name, tensor in (("seqused_q", seqused_q), ("seqused_k", seqused_k)):
        if tensor is None:
            continue
        if tensor.dtype != torch.int32:
            raise RuntimeError(f"MLA: {name} must have dtype int32")
        if tensor.device != q.device:
            raise RuntimeError(f"MLA: {name} must be on the same NPU device")
        if tensor.ndim != 1 or tensor.numel() != batch:
            raise RuntimeError(f"MLA: {name} must have shape [batch_size]")
        if not tensor.is_contiguous():
            raise RuntimeError(f"MLA: {name} must be contiguous")


def _normalize_mla_window(sq, sk, left, right, causal):
    # Fixed FA3 host rules. Call-level bounds are separate from device used.
    if not 0 <= sq <= 2**30 or not 0 <= sk <= 16384:
        raise RuntimeError("MLA: window bounds exceed the supported normalization domain")
    left = -1 if left >= sk - 1 else left
    right = -1 if right >= sq - 1 else right
    if causal and not (sq == 1 and left == -1 and right == -1):
        right = 0
    if left < -(2**31) or right < -(2**31):
        raise RuntimeError("MLA: window bounds exceed the supported normalization domain")
    kind = 1 if left < 0 and right == 0 else 2 if left >= 0 or right >= 0 else 0
    return kind, sk - 1 if left < 0 else left, sq - 1 if right < 0 else right


def _mla_window_bsnd_words(batch):
    if not 0 <= batch <= (2**32 - 1 - 413) // 8:
        raise RuntimeError("MLA: window metadata word count overflow")
    return 413 + 8 * batch


def _check_mla_window_bsnd_metadata(q, k, lengths, used_q, metadata, left, right, causal):
    _normalize_mla_window(q.shape[1], k.shape[1], left, right, causal)
    _check_mla_used(q, q.shape[0], used_q, lengths)
    if used_q is not None and metadata is not None:
        raise RuntimeError(
            "MLA compressed forward is not implemented yet for Q-used with explicit metadata"
        )
    words = _mla_window_bsnd_words(q.shape[0])
    if metadata is not None and (
        metadata.dtype != torch.int32
        or metadata.device != q.device
        or metadata.dim() != 1
        or metadata.numel() != words
        or not metadata.is_contiguous()
    ):
        raise RuntimeError(
            "MLA: window scheduler_metadata must be same-device contiguous int32[413 + 8 * batch_size]"
        )


def _mla_causal_bsnd_words(batch):
    if not 0 <= batch <= (2**32 - 1 - 410) // 8:
        raise RuntimeError("MLA: causal BSND metadata word count overflow")
    return 410 + 8 * batch


def _check_mla_causal_bsnd_metadata(q, cache_seqlens, seqused_q, metadata):
    if seqused_q is not None:
        _check_mla_bsnd_used(q, seqused_q, cache_seqlens, metadata)
    else:
        _check_mla_used(q, q.shape[0], None, cache_seqlens)
    words = _mla_causal_bsnd_words(q.shape[0])
    if metadata is not None:
        if (
            metadata.dtype != torch.int32
            or metadata.device != q.device
            or metadata.ndim != 1
            or metadata.shape[0] != words
            or not metadata.is_contiguous()
        ):
            raise RuntimeError(
                "MLA: causal scheduler_metadata must be same-device contiguous int32[410 + 8 * batch_size]"
            )


def _check_mla_forward_metadata(q, cache_seqlens, metadata):
    if cache_seqlens is not None or metadata is not None:
        _mla_bsnd_metadata_words(q.shape[0])
    if cache_seqlens is not None:
        _check_mla_cache_lengths(cache_seqlens, q.shape[0], q.device)
    if metadata is not None:
        if metadata.dtype != torch.int32:
            raise RuntimeError("MLA: scheduler_metadata must have dtype int32")
        if metadata.device != q.device:
            raise RuntimeError("MLA: scheduler_metadata must be on the same NPU device")
        if metadata.ndim != 1 or metadata.shape[0] != _mla_bsnd_metadata_words(q.shape[0]):
            raise RuntimeError("MLA: scheduler_metadata must have shape [401 + batch_size]")
        if not metadata.is_contiguous():
            raise RuntimeError("MLA: scheduler_metadata must be contiguous")


def _check_mla_paged_inputs(
    q,
    k,
    page_table,
    cache_seqlens,
    metadata,
    seqused_q=None,
    cu_seqlens_q=None,
    max_seqlen_q=None,
    window_left=-1,
    window_right=-1,
    causal=False,
    num_splits=1,
    has_softcap=False,
):
    # Mirror both paged plans and PagedForward's descriptor storage checks.
    # Physical pages and total TND tokens are independent of logical batch.
    tnd = cu_seqlens_q is not None
    is_window = (window_left, window_right) != (-1, -1)
    prefix_words = (
        756
        if num_splits == 0
        else 414
        if (num_splits != 1 or has_softcap)
        else 411
        if is_window
        else 408
    )
    sequence_words = 7 if (num_splits != 1 or has_softcap) else 5
    prefill = q.shape[-1] == 192
    if prefill:
        prefix_words, sequence_words = 372, (8 if tnd else 2) + 2 * k.shape[-2]
    batch, page = q.shape[0], k.shape[1]
    if tnd:
        if max_seqlen_q is None:
            raise RuntimeError("MLA: TND paged forward requires max_seqlen_q")
        if (
            cu_seqlens_q.ndim != 1
            or cu_seqlens_q.numel() < 1
            or cu_seqlens_q.dtype != torch.int32
            or not cu_seqlens_q.is_contiguous()
            or cu_seqlens_q.device != q.device
        ):
            raise RuntimeError(
                "MLA: cu_seqlens_q must be same-device contiguous int32[batch_size + 1]"
            )
        batch = cu_seqlens_q.numel() - 1
    if page_table.ndim != 2 or page_table.dtype != torch.int32 or page_table.shape[0] != batch:
        raise RuntimeError(
            "MLA: page_table must have shape [batch_size, max_pages] and dtype int32"
        )
    if page_table.device != q.device:
        raise RuntimeError("MLA: page_table must be on the same NPU device")
    capacity = page_table.shape[1] * page
    if (
        batch > ((1 << 32) - 1 - prefix_words) // sequence_words
        or k.shape[0] > (1 << 31)
        or not 16 <= page <= 1024
        or page % 16 != 0
        or not _fits_mla_dense_shape(
            0 if tnd else batch, max_seqlen_q if tnd else q.shape[1], q.shape[-2], capacity
        )
        or (
            tnd
            and (
                q.shape[0] > (1 << 31) - 1
                or q.shape[0] * q.shape[1] > ((1 << 63) - 1) // 1024
                or q.shape[0] > batch * max_seqlen_q
            )
        )
    ):
        raise RuntimeError("MLA: paged shape bounds or page size exceeded")
    if prefill:
        _mla_prefill_bsnd_budget(
            batch,
            max_seqlen_q if tnd else q.shape[1],
            capacity,
            q.shape[-2],
            k.shape[-2],
            num_splits,
            window_left,
            window_right,
            causal,
            total_q=q.shape[0] if tnd else None,
            total_k=batch * capacity if tnd else None,
            page_size=page,
        )
    elif num_splits != 1 or has_softcap:
        _check_mla_paged_split_budget(
            batch,
            max_seqlen_q if tnd else q.shape[1],
            q.shape[-2],
            capacity,
            page,
            num_splits,
            total_q=q.shape[0] if tnd else None,
            has_softcap=has_softcap,
        )
    if is_window or (num_splits != 1 or has_softcap):
        _normalize_mla_window(
            max_seqlen_q if tnd else q.shape[1], capacity, window_left, window_right, causal
        )
    if page_table.stride(0) < 0 or page_table.stride(1) != 1:
        raise RuntimeError(
            "MLA: page_table requires nonnegative row stride and contiguous last dimension"
        )
    if seqused_q is not None and metadata is not None and not (prefill and tnd):
        raise RuntimeError("MLA: Q-used paged metadata cannot be explicitly reused")
    if cache_seqlens is not None:
        _check_mla_cache_lengths(cache_seqlens, batch, q.device)
    if seqused_q is not None:
        if (
            seqused_q.dtype != torch.int32
            or seqused_q.ndim != 1
            or seqused_q.numel() != batch
            or not seqused_q.is_contiguous()
            or seqused_q.device != q.device
        ):
            raise RuntimeError("MLA: seqused_q must be same-device contiguous int32[batch_size]")
    if metadata is not None:
        if (
            metadata.dtype != torch.int32
            or metadata.ndim != 1
            or metadata.numel() != prefix_words + sequence_words * batch
            or not metadata.is_contiguous()
            or metadata.device != q.device
        ):
            raise RuntimeError(
                "MLA: scheduler_metadata must be same-device contiguous int32 with the selected paged protocol size"
            )
    if not torch.compiler.is_compiling():
        for tensor in (page_table, cache_seqlens, seqused_q, metadata, cu_seqlens_q):
            if tensor is None:
                continue
            offset = tensor.storage_offset()
            span = (
                0
                if tensor.numel() == 0
                else 1 + sum((n - 1) * s for n, s in zip(tensor.shape, tensor.stride()))
            )
            if offset < 0 or (
                span
                and (
                    offset + span > ((1 << 63) - 1) // 4
                    or (offset + span) * 4 > tensor.untyped_storage().size()
                )
            ):
                raise RuntimeError("MLA: paged descriptor exceeds storage or signed byte bounds")
    return batch


@register_fake("flash_attn_npu_3::_mla_paged_bsnd_forward")
def _mla_paged_bsnd_forward_fake(
    q,
    k,
    v,
    qv,
    page_table,
    softmax_scale=None,
    cache_seqlens=None,
    seqused_q=None,
    scheduler_metadata=None,
    causal=False,
    window_left=-1,
    window_right=-1,
):
    check_inputs(q, k, v, qv, paged_kv=True)
    _check_mla_dense_options(softmax_scale=softmax_scale)
    _check_mla_paged_inputs(
        q,
        k,
        page_table,
        cache_seqlens,
        scheduler_metadata,
        seqused_q,
        window_left=window_left,
        window_right=window_right,
        causal=causal,
    )
    return _mla_fake_outputs(q, v)


@register_fake("flash_attn_npu_3::_mla_paged_tnd_forward")
def _mla_paged_tnd_forward_fake(
    q,
    k,
    v,
    qv,
    page_table,
    cu_seqlens_q,
    max_seqlen_q,
    softmax_scale=None,
    cache_seqlens=None,
    seqused_q=None,
    scheduler_metadata=None,
    causal=False,
    window_left=-1,
    window_right=-1,
):
    check_inputs(q, k, v, qv, varlen_q=True, paged_kv=True)
    _check_mla_dense_options(softmax_scale=softmax_scale, causal=causal, allow_causal=True)
    _check_mla_paged_inputs(
        q,
        k,
        page_table,
        cache_seqlens,
        scheduler_metadata,
        seqused_q,
        cu_seqlens_q,
        max_seqlen_q,
        window_left,
        window_right,
        causal,
    )
    return _mla_fake_outputs(q, v)


def _check_mla_paged_split_budget(
    batch, sq, heads, capacity, page, splits, total_q=None, has_softcap=False
):
    if splits != 0 and not (1 if has_softcap else 2) <= splits <= 256:
        raise RuntimeError("MLA: paged split requires num_splits=0 or [2,256]")
    fixed = 756 if splits == 0 else 414
    count = 26 if splits == 0 else splits
    if not 0 <= batch <= (2**32 - 1 - fixed) // 7:
        raise RuntimeError("MLA: paged split metadata word count overflow")
    if (
        (count * heads * max(total_q, 1) * 512 * 4 > 2**63 - 1)
        if total_q is not None
        else (
            count * max(batch, 1) * heads * max(sq, 1) * 512 > 2**63 - 1
            or count * batch * heads * sq * 512 * 4 > 2**63 - 1
        )
    ):
        raise RuntimeError("MLA: paged split accumulator overflow")
    if batch and sq and capacity and (total_q is None or total_q) and 16 <= page <= 1024:
        base = 512 // page * page if page < 512 and 512 % page else 512
        m = ((min(sq * heads, 256) + 15) // 16) * 16
        n = ((min(capacity, base) + 31) // 32) * 32
        reserve = 2 * 26 * (6 * m * n + 8 * m * 512 + 8 * 256)
        rows = (3 * 26 if splits == 0 else batch * ((sq * heads + 255) // 256) * splits) * 256
        if total_q is not None and splits != 0:
            rows = (total_q * heads + min(batch, total_q) * 255) * splits
        if reserve + rows * 528 * 4 > 2**63 - 1:
            raise RuntimeError("MLA: paged split workspace overflow")


@register_fake("flash_attn_npu_3::_mla_split_paged_bsnd_forward")
def _mla_split_paged_bsnd_forward_fake(
    q,
    k,
    v,
    qv,
    page_table,
    num_splits,
    window_left=-1,
    window_right=-1,
    causal=False,
    softmax_scale=None,
    cache_seqlens=None,
    seqused_q=None,
    scheduler_metadata=None,
    softcap=0.0,
    partials=True,
):
    check_inputs(q, k, v, qv, paged_kv=True)
    _check_mla_dense_options(softmax_scale=softmax_scale, softcap=softcap, allow_softcap=True)
    _check_mla_paged_inputs(
        q,
        k,
        page_table,
        cache_seqlens,
        scheduler_metadata,
        seqused_q,
        window_left=window_left,
        window_right=window_right,
        causal=causal,
        num_splits=num_splits,
        has_softcap=softcap > 0,
    )
    _check_mla_paged_split_budget(
        q.shape[0],
        q.shape[1],
        q.shape[2],
        page_table.shape[1] * k.shape[1],
        k.shape[1],
        num_splits,
        has_softcap=softcap > 0,
    )
    return _mla_fake_outputs(q, v, num_splits, partials)


@register_fake("flash_attn_npu_3::_mla_split_paged_tnd_forward")
def _mla_split_paged_tnd_forward_fake(
    q,
    k,
    v,
    qv,
    page_table,
    cu_seqlens_q,
    max_seqlen_q,
    num_splits,
    window_left=-1,
    window_right=-1,
    causal=False,
    softmax_scale=None,
    cache_seqlens=None,
    seqused_q=None,
    scheduler_metadata=None,
    softcap=0.0,
    partials=True,
):
    _check_mla_dense_options(softmax_scale=softmax_scale, softcap=softcap, allow_softcap=True)
    if num_splits != 0 and not (1 if softcap > 0 else 2) <= num_splits <= 256:
        raise RuntimeError("MLA: TND paged split requires num_splits=0 or [2,256]")
    check_inputs(q, k, v, qv, varlen_q=True, paged_kv=True)
    _check_mla_paged_inputs(
        q,
        k,
        page_table,
        cache_seqlens,
        scheduler_metadata,
        seqused_q,
        cu_seqlens_q,
        max_seqlen_q,
        window_left,
        window_right,
        causal,
        num_splits,
        has_softcap=softcap > 0,
    )
    return _mla_fake_outputs(q, v, num_splits, partials)


def _mla_split_bsnd_words(batch, splits=2):
    fixed = 758 if splits == 0 else 416
    if not 0 <= batch <= (2**32 - 1 - fixed) // 10:
        raise RuntimeError("MLA: split metadata word count overflow")
    return fixed + 10 * batch


def _check_mla_split_bsnd(
    q, k, lengths, used_q, metadata, splits, left, right, causal, softcap=0.0
):
    if splits != 0 and not (1 if softcap > 0 else 2) <= splits <= 256:
        raise RuntimeError("MLA: split requires num_splits=0 or [2,256]")
    batch, sq, heads = q.shape[:3]
    _normalize_mla_window(sq, k.shape[1], left, right, causal)
    _check_mla_used(q, batch, used_q, lengths)
    words = _mla_split_bsnd_words(batch, splits)
    if metadata is not None:
        if used_q is not None:
            raise RuntimeError(
                "MLA compressed forward is not implemented yet for Q-used with explicit metadata"
            )
        if (
            metadata.device != q.device
            or metadata.dtype != torch.int32
            or metadata.ndim != 1
            or metadata.numel() != words
            or not metadata.is_contiguous()
        ):
            raise RuntimeError(
                "MLA: split scheduler_metadata must be same-device contiguous int32 with the selected protocol size"
            )
    if splits == 0:
        # Static public capacity, independent of device used lengths/native FD count.
        if 26 * max(batch, 1) * heads * max(sq, 1) * 512 > 2**63 - 1:
            raise RuntimeError("MLA: auto split accumulator stride overflow")
        if 26 * batch * heads * sq * 512 * 4 > 2**63 - 1:
            raise RuntimeError("MLA: auto split workspace or accumulator overflow")
        return
    # Match the host's conservative M=256,N=512 envelope before allocating fake tensors.
    if batch and sq and k.shape[1]:
        rows = batch * ((sq * heads + 255) // 256) * splits * 256
        if rows * 528 * 4 + 2 * 26 * (6 * 256 * 512 + 8 * 256 * 512 + 8 * 256) > 2**63 - 1:
            raise RuntimeError("MLA: split workspace or accumulator overflow")
    if splits * batch * heads * sq * 512 * 4 > 2**63 - 1:
        raise RuntimeError("MLA: split workspace or accumulator overflow")


@register_fake("flash_attn_npu_3::_mla_split_bsnd_forward")
def _mla_split_bsnd_forward_fake(
    q,
    k,
    v,
    qv,
    num_splits,
    window_left=-1,
    window_right=-1,
    causal=False,
    softmax_scale=None,
    cache_seqlens=None,
    seqused_q=None,
    scheduler_metadata=None,
    softcap=0.0,
    partials=True,
):
    check_inputs(q, k, v, qv)
    _check_mla_dense_options(softmax_scale=softmax_scale, softcap=softcap, allow_softcap=True)
    _check_mla_split_bsnd(
        q,
        k,
        cache_seqlens,
        seqused_q,
        scheduler_metadata,
        num_splits,
        window_left,
        window_right,
        causal,
        softcap,
    )
    return _mla_fake_outputs(q, v, num_splits, partials)


@register_fake("flash_attn_npu_3::_mla_window_bsnd_forward")
def _mla_window_bsnd_forward_fake(
    q,
    k,
    v,
    qv,
    window_left,
    window_right,
    causal,
    softmax_scale=None,
    cache_seqlens=None,
    seqused_q=None,
    scheduler_metadata=None,
):
    check_inputs(q, k, v, qv)
    _check_mla_dense_options(softmax_scale=softmax_scale)
    _check_mla_window_bsnd_metadata(
        q, k, cache_seqlens, seqused_q, scheduler_metadata, window_left, window_right, causal
    )
    return _mla_fake_outputs(q, v)


@register_fake("flash_attn_npu_3::_mla_window_bsnd_cache")
def _mla_window_bsnd_cache_fake(
    q,
    k,
    v,
    qv,
    window_left,
    window_right,
    causal,
    softmax_scale=None,
    cache_seqlens=None,
    scheduler_metadata=None,
):
    out, lse = _mla_window_bsnd_forward_fake(
        q,
        k,
        v,
        qv,
        window_left,
        window_right,
        causal,
        softmax_scale,
        cache_seqlens,
        None,
        scheduler_metadata,
    )
    return out, lse, torch.empty(0, device=q.device), torch.empty(0, device=q.device)


@register_fake("flash_attn_npu_3::_mla_causal_bsnd_forward")
def _mla_causal_bsnd_forward_fake(
    q, k, v, qv, softmax_scale=None, cache_seqlens=None, seqused_q=None, scheduler_metadata=None
):
    check_inputs(q, k, v, qv)
    _check_mla_dense_options(softmax_scale=softmax_scale, causal=True, allow_causal=True)
    _check_mla_causal_bsnd_metadata(q, cache_seqlens, seqused_q, scheduler_metadata)
    return _mla_fake_outputs(q, v)


@register_fake("flash_attn_npu_3::_mla_causal_bsnd_cache")
def _mla_causal_bsnd_cache_fake(
    q, k, v, qv, softmax_scale=None, cache_seqlens=None, scheduler_metadata=None
):
    out, lse = _mla_causal_bsnd_forward_fake(
        q, k, v, qv, softmax_scale, cache_seqlens, None, scheduler_metadata
    )
    return out, lse, torch.empty(0, device=q.device), torch.empty(0, device=q.device)


@register_fake("flash_attn_npu_3::_mla_dense_forward")
def _mla_dense_forward_fake(
    q, k, v, qv, softmax_scale=None, cache_seqlens=None, scheduler_metadata=None
):
    check_inputs(q, k, v, qv)
    _check_mla_dense_options(softmax_scale=softmax_scale)
    _check_mla_forward_metadata(q, cache_seqlens, scheduler_metadata)
    return _mla_fake_outputs(q, v)


@register_fake("flash_attn_npu_3::_mla_tnd_forward")
def _mla_tnd_forward_fake(
    q,
    k,
    v,
    qv,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    softmax_scale=None,
    seqused_q=None,
    seqused_k=None,
):
    check_inputs(q, k, v, qv, varlen_q=True, varlen_k=True)
    _check_mla_dense_options(softmax_scale=softmax_scale)
    _check_mla_tnd_inputs(q, k, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)
    if seqused_q is not None or seqused_k is not None:
        _check_mla_used(q, cu_seqlens_q.numel() - 1, seqused_q, seqused_k)
    return _mla_fake_outputs(q, v)


@register_fake("flash_attn_npu_3::_mla_causal_tnd_forward")
def _mla_causal_tnd_forward_fake(
    q,
    k,
    v,
    qv,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    softmax_scale=None,
    seqused_q=None,
    seqused_k=None,
):
    result = _mla_tnd_forward_fake(
        q,
        k,
        v,
        qv,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        seqused_q,
        seqused_k,
    )
    # v8 always has allocation/used spans, including when used is implicit.
    _check_mla_used(q, cu_seqlens_q.numel() - 1, seqused_q, seqused_k)
    return result


@register_fake("flash_attn_npu_3::_mla_split_mixed_forward")
def _mla_split_mixed_forward_fake(
    q,
    k,
    v,
    qv,
    cu_seqlens_q,
    max_seqlen_q,
    num_splits,
    window_left=-1,
    window_right=-1,
    causal=False,
    softmax_scale=None,
    cache_seqlens=None,
    seqused_q=None,
    scheduler_metadata=None,
    softcap=0.0,
    partials=True,
):
    check_inputs(q, k, v, qv, varlen_q=True)
    _check_mla_dense_options(softmax_scale=softmax_scale, softcap=softcap, allow_softcap=True)
    _check_mla_mixed_inputs(q, k, cu_seqlens_q, max_seqlen_q, cache_seqlens, None, seqused_q)
    _check_mla_used(q, k.shape[0], seqused_q, cache_seqlens)
    if seqused_q is not None and scheduler_metadata is not None:
        raise RuntimeError("MLA: Q-used with explicit metadata is not supported")
    if num_splits != 0 and not (1 if softcap > 0 else 2) <= num_splits <= 256:
        raise RuntimeError("MLA: mixed split requires num_splits=0 or [2,256]")
    words = _mla_split_bsnd_words(k.shape[0], num_splits)
    _normalize_mla_window(max_seqlen_q, k.shape[1], window_left, window_right, causal)
    if scheduler_metadata is not None and (
        scheduler_metadata.dtype != torch.int32
        or scheduler_metadata.ndim != 1
        or scheduler_metadata.numel() != words
        or not scheduler_metadata.is_contiguous()
        or scheduler_metadata.device != q.device
    ):
        raise RuntimeError(
            "MLA: mixed split metadata must be same-device contiguous int32 with the selected protocol size"
        )
    capacity = 26 if num_splits == 0 else num_splits
    total, heads = q.shape[:2]
    limit = 2**63 - 1
    if capacity * heads * max(total, 1) * 512 * 4 > limit:
        raise RuntimeError("MLA: mixed split accumulator overflow")
    if total and k.shape[0] and k.shape[1]:
        active = min(k.shape[0], total)
        groups = min(
            active * ((max_seqlen_q * heads + 255) // 256), (total * heads + active * 255) // 256
        )
        if (2 * 26 + 26 if num_splits == 0 else groups * num_splits) * 256 * 528 * 4 + 2 * 26 * (
            6 * 256 * 512 + 8 * 256 * 512 + 8 * 256
        ) > limit:
            raise RuntimeError("MLA: mixed split workspace overflow")
    return _mla_fake_outputs(q, v, num_splits, partials)


@register_fake("flash_attn_npu_3::_mla_split_mixed_metadata")
def _mla_split_mixed_metadata_fake(
    batch_size,
    max_seqlen_q,
    max_seqlen_k,
    num_heads_q,
    qkv_dtype,
    cache_seqlens,
    cu_seqlens_q,
    window_left,
    window_right,
    causal,
    num_splits,
    has_softcap=False,
):
    _mla_tnd_bsnd_metadata_fake(
        batch_size, max_seqlen_q, max_seqlen_k, num_heads_q, qkv_dtype, cache_seqlens, cu_seqlens_q
    )
    _normalize_mla_window(max_seqlen_q, max_seqlen_k, window_left, window_right, causal)
    if num_splits != 0 and not (1 if has_softcap else 2) <= num_splits <= 256:
        raise RuntimeError("MLA: mixed split requires num_splits=0 or [2,256]")
    return cache_seqlens.new_empty((_mla_split_bsnd_words(batch_size, num_splits),))


@register_fake("flash_attn_npu_3::_mla_split_tnd_forward")
def _mla_split_tnd_forward_fake(
    q,
    k,
    v,
    qv,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    num_splits,
    window_left=-1,
    window_right=-1,
    causal=False,
    softmax_scale=None,
    seqused_q=None,
    seqused_k=None,
    scheduler_metadata=None,
    softcap=0.0,
    partials=True,
):
    _check_mla_dense_options(softcap=softcap, allow_softcap=True)
    out, lse = _mla_causal_tnd_forward_fake(
        q,
        k,
        v,
        qv,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        seqused_q,
        seqused_k,
    )
    if num_splits != 0 and not (1 if softcap > 0 else 2) <= num_splits <= 256:
        raise RuntimeError("MLA: TND split requires 0 or [2,256], or 1 with softcap")
    words = _mla_split_bsnd_words(cu_seqlens_q.numel() - 1, num_splits)
    _normalize_mla_window(max_seqlen_q, max_seqlen_k, window_left, window_right, causal)
    if scheduler_metadata is not None and (
        scheduler_metadata.dtype != torch.int32
        or scheduler_metadata.ndim != 1
        or scheduler_metadata.numel() != words
        or not scheduler_metadata.is_contiguous()
        or scheduler_metadata.device != q.device
    ):
        raise RuntimeError(
            "MLA: TND split metadata must be same-device contiguous int32 with the selected protocol size"
        )
    total, heads = q.shape[:2]
    batch = cu_seqlens_q.numel() - 1
    # Match the static host TND envelope without reading device prefixes/used.
    # Byte and empty-shape stride checks precede fake accumulator allocation.
    capacity = 26 if num_splits == 0 else num_splits
    limit = 2**63 - 1
    if capacity * heads * max(total, 1) * 512 > limit:
        raise RuntimeError("MLA: TND split accumulator stride overflow")
    if capacity * heads * total * 512 * 4 > limit:
        raise RuntimeError("MLA: TND split workspace or accumulator overflow")
    if num_splits != 0 and total and k.shape[0]:
        active = min(batch, total)
        groups = min(
            active * ((max_seqlen_q * heads + 255) // 256), (total * heads + active * 255) // 256
        )
        if (
            groups * num_splits * 256 * 528 * 4 + 2 * 26 * (6 * 256 * 512 + 8 * 256 * 512 + 8 * 256)
            > limit
        ):
            raise RuntimeError("MLA: TND split workspace or accumulator overflow")
    return _mla_fake_outputs(q, v, num_splits, partials, (out, lse))


@register_fake("flash_attn_npu_3::_mla_split_tnd_metadata")
def _mla_split_tnd_metadata_fake(
    q,
    k,
    v,
    qv,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    num_splits,
    window_left=-1,
    window_right=-1,
    causal=False,
    seqused_q=None,
    seqused_k=None,
    has_softcap=False,
):
    _mla_split_tnd_forward_fake(
        q,
        k,
        v,
        qv,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        num_splits,
        window_left,
        window_right,
        causal,
        None,
        seqused_q,
        seqused_k,
        softcap=1.0 if has_softcap else 0.0,
    )
    return q.new_empty(
        (_mla_split_bsnd_words(cu_seqlens_q.numel() - 1, num_splits),), dtype=torch.int32
    )


@register_fake("flash_attn_npu_3::_mla_window_tnd_forward")
def _mla_window_tnd_forward_fake(
    q,
    k,
    v,
    qv,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    window_left,
    window_right,
    causal,
    softmax_scale=None,
    seqused_q=None,
    seqused_k=None,
):
    result = _mla_causal_tnd_forward_fake(
        q,
        k,
        v,
        qv,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        seqused_q,
        seqused_k,
    )
    _mla_window_bsnd_words(cu_seqlens_q.numel() - 1)
    _normalize_mla_window(max_seqlen_q, max_seqlen_k, window_left, window_right, causal)
    return result


@register_fake("flash_attn_npu_3::_mla_window_mixed_forward")
def _mla_window_mixed_forward_fake(
    q,
    k,
    v,
    qv,
    cu_seqlens_q,
    max_seqlen_q,
    window_left,
    window_right,
    causal,
    softmax_scale=None,
    cache_seqlens=None,
    seqused_q=None,
    scheduler_metadata=None,
):
    check_inputs(q, k, v, qv, varlen_q=True)
    _check_mla_dense_options(softmax_scale=softmax_scale, causal=True, allow_causal=True)
    _check_mla_window_mixed_inputs(
        q,
        k,
        cu_seqlens_q,
        max_seqlen_q,
        cache_seqlens,
        scheduler_metadata,
        seqused_q,
        window_left=window_left,
        window_right=window_right,
        causal=causal,
    )
    return _mla_fake_outputs(q, v)


@register_fake("flash_attn_npu_3::_mla_window_mixed_cache")
def _mla_window_mixed_cache_fake(
    q,
    k,
    v,
    qv,
    cu_seqlens_q,
    max_seqlen_q,
    window_left,
    window_right,
    causal,
    softmax_scale=None,
    cache_seqlens=None,
    scheduler_metadata=None,
):
    out, lse = _mla_window_mixed_forward_fake(
        q,
        k,
        v,
        qv,
        cu_seqlens_q,
        max_seqlen_q,
        window_left,
        window_right,
        causal,
        softmax_scale,
        cache_seqlens,
        scheduler_metadata=scheduler_metadata,
    )
    return out, lse, torch.empty(0, device=q.device), torch.empty(0, device=q.device)


@register_fake("flash_attn_npu_3::_mla_causal_mixed_forward")
def _mla_causal_mixed_forward_fake(
    q,
    k,
    v,
    qv,
    cu_seqlens_q,
    max_seqlen_q,
    softmax_scale=None,
    cache_seqlens=None,
    seqused_q=None,
    scheduler_metadata=None,
):
    check_inputs(q, k, v, qv, varlen_q=True)
    _check_mla_dense_options(softmax_scale=softmax_scale, causal=True, allow_causal=True)
    _check_mla_causal_mixed_inputs(
        q, k, cu_seqlens_q, max_seqlen_q, cache_seqlens, scheduler_metadata, seqused_q
    )
    return _mla_fake_outputs(q, v)


@register_fake("flash_attn_npu_3::_mla_causal_mixed_cache")
def _mla_causal_mixed_cache_fake(
    q,
    k,
    v,
    qv,
    cu_seqlens_q,
    max_seqlen_q,
    softmax_scale=None,
    cache_seqlens=None,
    scheduler_metadata=None,
):
    out, lse = _mla_causal_mixed_forward_fake(
        q,
        k,
        v,
        qv,
        cu_seqlens_q,
        max_seqlen_q,
        softmax_scale,
        cache_seqlens,
        scheduler_metadata=scheduler_metadata,
    )
    return out, lse, torch.empty(0, device=q.device), torch.empty(0, device=q.device)


@register_fake("flash_attn_npu_3::_mla_tnd_bsnd_forward")
def _mla_tnd_bsnd_forward_fake(
    q,
    k,
    v,
    qv,
    cu_seqlens_q,
    max_seqlen_q,
    softmax_scale=None,
    cache_seqlens=None,
    scheduler_metadata=None,
):
    check_inputs(q, k, v, qv, varlen_q=True)
    _check_mla_dense_options(softmax_scale=softmax_scale)
    _check_mla_mixed_inputs(q, k, cu_seqlens_q, max_seqlen_q, cache_seqlens, scheduler_metadata)
    return _mla_fake_outputs(q, v)


def _check_mla_gqa_options(pack, margin):
    if pack is not None and not isinstance(pack, bool):
        raise RuntimeError("MLA: pack_gqa must be None or bool")
    # Fake checks the architecture envelope; real host checks this chip's count.
    if not isinstance(margin, int) or not 0 <= margin < 26:
        raise RuntimeError("MLA: sm_margin must leave at least one physical AIC")


def _mla_gqa_active(heads, kv_heads, pack, margin):
    return margin != 0 or (pack is False and heads != kv_heads)


def _mla_gqa_budget(
    batch,
    sq,
    sk,
    heads,
    kv_heads,
    dim,
    splits,
    left,
    right,
    causal,
    pack,
    margin,
    total_q=None,
    total_k=None,
    page=0,
):
    _check_mla_gqa_options(pack, margin)
    if dim not in (128, 512) or kv_heads <= 0 or heads % kv_heads or (dim == 512 and kv_heads != 1):
        raise RuntimeError("MLA: GQA representation is unsupported")
    groups = heads if pack is False else kv_heads
    _mla_prefill_bsnd_budget(
        batch, sq, sk, heads, groups, splits, left, right, causal, total_q, total_k, page
    )
    words = 380 + batch * (8 + 2 * groups)
    if not 0 <= batch <= 2**32 - 1 or words > 2**32 - 1:
        raise RuntimeError("MLA: GQA metadata word count overflow")
    group = heads // groups
    n = (
        (8192 if group < 5 else 4096 if group < 10 else 2048)
        if sq <= 16 and total_q is None
        else 512
    )
    if _normalize_mla_window(sq, sk, left, right, causal)[0]:
        n = min(n, 1024)
    if dim == 512:
        n = 512
    if page and n > page and n % page:
        n = n // page * page
    n = min(n, sk) or 32
    aligned = (n + 31) // 32 * 32
    m = (
        256
        if dim == 512
        else (
            512
            if aligned <= 512
            else 256
            if aligned <= 1024
            else 128
            if aligned <= 2048
            else 64
            if aligned <= 4096
            else 32
        )
    )
    count = batch * groups * ((sq * group + m - 1) // m)
    if total_q is not None:
        count = min(count, (total_q * group // m + batch) * groups)
    scratch = 26 * (12 * m * aligned + 16 * m * dim + 16 * m)
    if scratch + count * (splits or 26) * m * (dim + 16) * 4 > 2**63 - 1:
        raise RuntimeError("MLA: GQA workspace overflow")
    return words


@register_fake("flash_attn_npu_3::_mla_gqa_forward")
def _mla_gqa_forward_fake(
    q,
    k,
    v,
    qv,
    num_splits,
    window_left,
    window_right,
    causal,
    softmax_scale,
    seqused_q,
    seqused_k,
    scheduler_metadata,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    page_table,
    softcap,
    pack_gqa,
    sm_margin,
    partials=True,
):
    tnd = cu_seqlens_q is not None
    if not check_inputs(
        q,
        k,
        v,
        qv,
        varlen_q=tnd,
        varlen_k=cu_seqlens_k is not None,
        paged_kv=page_table is not None,
    ):
        raise RuntimeError("MLA: GQA entry requires compressed 64/512 or prefill 192/128 tensors")
    _check_mla_dense_options(
        softmax_scale=softmax_scale,
        num_splits=num_splits,
        allow_split=True,
        softcap=softcap,
        allow_softcap=True,
        pack_gqa=pack_gqa,
        sm_margin=sm_margin,
    )
    if cu_seqlens_k is not None:
        _check_mla_tnd_inputs(q, k, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)
    elif tnd and page_table is None:
        _check_mla_mixed_inputs(q, k, cu_seqlens_q, max_seqlen_q, seqused_k, None, seqused_q)
    batch = cu_seqlens_q.numel() - 1 if tnd else q.shape[0]
    _check_mla_used(q, batch, seqused_q, seqused_k)
    if page_table is not None:
        _check_mla_paged_inputs(
            q,
            k,
            page_table,
            seqused_k,
            None,
            seqused_q,
            cu_seqlens_q,
            max_seqlen_q,
            window_left,
            window_right,
            causal,
            num_splits,
            has_softcap=softcap > 0,
        )
    sq = max_seqlen_q if tnd else q.shape[1]
    sk = (
        max_seqlen_k
        if cu_seqlens_k is not None
        else k.shape[1] * (page_table.shape[1] if page_table is not None else 1)
    )
    words = _mla_gqa_budget(
        batch,
        sq,
        sk,
        q.shape[-2],
        k.shape[-2],
        v.shape[-1],
        num_splits,
        window_left,
        window_right,
        causal,
        pack_gqa,
        sm_margin,
        q.shape[0] if tnd else None,
        (k.shape[0] if cu_seqlens_k is not None else batch * sk) if tnd else None,
        k.shape[1] if page_table is not None else 0,
    )
    if scheduler_metadata is not None:
        if seqused_q is not None and (not tnd or (qv is not None and cu_seqlens_k is None)):
            raise RuntimeError("MLA: Q-used with explicit metadata is not implemented yet")
        if (
            scheduler_metadata.dtype != torch.int32
            or scheduler_metadata.device != q.device
            or scheduler_metadata.shape != (words,)
            or not scheduler_metadata.is_contiguous()
        ):
            raise RuntimeError(
                "MLA: GQA scheduler_metadata must be same-device contiguous int32 with matching word count"
            )
    return _mla_fake_outputs(q, v, num_splits, partials)


@register_fake("flash_attn_npu_3::_mla_gqa_metadata")
def _mla_gqa_metadata_fake(
    batch_size,
    max_seqlen_q,
    max_seqlen_k,
    num_heads_q,
    num_heads_kv,
    headdim_v,
    qkv_dtype,
    cache_seqlens,
    cu_seqlens_q,
    num_splits,
    window_left,
    window_right,
    causal,
    page_size,
    has_softcap,
    pack_gqa,
    sm_margin,
    seqused_q=None,
):
    _check_mla_cache_lengths(cache_seqlens, batch_size, cache_seqlens.device)
    if seqused_q is not None and (cu_seqlens_q is None or headdim_v != 128):
        raise RuntimeError("MLA: Q-used metadata requires prefill TND Q/cache")
    _check_mla_used(cache_seqlens, batch_size, seqused_q, None)
    if cu_seqlens_q is not None:
        _check_mla_metadata_query_prefix(cu_seqlens_q, batch_size, cache_seqlens.device)
    if qkv_dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("MLA: qkv_dtype must be FP16 or BF16")
    words = _mla_gqa_budget(
        batch_size,
        max_seqlen_q,
        max_seqlen_k,
        num_heads_q,
        num_heads_kv,
        headdim_v,
        num_splits,
        window_left,
        window_right,
        causal,
        pack_gqa,
        sm_margin,
        0 if cu_seqlens_q is not None else None,
        0 if cu_seqlens_q is not None else None,
        page_size,
    )
    return cache_seqlens.new_empty((words,))


def _mla_prefill_bsnd_budget(
    batch,
    sq,
    sk,
    heads,
    kv_heads,
    splits,
    left,
    right,
    causal,
    total_q=None,
    total_k=None,
    page_size=0,
):
    if (
        not _fits_mla_dense_shape(batch if total_q is None else 0, sq, heads, sk)
        or kv_heads <= 0
        or heads % kv_heads
        or not 0 <= splits <= 256
    ):
        raise RuntimeError("MLA: prefill BSND shape/split bounds exceeded")
    if page_size and (not 16 <= page_size <= 1024 or page_size % 16 or sk % page_size):
        raise RuntimeError("MLA: prefill paged capacity/page size invalid")
    words = (
        ((372 if page_size else 368) + batch * (2 + 2 * kv_heads))
        if total_q is None
        else (372 + batch * (8 + 2 * kv_heads))
    )
    if words > 2**32 - 1 or batch * kv_heads > (2**32 - 1 - 758) // 10:
        raise RuntimeError("MLA: prefill metadata word count overflow")
    kind, _, _ = _normalize_mla_window(sq, sk, left, right, causal)
    group = heads // kv_heads
    n = (
        (8192 if group < 5 else 4096 if group < 10 else 2048)
        if sq <= 16 and total_q is None
        else 512
    )
    n = min(n, 1024) if kind else n
    if page_size and n > page_size and n % page_size:
        n = n // page_size * page_size
    n = min(n, sk) or 32
    aligned = (n + 31) // 32 * 32
    m = (
        512
        if aligned <= 512
        else 256
        if aligned <= 1024
        else 128
        if aligned <= 2048
        else 64
        if aligned <= 4096
        else 32
    )
    capacity = splits or 26
    rows = max(batch, 1) * max(sq, 1) if total_q is None else max(total_q, 1)
    if rows * heads * capacity * 512 > 2**63 - 1:
        raise RuntimeError("MLA: prefill accumulator overflow")
    if batch and sq and sk and (total_q is None or (total_q and total_k)):
        scratch = 26 * (12 * m * aligned + 16 * m * 128)
        groups = batch * kv_heads * ((sq * group + m - 1) // m)
        if total_q is not None:
            groups = min(groups, (total_q * group // m + batch) * kv_heads)
        workspace = scratch + (0 if splits == 1 else groups * capacity * m * 144 * 4)
        if workspace > 2**63 - 1:
            raise RuntimeError("MLA: prefill workspace overflow")
    return words


@register_fake("flash_attn_npu_3::_mla_prefill_bsnd_metadata")
def _mla_prefill_bsnd_metadata_fake(
    batch_size,
    max_seqlen_q,
    max_seqlen_k,
    num_heads_q,
    num_heads_kv,
    qkv_dtype,
    cache_seqlens,
    num_splits=1,
    window_left=-1,
    window_right=-1,
    causal=False,
    page_size=0,
    has_softcap=False,
):
    words = _mla_prefill_bsnd_budget(
        batch_size,
        max_seqlen_q,
        max_seqlen_k,
        num_heads_q,
        num_heads_kv,
        num_splits,
        window_left,
        window_right,
        causal,
        page_size=page_size,
    )
    if qkv_dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("MLA: qkv_dtype must be FP16 or BF16")
    _check_mla_cache_lengths(cache_seqlens, batch_size, cache_seqlens.device)
    return cache_seqlens.new_empty((words,))


@register_fake("flash_attn_npu_3::_mla_prefill_bsnd_forward")
def _mla_prefill_bsnd_forward_fake(
    q,
    k,
    v,
    num_splits=1,
    window_left=-1,
    window_right=-1,
    causal=False,
    softmax_scale=None,
    cache_seqlens=None,
    seqused_q=None,
    scheduler_metadata=None,
    page_table=None,
    softcap=0.0,
    partials=True,
):
    check_inputs(q, k, v, paged_kv=page_table is not None)
    if q.shape[-1] != 192 or v.shape[-1] != 128:
        raise RuntimeError("MLA: prefill requires 192/128 tensors")
    _check_mla_dense_options(
        softmax_scale=softmax_scale,
        num_splits=num_splits,
        allow_split=True,
        softcap=softcap,
        allow_softcap=True,
    )
    _check_mla_used(q, q.shape[0], seqused_q, cache_seqlens)
    if page_table is not None:
        _check_mla_paged_inputs(
            q,
            k,
            page_table,
            cache_seqlens,
            scheduler_metadata,
            seqused_q,
            window_left=window_left,
            window_right=window_right,
            causal=causal,
            num_splits=num_splits,
        )
    words = _mla_prefill_bsnd_budget(
        q.shape[0],
        q.shape[1],
        k.shape[1] * (page_table.shape[1] if page_table is not None else 1),
        q.shape[2],
        k.shape[2],
        num_splits,
        window_left,
        window_right,
        causal,
        page_size=k.shape[1] if page_table is not None else 0,
    )
    if scheduler_metadata is not None:
        if seqused_q is not None:
            raise RuntimeError("MLA: prefill Q-used with explicit metadata is not implemented yet")
        if (
            scheduler_metadata.dtype != torch.int32
            or scheduler_metadata.device != q.device
            or scheduler_metadata.shape != (words,)
            or not scheduler_metadata.is_contiguous()
        ):
            raise RuntimeError(
                "MLA: prefill scheduler_metadata must be same-device contiguous int32 with matching word count"
            )
    return _mla_fake_outputs(q, v, num_splits, partials)


def _mla_prefill_forward(
    q,
    k,
    v,
    splits,
    left,
    right,
    causal,
    scale,
    used_k,
    used_q,
    metadata,
    cu_q=None,
    cu_k=None,
    max_q=None,
    max_k=None,
    page_table=None,
    softcap=0.0,
    partials=True,
):
    options = dict(
        num_splits=splits,
        window_left=left,
        window_right=right,
        causal=causal,
        softmax_scale=scale,
        seqused_q=used_q,
        scheduler_metadata=metadata,
        softcap=softcap,
        partials=partials,
    )
    ops = torch.ops.flash_attn_npu_3
    if page_table is None and cu_q is not None and cu_k is not None:
        _check_mla_tnd_inputs(q, k, cu_q, cu_k, max_q, max_k)
        return ops._mla_prefill_tnd_forward(
            q, k, v, cu_q, cu_k, max_q, max_k, seqused_k=used_k, **options
        )
    if cu_q is not None:
        if page_table is None:
            _check_mla_mixed_inputs(q, k, cu_q, max_q, used_k, None, used_q)
        return ops._mla_prefill_mixed_forward(
            q, k, v, cu_q, max_q, cache_seqlens=used_k, page_table=page_table, **options
        )
    return ops._mla_prefill_bsnd_forward(
        q, k, v, cache_seqlens=used_k, page_table=page_table, **options
    )


@register_fake("flash_attn_npu_3::_mla_prefill_mixed_metadata")
def _mla_prefill_mixed_metadata_fake(
    batch_size,
    max_seqlen_q,
    max_seqlen_k,
    num_heads_q,
    num_heads_kv,
    qkv_dtype,
    cache_seqlens,
    cu_seqlens_q,
    num_splits=1,
    window_left=-1,
    window_right=-1,
    causal=False,
    seqused_q=None,
    page_size=0,
    has_softcap=False,
):
    if qkv_dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("MLA: qkv_dtype must be FP16 or BF16")
    _check_mla_cache_lengths(cache_seqlens, batch_size, cache_seqlens.device)
    _check_mla_metadata_query_prefix(cu_seqlens_q, batch_size, cache_seqlens.device)
    _check_mla_used(cache_seqlens, batch_size, seqused_q, cache_seqlens)
    words = _mla_prefill_bsnd_budget(
        batch_size,
        max_seqlen_q,
        max_seqlen_k,
        num_heads_q,
        num_heads_kv,
        num_splits,
        window_left,
        window_right,
        causal,
        total_q=0,
        total_k=0,
        page_size=page_size,
    )
    return cache_seqlens.new_empty((words,))


@register_fake("flash_attn_npu_3::_mla_prefill_mixed_forward")
def _mla_prefill_mixed_forward_fake(
    q,
    k,
    v,
    cu_seqlens_q,
    max_seqlen_q,
    num_splits=1,
    window_left=-1,
    window_right=-1,
    causal=False,
    softmax_scale=None,
    cache_seqlens=None,
    seqused_q=None,
    scheduler_metadata=None,
    page_table=None,
    softcap=0.0,
    partials=True,
):
    _check_mla_dense_options(softcap=softcap, allow_softcap=True)
    check_inputs(q, k, v, varlen_q=True, paged_kv=page_table is not None)
    if q.shape[-1] != 192 or v.shape[-1] != 128:
        raise RuntimeError("MLA: prefill requires 192/128 tensors")
    if page_table is not None:
        _check_mla_paged_inputs(
            q,
            k,
            page_table,
            cache_seqlens,
            scheduler_metadata,
            seqused_q,
            cu_seqlens_q,
            max_seqlen_q,
            window_left,
            window_right,
            causal,
            num_splits,
            has_softcap=softcap > 0,
        )
    else:
        _check_mla_mixed_inputs(q, k, cu_seqlens_q, max_seqlen_q, cache_seqlens, None, seqused_q)
    batch = cu_seqlens_q.numel() - 1
    capacity_k = k.shape[1] * (page_table.shape[1] if page_table is not None else 1)
    _check_mla_dense_options(softmax_scale=softmax_scale, num_splits=num_splits, allow_split=True)
    words = _mla_prefill_bsnd_budget(
        batch,
        max_seqlen_q,
        capacity_k,
        q.shape[1],
        k.shape[2],
        num_splits,
        window_left,
        window_right,
        causal,
        total_q=q.shape[0],
        total_k=batch * capacity_k,
        page_size=k.shape[1] if page_table is not None else 0,
    )
    if scheduler_metadata is not None and (
        scheduler_metadata.dtype != torch.int32
        or scheduler_metadata.device != q.device
        or scheduler_metadata.shape != (words,)
        or not scheduler_metadata.is_contiguous()
    ):
        raise RuntimeError(
            "MLA: prefill scheduler_metadata must be same-device contiguous int32 with matching word count"
        )
    return _mla_fake_outputs(q, v, num_splits, partials)


def _check_prefill_tnd(
    q, k, v, cu_q, cu_k, max_q, max_k, splits, left, right, causal, used_q, used_k
):
    check_inputs(q, k, v, varlen_q=True, varlen_k=True)
    if q.shape[-1] != 192 or v.shape[-1] != 128:
        raise RuntimeError("MLA: prefill requires 192/128 tensors")
    _check_mla_tnd_inputs(q, k, cu_q, cu_k, max_q, max_k)
    _check_mla_used(q, cu_q.numel() - 1, used_q, used_k)
    return _mla_prefill_bsnd_budget(
        cu_q.numel() - 1,
        max_q,
        max_k,
        q.shape[1],
        k.shape[1],
        splits,
        left,
        right,
        causal,
        total_q=q.shape[0],
        total_k=k.shape[0],
    )


@register_fake("flash_attn_npu_3::_mla_prefill_tnd_metadata")
def _mla_prefill_tnd_metadata_fake(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    num_splits=1,
    window_left=-1,
    window_right=-1,
    causal=False,
    seqused_q=None,
    seqused_k=None,
    has_softcap=False,
):
    words = _check_prefill_tnd(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        num_splits,
        window_left,
        window_right,
        causal,
        seqused_q,
        seqused_k,
    )
    return q.new_empty((words,), dtype=torch.int32)


@register_fake("flash_attn_npu_3::_mla_prefill_tnd_forward")
def _mla_prefill_tnd_forward_fake(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    num_splits=1,
    window_left=-1,
    window_right=-1,
    causal=False,
    softmax_scale=None,
    seqused_q=None,
    seqused_k=None,
    scheduler_metadata=None,
    softcap=0.0,
    partials=True,
):
    _check_mla_dense_options(softcap=softcap, allow_softcap=True)
    words = _check_prefill_tnd(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        num_splits,
        window_left,
        window_right,
        causal,
        seqused_q,
        seqused_k,
    )
    _check_mla_dense_options(softmax_scale=softmax_scale, num_splits=num_splits, allow_split=True)
    if scheduler_metadata is not None and (
        scheduler_metadata.dtype != torch.int32
        or scheduler_metadata.device != q.device
        or scheduler_metadata.shape != (words,)
        or not scheduler_metadata.is_contiguous()
    ):
        raise RuntimeError(
            "MLA: prefill scheduler_metadata must be same-device contiguous int32 with matching word count"
        )
    return _mla_fake_outputs(q, v, num_splits, partials)


def _mla_split_forward(
    q,
    k,
    v,
    qv,
    *,
    num_splits,
    window_size=(-1, -1),
    causal=False,
    softmax_scale=None,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    seqused_q=None,
    seqused_k=None,
    page_table=None,
    scheduler_metadata=None,
    softcap=0.0,
    partials=True,
):
    """Select the existing split operator; its registered fake uses the same route."""
    ops = torch.ops.flash_attn_npu_3
    options = dict(
        num_splits=num_splits,
        window_left=window_size[0],
        window_right=window_size[1],
        causal=causal,
        softmax_scale=softmax_scale,
        seqused_q=seqused_q,
        scheduler_metadata=scheduler_metadata,
    )
    if page_table is not None:
        if cu_seqlens_q is not None:
            if max_seqlen_q is None:
                raise RuntimeError("MLA: TND paged split requires max_seqlen_q")
            return ops._mla_split_paged_tnd_forward(
                q,
                k,
                v,
                qv,
                page_table,
                cu_seqlens_q,
                max_seqlen_q,
                cache_seqlens=seqused_k,
                softcap=softcap,
                **options,
                partials=partials,
            )
        return ops._mla_split_paged_bsnd_forward(
            q,
            k,
            v,
            qv,
            page_table,
            cache_seqlens=seqused_k,
            softcap=softcap,
            **options,
            partials=partials,
        )
    if cu_seqlens_q is not None:
        if cu_seqlens_k is not None:
            return ops._mla_split_tnd_forward(
                q,
                k,
                v,
                qv,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                seqused_k=seqused_k,
                softcap=softcap,
                **options,
                partials=partials,
            )
        return ops._mla_split_mixed_forward(
            q,
            k,
            v,
            qv,
            cu_seqlens_q,
            max_seqlen_q,
            cache_seqlens=seqused_k,
            softcap=softcap,
            **options,
            partials=partials,
        )
    return ops._mla_split_bsnd_forward(
        q,
        k,
        v,
        qv,
        cache_seqlens=seqused_k,
        softcap=softcap,
        **options,
        partials=partials,
    )


def forward(
    q,
    k,
    v,
    qv,
    *,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    num_splits=1,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    seqused_q=None,
    seqused_k=None,
    page_table=None,
    scheduler_metadata=None,
    return_accumulators=False,
    attention_chunk=0,
    softcap=0.0,
    pack_gqa=None,
    sm_margin=0,
    rotary_interleaved=True,
    unsupported_options=(),
):
    """Shared forward-only adaptation of the three FA3 public entry points.

    Follow pinned FA3 d15f153's public-wrapper/forward/backend boundary. NPU
    operators own tensor validation and real/fake execution. Keep the existing
    four-result cache operators to avoid extra Python device allocations.
    """
    for name, value in unsupported_options:
        if value is not None:
            raise RuntimeError(f"MLA compressed forward is not implemented yet for {name}")
    _check_mla_dense_options(
        softmax_scale=softmax_scale,
        causal=causal,
        allow_causal=True,
        window_size=window_size,
        allow_window=True,
        num_splits=num_splits,
        allow_split=True,
        attention_chunk=attention_chunk,
        softcap=softcap,
        allow_softcap=True,
        pack_gqa=pack_gqa,
        sm_margin=sm_margin,
        rotary_interleaved=rotary_interleaved,
    )
    if cu_seqlens_q is None and max_seqlen_q is not None:
        raise RuntimeError(
            "MLA compressed forward is not implemented yet for max_seqlen_q without cu_seqlens_q"
        )
    if page_table is not None and cu_seqlens_q is not None and max_seqlen_q is None:
        raise RuntimeError("MLA: TND paged forward requires max_seqlen_q")
    if cu_seqlens_k is None and isinstance(seqused_k, int):
        # Integer convenience input needs valid geometry before creating
        # its device tensor. Tensor/None lengths go straight to the op.
        check_inputs(
            q,
            k,
            v,
            qv,
            varlen_q=cu_seqlens_q is not None,
            paged_kv=page_table is not None,
        )
        batch = k.shape[0]
        if page_table is not None:
            batch = _check_mla_paged_inputs(
                q,
                k,
                page_table,
                None,
                None
                if _mla_gqa_active(q.shape[-2], k.shape[-2], pack_gqa, sm_margin)
                else scheduler_metadata,
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=max_seqlen_q,
                window_left=window_size[0],
                window_right=window_size[1],
                causal=causal,
                num_splits=num_splits,
                has_softcap=softcap > 0,
            )
        capacity = k.shape[1] * (page_table.shape[1] if page_table is not None else 1)
        if not 0 <= seqused_k <= capacity:
            raise RuntimeError("MLA: cache_seqlens must be between zero and KV capacity")
        seqused_k = torch.full(
            (batch,),
            seqused_k,
            dtype=torch.int32,
            device=k.device,
        )
    if _mla_gqa_active(q.shape[-2], k.shape[-2], pack_gqa, sm_margin):
        result = torch.ops.flash_attn_npu_3._mla_gqa_forward(
            q,
            k,
            v,
            qv,
            num_splits,
            *window_size,
            causal,
            softmax_scale,
            seqused_q,
            seqused_k,
            scheduler_metadata,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q or 0,
            max_seqlen_k or 0,
            page_table,
            softcap,
            pack_gqa,
            sm_margin,
            partials=return_accumulators,
        )
        return result if return_accumulators else result[:2]
    if qv is None:
        result = _mla_prefill_forward(
            q,
            k,
            v,
            num_splits,
            *window_size,
            causal,
            softmax_scale,
            seqused_k,
            seqused_q,
            scheduler_metadata,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            page_table,
            softcap=softcap,
            partials=return_accumulators,
        )
        return result if return_accumulators else result[:2]
    if num_splits != 1 or softcap != 0:
        result = _mla_split_forward(
            q,
            k,
            v,
            qv,
            num_splits=num_splits,
            window_size=window_size,
            causal=causal,
            softmax_scale=softmax_scale,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
            page_table=page_table,
            scheduler_metadata=scheduler_metadata,
            softcap=softcap,
            partials=return_accumulators,
        )
        return result if return_accumulators else result[:2]
    ops = torch.ops.flash_attn_npu_3
    cache_options = dict(
        softmax_scale=softmax_scale,
        cache_seqlens=seqused_k,
        scheduler_metadata=scheduler_metadata,
    )
    if page_table is not None:
        cache_options.update(causal=causal, window_left=window_size[0], window_right=window_size[1])
        if cu_seqlens_q is not None:
            out, lse = ops._mla_paged_tnd_forward(
                q,
                k,
                v,
                qv,
                page_table,
                cu_seqlens_q,
                max_seqlen_q,
                **cache_options,
            )
        else:
            out, lse = ops._mla_paged_bsnd_forward(q, k, v, qv, page_table, **cache_options)
    elif cu_seqlens_k is not None:
        tnd_options = dict(softmax_scale=softmax_scale, seqused_q=seqused_q, seqused_k=seqused_k)
        if window_size != (-1, -1):
            out, lse = ops._mla_window_tnd_forward(
                q,
                k,
                v,
                qv,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                *window_size,
                causal,
                **tnd_options,
            )
        else:
            op = ops._mla_causal_tnd_forward if causal else ops._mla_tnd_forward
            out, lse = op(
                q,
                k,
                v,
                qv,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                **tnd_options,
            )
    elif cu_seqlens_q is not None:
        if max_seqlen_q is None:
            raise RuntimeError("MLA: mixed TND/BSND requires max_seqlen_q")
        if window_size != (-1, -1):
            op = (
                ops._mla_window_mixed_cache
                if return_accumulators
                else ops._mla_window_mixed_forward
            )
            return op(
                q, k, v, qv, cu_seqlens_q, max_seqlen_q, *window_size, causal, **cache_options
            )
        if causal:
            op = (
                ops._mla_causal_mixed_cache
                if return_accumulators
                else ops._mla_causal_mixed_forward
            )
            return op(q, k, v, qv, cu_seqlens_q, max_seqlen_q, **cache_options)
        out, lse = ops._mla_tnd_bsnd_forward(
            q, k, v, qv, cu_seqlens_q, max_seqlen_q, **cache_options
        )
    elif window_size != (-1, -1):
        op = ops._mla_window_bsnd_cache if return_accumulators else ops._mla_window_bsnd_forward
        return op(q, k, v, qv, *window_size, causal, **cache_options)
    elif causal:
        op = ops._mla_causal_bsnd_cache if return_accumulators else ops._mla_causal_bsnd_forward
        return op(q, k, v, qv, **cache_options)
    else:
        out, lse = ops._mla_dense_forward(q, k, v, qv, **cache_options)
    if return_accumulators:
        return out, lse, torch.empty(0, device=out.device), torch.empty(0, device=out.device)
    return out, lse


def _check_mla_int8_config(batch, sq, heads, capacity, splits, causal):
    if not (
        0 <= batch <= (2**32 - 1 - 462)
        and 0 <= sq <= 32
        and heads in (1, 2, 4, 8, 16, 32, 64, 128)
        and 0 <= capacity <= 2**31 - 1
        and capacity % 128 == 0
        and splits in (0, 1)
        and (sq <= 1 or causal)
        and batch * max(sq, 1) * heads <= 2**31 - 1
        and batch * heads * 512 <= 2**32 - 1
    ):
        raise RuntimeError(
            "MLA INT8: Q<=32, power-of-two Hq<=128, page=128, multi-Q causal and num_splits=0/1 required"
        )


def _check_mla_int8_storage(tensor, device, dtype, last_contiguous=True):
    if (
        tensor is None
        or tensor.device != device
        or device.type != "npu"
        or tensor.dtype != dtype
        or tensor.requires_grad
    ):
        raise RuntimeError("MLA INT8: wrong dtype/device, missing scale or unsupported gradients")
    if (
        tensor.ndim == 0
        or (last_contiguous and tensor.stride(-1) != 1)
        or any(s < 0 for s in tensor.stride())
    ):
        raise RuntimeError("MLA INT8: invalid stride; data last dimension must be contiguous")
    if not torch.compiler.is_compiling() and tensor.numel():
        span = 1 + sum((n - 1) * s for n, s in zip(tensor.shape, tensor.stride()))
        end = (tensor.storage_offset() + span) * tensor.element_size()
        if (
            tensor.storage_offset() < 0
            or end > 2**63 - 1
            or end > tensor.untyped_storage().nbytes()
        ):
            raise RuntimeError("MLA INT8: view exceeds storage/byte bounds")


@register_fake("flash_attn_npu_3::_mla_int8_metadata")
def _mla_int8_metadata_fake(batch, sq, heads, capacity, lengths, splits, causal, prefix=None):
    _check_mla_int8_config(batch, sq, heads, capacity, splits, causal)
    _check_mla_cache_lengths(lengths, batch, lengths.device)
    if lengths.device.type != "npu":
        raise RuntimeError("MLA INT8: lengths must be NPU tensors")
    if prefix is not None:
        _check_mla_metadata_query_prefix(prefix, batch, lengths.device)
    return torch.empty(
        (462 + batch * (2 if prefix is not None else 1) + int(prefix is not None),),
        device=lengths.device,
        dtype=torch.int32,
    )


def _mla_int8_forward_fake(
    q,
    k,
    v,
    qv,
    table,
    qs,
    ks,
    vs,
    lengths,
    metadata,
    scale,
    causal,
    splits,
    partials,
    prefix=None,
    max_q=0,
):
    tnd = prefix is not None
    if (
        q.ndim != (3 if tnd else 4)
        or qv.ndim != q.ndim
        or k.ndim != 4
        or v.ndim != 4
        or table.ndim != 2
    ):
        raise RuntimeError("MLA INT8: BSND/TND Q and ND PA cache required")
    for t in (q, k):
        _check_mla_int8_storage(t, q.device, torch.bfloat16)
    for t in (qv, v):
        _check_mla_int8_storage(t, q.device, torch.int8)
    for t in (qs, ks, vs):
        _check_mla_int8_storage(t, q.device, torch.float32, False)
    _check_mla_int8_storage(table, q.device, torch.int32)
    b, sq, h, d = (table.shape[0], max_q, q.shape[1], q.shape[2]) if tnd else q.shape
    capacity = table.shape[1] * 128
    if tnd:
        _check_mla_metadata_query_prefix(prefix, b, q.device)
        if q.shape[0] > 1048576 or q.shape[0] * h > 2**31 - 1:
            raise RuntimeError("MLA INT8: TND allocation bounds exceeded")
    _check_mla_int8_config(b, sq, h, capacity, splits, causal)
    if (
        d != 64
        or qv.shape != (*q.shape[:-1], 512)
        or qs.shape != q.shape[:-1]
        or ks.shape != (1,)
        or vs.shape != (1,)
        or table.shape[0] != b
        or k.shape != (*v.shape[:-1], 64)
        or v.shape[1:] != (128, 1, 512)
        or v.shape[0] > 2**31
    ):
        raise RuntimeError("MLA INT8: invalid Q/content/scale/page-table/cache shapes")
    if lengths is not None:
        _check_mla_cache_lengths(lengths, b, q.device)
    if metadata is not None:
        if (
            metadata.device != q.device
            or metadata.dtype != torch.int32
            or metadata.shape != (462 + b * (2 if tnd else 1) + int(tnd),)
            or not metadata.is_contiguous()
        ):
            raise RuntimeError(
                "MLA INT8: metadata must be same-device contiguous int32 with matching layout size"
            )
    if scale is not None and (not math.isfinite(scale) or abs(scale) > 3.4028234663852886e38):
        raise RuntimeError("MLA INT8: softmax_scale must be finite in FP32")
    lse_shape = (h, q.shape[0]) if tnd else (b, h, sq)
    return (
        torch.empty((*q.shape[:-1], 512), device=q.device, dtype=torch.bfloat16),
        torch.empty(lse_shape if partials else (0,), device=q.device, dtype=torch.float32),
        torch.empty(
            (26, *lse_shape, 512) if partials and splits == 0 else (0,),
            device=q.device,
            dtype=torch.float32,
        ),
        torch.empty(
            (26, *lse_shape) if partials and splits == 0 else (0,),
            device=q.device,
            dtype=torch.float32,
        ),
    )


@register_fake("flash_attn_npu_3::_mla_int8_bsnd")
def _mla_int8_bsnd_fake(
    q, k, v, qv, table, qs, ks, vs, lengths, metadata, scale, causal, splits, partials
):
    return _mla_int8_forward_fake(
        q, k, v, qv, table, qs, ks, vs, lengths, metadata, scale, causal, splits, partials
    )


@register_fake("flash_attn_npu_3::_mla_int8_tnd")
def _mla_int8_tnd_fake(
    q,
    k,
    v,
    qv,
    table,
    qs,
    ks,
    vs,
    lengths,
    metadata,
    scale,
    causal,
    splits,
    partials,
    prefix,
    max_q,
):
    return _mla_int8_forward_fake(
        q,
        k,
        v,
        qv,
        table,
        qs,
        ks,
        vs,
        lengths,
        metadata,
        scale,
        causal,
        splits,
        partials,
        prefix,
        max_q,
    )


def _check_mla_metadata_options(
    num_heads_kv,
    max_seqlen_k_new,
    causal,
    window_size,
    attention_chunk,
    has_softcap,
    num_splits,
    sm_margin,
    optional_args,
    allow_causal=False,
    allow_window=False,
    allow_split=False,
    prefill=False,
    allow_softcap=False,
):
    if not prefill and num_heads_kv != 1:
        raise RuntimeError(
            "MLA scheduler metadata is not implemented yet outside the dense shape bounds"
        )
    if (
        any(arg is not None for arg in optional_args)
        or max_seqlen_k_new != 0
        or (causal and not allow_causal)
        or (window_size != (-1, -1) and not allow_window)
        or attention_chunk != 0
        or (has_softcap and not allow_softcap)
        or (num_splits != 1 and not (allow_split and 0 <= num_splits <= 256))
    ):
        raise RuntimeError(
            "MLA scheduler metadata is not implemented yet for these length, mask, split, "
            "or scheduling options"
        )


def _fits_mla_dense_shape(batch, sq, heads, sk):
    # Match mla_tiling.h: source ceil-div, query/LSE DMA strides and tensor bytes.
    return (
        0 <= batch <= 2**32 - 1
        and 0 <= sq <= 2**30
        and 0 < heads <= (2**32 - 1) // 512
        and sq * heads <= 2**32 - 256
        and batch * sq * heads <= (2**63 - 1) // 1024
        and 0 <= sk <= 16384
    )


def _mla_bsnd_metadata_words(batch):
    if not 0 <= batch <= 2**32 - 1 - 401:
        raise RuntimeError("MLA: BSND metadata word count overflow")
    return 401 + batch


@register_fake("flash_attn_npu_3::_mla_window_mixed_metadata")
def _mla_window_mixed_metadata_fake(
    batch_size,
    max_seqlen_q,
    max_seqlen_k,
    num_heads_q,
    qkv_dtype,
    cache_seqlens,
    cu_seqlens_q,
    window_left,
    window_right,
    causal,
):
    _mla_tnd_bsnd_metadata_fake(
        batch_size, max_seqlen_q, max_seqlen_k, num_heads_q, qkv_dtype, cache_seqlens, cu_seqlens_q
    )
    _normalize_mla_window(max_seqlen_q, max_seqlen_k, window_left, window_right, causal)
    return torch.empty(
        (_mla_window_bsnd_words(batch_size),), dtype=torch.int32, device=cache_seqlens.device
    )


@register_fake("flash_attn_npu_3::_mla_causal_mixed_metadata")
def _mla_causal_mixed_metadata_fake(
    batch_size,
    max_seqlen_q,
    max_seqlen_k,
    num_heads_q,
    qkv_dtype,
    cache_seqlens,
    cu_seqlens_q,
):
    _mla_tnd_bsnd_metadata_fake(
        batch_size, max_seqlen_q, max_seqlen_k, num_heads_q, qkv_dtype, cache_seqlens, cu_seqlens_q
    )
    return torch.empty(
        (_mla_causal_bsnd_words(batch_size),), dtype=torch.int32, device=cache_seqlens.device
    )


@register_fake("flash_attn_npu_3::_mla_split_bsnd_metadata")
def _mla_split_bsnd_metadata_fake(
    batch_size,
    max_seqlen_q,
    max_seqlen_k,
    num_heads_q,
    qkv_dtype,
    cache_seqlens,
    window_left,
    window_right,
    causal,
    num_splits,
    has_softcap=False,
):
    _mla_causal_bsnd_metadata_fake(
        batch_size, max_seqlen_q, max_seqlen_k, num_heads_q, qkv_dtype, cache_seqlens
    )
    _normalize_mla_window(max_seqlen_q, max_seqlen_k, window_left, window_right, causal)
    if num_splits != 0 and not (1 if has_softcap else 2) <= num_splits <= 256:
        raise RuntimeError("MLA: split requires num_splits=0 or [2,256]")
    return torch.empty(
        (_mla_split_bsnd_words(batch_size, num_splits),),
        dtype=torch.int32,
        device=cache_seqlens.device,
    )


@register_fake("flash_attn_npu_3::_mla_window_bsnd_metadata")
def _mla_window_bsnd_metadata_fake(
    batch_size,
    max_seqlen_q,
    max_seqlen_k,
    num_heads_q,
    qkv_dtype,
    cache_seqlens,
    window_left,
    window_right,
    causal,
):
    _mla_causal_bsnd_metadata_fake(
        batch_size, max_seqlen_q, max_seqlen_k, num_heads_q, qkv_dtype, cache_seqlens
    )
    _normalize_mla_window(max_seqlen_q, max_seqlen_k, window_left, window_right, causal)
    return torch.empty(
        (_mla_window_bsnd_words(batch_size),), dtype=torch.int32, device=cache_seqlens.device
    )


@register_fake("flash_attn_npu_3::_mla_causal_bsnd_metadata")
def _mla_causal_bsnd_metadata_fake(
    batch_size, max_seqlen_q, max_seqlen_k, num_heads_q, qkv_dtype, cache_seqlens
):
    if not _fits_mla_dense_shape(batch_size, max_seqlen_q, num_heads_q, max_seqlen_k):
        raise RuntimeError("MLA: causal BSND shape bounds exceeded")
    if qkv_dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("MLA: qkv_dtype must be FP16 or BF16")
    if cache_seqlens.device.type != "npu":
        raise RuntimeError("MLA: cache_seqlens must be on the same NPU device")
    _check_mla_cache_lengths(cache_seqlens, batch_size, cache_seqlens.device)
    return torch.empty(
        (_mla_causal_bsnd_words(batch_size),), dtype=torch.int32, device=cache_seqlens.device
    )


@register_fake("flash_attn_npu_3::_mla_dense_metadata")
def _mla_dense_metadata_fake(
    batch_size, max_seqlen_q, max_seqlen_k, num_heads_q, qkv_dtype, cache_seqlens
):
    if not _fits_mla_dense_shape(batch_size, max_seqlen_q, num_heads_q, max_seqlen_k):
        raise RuntimeError(
            "MLA scheduler metadata is not implemented yet outside the dense shape bounds"
        )
    if qkv_dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("MLA: qkv_dtype must be FP16 or BF16")
    _check_mla_cache_lengths(cache_seqlens, batch_size, cache_seqlens.device)
    return torch.empty(
        (_mla_bsnd_metadata_words(batch_size),), dtype=torch.int32, device=cache_seqlens.device
    )


def _check_mla_metadata_query_prefix(cu_seqlens_q, batch_size, device):
    if cu_seqlens_q.dtype != torch.int32:
        raise RuntimeError("MLA: cu_seqlens_q must have dtype int32")
    if cu_seqlens_q.device != device:
        raise RuntimeError("MLA: cu_seqlens_q must be on the same NPU device")
    if cu_seqlens_q.ndim != 1 or cu_seqlens_q.numel() != batch_size + 1:
        raise RuntimeError("MLA: cu_seqlens_q must have shape [batch_size + 1]")
    if not cu_seqlens_q.is_contiguous():
        raise RuntimeError("MLA: cu_seqlens_q must be contiguous")


@register_fake("flash_attn_npu_3::_mla_paged_metadata")
def _mla_paged_metadata_fake(
    batch_size,
    max_seqlen_q,
    max_seqlen_k,
    num_heads_q,
    page_size,
    qkv_dtype,
    cache_seqlens,
    cu_seqlens_q,
    causal=False,
    window_left=-1,
    window_right=-1,
    num_splits=1,
    has_softcap=False,
):
    is_window = (window_left, window_right) != (-1, -1)
    prefix_words = (
        756
        if num_splits == 0
        else 414
        if (num_splits != 1 or has_softcap)
        else 411
        if is_window
        else 408
    )
    sequence_words = 7 if (num_splits != 1 or has_softcap) else 5
    if num_splits != 1 or has_softcap:
        if cu_seqlens_q is not None:
            if num_splits != 0 and not (1 if has_softcap else 2) <= num_splits <= 256:
                raise RuntimeError("MLA: TND paged split requires num_splits=0 or [2,256]")
            # The total allocation is device data; AICPU checks its budget.
        else:
            _check_mla_paged_split_budget(
                batch_size,
                max_seqlen_q,
                num_heads_q,
                max_seqlen_k,
                page_size,
                num_splits,
                has_softcap=has_softcap,
            )
    if not 0 <= batch_size <= (2**32 - 1 - prefix_words) // sequence_words:
        raise RuntimeError("MLA: paged metadata word count overflow")
    if not (16 <= page_size <= 1024 and page_size % 16 == 0):
        raise RuntimeError("MLA: page_size must be a multiple of 16 in [16, 1024]")
    batch_bound = batch_size if cu_seqlens_q is None else 0
    if (
        not _fits_mla_dense_shape(batch_bound, max_seqlen_q, num_heads_q, max_seqlen_k)
        or max_seqlen_k % page_size != 0
    ):
        raise RuntimeError(
            "MLA: paged metadata shape bounds exceeded or max_seqlen_k is not divisible by page_size"
        )
    if qkv_dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("MLA: qkv_dtype must be FP16 or BF16")
    _check_mla_cache_lengths(cache_seqlens, batch_size, cache_seqlens.device)
    if cu_seqlens_q is not None:
        _check_mla_metadata_query_prefix(cu_seqlens_q, batch_size, cache_seqlens.device)
    if is_window or (num_splits != 1 or has_softcap):
        _normalize_mla_window(max_seqlen_q, max_seqlen_k, window_left, window_right, causal)
    return torch.empty(
        (prefix_words + sequence_words * batch_size,),
        dtype=torch.int32,
        device=cache_seqlens.device,
    )


@register_fake("flash_attn_npu_3::_mla_tnd_bsnd_metadata")
def _mla_tnd_bsnd_metadata_fake(
    batch_size, max_seqlen_q, max_seqlen_k, num_heads_q, qkv_dtype, cache_seqlens, cu_seqlens_q
):
    if not 0 <= batch_size <= (2**32 - 1 - 406) // 5:
        raise RuntimeError("MLA: mixed metadata word count overflow")
    if not _fits_mla_dense_shape(0, max_seqlen_q, num_heads_q, max_seqlen_k):
        raise RuntimeError("MLA: mixed metadata shape or max_seqlen bounds exceeded")
    if qkv_dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("MLA: qkv_dtype must be FP16 or BF16")
    _check_mla_cache_lengths(cache_seqlens, batch_size, cache_seqlens.device)
    _check_mla_metadata_query_prefix(cu_seqlens_q, batch_size, cache_seqlens.device)
    return torch.empty((406 + 5 * batch_size,), dtype=torch.int32, device=cache_seqlens.device)


def maybe_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_new: Optional[torch.Tensor] = None,
    v_new: Optional[torch.Tensor] = None,
    qv: Optional[torch.Tensor] = None,
    out_: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    cu_seqlens_k_new: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    page_table: Optional[torch.Tensor] = None,
    kv_batch_idx: Optional[torch.Tensor] = None,
    leftpad_k: Optional[torch.Tensor] = None,
    rotary_cos: Optional[torch.Tensor] = None,
    rotary_sin: Optional[torch.Tensor] = None,
    seqlens_rotary: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size_left: int = -1,
    window_size_right: int = -1,
    attention_chunk: int = 0,
    softcap: float = 0.0,
    rotary_interleaved: bool = True,
    scheduler_metadata: Optional[torch.Tensor] = None,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    sm_margin: int = 0,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Share MLA validation and alternate routes between the raw forward and its fake."""
    if check_inputs(
        q,
        k,
        v,
        qv,
        varlen_q=cu_seqlens_q is not None,
        varlen_k=cu_seqlens_k is not None,
        paged_kv=page_table is not None,
    ):
        _check_mla_dense_options(
            softmax_scale=softmax_scale,
            causal=causal,
            allow_causal=True,
            allow_window=True,
            window_size=(window_size_left, window_size_right),
            attention_chunk=attention_chunk,
            softcap=softcap,
            allow_softcap=True,
            num_splits=num_splits,
            allow_split=cu_seqlens_k is None or cu_seqlens_q is not None,
            pack_gqa=pack_gqa,
            sm_margin=sm_margin,
            rotary_interleaved=rotary_interleaved,
            optional_tensors=(
                k_new,
                v_new,
                out_,
                cu_seqlens_k_new,
                max_seqlen_q if cu_seqlens_q is None else None,
                max_seqlen_k if cu_seqlens_k is None else None,
                kv_batch_idx,
                leftpad_k,
                rotary_cos,
                rotary_sin,
                seqlens_rotary,
                q_descale,
                k_descale,
                v_descale,
            ),
        )
        if _mla_gqa_active(q.shape[-2], k.shape[-2], pack_gqa, sm_margin):
            return torch.ops.flash_attn_npu_3._mla_gqa_forward(
                q,
                k,
                v,
                qv,
                num_splits,
                window_size_left,
                window_size_right,
                causal,
                softmax_scale,
                seqused_q,
                seqused_k,
                scheduler_metadata,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q or 0,
                max_seqlen_k or 0,
                page_table,
                softcap,
                pack_gqa,
                sm_margin,
            )
        if qv is None:
            return _mla_prefill_forward(
                q,
                k,
                v,
                num_splits,
                window_size_left,
                window_size_right,
                causal,
                softmax_scale,
                seqused_k,
                seqused_q,
                scheduler_metadata,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                page_table,
                softcap=softcap,
            )
        if num_splits != 1 or softcap != 0:
            return _mla_split_forward(
                q,
                k,
                v,
                qv,
                num_splits=num_splits,
                window_size=(window_size_left, window_size_right),
                causal=causal,
                softmax_scale=softmax_scale,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                seqused_q=seqused_q,
                seqused_k=seqused_k,
                page_table=page_table,
                scheduler_metadata=scheduler_metadata,
                softcap=softcap,
            )
        if page_table is not None:
            _check_mla_paged_inputs(
                q,
                k,
                page_table,
                seqused_k,
                scheduler_metadata,
                seqused_q,
                cu_seqlens_q,
                max_seqlen_q,
                window_left=window_size_left,
                window_right=window_size_right,
                causal=causal,
            )
        elif cu_seqlens_q is not None:
            if cu_seqlens_k is None:
                if (window_size_left, window_size_right) != (-1, -1):
                    _check_mla_window_mixed_inputs(
                        q,
                        k,
                        cu_seqlens_q,
                        max_seqlen_q,
                        seqused_k,
                        scheduler_metadata,
                        seqused_q,
                        window_left=window_size_left,
                        window_right=window_size_right,
                        causal=causal,
                    )
                else:
                    (_check_mla_causal_mixed_inputs if causal else _check_mla_mixed_inputs)(
                        q, k, cu_seqlens_q, max_seqlen_q, seqused_k, scheduler_metadata, seqused_q
                    )
            else:
                if scheduler_metadata is not None:
                    raise RuntimeError(
                        "MLA compressed forward is not implemented yet for TND explicit metadata"
                    )
                _check_mla_tnd_inputs(q, k, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)
                if causal or seqused_q is not None or seqused_k is not None:
                    _check_mla_used(q, cu_seqlens_q.numel() - 1, seqused_q, seqused_k)
                if (window_size_left, window_size_right) != (-1, -1):
                    _mla_window_bsnd_words(cu_seqlens_q.numel() - 1)
                    _normalize_mla_window(
                        max_seqlen_q, max_seqlen_k, window_size_left, window_size_right, causal
                    )
        elif (window_size_left, window_size_right) != (-1, -1):
            _check_mla_window_bsnd_metadata(
                q,
                k,
                seqused_k,
                seqused_q,
                scheduler_metadata,
                window_size_left,
                window_size_right,
                causal,
            )
        elif causal:
            _check_mla_causal_bsnd_metadata(q, seqused_k, seqused_q, scheduler_metadata)
        elif seqused_q is not None:
            _check_mla_bsnd_used(q, seqused_q, seqused_k, scheduler_metadata)
        else:
            _check_mla_forward_metadata(q, seqused_k, scheduler_metadata)
    return None


def get_scheduler_metadata(
    batch_size,
    max_seqlen_q,
    max_seqlen_k,
    num_heads_q,
    num_heads_kv,
    headdim,
    cache_seqlens: torch.Tensor,
    qkv_dtype=torch.bfloat16,
    headdim_v=None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k_new: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    page_size: Optional[int] = None,
    max_seqlen_k_new=0,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    attention_chunk=0,
    has_softcap=False,
    num_splits=0,  # Can be tuned for speed
    pack_gqa=None,  # Can be tuned for speed
    sm_margin=0,  # Can be tuned if some SMs are used for communication
):
    if qkv_dtype == torch.int8:
        if headdim != 64 or headdim_v != 512 or num_heads_kv != 1 or page_size != 128:
            raise RuntimeError(
                "MLA INT8: metadata requires BSND/TND Q, 64/512 shared content and ND PA page=128"
            )
        _check_mla_metadata_options(
            num_heads_kv,
            max_seqlen_k_new,
            causal,
            window_size,
            attention_chunk,
            has_softcap,
            num_splits,
            sm_margin,
            (cu_seqlens_k_new, cache_leftpad),
            allow_causal=True,
            allow_split=True,
        )
        _check_mla_gqa_options(pack_gqa, sm_margin)
        if num_splits not in (0, 1) or _mla_gqa_active(num_heads_q, 1, pack_gqa, sm_margin):
            raise RuntimeError("MLA INT8: unsupported split/pack_gqa/sm_margin")
        return torch.ops.flash_attn_npu_3._mla_int8_metadata(
            batch_size,
            max_seqlen_q,
            num_heads_q,
            max_seqlen_k,
            cache_seqlens,
            num_splits,
            causal,
            cu_seqlens_q,
        )
    if check_metadata(headdim, headdim_v):
        _check_mla_metadata_options(
            num_heads_kv,
            max_seqlen_k_new,
            causal,
            (window_size[0], window_size[1]),
            attention_chunk,
            has_softcap,
            num_splits,
            sm_margin,
            (cu_seqlens_k_new, cache_leftpad),
            allow_causal=True,
            allow_window=True,
            allow_split=True,
            prefill=headdim == 192,
            allow_softcap=True,
        )
        _check_mla_gqa_options(pack_gqa, sm_margin)
        if _mla_gqa_active(num_heads_q, num_heads_kv, pack_gqa, sm_margin):
            if page_size is not None and page_size == 0:
                raise RuntimeError("MLA: page_size must be nonzero")
            return torch.ops.flash_attn_npu_3._mla_gqa_metadata(
                batch_size,
                max_seqlen_q,
                max_seqlen_k,
                num_heads_q,
                num_heads_kv,
                headdim_v,
                qkv_dtype,
                cache_seqlens,
                cu_seqlens_q,
                num_splits,
                window_size[0],
                window_size[1],
                causal,
                page_size or 0,
                has_softcap,
                pack_gqa,
                sm_margin,
            )
        if headdim == 192:
            if page_size is not None and page_size == 0:
                raise RuntimeError("MLA: prefill page_size must be 16..1024 in multiples of 16")
            if cu_seqlens_q is not None:
                return torch.ops.flash_attn_npu_3._mla_prefill_mixed_metadata(
                    batch_size,
                    max_seqlen_q,
                    max_seqlen_k,
                    num_heads_q,
                    num_heads_kv,
                    qkv_dtype,
                    cache_seqlens,
                    cu_seqlens_q,
                    num_splits,
                    window_size[0],
                    window_size[1],
                    causal,
                    None,
                    page_size or 0,
                    has_softcap=has_softcap,
                )
            return torch.ops.flash_attn_npu_3._mla_prefill_bsnd_metadata(
                batch_size,
                max_seqlen_q,
                max_seqlen_k,
                num_heads_q,
                num_heads_kv,
                qkv_dtype,
                cache_seqlens,
                num_splits,
                window_size[0],
                window_size[1],
                causal,
                page_size or 0,
                has_softcap=has_softcap,
            )
        if num_splits != 1 or has_softcap:
            if page_size is not None:
                return torch.ops.flash_attn_npu_3._mla_paged_metadata(
                    batch_size,
                    max_seqlen_q,
                    max_seqlen_k,
                    num_heads_q,
                    page_size,
                    qkv_dtype,
                    cache_seqlens,
                    cu_seqlens_q,
                    causal,
                    window_size[0],
                    window_size[1],
                    num_splits,
                    has_softcap=has_softcap,
                )
            if cu_seqlens_q is not None:
                return torch.ops.flash_attn_npu_3._mla_split_mixed_metadata(
                    batch_size,
                    max_seqlen_q,
                    max_seqlen_k,
                    num_heads_q,
                    qkv_dtype,
                    cache_seqlens,
                    cu_seqlens_q,
                    window_size[0],
                    window_size[1],
                    causal,
                    num_splits,
                    has_softcap=has_softcap,
                )
            return torch.ops.flash_attn_npu_3._mla_split_bsnd_metadata(
                batch_size,
                max_seqlen_q,
                max_seqlen_k,
                num_heads_q,
                qkv_dtype,
                cache_seqlens,
                window_size[0],
                window_size[1],
                causal,
                num_splits,
                has_softcap=has_softcap,
            )
        if page_size is not None:
            return torch.ops.flash_attn_npu_3._mla_paged_metadata(
                batch_size,
                max_seqlen_q,
                max_seqlen_k,
                num_heads_q,
                page_size,
                qkv_dtype,
                cache_seqlens,
                cu_seqlens_q,
                causal,
                window_size[0],
                window_size[1],
            )
        if window_size != (-1, -1):
            if cu_seqlens_q is not None:
                return torch.ops.flash_attn_npu_3._mla_window_mixed_metadata(
                    batch_size,
                    max_seqlen_q,
                    max_seqlen_k,
                    num_heads_q,
                    qkv_dtype,
                    cache_seqlens,
                    cu_seqlens_q,
                    window_size[0],
                    window_size[1],
                    causal,
                )
            return torch.ops.flash_attn_npu_3._mla_window_bsnd_metadata(
                batch_size,
                max_seqlen_q,
                max_seqlen_k,
                num_heads_q,
                qkv_dtype,
                cache_seqlens,
                window_size[0],
                window_size[1],
                causal,
            )
        if cu_seqlens_q is not None:
            op = (
                torch.ops.flash_attn_npu_3._mla_causal_mixed_metadata
                if causal
                else torch.ops.flash_attn_npu_3._mla_tnd_bsnd_metadata
            )
            return op(
                batch_size,
                max_seqlen_q,
                max_seqlen_k,
                num_heads_q,
                qkv_dtype,
                cache_seqlens,
                cu_seqlens_q,
            )
        if causal:
            return torch.ops.flash_attn_npu_3._mla_causal_bsnd_metadata(
                batch_size, max_seqlen_q, max_seqlen_k, num_heads_q, qkv_dtype, cache_seqlens
            )
        return torch.ops.flash_attn_npu_3._mla_dense_metadata(
            batch_size, max_seqlen_q, max_seqlen_k, num_heads_q, qkv_dtype, cache_seqlens
        )


def int8_forward(
    q,
    k_cache,
    v_cache,
    qv,
    page_table,
    mla_q_scale,
    mla_k_scale,
    mla_v_scale,
    cache_seqlens,
    scheduler_metadata,
    cu_seqlens_q,
    max_seqlen_q,
    softmax_scale,
    causal,
    window_size,
    attention_chunk,
    softcap,
    num_splits,
    pack_gqa,
    sm_margin,
    rotary_interleaved,
    return_softmax_lse,
    optional_tensors,
):
    if (
        qv is None
        or page_table is None
        or any(t is None for t in (mla_q_scale, mla_k_scale, mla_v_scale))
    ):
        raise RuntimeError(
            "MLA INT8: qv, ND page_table and all three mla_*_scale tensors are required"
        )
    tnd = q.ndim == 3
    if (tnd and (cu_seqlens_q is None or max_seqlen_q is None)) or (
        not tnd and (q.ndim != 4 or cu_seqlens_q is not None or max_seqlen_q is not None)
    ):
        raise RuntimeError(
            "MLA INT8: TND requires cu_seqlens_q/max_seqlen_q; BSND does not accept them"
        )
    _check_mla_dense_options(
        softmax_scale=softmax_scale,
        causal=causal,
        allow_causal=True,
        window_size=window_size,
        attention_chunk=attention_chunk,
        softcap=softcap,
        num_splits=num_splits,
        allow_split=True,
        pack_gqa=pack_gqa,
        sm_margin=sm_margin,
        rotary_interleaved=rotary_interleaved,
        optional_tensors=optional_tensors,
    )
    if num_splits not in (0, 1) or _mla_gqa_active(q.shape[-2], 1, pack_gqa, sm_margin):
        raise RuntimeError(
            "MLA INT8: only native num_splits=0/1 and default effective pack_gqa/sm_margin are connected"
        )
    if isinstance(cache_seqlens, int):
        if not 0 <= cache_seqlens <= 2**31 - 1:
            raise RuntimeError("MLA INT8: invalid cache_seqlens")
        cache_seqlens = torch.full(
            (page_table.shape[0] if tnd else q.shape[0],),
            cache_seqlens,
            dtype=torch.int32,
            device=q.device,
        )
    op = (
        torch.ops.flash_attn_npu_3._mla_int8_tnd
        if tnd
        else torch.ops.flash_attn_npu_3._mla_int8_bsnd
    )
    extra = (cu_seqlens_q, max_seqlen_q) if tnd else ()
    result = op(
        q,
        k_cache,
        v_cache,
        qv,
        page_table,
        mla_q_scale,
        mla_k_scale,
        mla_v_scale,
        cache_seqlens,
        scheduler_metadata,
        softmax_scale,
        causal,
        num_splits,
        return_softmax_lse,
        *extra,
    )
    return result if return_softmax_lse else result[0]
