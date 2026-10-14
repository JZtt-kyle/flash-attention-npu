# Copyright (c) 2023, Tri Dao.
# Modified by Minghua Shen, 2026

from typing import Optional, Union, Tuple

import torch

# isort: off
# We need to import the kernels after importing torch
from . import flash_attn_npu_3  # Registers operators with PyTorch

# isort: on

if torch.__version__ >= "2.4.0":
    _torch_custom_op_wrapper = torch.library.custom_op
    _torch_register_fake_wrapper = torch.library.register_fake
else:

    def noop_custom_op_wrapper(name, fn=None, /, *, mutates_args, device_types=None, schema=None):
        def wrap(func):
            return func

        if fn is None:
            return wrap
        return fn

    def noop_register_fake_wrapper(op, fn=None, /, *, lib=None, _stacklevel=1):
        def wrap(func):
            return func

        if fn is None:
            return wrap
        return fn

    _torch_custom_op_wrapper = noop_custom_op_wrapper
    _torch_register_fake_wrapper = noop_register_fake_wrapper


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def round_multiple(x, m):
    return (x + m - 1) // m * m


def _check_fwd_dimension(value, name):
    if not 0 <= value <= 2**31 - 1:
        raise RuntimeError(
            f"{name} must be in [0, {2**31 - 1}] for the 32-bit forward scheduler; got {value}"
        )


_HEADDIM_BWD_ALIGN = 64


def _pad_bwd_headdim(dout, q, k, v, out, head_size_og):
    """
    Pad headdim to a multiple of 64 for the bwd kernel.
    """
    q_dtype = q.dtype
    if q_dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"mha_bwd only supports FP16 and BF16, got {q_dtype}")
    for name, t in (("k", k), ("v", v), ("out", out), ("dout", dout)):
        if t.dtype != q_dtype:
            raise ValueError(
                f"mha_bwd: q/k/v/out/dout must have the same dtype, "
                f"got q={q_dtype}, {name}={t.dtype}"
            )
    if dout.size(-1) != head_size_og:
        raise ValueError(
            f"dout headdim ({dout.size(-1)}) must equal original q/k/v headdim ({head_size_og})"
        )
    qkv_out_headdims = [t.size(-1) for t in (q, k, v, out)]
    if len(set(qkv_out_headdims)) != 1:
        raise ValueError(
            f"q/k/v/out must share the same headdim, got {qkv_out_headdims} "
            "(unequal headdim is not supported)"
        )
    ctx_headdim = qkv_out_headdims[0]
    if ctx_headdim <= 0 or ctx_headdim > 256:
        raise ValueError(f"headdim must be in (0, 256], got {ctx_headdim} ")
    target = round_multiple(ctx_headdim, _HEADDIM_BWD_ALIGN)

    def _pad(t):
        cur = t.size(-1)
        if cur == target:
            return t
        if cur > target:
            raise ValueError(f"headdim {cur} > pad target {target}")
        return torch.nn.functional.pad(t, [0, target - cur])

    return _pad(dout), _pad(q), _pad(k), _pad(v), _pad(out), head_size_og


def _validate_varlen_query(q, cu_seqlens_q):
    if q.dim() != 3:
        raise RuntimeError(f"q must be 3-dimensional; got {q.dim()} dimensions")
    if cu_seqlens_q.numel() <= 1:
        raise RuntimeError("batch size must be positive")
    if q.shape[0] == 0:
        raise RuntimeError("total_q must be positive (q must contain at least one query token)")


def round_up_headdim(head_size: int) -> int:
    return round_multiple(head_size, _HEADDIM_BWD_ALIGN)


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


def _check_mla_inputs(q, k, v, qv=None, *, varlen_q=False, varlen_k=False, paged_kv=False):
    """Recognize validated MLA before copies or ordinary FA dispatch.

    Uses only tensor metadata, including for fake tensors. Keep in sync with
    ascend910/flash_attn_npu_3/mla/mla_validation.h. Outer strides and storage
    offsets are legal for BSND, TND and mixed views; no view is materialized here.
    """

    def check(condition, message):
        if not condition:
            raise RuntimeError(message)

    check(q.ndim == (3 if varlen_q else 4), "FlashAttention: q rank must match BSND/TND layout")
    check(
        k.ndim == (3 if varlen_k else 4) and v.ndim == k.ndim,
        "FlashAttention: k/v rank must match dense, TND or paged KV layout",
    )
    check(q.shape[-1] == k.shape[-1], "FlashAttention: q and k head dimensions must match")
    if qv is None and q.shape[-1] == v.shape[-1]:
        return False
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
    raise RuntimeError("NPU FlashAttention MLA prefill forward is not implemented yet")


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
):
    if any(t is not None for t in optional_tensors) or not rotary_interleaved:
        raise RuntimeError(
            "MLA compressed forward is not implemented yet for optional tensors or length/rotary options"
        )
    if (
        (causal and not allow_causal)
        or (window_size != (-1, -1) and not allow_window)
        or attention_chunk != 0
        or softcap != 0.0
        or (num_splits != 1 and not (allow_split and 0 <= num_splits <= 256))
        or pack_gqa is not None
        or sm_margin != 0
    ):
        raise RuntimeError(
            "MLA compressed forward is not implemented yet for mask, split, softcap or scheduling options"
        )
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


def _check_mla_metadata(headdim, headdim_v):
    if headdim == headdim_v:
        return False
    # Select only the schedule family. Forward still requires a real qv.
    if (headdim, headdim_v) != (64, 512):
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
):
    # Mirror both paged plans and PagedForward's descriptor storage checks.
    # Physical pages and total TND tokens are independent of logical batch.
    tnd = cu_seqlens_q is not None
    is_window = (window_left, window_right) != (-1, -1)
    prefix_words = 411 if is_window else 408
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
        batch > ((1 << 32) - 1 - prefix_words) // 5
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
    if is_window:
        _normalize_mla_window(q.shape[1], capacity, window_left, window_right, causal)
    if page_table.stride(0) < 0 or page_table.stride(1) != 1:
        raise RuntimeError(
            "MLA: page_table requires nonnegative row stride and contiguous last dimension"
        )
    if seqused_q is not None and metadata is not None:
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
            or metadata.numel() != prefix_words + 5 * batch
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


@_torch_register_fake_wrapper("flash_attn_npu_3::_mla_paged_bsnd_forward")
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
    _check_mla_inputs(q, k, v, qv, paged_kv=True)
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
    return (
        q.new_empty((*q.shape[:-1], 512)),
        q.new_empty((q.shape[0], q.shape[2], q.shape[1]), dtype=torch.float32),
    )


@_torch_register_fake_wrapper("flash_attn_npu_3::_mla_paged_tnd_forward")
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
    _check_mla_inputs(q, k, v, qv, varlen_q=True, paged_kv=True)
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
    return (
        q.new_empty((*q.shape[:-1], 512)),
        q.new_empty((q.shape[1], q.shape[0]), dtype=torch.float32),
    )


def _mla_split_bsnd_words(batch, splits=2):
    fixed = 758 if splits == 0 else 416
    if not 0 <= batch <= (2**32 - 1 - fixed) // 10:
        raise RuntimeError("MLA: split metadata word count overflow")
    return fixed + 10 * batch


def _check_mla_split_bsnd(q, k, lengths, used_q, metadata, splits, left, right, causal):
    if splits != 0 and not 2 <= splits <= 256:
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


@_torch_register_fake_wrapper("flash_attn_npu_3::_mla_split_bsnd_forward")
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
):
    _check_mla_inputs(q, k, v, qv)
    _check_mla_dense_options(softmax_scale=softmax_scale)
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
    )
    batch, sq, heads = q.shape[:3]
    capacity = 26 if num_splits == 0 else num_splits
    return (
        q.new_empty((batch, sq, heads, 512)),
        q.new_empty((batch, heads, sq), dtype=torch.float32),
        q.new_empty((capacity, batch, heads, sq, 512), dtype=torch.float32),
        q.new_empty((capacity, batch, heads, sq), dtype=torch.float32),
    )


@_torch_register_fake_wrapper("flash_attn_npu_3::_mla_window_bsnd_forward")
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
    _check_mla_inputs(q, k, v, qv)
    _check_mla_dense_options(softmax_scale=softmax_scale)
    _check_mla_window_bsnd_metadata(
        q, k, cache_seqlens, seqused_q, scheduler_metadata, window_left, window_right, causal
    )
    return (
        q.new_empty((*q.shape[:-1], 512)),
        q.new_empty((q.shape[0], q.shape[2], q.shape[1]), dtype=torch.float32),
    )


@_torch_register_fake_wrapper("flash_attn_npu_3::_mla_window_bsnd_cache")
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


@_torch_register_fake_wrapper("flash_attn_npu_3::_mla_causal_bsnd_forward")
def _mla_causal_bsnd_forward_fake(
    q, k, v, qv, softmax_scale=None, cache_seqlens=None, seqused_q=None, scheduler_metadata=None
):
    _check_mla_inputs(q, k, v, qv)
    _check_mla_dense_options(softmax_scale=softmax_scale, causal=True, allow_causal=True)
    _check_mla_causal_bsnd_metadata(q, cache_seqlens, seqused_q, scheduler_metadata)
    return (
        torch.empty((*q.shape[:-1], 512), dtype=q.dtype, device=q.device),
        torch.empty((q.shape[0], q.shape[2], q.shape[1]), dtype=torch.float32, device=q.device),
    )


@_torch_register_fake_wrapper("flash_attn_npu_3::_mla_causal_bsnd_cache")
def _mla_causal_bsnd_cache_fake(
    q, k, v, qv, softmax_scale=None, cache_seqlens=None, scheduler_metadata=None
):
    out, lse = _mla_causal_bsnd_forward_fake(
        q, k, v, qv, softmax_scale, cache_seqlens, None, scheduler_metadata
    )
    return out, lse, torch.empty(0, device=q.device), torch.empty(0, device=q.device)


@_torch_register_fake_wrapper("flash_attn_npu_3::_mla_dense_forward")
def _mla_dense_forward_fake(
    q, k, v, qv, softmax_scale=None, cache_seqlens=None, scheduler_metadata=None
):
    _check_mla_inputs(q, k, v, qv)
    _check_mla_dense_options(softmax_scale=softmax_scale)
    _check_mla_forward_metadata(q, cache_seqlens, scheduler_metadata)
    return (
        q.new_empty((*q.shape[:-1], v.shape[-1])),
        q.new_empty((q.shape[0], q.shape[2], q.shape[1]), dtype=torch.float32),
    )


@_torch_register_fake_wrapper("flash_attn_npu_3::_mla_tnd_forward")
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
    _check_mla_inputs(q, k, v, qv, varlen_q=True, varlen_k=True)
    _check_mla_dense_options(softmax_scale=softmax_scale)
    _check_mla_tnd_inputs(q, k, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)
    if seqused_q is not None or seqused_k is not None:
        _check_mla_used(q, cu_seqlens_q.numel() - 1, seqused_q, seqused_k)
    return (
        q.new_empty((*q.shape[:-1], v.shape[-1])),
        q.new_empty((q.shape[1], q.shape[0]), dtype=torch.float32),
    )


@_torch_register_fake_wrapper("flash_attn_npu_3::_mla_causal_tnd_forward")
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


@_torch_register_fake_wrapper("flash_attn_npu_3::_mla_split_tnd_forward")
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
):
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
    if not 2 <= num_splits <= 256:
        raise RuntimeError("MLA: TND split requires num_splits in [2,256]")
    words = _mla_split_bsnd_words(cu_seqlens_q.numel() - 1)
    _normalize_mla_window(max_seqlen_q, max_seqlen_k, window_left, window_right, causal)
    if scheduler_metadata is not None and (
        scheduler_metadata.dtype != torch.int32
        or scheduler_metadata.ndim != 1
        or scheduler_metadata.numel() != words
        or not scheduler_metadata.is_contiguous()
        or scheduler_metadata.device != q.device
    ):
        raise RuntimeError(
            "MLA: TND split metadata must be same-device contiguous int32 with the v19 size"
        )
    total, heads = q.shape[:2]
    batch = cu_seqlens_q.numel() - 1
    # Match the static host TND envelope without reading device prefixes/used.
    # Byte and empty-shape stride checks precede fake accumulator allocation.
    limit = 2**63 - 1
    if num_splits * heads * max(total, 1) * 512 > limit:
        raise RuntimeError("MLA: TND split accumulator stride overflow")
    if num_splits * heads * total * 512 * 4 > limit:
        raise RuntimeError("MLA: TND split workspace or accumulator overflow")
    if total and k.shape[0]:
        active = min(batch, total)
        groups = min(
            active * ((max_seqlen_q * heads + 255) // 256), (total * heads + active * 255) // 256
        )
        if (
            groups * num_splits * 256 * 528 * 4 + 2 * 26 * (6 * 256 * 512 + 8 * 256 * 512 + 8 * 256)
            > limit
        ):
            raise RuntimeError("MLA: TND split workspace or accumulator overflow")
    return (
        out,
        lse,
        q.new_empty((num_splits, q.shape[1], q.shape[0], 512), dtype=torch.float32),
        q.new_empty((num_splits, q.shape[1], q.shape[0]), dtype=torch.float32),
    )


@_torch_register_fake_wrapper("flash_attn_npu_3::_mla_split_tnd_metadata")
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
    )
    return q.new_empty((_mla_split_bsnd_words(cu_seqlens_q.numel() - 1),), dtype=torch.int32)


@_torch_register_fake_wrapper("flash_attn_npu_3::_mla_window_tnd_forward")
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


@_torch_register_fake_wrapper("flash_attn_npu_3::_mla_window_mixed_forward")
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
    _check_mla_inputs(q, k, v, qv, varlen_q=True)
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
    return q.new_empty((*q.shape[:-1], 512)), q.new_empty(
        (q.shape[1], q.shape[0]), dtype=torch.float32
    )


@_torch_register_fake_wrapper("flash_attn_npu_3::_mla_window_mixed_cache")
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


@_torch_register_fake_wrapper("flash_attn_npu_3::_mla_causal_mixed_forward")
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
    _check_mla_inputs(q, k, v, qv, varlen_q=True)
    _check_mla_dense_options(softmax_scale=softmax_scale, causal=True, allow_causal=True)
    _check_mla_causal_mixed_inputs(
        q, k, cu_seqlens_q, max_seqlen_q, cache_seqlens, scheduler_metadata, seqused_q
    )
    return q.new_empty((*q.shape[:-1], 512)), q.new_empty(
        (q.shape[1], q.shape[0]), dtype=torch.float32
    )


@_torch_register_fake_wrapper("flash_attn_npu_3::_mla_causal_mixed_cache")
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


@_torch_register_fake_wrapper("flash_attn_npu_3::_mla_tnd_bsnd_forward")
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
    _check_mla_inputs(q, k, v, qv, varlen_q=True)
    _check_mla_dense_options(softmax_scale=softmax_scale)
    _check_mla_mixed_inputs(q, k, cu_seqlens_q, max_seqlen_q, cache_seqlens, scheduler_metadata)
    return (
        q.new_empty((*q.shape[:-1], v.shape[-1])),
        q.new_empty((q.shape[1], q.shape[0]), dtype=torch.float32),
    )


@_torch_custom_op_wrapper(
    "flash_attn_npu_3::_flash_attn_forward", mutates_args=(), device_types="npu"
)
def _flash_attn_forward(
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if _check_mla_inputs(
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
            num_splits=num_splits,
            allow_split=page_table is None
            and (
                (cu_seqlens_q is None and cu_seqlens_k is None)
                or (num_splits > 1 and cu_seqlens_q is not None and cu_seqlens_k is not None)
            ),
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
        if num_splits != 1:
            if cu_seqlens_q is not None:
                return torch.ops.flash_attn_npu_3._mla_split_tnd_forward(
                    q,
                    k,
                    v,
                    qv,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    max_seqlen_q,
                    max_seqlen_k,
                    num_splits,
                    window_size_left,
                    window_size_right,
                    causal,
                    softmax_scale,
                    seqused_q,
                    seqused_k,
                    scheduler_metadata,
                )
            return torch.ops.flash_attn_npu_3._mla_split_bsnd_forward(
                q,
                k,
                v,
                qv,
                num_splits,
                window_size_left,
                window_size_right,
                causal,
                softmax_scale,
                seqused_k,
                seqused_q,
                scheduler_metadata,
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
    q, k, k_new, v_new = [maybe_contiguous(x) for x in (q, k, k_new, v_new)]
    v = v.contiguous() if v.stride(-1) != 1 and v.stride(-3) != 1 else v
    cu_seqlens_q, cu_seqlens_k, cu_seqlens_k_new = [
        maybe_contiguous(x) for x in (cu_seqlens_q, cu_seqlens_k, cu_seqlens_k_new)
    ]
    seqused_q, seqused_k = [maybe_contiguous(x) for x in (seqused_q, seqused_k)]
    page_table, kv_batch_idx, leftpad_k = [
        maybe_contiguous(x) for x in (page_table, kv_batch_idx, leftpad_k)
    ]
    rotary_cos, rotary_sin = [maybe_contiguous(x) for x in (rotary_cos, rotary_sin)]
    seqlens_rotary = maybe_contiguous(seqlens_rotary)
    out, softmax_lse, out_accum, softmax_lse_accum = flash_attn_npu_3.fwd(
        q,
        k,
        v,
        k_new,
        v_new,
        qv,
        out_,
        cu_seqlens_q,
        cu_seqlens_k,
        cu_seqlens_k_new,
        seqused_q,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        page_table,
        kv_batch_idx,
        leftpad_k,
        rotary_cos,
        rotary_sin,
        seqlens_rotary,
        q_descale,
        k_descale,
        v_descale,
        softmax_scale,
        causal,
        window_size_left,
        window_size_right,
        attention_chunk,
        softcap,
        rotary_interleaved,
        scheduler_metadata,
        num_splits,
        pack_gqa,
        sm_margin,
    )

    if out_accum is None:
        out_accum = torch.tensor([], device=out.device)

    if softmax_lse_accum is None:
        softmax_lse_accum = torch.tensor([], device=out.device)

    return out, softmax_lse, out_accum, softmax_lse_accum


@_torch_register_fake_wrapper("flash_attn_npu_3::_flash_attn_forward")
def _flash_attn_forward_fake(
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Symbolic fake implementation of flash attention forward.
    Returns tensors with the correct shapes and dtypes without actual computation.
    """
    if _check_mla_inputs(
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
            num_splits=num_splits,
            allow_split=page_table is None
            and (
                (cu_seqlens_q is None and cu_seqlens_k is None)
                or (num_splits > 1 and cu_seqlens_q is not None and cu_seqlens_k is not None)
            ),
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
        if num_splits != 1:
            if cu_seqlens_q is not None:
                return _mla_split_tnd_forward_fake(
                    q,
                    k,
                    v,
                    qv,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    max_seqlen_q,
                    max_seqlen_k,
                    num_splits,
                    window_size_left,
                    window_size_right,
                    causal,
                    softmax_scale,
                    seqused_q,
                    seqused_k,
                    scheduler_metadata,
                )
            return _mla_split_bsnd_forward_fake(
                q,
                k,
                v,
                qv,
                num_splits,
                window_size_left,
                window_size_right,
                causal,
                softmax_scale,
                seqused_k,
                seqused_q,
                scheduler_metadata,
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

    # Determine if we're in varlen mode
    is_varlen_q = cu_seqlens_q is not None
    if is_varlen_q:
        _validate_varlen_query(q, cu_seqlens_q)
        _check_mla_inputs(q, k, v, qv, varlen_q=True, varlen_k=True)

    # Get dimensions from query tensor
    if is_varlen_q:
        # varlen mode: q is (total_q, num_heads, head_size)
        total_q, num_heads, head_size = q.shape
        batch_size = cu_seqlens_q.shape[0] - 1

        if max_seqlen_q is None:
            raise ValueError("max_seqlen_q must be provided if cu_seqlens_q is provided")
        seqlen_q = max_seqlen_q
    else:
        # batch mode: q is (batch_size, seqlen_q, num_heads, head_size)
        batch_size, seqlen_q, num_heads, head_size = q.shape
        total_q = batch_size * q.shape[1]

    _check_fwd_dimension(batch_size, "batch_size")
    _check_fwd_dimension(seqlen_q, "max_seqlen_q" if is_varlen_q else "seqlen_q")

    # Get value head dimension
    head_size_v = v.shape[-1]

    # Determine output dtype (FP8 inputs produce BF16 outputs)
    q_type = q.dtype
    if q_type == torch.float8_e4m3fn:
        out_dtype = torch.bfloat16
    else:
        out_dtype = q_type

    # Create output tensor
    if out_ is not None:
        # If out_ is provided, _flash_attn_forward becomes non-functional
        raise TypeError(
            "Tracing (torch.compile/torch.export) with pre-allocated output tensor is not supported."
        )

    if is_varlen_q:
        out = torch.empty((total_q, num_heads, head_size_v), dtype=out_dtype, device=q.device)
    else:
        out = torch.empty(
            (batch_size, seqlen_q, num_heads, head_size_v), dtype=out_dtype, device=q.device
        )

    # Create softmax_lse tensor
    if is_varlen_q:
        softmax_lse = torch.empty((num_heads, total_q), dtype=torch.float32, device=q.device)
    else:
        softmax_lse = torch.empty(
            (batch_size, num_heads, seqlen_q), dtype=torch.float32, device=q.device
        )

    # TODO(guilhermeleobas): Implement "get_num_splits"
    # There's an heuristic to compute num_splits when "num_splits <= 0"
    # assert that num_splits is > 0 for now
    if num_splits <= 0:
        raise ValueError(
            f"tracing (torch.compile/torch.export) with num_splits <= 0 not supported. Got {num_splits=}"
        )

    if num_splits > 1:
        if is_varlen_q:
            out_accum = torch.empty(
                (num_splits, num_heads, total_q, head_size_v), dtype=torch.float32, device=q.device
            )
            softmax_lse_accum = torch.empty(
                (num_splits, num_heads, total_q), dtype=torch.float32, device=q.device
            )
        else:
            out_accum = torch.empty(
                (num_splits, batch_size, num_heads, seqlen_q, head_size_v),
                dtype=torch.float32,
                device=q.device,
            )
            softmax_lse_accum = torch.empty(
                (num_splits, batch_size, num_heads, seqlen_q), dtype=torch.float32, device=q.device
            )
    else:
        # Tensors are not set when num_splits < 1
        out_accum = torch.tensor([], device=out.device)
        softmax_lse_accum = torch.tensor([], device=out.device)

    return out, softmax_lse, out_accum, softmax_lse_accum


@_torch_custom_op_wrapper(
    "flash_attn_npu_3::_flash_attn_backward", mutates_args=("dq", "dk", "dv"), device_types="npu"
)
def _flash_attn_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    sequed_q: Optional[torch.Tensor] = None,
    sequed_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    dq: Optional[torch.Tensor] = None,
    dk: Optional[torch.Tensor] = None,
    dv: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    is_causal: bool = False,
    window_size_left: int = -1,
    window_size_right: int = -1,
    softcap: float = 0.0,
    deterministic: bool = False,
    sm_margin: int = 0,
) -> torch.Tensor:
    # dq, dk, dv are allocated by us so they should already be contiguous
    dout, q, k, v, out = [maybe_contiguous(x) for x in (dout, q, k, v, out)]
    (
        dq,
        dk,
        dv,
        softmax_d,
    ) = flash_attn_npu_3.bwd(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        dq,
        dk,
        dv,
        cu_seqlens_q,
        cu_seqlens_k,
        sequed_q,
        sequed_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        is_causal,
        window_size_left,
        window_size_right,
        softcap,
        deterministic,
        sm_margin,
    )
    return softmax_d


@_torch_register_fake_wrapper("flash_attn_npu_3::_flash_attn_backward")
def _flash_attn_backward_fake(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    sequed_q: Optional[torch.Tensor] = None,
    sequed_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    dq: Optional[torch.Tensor] = None,
    dk: Optional[torch.Tensor] = None,
    dv: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    is_causal: bool = False,
    window_size_left: int = -1,
    window_size_right: int = -1,
    softcap: float = 0.0,
    deterministic: bool = False,
    sm_margin: int = 0,
) -> torch.Tensor:

    is_varlen_q = cu_seqlens_q is not None
    is_varlen_k = cu_seqlens_q is not None
    is_varlen = is_varlen_q or is_varlen_k or sequed_q is not None or sequed_k is not None

    if not is_varlen_q:
        batch_size = q.size(0)
        seqlen_q = q.size(1)
        seqlen_k = k.size(1)
        total_q = batch_size * q.size(1)
    else:
        batch_size = cu_seqlens_q.size(0) - 1
        total_q = q.size(0)
        seqlen_q = max_seqlen_q
        seqlen_k = max_seqlen_k

    if window_size_left >= seqlen_k - 1:
        window_size_left = -1

    if window_size_right >= seqlen_q - 1:
        window_size_right = -1

    if is_causal:
        window_size_right = 0

    is_causal = window_size_left < 0 and window_size_right == 0

    kBlockM = 128

    num_heads = q.shape[-2]
    seqlen_q_rounded = round_multiple(seqlen_q, kBlockM)

    total_q_padded_rounded = round_multiple(total_q + batch_size * kBlockM, kBlockM)

    dq = torch.empty_like(q) if dq is None else dq
    dk = torch.empty_like(k) if dk is None else dk
    dv = torch.empty_like(v) if dv is None else dv

    if not is_varlen:
        softmax_d = torch.empty(
            (batch_size, num_heads, seqlen_q_rounded), dtype=torch.float32, device=q.device
        )
    else:
        softmax_d = torch.empty(
            (num_heads, total_q_padded_rounded), dtype=torch.float32, device=q.device
        )

    return softmax_d


def setup_context(ctx, inputs, output):
    q, k, v = inputs[:3]
    out, softmax_lse, _, _ = output
    ctx.save_for_backward(q, k, v, out, softmax_lse)
    ctx.head_size_og = q.size(-1)
    ctx.softmax_scale = inputs[-11]
    ctx.causal = inputs[-10]
    ctx.window_size = [inputs[-9], inputs[-8]]
    ctx.attention_chunk = inputs[-7]
    ctx.softcap = inputs[-6]
    ctx.sm_margin = inputs[-1]


def _backward(ctx, dout, *grads):
    q, k, v, out, softmax_lse = ctx.saved_tensors
    dout, q, k, v, out, head_size_og = _pad_bwd_headdim(dout, q, k, v, out, ctx.head_size_og)
    dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
    _flash_attn_backward(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        None,
        None,  # cu_seqlens_q, cu_seqlens_k,
        None,
        None,  # sequed_q, sequed_k,
        None,
        None,  # max_seqlen_q, max_seqlen_k,
        dq,
        dk,
        dv,
        ctx.softmax_scale,
        ctx.causal,
        ctx.window_size[0],
        ctx.window_size[1],
        ctx.softcap,
        False,  # deterministic
        ctx.sm_margin,
    )
    dq = dq[..., :head_size_og]
    dk = dk[..., :head_size_og]
    dv = dv[..., :head_size_og]
    return dq, dk, dv, *((None,) * 21)


class FlashAttnQKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv,
        softmax_scale,
        causal,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        window_size=(-1, -1),
        attention_chunk=0,
        softcap=0.0,
        deterministic=False,
        num_heads_q=None,
        sm_margin=0,
        return_softmax=False,
    ):
        if softmax_scale is None:
            softmax_scale = qkv.shape[-1] ** (-0.5)
        if qkv.dim() == 5:
            assert qkv.shape[-3] == 3
            q, k, v = qkv.unbind(dim=-3)
        else:
            assert qkv.dim() == 4
            assert num_heads_q is not None
            num_heads_k = (qkv.shape[2] - num_heads_q) // 2
            assert num_heads_k * 2 + num_heads_q == qkv.shape[2]
            q, k, v = qkv.split([num_heads_q, num_heads_k, num_heads_k], dim=-2)
        out, softmax_lse, *rest = _flash_attn_forward(
            q,
            k,
            v,
            None,
            None,  # k_new, v_new
            None,  # qv
            None,  # out
            None,
            None,
            None,  # cu_seqlens_q/k/k_new
            None,
            None,  # seqused_q/k
            None,
            None,  # max_seqlen_q/k
            None,
            None,
            None,  # page_table, kv_batch_idx, leftpad_k,
            None,
            None,
            None,  # rotary_cos/sin, seqlens_rotary
            q_descale,
            k_descale,
            v_descale,
            softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            attention_chunk=attention_chunk,
            softcap=softcap,
            sm_margin=sm_margin,
        )
        # ctx.save_for_backward(q, k, v, out_padded, softmax_lse)
        ctx.save_for_backward(q, k, v, out, softmax_lse)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.attention_chunk = attention_chunk
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        ctx.ndim = qkv.dim()
        ctx.sm_margin = sm_margin
        ctx.head_size_og = q.size(-1)
        return (out, softmax_lse) if return_softmax else out

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse = ctx.saved_tensors
        assert ctx.attention_chunk == 0, "FA3 backward does not support attention_chunk"
        dout, q, k, v, out, head_size_og = _pad_bwd_headdim(dout, q, k, v, out, ctx.head_size_og)
        if ctx.ndim == 5:
            qkv_shape = q.shape[:-2] + (3, *q.shape[-2:])
            dqkv = torch.empty(qkv_shape, dtype=q.dtype, device=q.device)
            dq, dk, dv = dqkv.unbind(dim=-3)
        else:
            num_heads_q = q.shape[2]
            num_heads_k = k.shape[2]
            qkv_shape = q.shape[:-2] + (num_heads_q + num_heads_k * 2, *q.shape[-1:])
            dqkv = torch.empty(qkv_shape, dtype=q.dtype, device=q.device)
            dq, dk, dv = dqkv.split([num_heads_q, num_heads_k, num_heads_k], dim=-2)
        _flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            None,
            None,  # cu_seqlens_q, cu_seqlens_k,
            None,
            None,  # sequed_q, sequed_k,
            None,
            None,  # max_seqlen_q, max_seqlen_k,
            dq,
            dk,
            dv,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size[0],
            ctx.window_size[1],
            ctx.softcap,
            ctx.deterministic,
            ctx.sm_margin,
        )
        dqkv = dqkv[..., :head_size_og]  # We could have padded the head dimension
        return dqkv, None, None, None, None, None, None, None, None, None, None, None, None


class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        softmax_scale,
        causal,
        qv=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        window_size=(-1, -1),
        attention_chunk=0,
        softcap=0.0,
        num_splits=1,
        pack_gqa=None,
        deterministic=False,
        sm_margin=0,
        return_softmax=False,
    ):
        is_mla = _check_mla_inputs(q, k, v, qv)
        if is_mla:
            _check_mla_dense_options(
                softmax_scale=softmax_scale,
                causal=causal,
                allow_causal=True,
                window_size=window_size,
                attention_chunk=attention_chunk,
                softcap=softcap,
                num_splits=num_splits,
                pack_gqa=pack_gqa,
                sm_margin=sm_margin,
                optional_tensors=(q_descale, k_descale, v_descale),
            )
        if softmax_scale is None:
            softmax_scale = (q.shape[-1] + (qv.shape[-1] if qv is not None else 0)) ** (-0.5)
        # MLA generates and consumes its private AICPU record inside DenseForward.
        # Explicit reuse is available through the read-only kvcache entry.
        scheduler_metadata = None
        if not is_mla:
            # Training interfaces do not expose scheduler_metadata (aligned with
            # official flash-attn); the scheduler metadata is computed internally on
            # the AICPU so no D2H/H2D sync breaks the pipeline.
            batch_size, seqlen_q, num_heads, head_size = q.shape
            seqlen_k, num_heads_k = k.shape[1], k.shape[2]
            cache_seqlens = torch.full((batch_size,), seqlen_k, dtype=torch.int32, device=q.device)
            scheduler_metadata = get_scheduler_metadata(
                batch_size,
                seqlen_q,
                seqlen_k,
                num_heads,
                num_heads_k,
                head_size,
                cache_seqlens,
                qkv_dtype=q.dtype,
                headdim_v=v.shape[-1],
                causal=causal,
                window_size=window_size,
                has_softcap=softcap > 0.0,
                num_splits=num_splits,
                sm_margin=sm_margin,
            )
        # out, q, k, v, out_padded, softmax_lse = _flash_attn_forward(
        out, softmax_lse, *rest = _flash_attn_forward(
            q,
            k,
            v,
            None,
            None,  # k_new, v_new
            qv,  # qv
            None,  # out
            None,
            None,
            None,  # cu_seqlens_q/k/k_new
            None,
            None,  # seqused_q/k
            None,
            None,  # max_seqlen_q/k
            None,
            None,
            None,  # page_table, kv_batch_idx, leftpad_k,
            None,
            None,
            None,  # rotary_cos/sin, seqlens_rotary
            q_descale,
            k_descale,
            v_descale,
            softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            attention_chunk=attention_chunk,
            softcap=softcap,
            scheduler_metadata=scheduler_metadata,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            sm_margin=sm_margin,
        )
        # ctx.save_for_backward(q, k, v, out_padded, softmax_lse)
        ctx.save_for_backward(q, k, v, out, softmax_lse)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.attention_chunk = attention_chunk
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        ctx.sm_margin = sm_margin
        ctx.head_size_og = q.size(-1)
        return (out, softmax_lse) if return_softmax else out

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse = ctx.saved_tensors
        assert ctx.attention_chunk == 0, "FA3 backward does not support attention_chunk"
        dout, q, k, v, out, head_size_og = _pad_bwd_headdim(dout, q, k, v, out, ctx.head_size_og)
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        _flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            None,
            None,  # cu_seqlens_q, cu_seqlens_k,
            None,
            None,  # sequed_q, sequed_k,
            None,
            None,  # max_seqlen_q, max_seqlen_k,
            dq,
            dk,
            dv,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size[0],
            ctx.window_size[1],
            ctx.softcap,
            ctx.deterministic,
            ctx.sm_margin,
        )
        dq = dq[..., :head_size_og]  # We could have padded the head dimension
        dk = dk[..., :head_size_og]
        dv = dv[..., :head_size_og]
        return (
            dq,
            dk,
            dv,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class FlashAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        qv=None,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        window_size=(-1, -1),
        attention_chunk=0,
        softcap=0.0,
        num_splits=1,
        pack_gqa=None,
        deterministic=False,
        sm_margin=0,
        return_softmax=False,
    ):
        if _check_mla_inputs(q, k, v, qv, varlen_q=True, varlen_k=True):
            _check_mla_dense_options(
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                attention_chunk=attention_chunk,
                softcap=softcap,
                num_splits=num_splits,
                pack_gqa=pack_gqa,
                sm_margin=sm_margin,
                optional_tensors=(q_descale, k_descale, v_descale),
            )
            _check_mla_tnd_inputs(q, k, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)
            if seqused_q is not None or seqused_k is not None:
                _check_mla_used(q, cu_seqlens_q.numel() - 1, seqused_q, seqused_k)
            out, lse = torch.ops.flash_attn_npu_3._mla_tnd_forward(
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
            return (out, lse) if return_softmax else out
        _validate_varlen_query(q, cu_seqlens_q)
        if softmax_scale is None:
            softmax_scale = (q.shape[-1] + (qv.shape[-1] if qv is not None else 0)) ** (-0.5)
        # Training interfaces do not expose scheduler_metadata (aligned with
        # official flash-attn); the scheduler metadata is computed internally on
        # the AICPU so no D2H/H2D sync breaks the pipeline. cache_seqlens holds
        # the per-batch KV lengths and stays on device.
        batch_size = cu_seqlens_q.shape[0] - 1
        num_heads, head_size = q.shape[1], q.shape[2]
        num_heads_k = k.shape[1]
        if seqused_k is not None:
            cache_seqlens = seqused_k
        else:
            cache_seqlens = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
        if seqused_q is not None:
            seqlens_q = seqused_q
        else:
            seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
        # Pass the per-batch used lengths directly to the internal scheduler.
        scheduler_metadata = _get_scheduler_metadata_op(
            batch_size,
            max_seqlen_q,
            max_seqlen_k,
            num_heads,
            num_heads_k,
            head_size,
            headdim_v=v.shape[-1],
            qkv_dtype=q.dtype,
            cache_seqlens=maybe_contiguous(cache_seqlens),
            seqlens_q=seqlens_q,
            cu_seqlens_k=None,
            cu_seqlens_k_new=None,
            cache_leftpad=None,
            page_size=None,
            max_seqlen_k_new=0,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            attention_chunk=0,
            has_softcap=softcap > 0.0,
            num_splits=num_splits,
            pack_gqa=None,
            sm_margin=sm_margin,
        )
        # out, q, k, v, out_padded, softmax_lse = _flash_attn_varlen_forward(
        out, softmax_lse, *rest = _flash_attn_forward(
            q,
            k,
            v,
            None,
            None,  # k_new, v_new
            qv,  # qv
            None,  # out
            cu_seqlens_q,
            cu_seqlens_k,
            None,  # cu_seqlens_k_new
            seqused_q,
            seqused_k,
            max_seqlen_q,
            max_seqlen_k,
            None,
            None,
            None,  # page_table, kv_batch_idx, leftpad_k,
            None,
            None,
            None,  # rotary_cos/sin, seqlens_rotary
            q_descale,
            k_descale,
            v_descale,
            softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            attention_chunk=attention_chunk,
            softcap=softcap,
            scheduler_metadata=scheduler_metadata,
            num_splits=num_splits,
            pack_gqa=pack_gqa,
            sm_margin=sm_margin,
        )
        # ctx.save_for_backward(q, k, v, out_padded, softmax_lse, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k)
        ctx.save_for_backward(
            q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k
        )
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.attention_chunk = attention_chunk
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        ctx.sm_margin = sm_margin
        ctx.head_size_og = q.size(-1)
        return (out, softmax_lse) if return_softmax else out

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k = (
            ctx.saved_tensors
        )
        assert ctx.attention_chunk == 0, "FA3 backward does not support attention_chunk"
        dout, q, k, v, out, head_size_og = _pad_bwd_headdim(dout, q, k, v, out, ctx.head_size_og)
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        _flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            dq,
            dk,
            dv,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size[0],
            ctx.window_size[1],
            ctx.softcap,
            ctx.deterministic,
            ctx.sm_margin,
        )
        dq = dq[..., :head_size_og]  # We could have padded the head dimension
        dk = dk[..., :head_size_og]
        dv = dv[..., :head_size_og]
        return (
            dq,
            dk,
            dv,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def flash_attn_qkvpacked_func(
    qkv,
    softmax_scale=None,
    causal=False,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    window_size=(-1, -1),
    attention_chunk=0,
    softcap=0.0,
    deterministic=False,
    num_heads_q=None,
    sm_margin=0,
    return_attn_probs=False,
):
    """dropout_p should be set to 0.0 during evaluation
    If Q, K, V are already stacked into 1 tensor, this function will be faster than
    calling flash_attn_func on Q, K, V since the backward pass avoids explicit concatenation
    of the gradients of Q, K, V.
    For multi-query and grouped-query attention (MQA/GQA), please see
    flash_attn_kvpacked_func and flash_attn_func.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between [i - window_size[0], i + window_size[1]] inclusive.

    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, headdim)
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        softcap: float. Anything > 0 activates softcapping attention.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    return FlashAttnQKVPackedFunc.apply(
        qkv,
        softmax_scale,
        causal,
        q_descale,
        k_descale,
        v_descale,
        window_size,
        attention_chunk,
        softcap,
        deterministic,
        num_heads_q,
        sm_margin,
        return_attn_probs,
    )


def flash_attn_func(
    q,
    k,
    v,
    softmax_scale=None,
    causal=False,
    qv=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    window_size=(-1, -1),
    attention_chunk=0,
    softcap=0.0,
    num_splits=1,
    pack_gqa=None,
    deterministic=False,
    sm_margin=0,
    return_attn_probs=False,
):
    """Compute attention for BSND queries, keys and values.

    For compressed MLA, q/k contain 64-dimensional RoPE components, qv is
    512-dimensional query content, and v is shared content K/V with one head.
    Current MLA options and limits are documented in docs/mla/status.md.

    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k: (batch_size, seqlen, nheads_k, headdim)
        v: (batch_size, seqlen, nheads_k, headdim_v)
        qv [optional]: (batch_size, seqlen, nheads, headdim_v).
        softmax_scale: float. Scales q @ k.T + qv @ v.T when qv is supplied, otherwise q @ k.T.
            Defaults to 1 / sqrt(headdim + headdim_v) with qv, or 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Returns (out, softmax_lse) when True.
    Return:
        out: (batch_size, seqlen, nheads, headdim_v).
        softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The
            FP32 natural log of each row's softmax normalization factor, including
            the qv contribution for MLA.
    """
    # Causal MLA is forward-only. Its real/fake dispatcher validates every
    # tensor, including rejecting gradients, without an autograd context.
    if (
        (causal or window_size != (-1, -1) or num_splits != 1)
        and qv is not None
        and q.device.type == "npu"
    ):
        _check_mla_dense_options(
            causal=causal,
            allow_causal=True,
            allow_window=True,
            window_size=window_size,
            attention_chunk=attention_chunk,
            softcap=softcap,
            num_splits=num_splits,
            allow_split=True,
            pack_gqa=pack_gqa,
            sm_margin=sm_margin,
            optional_tensors=(q_descale, k_descale, v_descale),
        )
        if num_splits != 1:
            out, softmax_lse, _, _ = torch.ops.flash_attn_npu_3._mla_split_bsnd_forward(
                q, k, v, qv, num_splits, window_size[0], window_size[1], causal, softmax_scale
            )
        elif window_size != (-1, -1):
            out, softmax_lse = torch.ops.flash_attn_npu_3._mla_window_bsnd_forward(
                q, k, v, qv, window_size[0], window_size[1], causal, softmax_scale
            )
        else:
            out, softmax_lse = torch.ops.flash_attn_npu_3._mla_causal_bsnd_forward(
                q, k, v, qv, softmax_scale
            )
        return (out, softmax_lse) if return_attn_probs else out
    return FlashAttnFunc.apply(
        q,
        k,
        v,
        softmax_scale,
        causal,
        qv,
        q_descale,
        k_descale,
        v_descale,
        window_size,
        attention_chunk,
        softcap,
        num_splits,
        pack_gqa,
        deterministic,
        sm_margin,
        return_attn_probs,
    )


def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    seqused_q=None,
    seqused_k=None,
    softmax_scale=None,
    causal=False,
    qv=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    window_size=(-1, -1),
    attention_chunk=0,
    softcap=0.0,
    num_splits=1,
    pack_gqa=None,
    deterministic=False,
    sm_margin=0,
    return_attn_probs=False,
):
    """Compute attention for TND inputs with int32 allocation prefixes.

    cu_seqlens_q/k have shape (batch_size + 1,) and start at zero;
    max_seqlen_q/k bound each allocated segment. Optional int32 seqused_q/k
    have shape (batch_size,) and limit the used prefix without moving the
    next sample's allocation. Output keeps the allocated query shape with
    last dimension v.shape[-1]; unused query slots have no value guarantee.
    return_attn_probs=True returns (out, lse), with FP32 lse shaped (nheads, total_q).

    Compressed MLA uses q/k dimensions 64 and qv/v dimensions 512. Metadata
    is generated internally on every call (v19 for explicit num_splits=2..256; v13 for nondefault windows, v8 for right-down causal; otherwise
    v3 without used lengths, v5 with either used array). There is no explicit metadata or page-table parameter;
    paged MLA uses flash_attn_with_kvcache. See docs/mla/status.md for limits.
    """
    if (
        (causal or window_size != (-1, -1) or num_splits != 1)
        and qv is not None
        and q.device.type == "npu"
    ):
        _check_mla_dense_options(
            causal=causal,
            allow_window=True,
            allow_causal=True,
            softmax_scale=softmax_scale,
            window_size=window_size,
            attention_chunk=attention_chunk,
            softcap=softcap,
            num_splits=num_splits,
            allow_split=num_splits > 1,
            pack_gqa=pack_gqa,
            sm_margin=sm_margin,
            optional_tensors=(q_descale, k_descale, v_descale),
        )
        if num_splits != 1:
            out, lse, _, _ = torch.ops.flash_attn_npu_3._mla_split_tnd_forward(
                q,
                k,
                v,
                qv,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                num_splits,
                window_size[0],
                window_size[1],
                causal,
                softmax_scale,
                seqused_q,
                seqused_k,
            )
            return (out, lse) if return_attn_probs else out
        if window_size != (-1, -1):
            out, lse = torch.ops.flash_attn_npu_3._mla_window_tnd_forward(
                q,
                k,
                v,
                qv,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                window_size[0],
                window_size[1],
                causal,
                softmax_scale,
                seqused_q,
                seqused_k,
            )
            return (out, lse) if return_attn_probs else out
        out, lse = torch.ops.flash_attn_npu_3._mla_causal_tnd_forward(
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
        return (out, lse) if return_attn_probs else out
    return FlashAttnVarlenFunc.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        qv,
        q_descale,
        k_descale,
        v_descale,
        window_size,
        attention_chunk,
        softcap,
        num_splits,
        pack_gqa,
        deterministic,
        sm_margin,
        return_attn_probs,
    )


def flash_attn_combine(out_partial, lse_partial, out=None, out_dtype=None):
    return flash_attn_npu_3.fwd_combine(out_partial, lse_partial, out, out_dtype)


def flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    qv=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens: Optional[Union[(int, torch.Tensor)]] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k_new: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    rotary_seqlens: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    attention_chunk=0,
    softcap=0.0,  # 0.0 means deactivated
    rotary_interleaved=True,
    scheduler_metadata=None,
    num_splits=0,  # Can be tuned for speed
    pack_gqa=None,  # Can be tuned for speed
    sm_margin=0,  # Can be tuned if some SMs are used for communication
    return_softmax_lse=False,
):
    """
    NPU compressed MLA supports read-only BSND or ND paged caches with BSND/TND
    queries, including right-down causal and window masks. Nonpaged BSND queries
    support automatic num_splits=0, single split 1, and explicit splits 2..256.
    Other MLA layouts require num_splits=1. Automatic requests return 26 FP32
    accumulator slots; the native partition determines the actual splits per M
    block and unused slots are 0/-inf. Current limits are in docs/mla/status.md.
    Generic FA3 append and rotary options below remain unsupported for MLA.

    If k and v are not None, k_cache and v_cache will be updated *inplace* with the new values from
    k and v. This is useful for incremental decoding: you can pass in the cached keys/values from
    the previous step, and update them with the new keys/values from the current step, and do
    attention with the updated cache, all in 1 kernel.

    If you pass in k / v, you must make sure that the cache is large enough to hold the new values.
    For example, the KV cache could be pre-allocated with the max sequence length, and you can use
    cache_seqlens to keep track of the current sequence lengths of each sequence in the batch.

    Also apply rotary embedding if rotary_cos and rotary_sin are passed in. The key @k will be
    rotated by rotary_cos and rotary_sin at indices cache_seqlens, cache_seqlens + 1, etc.
    If causal or local (i.e., window_size != (-1, -1)), the query @q will be rotated by rotary_cos
    and rotary_sin at indices cache_seqlens, cache_seqlens + 1, etc.
    If not causal and not local, the query @q will be rotated by rotary_cos and rotary_sin at
    indices cache_seqlens only (i.e. we consider all tokens in @q to be at position cache_seqlens).

    See tests/test_flash_attn.py::test_flash_attn_kvcache for examples of how to use this function.

    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Note: Does not support backward pass.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim), or (total_q, nheads, headdim)
            with cu_seqlens_q and max_seqlen_q for TND queries.
        k_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim) if there's no page_table,
            or (num_blocks, page_block_size, nheads_k, headdim) if there's a page_table (i.e. paged KV cache)
            For MLA, page_block_size must be a multiple of 16 in [16, 1024].
        v_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim_v) if there's no page_table,
            or (num_blocks, page_block_size, nheads_k, headdim_v) if there's a page_table (i.e. paged KV cache)
        k [optional]: (batch_size, seqlen_new, nheads_k, headdim). If not None, we concatenate
            k with k_cache, starting at the indices specified by cache_seqlens.
        v [optional]: (batch_size, seqlen_new, nheads_k, headdim_v). Similar to k.
        qv [optional]: q's layout with last dimension headdim_v.
        rotary_cos [optional]: (seqlen_ro, rotary_dim / 2). If not None, we apply rotary embedding
            to k and q. Only applicable if k and v are passed in. rotary_dim must be divisible by 16.
        rotary_sin [optional]: (seqlen_ro, rotary_dim / 2). Similar to rotary_cos.
        cache_seqlens: int, or (batch_size,), dtype torch.int32. The sequence lengths of the
            KV cache. For MLA, None uses full logical capacity; an integer is broadcast
            across logical batches, including TND queries. Paged capacity is
            page_table.shape[1] * k_cache.shape[1], at most 16384 per sample.
        cache_batch_idx: (batch_size,), dtype torch.int32. The indices used to index into the KV cache.
            If None, we assume that the batch indices are [0, 1, 2, ..., batch_size - 1].
            If the indices are not distinct, and k and v are provided, the values updated in the cache
                 might come from any of the duplicate indices.
        cache_leftpad: (batch_size,), dtype torch.int32. The index that the KV cache starts. If None, assume 0.
        page_table [optional]: (batch_size, max_num_blocks_per_seq), dtype torch.int32.
        softmax_scale: float. Scales q @ k.T + qv @ v.T when qv is supplied, otherwise q @ k.T.
            Defaults to 1 / sqrt(headdim + headdim_v) with qv, or 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        softcap: float. Anything > 0 activates softcapping attention.
        rotary_interleaved: bool. Only applicable if rotary_cos and rotary_sin are passed in.
            If True, rotary embedding will combine dimensions 0 & 1, 2 & 3, etc. If False,
            rotary embedding will combine dimensions 0 & rotary_dim / 2, 1 & rotary_dim / 2 + 1
            (i.e. GPT-NeoX style).
        num_splits: int. If > 1, split the key/value into this many chunks along the sequence.
           If num_splits == 1, we don't split the key/value. If num_splits == 0, we use a heuristic
           to automatically determine the number of splits.
           Don't change this unless you know what you are doing.
        scheduler_metadata: Optional scheduler metadata precomputed on the AICPU by
            get_scheduler_metadata. The forward uses the AICPU-computed schedule,
            avoiding host-side computation that would sync on cache_seqlens.
            Each forward supplies its numerical scale and softcap through tiling. Creation arguments are
            validated against this call: causal / window_size / has_softcap /
            num_splits must match, and max_seqlen_k must equal the effective KV cache
            capacity seen by the forward (page_table: page_block_size * page_table.shape[1];
            otherwise k_cache.shape[1]) — do not overprovision it. Pass the tensor
            returned by get_scheduler_metadata unchanged (do not clone/copy it).
            softmax_scale and the numerical softcap value come from this forward
            call and may change when reusing metadata with the same has_softcap.
            MLA uses int32 v2 records for unmasked BSND queries, v7 for causal BSND,
            v4/v9 for unmasked/causal TND queries with a BSND cache, v6 for unmasked paged
            caches, or v10/v11 for causal paged caches with BSND/TND queries.
            The device checks current lengths, TND query prefixes and all used pages.
            MLA records may be cloned or passed as contiguous
            offset views; scale is supplied to forward and is not fixed by the record.
            Reuse skips AICPU generation; host workspace budgeting still runs.
            Current MLA options and limits are documented in docs/mla/status.md.
        return_softmax_lse: bool. Whether to return the logsumexp of the attention scores.

    Return:
        out: query layout with last dimension v_cache.shape[-1].
        If return_softmax_lse=True, returns (out, softmax_lse, out_accum, softmax_lse_accum).
        softmax_lse is FP32 with shape (batch_size, nheads, seqlen) for BSND queries or
            (nheads, total_q) for TND queries. It contains the natural log of each row's
            softmax normalization factor, including the qv contribution for MLA.
        For MLA without split, both accumulators are empty tensors on the output device.
    """
    # The qv dispatchers validate tensors independently in real/fake
    # execution. Route before repeating those checks in Python; other forms
    # retain the common recognition/rejection path.
    is_mla = qv is not None and q.device.type == "npu"
    if not is_mla:
        is_mla = _check_mla_inputs(
            q,
            k_cache,
            v_cache,
            qv,
            varlen_q=cu_seqlens_q is not None,
            paged_kv=page_table is not None,
        )
    if is_mla:
        # The real/fake dispatcher checks scale. Keep its scalar validation
        # inside the op so torch.compile can capture this public wrapper.
        _check_mla_dense_options(
            causal=causal,
            allow_causal=True,
            allow_window=True,
            window_size=window_size,
            attention_chunk=attention_chunk,
            softcap=softcap,
            num_splits=num_splits,
            allow_split=cu_seqlens_q is None and page_table is None,
            pack_gqa=pack_gqa,
            sm_margin=sm_margin,
            rotary_interleaved=rotary_interleaved,
            optional_tensors=(
                k,
                v,
                cu_seqlens_k_new,
                max_seqlen_q if cu_seqlens_q is None else None,
                cache_batch_idx,
                cache_leftpad,
                rotary_cos,
                rotary_sin,
                rotary_seqlens,
                q_descale,
                k_descale,
                v_descale,
            ),
        )
        if page_table is not None and cu_seqlens_q is not None and max_seqlen_q is None:
            raise RuntimeError("MLA: TND paged forward requires max_seqlen_q")
        if isinstance(cache_seqlens, int):
            # Integer convenience input needs valid geometry before creating
            # its device tensor. Tensor/None lengths go straight to the op.
            _check_mla_inputs(
                q,
                k_cache,
                v_cache,
                qv,
                varlen_q=cu_seqlens_q is not None,
                paged_kv=page_table is not None,
            )
            batch = k_cache.shape[0]
            if page_table is not None:
                batch = _check_mla_paged_inputs(
                    q,
                    k_cache,
                    page_table,
                    None,
                    scheduler_metadata,
                    cu_seqlens_q=cu_seqlens_q,
                    max_seqlen_q=max_seqlen_q,
                    window_left=window_size[0],
                    window_right=window_size[1],
                    causal=causal,
                )
            capacity = k_cache.shape[1] * (page_table.shape[1] if page_table is not None else 1)
            if not 0 <= cache_seqlens <= capacity:
                raise RuntimeError("MLA: cache_seqlens must be between zero and KV capacity")
            cache_seqlens = torch.full(
                (batch,),
                cache_seqlens,
                dtype=torch.int32,
                device=k_cache.device,
            )
        if num_splits != 1:
            result = torch.ops.flash_attn_npu_3._mla_split_bsnd_forward(
                q,
                k_cache,
                v_cache,
                qv,
                num_splits,
                window_size[0],
                window_size[1],
                causal,
                softmax_scale,
                cache_seqlens,
                None,
                scheduler_metadata,
            )
            return result if return_softmax_lse else result[0]
        if page_table is not None and cu_seqlens_q is not None:
            out, softmax_lse = torch.ops.flash_attn_npu_3._mla_paged_tnd_forward(
                q,
                k_cache,
                v_cache,
                qv,
                page_table,
                cu_seqlens_q,
                max_seqlen_q,
                softmax_scale,
                cache_seqlens,
                scheduler_metadata=scheduler_metadata,
                causal=causal,
                window_left=window_size[0],
                window_right=window_size[1],
            )
        elif page_table is not None:
            out, softmax_lse = torch.ops.flash_attn_npu_3._mla_paged_bsnd_forward(
                q,
                k_cache,
                v_cache,
                qv,
                page_table,
                softmax_scale,
                cache_seqlens,
                scheduler_metadata=scheduler_metadata,
                causal=causal,
                window_left=window_size[0],
                window_right=window_size[1],
            )
        elif cu_seqlens_q is not None:
            if max_seqlen_q is None:
                raise RuntimeError("MLA: mixed TND/BSND requires max_seqlen_q")
            if window_size != (-1, -1):
                if return_softmax_lse:
                    return torch.ops.flash_attn_npu_3._mla_window_mixed_cache(
                        q,
                        k_cache,
                        v_cache,
                        qv,
                        cu_seqlens_q,
                        max_seqlen_q,
                        window_size[0],
                        window_size[1],
                        causal,
                        softmax_scale,
                        cache_seqlens,
                        scheduler_metadata,
                    )
                out, softmax_lse = torch.ops.flash_attn_npu_3._mla_window_mixed_forward(
                    q,
                    k_cache,
                    v_cache,
                    qv,
                    cu_seqlens_q,
                    max_seqlen_q,
                    window_size[0],
                    window_size[1],
                    causal,
                    softmax_scale,
                    cache_seqlens,
                    scheduler_metadata=scheduler_metadata,
                )
            elif causal:
                if return_softmax_lse:
                    return torch.ops.flash_attn_npu_3._mla_causal_mixed_cache(
                        q,
                        k_cache,
                        v_cache,
                        qv,
                        cu_seqlens_q,
                        max_seqlen_q,
                        softmax_scale,
                        cache_seqlens,
                        scheduler_metadata,
                    )
                out, softmax_lse = torch.ops.flash_attn_npu_3._mla_causal_mixed_forward(
                    q,
                    k_cache,
                    v_cache,
                    qv,
                    cu_seqlens_q,
                    max_seqlen_q,
                    softmax_scale,
                    cache_seqlens,
                    scheduler_metadata=scheduler_metadata,
                )
            else:
                out, softmax_lse = torch.ops.flash_attn_npu_3._mla_tnd_bsnd_forward(
                    q,
                    k_cache,
                    v_cache,
                    qv,
                    cu_seqlens_q,
                    max_seqlen_q,
                    softmax_scale,
                    cache_seqlens,
                    scheduler_metadata,
                )
        elif window_size != (-1, -1):
            if return_softmax_lse:
                return torch.ops.flash_attn_npu_3._mla_window_bsnd_cache(
                    q,
                    k_cache,
                    v_cache,
                    qv,
                    window_size[0],
                    window_size[1],
                    causal,
                    softmax_scale,
                    cache_seqlens,
                    scheduler_metadata,
                )
            out, softmax_lse = torch.ops.flash_attn_npu_3._mla_window_bsnd_forward(
                q,
                k_cache,
                v_cache,
                qv,
                window_size[0],
                window_size[1],
                causal,
                softmax_scale,
                cache_seqlens,
                scheduler_metadata=scheduler_metadata,
            )
        elif causal and return_softmax_lse:
            return torch.ops.flash_attn_npu_3._mla_causal_bsnd_cache(
                q, k_cache, v_cache, qv, softmax_scale, cache_seqlens, scheduler_metadata
            )
        elif causal:
            out, softmax_lse = torch.ops.flash_attn_npu_3._mla_causal_bsnd_forward(
                q,
                k_cache,
                v_cache,
                qv,
                softmax_scale,
                cache_seqlens,
                scheduler_metadata=scheduler_metadata,
            )
        else:
            out, softmax_lse = torch.ops.flash_attn_npu_3._mla_dense_forward(
                q, k_cache, v_cache, qv, softmax_scale, cache_seqlens, scheduler_metadata
            )
        if return_softmax_lse:
            return (
                out,
                softmax_lse,
                torch.empty(0, device=out.device),
                torch.empty(0, device=out.device),
            )
        return out
    assert k_cache.stride(-1) == 1, "k_cache must have contiguous last dimension"
    assert v_cache.stride(-1) == 1, "v_cache must have contiguous last dimension"
    if softmax_scale is None:
        softmax_scale = (q.shape[-1] + (qv.shape[-1] if qv is not None else 0)) ** (-0.5)
    if cache_seqlens is not None and isinstance(cache_seqlens, int):
        cache_seqlens = torch.full(
            (q.shape[0],), cache_seqlens, dtype=torch.int32, device=k_cache.device
        )
        cache_seqlens = maybe_contiguous(cache_seqlens)
    if scheduler_metadata is not None and not is_mla:
        _validate_scheduler_metadata(
            scheduler_metadata,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            seqlen_q=q.shape[1] if cu_seqlens_q is None else max_seqlen_q,
            varlen_q=cu_seqlens_q is not None,
            num_splits=num_splits,
            page_table=page_table,
            k_cache=k_cache,
            cache_seqlens=cache_seqlens,
        )
    out, softmax_lse, *rest = _flash_attn_forward(
        q,
        k_cache,
        v_cache,
        k,
        v,
        qv,
        None,  # out
        cu_seqlens_q,
        None,  # cu_seqlens_k
        cu_seqlens_k_new,
        None,  # seqused_q
        cache_seqlens,
        max_seqlen_q,
        None,  # max_seqlen_k
        page_table,
        cache_batch_idx,
        cache_leftpad,
        rotary_cos,
        rotary_sin,
        rotary_seqlens,
        q_descale,
        k_descale,
        v_descale,
        softmax_scale,
        causal=causal,
        window_size_left=window_size[0],
        window_size_right=window_size[1],
        attention_chunk=attention_chunk,
        softcap=softcap,
        rotary_interleaved=rotary_interleaved,
        scheduler_metadata=scheduler_metadata,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        sm_margin=sm_margin,
    )
    # return (out, softmax_lse) if return_softmax_lse else out
    return (out, softmax_lse, *rest) if return_softmax_lse else out


def _validate_scheduler_metadata(
    scheduler_metadata,
    *,
    causal,
    window_size,
    softcap,
    seqlen_q,
    varlen_q,
    num_splits,
    page_table,
    k_cache,
    cache_seqlens,
):
    """Reject scheduler_metadata created with arguments that do not match this
    call's mask layout, paged geometry, softcap feature and split schedule.
    Per-call numerical values are supplied in tiling without changing the shared schedule."""
    params = getattr(scheduler_metadata, "_fa_scheduler_params", None)
    if params is None:
        raise RuntimeError(
            "scheduler_metadata has no creation-argument fingerprint; it must be "
            "the unchanged tensor returned by get_scheduler_metadata"
        )
    expected = {
        "causal": bool(causal),
        "window_size": (int(window_size[0]), int(window_size[1])),
        "has_softcap": softcap > 0.0,
        "varlen_q": bool(varlen_q),
        "num_splits": int(num_splits),
    }
    if seqlen_q is not None:
        expected["max_seqlen_q"] = int(seqlen_q)
    if page_table is not None:
        # The fwd mask derivation and the page-table row stride both use the
        # paged capacity (page_block_size * max_num_blocks_per_seq) as KV bound.
        expected["page_size"] = int(k_cache.shape[1])
        expected["max_seqlen_k"] = int(k_cache.shape[1]) * int(page_table.shape[1])
    else:
        expected["page_size"] = None
        if varlen_q and k_cache.dim() == 3:
            if cache_seqlens is None:
                raise ValueError("cache_seqlens must be provided for a non-paged TND KV cache")
            expected["max_seqlen_k"] = int(cache_seqlens.max().item())
        else:
            expected["max_seqlen_k"] = int(k_cache.shape[1])
    mismatches = []
    for key, call_val in expected.items():
        meta_val = params.get(key, "<missing>")
        if meta_val != call_val:
            mismatches.append(f"{key}: metadata={meta_val!r} vs call={call_val!r}")
    if mismatches:
        raise ValueError(
            "scheduler_metadata arguments do not match this call: " + "; ".join(mismatches)
        )


# Real metadata contract:
#   all mask types:
#       shape = (2384,)
#   dtype  = torch.uint8
#   device = NPU
#   stride = (1,)
# The custom op prevents TorchDynamo from tracing into the raw pybind
# get_scheduler_metadata() implementation. Schema matches the V3 910
# pybind signature in flash_api.cpp (not the V4 parameter list).
_SCHEDULER_METADATA_TILING_BYTES = 2376


def _resolve_legacy_metadata_softcap(has_softcap, softcap):
    # Ordinary FA3 byte metadata still stores the numerical softcap and scale.
    # A feature flag cannot supply that value. Preserve old keyword calls, but
    # reject the flag-only form until this protocol can consume runtime values.
    if softcap is None:
        if has_softcap:
            raise RuntimeError(
                "NPU FlashAttention byte metadata with has_softcap=True is not implemented "
                "yet without the legacy softcap keyword"
            )
        return 0.0
    if not math.isfinite(softcap) or softcap < 0.0:
        raise RuntimeError("softcap must be finite and non-negative (0.0 disables softcap)")
    if has_softcap and softcap == 0.0:
        raise RuntimeError("has_softcap=True conflicts with legacy softcap=0.0")
    return softcap


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
):
    if num_heads_kv != 1:
        raise RuntimeError(
            "MLA scheduler metadata is not implemented yet outside the dense shape bounds"
        )
    if (
        any(arg is not None for arg in optional_args)
        or max_seqlen_k_new != 0
        or (causal and not allow_causal)
        or (window_size != (-1, -1) and not allow_window)
        or attention_chunk != 0
        or has_softcap
        or (num_splits != 1 and not (allow_split and 0 <= num_splits <= 256))
        or sm_margin != 0
    ):
        raise RuntimeError(
            "MLA scheduler metadata is not implemented yet for these length, mask, split, "
            "scheduling or legacy numerical options"
        )


def _fits_mla_dense_shape(batch, sq, heads, sk):
    # Match mla_schedule.h: source ceil-div, query/LSE DMA strides and tensor bytes.
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


@_torch_register_fake_wrapper("flash_attn_npu_3::_mla_window_mixed_metadata")
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


@_torch_register_fake_wrapper("flash_attn_npu_3::_mla_causal_mixed_metadata")
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


@_torch_register_fake_wrapper("flash_attn_npu_3::_mla_split_bsnd_metadata")
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
):
    _mla_causal_bsnd_metadata_fake(
        batch_size, max_seqlen_q, max_seqlen_k, num_heads_q, qkv_dtype, cache_seqlens
    )
    _normalize_mla_window(max_seqlen_q, max_seqlen_k, window_left, window_right, causal)
    if num_splits != 0 and not 2 <= num_splits <= 256:
        raise RuntimeError("MLA: split requires num_splits=0 or [2,256]")
    return torch.empty(
        (_mla_split_bsnd_words(batch_size, num_splits),),
        dtype=torch.int32,
        device=cache_seqlens.device,
    )


@_torch_register_fake_wrapper("flash_attn_npu_3::_mla_window_bsnd_metadata")
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


@_torch_register_fake_wrapper("flash_attn_npu_3::_mla_causal_bsnd_metadata")
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


@_torch_register_fake_wrapper("flash_attn_npu_3::_mla_dense_metadata")
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


@_torch_register_fake_wrapper("flash_attn_npu_3::_mla_paged_metadata")
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
):
    is_window = (window_left, window_right) != (-1, -1)
    prefix_words = 411 if is_window else 408
    if not 0 <= batch_size <= (2**32 - 1 - prefix_words) // 5:
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
    if is_window:
        _normalize_mla_window(max_seqlen_q, max_seqlen_k, window_left, window_right, causal)
    return torch.empty(
        (prefix_words + 5 * batch_size,), dtype=torch.int32, device=cache_seqlens.device
    )


@_torch_register_fake_wrapper("flash_attn_npu_3::_mla_tnd_bsnd_metadata")
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


@torch.library.custom_op(
    "flash_attn_npu_3::_get_scheduler_metadata",
    mutates_args=(),
)
def _get_scheduler_metadata_op(
    batch_size: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    num_heads_q: int,
    num_heads_kv: int,
    headdim: int,
    headdim_v: int,
    qkv_dtype: torch.dtype,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    cu_seqlens_k_new: Optional[torch.Tensor],
    seqused_q: Optional[torch.Tensor],
    cache_leftpad: Optional[torch.Tensor],
    page_size: Optional[int],
    max_seqlen_k_new: int,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    attention_chunk: int,
    has_softcap: bool,
    num_splits: int,
    pack_gqa: Optional[bool],
    sm_margin: int,
    softcap: Optional[float],
    softmax_scale: Optional[float],
) -> torch.Tensor:
    is_mla = _check_mla_metadata(headdim, headdim_v)
    resolved_softcap = None if is_mla else _resolve_legacy_metadata_softcap(has_softcap, softcap)
    scheduler_metadata = flash_attn_npu_3.get_scheduler_metadata(
        batch_size,
        max_seqlen_q,
        max_seqlen_k,
        num_heads_q,
        num_heads_kv,
        headdim,
        headdim_v,
        qkv_dtype,
        cache_seqlens,
        cu_seqlens_q,
        cu_seqlens_k,
        cu_seqlens_k_new,
        seqused_q,
        cache_leftpad,
        page_size,
        max_seqlen_k_new,
        causal,
        window_size_left,
        window_size_right,
        attention_chunk,
        has_softcap,
        num_splits,
        pack_gqa,
        sm_margin,
        softcap,
        softmax_scale,
    )
    if is_mla:
        # All compatibility checks are in the device record and the current
        # call; no Python attribute is needed to clone or reuse MLA metadata.
        return scheduler_metadata
    # Keep this side effect inside the opaque custom op so Dynamo does not
    # trace the setattr while still attaching the runtime fingerprint.
    if softmax_scale is None:
        softmax_scale = headdim ** (-0.5)
    scheduler_metadata._fa_scheduler_params = {
        "causal": bool(causal),
        "window_size": (int(window_size_left), int(window_size_right)),
        "softcap": float(resolved_softcap),
        "softmax_scale": float(softmax_scale),
        "page_size": None if page_size is None else int(page_size),
        "max_seqlen_q": int(max_seqlen_q),
        "max_seqlen_k": int(max_seqlen_k),
        "varlen_q": cu_seqlens_q is not None,
        "num_splits": int(num_splits),
    }
    return scheduler_metadata


@_torch_register_fake_wrapper("flash_attn_npu_3::_get_scheduler_metadata")
def _get_scheduler_metadata_fake(
    batch_size: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    num_heads_q: int,
    num_heads_kv: int,
    headdim: int,
    headdim_v: int,
    qkv_dtype: torch.dtype,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    cu_seqlens_k_new: Optional[torch.Tensor],
    seqused_q: Optional[torch.Tensor],
    cache_leftpad: Optional[torch.Tensor],
    page_size: Optional[int],
    max_seqlen_k_new: int,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    attention_chunk: int,
    has_softcap: bool,
    num_splits: int,
    pack_gqa: Optional[bool],
    sm_margin: int,
    softcap: Optional[float],
    softmax_scale: Optional[float],
) -> torch.Tensor:
    if _check_mla_metadata(headdim, headdim_v):
        _check_mla_metadata_options(
            num_heads_kv,
            max_seqlen_k_new,
            causal,
            (window_size_left, window_size_right),
            attention_chunk,
            has_softcap,
            num_splits,
            sm_margin,
            (
                cu_seqlens_k,
                cu_seqlens_k_new,
                seqused_q,
                cache_leftpad,
                pack_gqa,
                softcap,
                softmax_scale,
            ),
            allow_causal=True,
            allow_window=True,
            allow_split=cu_seqlens_q is None and page_size is None,
        )
        if num_splits != 1:
            return _mla_split_bsnd_metadata_fake(
                batch_size,
                max_seqlen_q,
                max_seqlen_k,
                num_heads_q,
                qkv_dtype,
                cache_seqlens,
                window_size_left,
                window_size_right,
                causal,
                num_splits,
            )
        if page_size is not None:
            return _mla_paged_metadata_fake(
                batch_size,
                max_seqlen_q,
                max_seqlen_k,
                num_heads_q,
                page_size,
                qkv_dtype,
                cache_seqlens,
                cu_seqlens_q,
                causal,
                window_size_left,
                window_size_right,
            )
        if (window_size_left, window_size_right) != (-1, -1):
            if cu_seqlens_q is not None:
                return _mla_window_mixed_metadata_fake(
                    batch_size,
                    max_seqlen_q,
                    max_seqlen_k,
                    num_heads_q,
                    qkv_dtype,
                    cache_seqlens,
                    cu_seqlens_q,
                    window_size_left,
                    window_size_right,
                    causal,
                )
            return _mla_window_bsnd_metadata_fake(
                batch_size,
                max_seqlen_q,
                max_seqlen_k,
                num_heads_q,
                qkv_dtype,
                cache_seqlens,
                window_size_left,
                window_size_right,
                causal,
            )
        if cu_seqlens_q is not None:
            return (_mla_causal_mixed_metadata_fake if causal else _mla_tnd_bsnd_metadata_fake)(
                batch_size,
                max_seqlen_q,
                max_seqlen_k,
                num_heads_q,
                qkv_dtype,
                cache_seqlens,
                cu_seqlens_q,
            )
        if causal:
            return _mla_causal_bsnd_metadata_fake(
                batch_size, max_seqlen_q, max_seqlen_k, num_heads_q, qkv_dtype, cache_seqlens
            )
        return _mla_dense_metadata_fake(
            batch_size, max_seqlen_q, max_seqlen_k, num_heads_q, qkv_dtype, cache_seqlens
        )
    _resolve_legacy_metadata_softcap(has_softcap, softcap)
    return torch.empty(
        (_SCHEDULER_METADATA_TILING_BYTES,),
        dtype=torch.uint8,
        device=cache_seqlens.device,
    )


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
    *,
    softcap=None,  # legacy ordinary-FA byte metadata only; 0.0 means deactivated
    softmax_scale=None,  # defaults to 1 / sqrt(headdim); must match the fwd call
):
    """Precompute NPU scheduling metadata with the official FA3 argument order.

    Ordinary FA3 returns byte metadata and still embeds numerical softcap/scale.
    Its legacy ``softcap`` and ``softmax_scale`` arguments are keyword-only; a
    positive legacy softcap enables softcapping even when has_softcap is omitted.
    ``has_softcap=True`` requires a positive legacy softcap for this protocol.
    MLA requires headdim=64, headdim_v=512 and matching qkv_dtype.
    Nonpaged BSND accepts num_splits=0..256; other layouts require 1.
    cache_seqlens must be a same-device contiguous int32 tensor of shape
    (batch_size,); None/integer lengths are kvcache conveniences only.
    Without page_size or cu_seqlens_q, automatic num_splits=0 returns a v18
    int32 record with 758 + 10 * batch_size words; explicit splits 2..256 use
    v17 with 416 + 10 * batch_size words. Both bind the normalized window and
    current lengths. For num_splits=1 it returns an int32 BSND-cache record:
    v2 for unmasked attention or v7 for right-down causal. Causal records contain
    410 + 8 * batch_size words and cannot be reused by an unmasked forward.
    With cu_seqlens_q and no page_size it generates a TND-query/BSND-cache record:
    v4 for unmasked attention or v9 for causal;
    max_seqlen_q is a per-query-segment bound, max_seqlen_k is cache capacity.
    Read-only flash_attn_with_kvcache accepts the matching v4/v9 record with TND queries
    and a BSND cache; the current query prefixes and KV lengths must match.
    With page_size it generates v6 for unmasked BSND/TND queries or v10 for causal
    BSND queries. Both contain 408 + 5 * batch_size words and are not interchangeable.
    page_size must be a
    multiple of 16 in [16, 1024], and max_seqlen_k is logical capacity, divisible
    by page_size and at most 16384. No page table or physical cache is inspected.
    Read-only MLA paged kvcache accepts the matching v6/v10 record; TND also
    requires cu_seqlens_q and max_seqlen_q at forward. The record does not bind
    a page table or physical cache; each forward validates current lengths,
    query prefixes and all used pages. Regenerate after lengths or prefixes change.
    There is no Q-used parameter here; public varlen accepts seqused_q/k and
    generates its own metadata internally. Scale is supplied only to forward;
    legacy numerical keywords are not supported. Length values are checked on device.
    """
    if headdim_v is None:
        headdim_v = headdim
    if _check_mla_metadata(headdim, headdim_v):
        _check_mla_metadata_options(
            num_heads_kv,
            max_seqlen_k_new,
            causal,
            (window_size[0], window_size[1]),
            attention_chunk,
            has_softcap,
            num_splits,
            sm_margin,
            (
                cu_seqlens_k_new,
                cache_leftpad,
                pack_gqa,
                softcap,
                softmax_scale,
            ),
            allow_causal=True,
            allow_window=True,
            allow_split=cu_seqlens_q is None and page_size is None,
        )
        if num_splits != 1:
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
    cache_seqlens = maybe_contiguous(cache_seqlens)
    # Route through the custom op so torch.compile / FakeTensor can use the
    # registered fake instead of tracing into the raw pybind extension.
    # Public API still hard-codes cu_seqlens_k / seqused_q as None (unchanged).
    scheduler_metadata = _get_scheduler_metadata_op(
        batch_size,
        max_seqlen_q,
        max_seqlen_k,
        num_heads_q,
        num_heads_kv,
        headdim,
        headdim_v,
        qkv_dtype,
        cache_seqlens,
        cu_seqlens_q,
        None,  # cu_seqlens_k
        cu_seqlens_k_new,
        None,  # seqused_q
        cache_leftpad,
        page_size,
        max_seqlen_k_new,
        causal,
        window_size[0],
        window_size[1],
        attention_chunk,
        has_softcap,
        num_splits,
        pack_gqa,
        sm_margin,
        softcap,
        softmax_scale,
    )
    return scheduler_metadata
