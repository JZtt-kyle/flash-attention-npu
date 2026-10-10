#pragma once

#include <torch/extension.h>
#include "mla_bsnd_metadata.h"
#include "mla_tnd_metadata.h"
#include "mla_tnd_bsnd_metadata.h"
#include "mla_used_metadata.h"

// Host-only boundary for the two representations in docs/mla/design.md.
// Contract: TriDao FA3 d15f1531a460ba456f41b01a774f33ab2db8febf,
// hopper/flash_api.cpp::mha_fwd. No CANN kernel or scheduling code is used here.
// Keep the corresponding Python/fake checks in _check_mla_inputs in sync.
namespace flash_attn::mla {

inline bool FitsDenseShape(int64_t batch, int64_t sq, int64_t heads, int64_t sk)
{
    return batch >= 0 && batch <= UINT32_MAX && sq >= 0 && sq <= kMlaMaxQueryLength && heads > 0 &&
           heads <= kMlaMaxQueryHeads && uint64_t(sq) * heads <= kMlaMaxDenseRows &&
           uint64_t(batch) * sq * heads <= kMlaMaxBatchRows && sk >= 0 && sk <= kMlaMaxKvLength;
}

inline bool CheckForwardBoundary(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                 const std::optional<at::Tensor>& qv, bool varlenQ, bool varlenK, bool pagedKV)
{
    TORCH_CHECK(q.dim() == (varlenQ ? 3 : 4), "FlashAttention: q rank must match BSND/TND layout");
    TORCH_CHECK(k.dim() == (varlenK ? 3 : 4) && v.dim() == k.dim(),
                "FlashAttention: k/v rank must match dense, TND or paged KV layout");
    TORCH_CHECK(q.size(-1) == k.size(-1), "FlashAttention: q and k head dimensions must match");
    if (!qv.has_value() && q.size(-1) == v.size(-1)) {
        return false;
    }

    TORCH_CHECK(!pagedKV || !varlenK, "MLA: page_table and cu_seqlens_k cannot be combined");
    TORCH_CHECK(q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16,
                "MLA: inputs must have FP16 or BF16 dtype");
    TORCH_CHECK(q.device().type() == at::kPrivateUse1, "MLA: inputs must be NPU tensors");
    auto checkTensor = [&q](const at::Tensor& t) {
        TORCH_CHECK(t.scalar_type() == q.scalar_type(), "MLA: all inputs must have the same dtype");
        TORCH_CHECK(t.device() == q.device(), "MLA: all inputs must be on the same device");
        TORCH_CHECK(t.dim() > 0 && t.stride(-1) == 1, "MLA: inputs must have contiguous last dimension");
    };
    checkTensor(q);
    checkTensor(k);
    checkTensor(v);
    for (int64_t axis = 0; axis < k.dim() - 1; ++axis) {
        TORCH_CHECK(k.size(axis) == v.size(axis), "MLA: k/v shapes must match except for head dimension");
    }
    const auto headsQ = q.size(-2);
    const auto headsKV = k.size(-2);
    TORCH_CHECK(headsQ > 0 && headsKV > 0 && headsQ % headsKV == 0,
                "MLA: positive KV head count must divide query head count");
    if (qv.has_value()) {
        checkTensor(*qv);
        TORCH_CHECK(qv->dim() == q.dim(), "MLA: qv shape must match q with the value head dimension");
        for (int64_t axis = 0; axis < q.dim() - 1; ++axis) {
            TORCH_CHECK(qv->size(axis) == q.size(axis), "MLA: qv shape must match q with the value head dimension");
        }
        TORCH_CHECK(qv->size(-1) == v.size(-1), "MLA: qv shape must match q with the value head dimension");
        TORCH_CHECK(q.size(-1) == 64 && v.size(-1) == 512,
                    "MLA: supported qv representation requires Dq=64 and Dv=512");
        TORCH_CHECK(headsKV == 1, "MLA: compressed representation requires one KV head");
        TORCH_CHECK((varlenQ || !varlenK) && !pagedKV,
                    "MLA compressed forward is not implemented yet for BSND Q with TND KV or paged KV");
        TORCH_CHECK(varlenQ || q.size(0) == k.size(0), "MLA: dense batch sizes must match");
        TORCH_CHECK(varlenQ || FitsDenseShape(q.size(0), q.size(1), headsQ, k.size(1)),
                    "MLA compressed forward is not implemented yet outside B>=0, nonnegative S1G within address/field "
                    "bounds, 0<=Sk<=16384");
        for (const auto& t : {q, k, v, *qv}) {
            TORCH_CHECK(t.is_contiguous(), "MLA compressed forward is not implemented yet for noncontiguous BSND/TND");
            TORCH_CHECK(!t.requires_grad(), "MLA compressed forward is not implemented yet for gradients");
        }
        return true;
    }
    TORCH_CHECK(q.size(-1) == 192 && v.size(-1) == 128,
                "MLA: supported representation without qv requires Dq=192 and Dv=128");
    TORCH_CHECK(false, "NPU FlashAttention MLA prefill forward is not implemented yet");
}

inline uint32_t CheckTndInputs(const at::Tensor& q, const at::Tensor& k, const at::Tensor& cuQ, const at::Tensor& cuK,
                               int64_t maxQ, int64_t maxK)
{
    TORCH_CHECK(cuQ.dim() == 1 && cuQ.numel() >= 1 && cuK.dim() == 1 && cuK.numel() == cuQ.numel(),
                "MLA: cu_seqlens_q/k must have matching shape [batch_size + 1]");
    for (const auto& t : {cuQ, cuK}) {
        TORCH_CHECK(t.scalar_type() == at::kInt, "MLA: cu_seqlens_q/k must have dtype int32");
        TORCH_CHECK(t.device() == q.device(), "MLA: cu_seqlens_q/k must be on the same NPU device");
        TORCH_CHECK(t.is_contiguous(), "MLA: cu_seqlens_q/k must be contiguous");
    }
    const int64_t batch = cuQ.numel() - 1;
    TORCH_CHECK(batch <= (UINT32_MAX - kMlaTndMetadataPrefixWords) / kMlaTndMetadataSequenceWords,
                "MLA: TND metadata word count overflow");
    TORCH_CHECK(q.size(0) <= INT32_MAX && k.size(0) <= INT32_MAX && q.size(1) <= kMlaMaxQueryHeads && maxQ >= 0 &&
                    maxQ <= kMlaMaxQueryLength && maxK >= 0 && maxK <= kMlaMaxKvLength &&
                    uint64_t(maxQ) * q.size(1) <= kMlaMaxDenseRows &&
                    uint64_t(q.size(0)) * q.size(1) <= kMlaMaxBatchRows,
                "MLA: compact TND shape or max_seqlen bounds exceeded");
    return kMlaTndMetadataPrefixWords + kMlaTndMetadataSequenceWords * batch;
}

inline bool CheckMetadataBoundary(int64_t headDim, int64_t headDimV)
{
    if (headDim == headDimV) {
        return false;
    }
    // This selects only the schedule family. Forward still requires qv and
    // independently checks the actual tensors and the record's geometry.
    TORCH_CHECK(headDim == 64 && headDimV == 512,
                "NPU FlashAttention MLA scheduler metadata is not implemented yet for these head dimensions");
    return true;
}

inline void CheckCacheLengths(const at::Tensor& lengths, int64_t batch, const at::Device& device)
{
    TORCH_CHECK(lengths.scalar_type() == at::kInt, "MLA: cache_seqlens must have dtype int32");
    TORCH_CHECK(lengths.device().type() == at::kPrivateUse1 && lengths.device() == device,
                "MLA: cache_seqlens must be on the same NPU device");
    TORCH_CHECK(lengths.dim() == 1 && lengths.size(0) == batch, "MLA: cache_seqlens must have shape [batch_size]");
    TORCH_CHECK(lengths.is_contiguous(), "MLA: cache_seqlens must be contiguous");
}

inline uint32_t CheckBsndMetadataWords(int64_t batch)
{
    TORCH_CHECK(batch >= 0 && batch <= UINT32_MAX - kMlaBsndKvLengthMetadataPrefixWords,
                "MLA: BSND metadata word count overflow");
    return kMlaBsndKvLengthMetadataPrefixWords + batch;
}

inline uint32_t CheckBsndUsedInputs(const at::Tensor& q, const at::Tensor& usedQuery,
                                    const std::optional<at::Tensor>& usedKv)
{
    const int64_t batch = q.size(0);
    TORCH_CHECK(batch <= (UINT32_MAX - kMlaUsedMetadataPrefixWords) / kMlaUsedMetadataSequenceWords,
                "MLA: used metadata word count overflow");
    const auto check = [&](const at::Tensor& t, const char* name) {
        TORCH_CHECK(t.scalar_type() == at::kInt, "MLA: ", name, " must have dtype int32");
        TORCH_CHECK(t.device() == q.device(), "MLA: ", name, " must be on the same NPU device");
        TORCH_CHECK(t.dim() == 1 && t.numel() == batch, "MLA: ", name, " must have shape [batch_size]");
        TORCH_CHECK(t.is_contiguous(), "MLA: ", name, " must be contiguous");
    };
    check(usedQuery, "seqused_q");
    if (usedKv) {
        check(*usedKv, "seqused_k");
    }
    return kMlaUsedMetadataPrefixWords + kMlaUsedMetadataSequenceWords * batch;
}

inline uint32_t CheckTndBsndMetadataInputs(int64_t batch, int64_t maxQ, int64_t sk, int64_t heads, at::ScalarType dtype,
                                           const at::Tensor& cacheSeqlens, const at::Tensor& cuQ)
{
    TORCH_CHECK(batch >= 0 && batch <= (UINT32_MAX - kMlaTndBsndMetadataPrefixWords) / kMlaTndBsndMetadataSequenceWords,
                "MLA: mixed metadata word count overflow");
    // Total Q is device-resident. Do not impose a rectangular B*maxQ bound.
    TORCH_CHECK(FitsDenseShape(0, maxQ, heads, sk), "MLA: mixed metadata shape or max_seqlen bounds exceeded");
    TORCH_CHECK(dtype == at::kHalf || dtype == at::kBFloat16, "MLA: qkv_dtype must be FP16 or BF16");
    CheckCacheLengths(cacheSeqlens, batch, cacheSeqlens.device());
    TORCH_CHECK(cuQ.scalar_type() == at::kInt, "MLA: cu_seqlens_q must have dtype int32");
    TORCH_CHECK(cuQ.device() == cacheSeqlens.device(), "MLA: cu_seqlens_q must be on the same NPU device");
    TORCH_CHECK(cuQ.dim() == 1 && cuQ.numel() == batch + 1, "MLA: cu_seqlens_q must have shape [batch_size + 1]");
    TORCH_CHECK(cuQ.is_contiguous(), "MLA: cu_seqlens_q must be contiguous");
    return kMlaTndBsndMetadataPrefixWords + kMlaTndBsndMetadataSequenceWords * batch;
}

inline void CheckDenseMetadataTensor(const at::Tensor& metadata, const at::Device& device, int64_t batch)
{
    TORCH_CHECK(metadata.scalar_type() == at::kInt, "MLA: scheduler_metadata must have dtype int32");
    TORCH_CHECK(metadata.device() == device, "MLA: scheduler_metadata must be on the same NPU device");
    TORCH_CHECK(metadata.dim() == 1 && metadata.size(0) == CheckBsndMetadataWords(batch),
                "MLA: scheduler_metadata must have shape [401 + batch_size]");
    TORCH_CHECK(metadata.is_contiguous(), "MLA: scheduler_metadata must be contiguous");
}

inline uint32_t CheckTndBsndForwardInputs(const at::Tensor& q, const at::Tensor& k, const at::Tensor& cuQ, int64_t maxQ,
                                          const std::optional<at::Tensor>& cacheSeqlens,
                                          const std::optional<at::Tensor>& metadata)
{
    const int64_t batch = k.size(0);
    TORCH_CHECK(batch <= (UINT32_MAX - kMlaTndBsndMetadataPrefixWords) / kMlaTndBsndMetadataSequenceWords,
                "MLA: mixed metadata word count overflow");
    const uint32_t words = kMlaTndBsndMetadataPrefixWords + kMlaTndBsndMetadataSequenceWords * batch;
    TORCH_CHECK(FitsDenseShape(0, maxQ, q.size(1), k.size(1)) && q.size(0) <= INT32_MAX &&
                    uint64_t(q.size(0)) * q.size(1) <= kMlaMaxBatchRows,
                "MLA: mixed shape or max_seqlen bounds exceeded");
    TORCH_CHECK(cuQ.scalar_type() == at::kInt, "MLA: cu_seqlens_q must have dtype int32");
    TORCH_CHECK(cuQ.device() == q.device(), "MLA: cu_seqlens_q must be on the same NPU device");
    TORCH_CHECK(cuQ.dim() == 1 && cuQ.numel() == batch + 1, "MLA: cu_seqlens_q must have shape [batch_size + 1]");
    TORCH_CHECK(cuQ.is_contiguous(), "MLA: cu_seqlens_q must be contiguous");
    if (cacheSeqlens) {
        CheckCacheLengths(*cacheSeqlens, batch, q.device());
    }
    if (metadata) {
        TORCH_CHECK(metadata->scalar_type() == at::kInt, "MLA: scheduler_metadata must have dtype int32");
        TORCH_CHECK(metadata->device() == q.device(), "MLA: scheduler_metadata must be on the same NPU device");
        TORCH_CHECK(metadata->dim() == 1 && metadata->numel() == words,
                    "MLA: scheduler_metadata must have shape [406 + 5 * batch_size]");
        TORCH_CHECK(metadata->is_contiguous(), "MLA: scheduler_metadata must be contiguous");
    }
    return words;
}

} // namespace flash_attn::mla
