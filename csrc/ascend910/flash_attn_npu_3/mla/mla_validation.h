#pragma once

#include <torch/extension.h>

// Host-only boundary for the two representations in docs/mla/design.md.
// Contract: TriDao FA3 d15f1531a460ba456f41b01a774f33ab2db8febf,
// hopper/flash_api.cpp::mha_fwd. No CANN kernel or scheduling code is used here.
// Keep the corresponding Python/fake checks in _check_mla_inputs in sync.
namespace flash_attn::mla {

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
        TORCH_CHECK(!varlenQ && !varlenK && !pagedKV,
                    "MLA compressed forward is not implemented yet for TND or paged KV");
        TORCH_CHECK(q.size(0) == k.size(0), "MLA: dense batch sizes must match");
        TORCH_CHECK(q.size(0) > 0 && q.size(0) <= UINT32_MAX && q.size(1) > 0 && q.size(1) <= 256 && headsQ <= 256 &&
                        uint64_t(q.size(1)) * uint64_t(headsQ) <= 256 && k.size(1) > 0 && k.size(1) <= 512,
                    "MLA compressed forward is not implemented yet outside B>0, 1<=Sq*Hq<=256, 1<=Sk<=512");
        for (const auto& t : {q, k, v, *qv}) {
            TORCH_CHECK(t.is_contiguous(), "MLA compressed forward is not implemented yet for noncontiguous BSND");
            TORCH_CHECK(!t.requires_grad(), "MLA compressed forward is not implemented yet for gradients");
        }
        return true;
    }
    TORCH_CHECK(q.size(-1) == 192 && v.size(-1) == 128,
                "MLA: supported representation without qv requires Dq=192 and Dv=128");
    TORCH_CHECK(false, "NPU FlashAttention MLA prefill forward is not implemented yet");
}

inline void CheckMetadataBoundary(int64_t headDim, int64_t headDimV)
{
    // Metadata has no qv argument: do not infer that qv is present from dimensions.
    TORCH_CHECK(headDim == headDimV, "NPU FlashAttention MLA scheduler metadata is not implemented yet");
}

} // namespace flash_attn::mla
