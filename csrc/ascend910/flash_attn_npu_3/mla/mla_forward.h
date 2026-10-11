#pragma once
#include <torch/extension.h>

namespace flash_attn::mla {
// Call only after the common Python/C++ dense boundary has validated inputs.
std::vector<at::Tensor> DenseForward(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                     const at::Tensor& qv, float scale,
                                     const std::optional<at::Tensor>& cacheSeqlens = std::nullopt,
                                     const std::optional<at::Tensor>& schedulerMetadata = std::nullopt);
at::Tensor DenseMetadata(int64_t batch, int64_t sq, int64_t heads, int64_t sk, at::ScalarType dtype,
                         const at::Tensor& cacheSeqlens);
at::Tensor TndBsndMetadataOp(int64_t batch, int64_t maxQ, int64_t sk, int64_t heads, at::ScalarType dtype,
                             const at::Tensor& cacheSeqlens, const at::Tensor& cuQ);
at::Tensor PagedMetadataOp(int64_t batch, int64_t maxQ, int64_t capacity, int64_t heads, int64_t pageSize,
                           at::ScalarType dtype, const at::Tensor& cacheSeqlens, const std::optional<at::Tensor>& cuQ);
std::vector<at::Tensor> TndForward(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& qv,
                                   const at::Tensor& cuQ, const at::Tensor& cuK, int64_t maxQ, int64_t maxK,
                                   float scale, const std::optional<at::Tensor>& usedQuery = std::nullopt,
                                   const std::optional<at::Tensor>& usedKv = std::nullopt);
std::vector<at::Tensor> TndBsndForward(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                       const at::Tensor& qv, const at::Tensor& cuQ, int64_t maxQ, float scale,
                                       const std::optional<at::Tensor>& cacheSeqlens,
                                       const std::optional<at::Tensor>& schedulerMetadata);
std::vector<at::Tensor> BsndUsedForward(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                        const at::Tensor& qv, const at::Tensor& usedQuery,
                                        const std::optional<at::Tensor>& usedKv, float scale);
std::vector<at::Tensor> TndBsndUsedForward(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                           const at::Tensor& qv, const at::Tensor& cuQ, int64_t maxQ,
                                           const at::Tensor& usedQuery, const std::optional<at::Tensor>& usedKv,
                                           float scale);
} // namespace flash_attn::mla
