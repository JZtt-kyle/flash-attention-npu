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
} // namespace flash_attn::mla
