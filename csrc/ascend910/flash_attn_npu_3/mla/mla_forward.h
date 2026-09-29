#pragma once
#include <torch/extension.h>

namespace flash_attn::mla {
// Call only after the common Python/C++ dense boundary has validated inputs.
std::vector<at::Tensor> DenseForward(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                     const at::Tensor& qv, float scale);
} // namespace flash_attn::mla
