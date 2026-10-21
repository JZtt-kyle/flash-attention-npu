#pragma once

#include <torch/extension.h>

namespace flash_attn {

// Static metadata checks are shared by the ordinary and MLA schedulers.
// Device-resident lengths and record contents are validated by their consumers.
inline void CheckMetadataTensor(const at::Tensor& tensor, const at::Device& device, int64_t count, const char* name,
                                at::ScalarType dtype = at::kInt)
{
    TORCH_CHECK(tensor.scalar_type() == dtype, name, " must have dtype ", dtype == at::kInt ? "int32" : "uint8");
    TORCH_CHECK((device.type() == at::kPrivateUse1 || device.is_meta()) && tensor.device() == device, name,
                " must be on the same NPU device");
    TORCH_CHECK(tensor.dim() == 1 && tensor.numel() == count, name, " must exactly match shape [", count, "]");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

} // namespace flash_attn
