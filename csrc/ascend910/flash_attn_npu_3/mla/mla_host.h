#pragma once

#include <cmath>
#include <limits>
#include "mla_launch.h"

namespace flash_attn::mla {

// Called only after representation and optional-argument validation. Keep the
// static guards in sync with the Python/fake boundary; reject before allocation.
inline std::vector<at::Tensor> ForwardDense(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                            const at::Tensor& qv, std::optional<float> scale)
{
    TORCH_CHECK(q.size(0) == k.size(0), "MLA: q and k/v batch dimensions must match");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() && qv.is_contiguous(),
                "NPU FlashAttention MLA compressed forward is not implemented yet for outer strides");
    TORCH_CHECK(q.size(0) > 0 && q.size(1) > 0 && k.size(1) > 0 && k.size(1) <= kOnlineKvLimit,
                "NPU FlashAttention MLA compressed forward is not implemented yet outside nonempty KV (1..512)");
    for (int axis = 0; axis < 3; ++axis) {
        TORCH_CHECK(q.size(axis) <= std::numeric_limits<uint32_t>::max(), "MLA: dense dimension exceeds uint32 range");
    }
    TORCH_CHECK(!q.requires_grad() && !k.requires_grad() && !v.requires_grad() && !qv.requires_grad(),
                "NPU FlashAttention MLA compressed forward is not implemented yet for gradients");
    const float softmaxScale = scale.value_or(1.0f / std::sqrt(576.0f));
    TORCH_CHECK(std::isfinite(softmaxScale), "MLA: softmax_scale must be finite");
    const c10::OptionalDeviceGuard deviceGuard(device_of(q));
    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const uint32_t cubeCores = platform->GetCoreNumAic();
    const uint32_t vectorCores = platform->GetCoreNumAiv();
    TORCH_CHECK(cubeCores > 0 && (vectorCores == cubeCores || vectorCores == 2 * cubeCores),
                "MLA: expected C1V1 or C1V2 resources");
    const uint64_t tasks = uint64_t(q.size(0)) * q.size(1) * ((q.size(2) + kHeadsPerTile - 1) / kHeadsPerTile);
    const uint32_t blocks = std::min<uint64_t>(cubeCores, tasks);
    auto out = at::empty({q.size(0), q.size(1), q.size(2), 512}, q.options());
    auto lse = at::empty({q.size(0), q.size(2), q.size(1)}, q.options().dtype(at::kFloat));
    const bool online = k.size(1) > kShortKvLimit;
    const uint32_t workspaceBytes = online ? kOnlineWorkspaceBytesPerCore : kWorkspaceBytesPerCore;
    auto workspace = at::empty({int64_t(blocks) * workspaceBytes}, q.options().dtype(at::kByte));
    uint64_t syncAddress = 0;
    uint32_t syncLength = 0;
    TORCH_CHECK(rtGetC2cCtrlAddr(&syncAddress, &syncLength) == RT_ERROR_NONE,
                "MLA: failed to get cross-core sync address");
    DenseParams params{reinterpret_cast<uint64_t>(q.data_ptr()),
                       reinterpret_cast<uint64_t>(k.data_ptr()),
                       reinterpret_cast<uint64_t>(v.data_ptr()),
                       reinterpret_cast<uint64_t>(qv.data_ptr()),
                       reinterpret_cast<uint64_t>(out.data_ptr()),
                       reinterpret_cast<uint64_t>(lse.data_ptr()),
                       reinterpret_cast<uint64_t>(workspace.data_ptr()),
                       tasks,
                       uint32_t(q.size(1)),
                       uint32_t(k.size(1)),
                       uint32_t(q.size(2)),
                       softmaxScale};
    auto stream = c10_npu::getCurrentNPUStream().stream(false);
    const bool bf16 = q.scalar_type() == at::kBFloat16;
    auto launch = [params, blocks, syncAddress, stream, bf16, online]() -> int {
        if (bf16) {
            if (online) {
                LaunchOnline<bfloat16_t>(params, blocks, syncAddress, stream);
            } else {
                LaunchShort<bfloat16_t>(params, blocks, syncAddress, stream);
            }
        } else {
            if (online) {
                LaunchOnline<half>(params, blocks, syncAddress, stream);
            } else {
                LaunchShort<half>(params, blocks, syncAddress, stream);
            }
        }
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2(online ? "ascendc_mla_online" : "ascendc_mla_short", launch);
    return {out, lse, at::empty({0}, q.options().dtype(at::kFloat)), at::empty({0}, q.options().dtype(at::kFloat))};
}

} // namespace flash_attn::mla
