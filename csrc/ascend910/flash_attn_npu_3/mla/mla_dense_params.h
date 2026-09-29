// Private device arguments for the first FP16 dense MLA milestone.
// No tensor addresses or scale are stored in the reusable schedule.
#pragma once
#include <cstdint>

namespace flash_attn::mla {
struct MlaDenseKernelSchedule {
    // Same endpoint capacity as MlaSchedule; not a device core count.
    static constexpr uint32_t kEndpointCapacity = 26;
    uint32_t batch, sq, heads, sk;
    uint32_t mBaseSize, s2BaseSize, usedCoreNum, blockDim, cvRatio;
    uint32_t mm1ResSize, mm2ResSize;
    uint32_t bN2End[kEndpointCapacity];
    uint32_t gS1End[kEndpointCapacity];
    uint32_t s2End[kEndpointCapacity];
};
} // namespace flash_attn::mla
