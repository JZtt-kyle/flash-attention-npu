// Host/AICPU launch ABI. No tensor length arrays are accepted in this dense step.
#pragma once
#include "mla_metadata.h"

namespace flash_attn::mla {
struct MlaMetadataArgs {
    uint64_t outputAddr;
    uint32_t outputWords;
    uint32_t dtype;
    MlaDenseShape shape;
    MlaPlatformInfo platform;
};
static_assert(std::is_standard_layout_v<MlaMetadataArgs> && std::is_trivially_copyable_v<MlaMetadataArgs>);
} // namespace flash_attn::mla
