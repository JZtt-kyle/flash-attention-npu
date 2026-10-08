// Private host/AICPU launch ABI. Addresses never belong in serialized records.
#pragma once
#include "mla_metadata.h"

namespace flash_attn::mla {
struct MlaMetadataArgs {
    uint64_t outputAddr;
    uint32_t outputWords;
    uint32_t dtype;
    MlaDenseShape shape;
    MlaPlatformInfo platform;
    // (0, 0) means implicit full capacity for the internal dense producer.
    // Otherwise the caller supplies B contiguous int32 lengths on this device,
    // retains their storage and orders writes before the AICPU launch.
    uint64_t cacheSeqlensAddr;
    uint64_t cacheSeqlensCount;
};
static_assert(std::is_standard_layout_v<MlaMetadataArgs> && std::is_trivially_copyable_v<MlaMetadataArgs>);
} // namespace flash_attn::mla
