// Private host/AICPU launch ABI; addresses are never serialized in v3 records.
#pragma once
#include "mla_tnd_schedule.h"
#include <type_traits>

namespace flash_attn::mla {
struct MlaTndMetadataArgs {
    uint64_t outputAddr;
    uint32_t outputWords, dtype;
    MlaTndShape shape;
    MlaPlatformInfo platform;
    uint64_t cuQueryAddr, queryCount;
    uint64_t cuKvAddr, kvCount;
};
static_assert(std::is_standard_layout_v<MlaTndMetadataArgs> && std::is_trivially_copyable_v<MlaTndMetadataArgs>);
} // namespace flash_attn::mla
