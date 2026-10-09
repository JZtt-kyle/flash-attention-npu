// Private host/AICPU launch ABI; addresses are never serialized in v4 records.
#pragma once
#include "mla_tnd_bsnd_schedule.h"
#include <type_traits>

namespace flash_attn::mla {
struct MlaTndBsndMetadataArgs {
    uint64_t outputAddr;
    uint32_t outputWords, dtype;
    MlaTndBsndShape shape;
    MlaPlatformInfo platform;
    uint64_t cuQueryAddr, queryCount;
    uint64_t kvLengthsAddr, kvCount;
};
static_assert(std::is_standard_layout_v<MlaTndBsndMetadataArgs> &&
              std::is_trivially_copyable_v<MlaTndBsndMetadataArgs>);
} // namespace flash_attn::mla
