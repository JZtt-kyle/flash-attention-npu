// BSND right-down causal protocol. The v5 allocation/used wire layout is
// shared, but version/kind are distinct: a no-mask consumer must reject v7.
#pragma once
#include "mla_used_metadata.h"

namespace flash_attn::mla {
constexpr uint32_t kMlaCausalBsndMetadataVersion = 7;
constexpr uint32_t kMlaMetadataCausalBsnd = 7;
constexpr uint32_t kMlaCausalTndMetadataVersion = 8;
constexpr uint32_t kMlaMetadataCausalTnd = 8;
constexpr uint32_t kMlaCausalMixedMetadataVersion = 9;
constexpr uint32_t kMlaMetadataCausalMixed = 9;

// Immutable host-generated candidate for full allocation lengths. The AICPU
// producer validates current inputs and geometry before using this partition.
// No pointer, device length value, tensor or serialized metadata is cached.
struct MlaCausalFullSchedule {
    uint32_t enabled = 0;
    uint32_t batch, queryCapacity, kvCapacity, heads;
    uint32_t aicNum, cvRatio, mBaseSize, s2BaseSize, usedCoreNum;
    std::array<uint32_t, kMlaMaxAic> bN2End, gS1End, s2End;
    int64_t maxCost;
};
static_assert(std::is_standard_layout_v<MlaCausalFullSchedule> && std::is_trivially_copyable_v<MlaCausalFullSchedule>);

struct MlaCausalMetadataArgs {
    MlaUsedMetadataArgs input;
    MlaCausalFullSchedule full;
};
static_assert(std::is_standard_layout_v<MlaCausalMetadataArgs> && std::is_trivially_copyable_v<MlaCausalMetadataArgs>);

// A failed static candidate is simply unavailable; actual used lengths can
// still yield a valid schedule. The normal producer remains the fallback.
bool BuildCausalFullSchedule(const MlaUsedShape& shape, const MlaPlatformInfo& platform, MlaCausalFullSchedule& output);

// Same buffer/descriptor/failure guarantees as BuildUsedMetadata. Only BSND
// Q/KV are supported. Effective Sq/Sk come from the serialized length snapshot,
// never from capacities. Empty allocations retain validation-only scheduling.
bool BuildCausalBsndMetadata(const MlaUsedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype,
                             uint32_t* output, uint32_t outputWords, const MlaUsedInputs& inputs,
                             const MlaCausalFullSchedule* full = nullptr);
// TND Q/KV retain both allocation prefixes and optional used arrays. No
// rectangular full-length candidate is valid for these device-owned spans.
bool BuildCausalTndMetadata(const MlaUsedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype,
                            uint32_t* output, uint32_t outputWords, const MlaUsedInputs& inputs);
// TND Q + BSND KV. Public metadata derives Q allocation size from its prefix
// after descriptor validation; forward generation uses the tensor's size.
bool BuildCausalMixedMetadata(const MlaUsedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype,
                              uint32_t* output, uint32_t outputWords, const MlaUsedInputs& inputs,
                              bool queryTotalFromPrefix = false);
} // namespace flash_attn::mla
