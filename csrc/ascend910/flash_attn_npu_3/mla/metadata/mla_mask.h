#pragma once
#include "mla_sequence.h"
#include "../tiling/mla_window.h"
#include "../mla_params.h"
#include "mla_paged.h"

// BSND right-down causal protocol. The v5 allocation/used wire layout is
// shared, but version/kind are distinct: a no-mask consumer must reject v7.
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

// Nonpaged window protocols; source/license: tiling/mla_window.cpp and LICENSE.cann / LICENSE.fa3.
namespace flash_attn::mla {
constexpr uint32_t kMlaWindowBsndMetadataVersion = 12;
constexpr uint32_t kMlaMetadataWindowBsnd = 12;
constexpr uint32_t kMlaWindowTndMetadataVersion = 13;
constexpr uint32_t kMlaMetadataWindowTnd = 13;
constexpr uint32_t kMlaWindowMixedMetadataVersion = 14;
constexpr uint32_t kMlaMetadataWindowMixed = 14;
// Three signed int32 tail words: normalized kind, left, right. Allocations in
// the used prefix retain the call-level bounds independently of used lengths.
constexpr uint32_t kMlaWindowMetadataTailWords = 3;
struct MlaWindowMetadataArgs {
    MlaUsedMetadataArgs input;
    MlaWindowConfig window;
};
static_assert(std::is_standard_layout_v<MlaWindowMetadataArgs> && std::is_trivially_copyable_v<MlaWindowMetadataArgs>);
inline bool WindowMetadataWords(int64_t batch, uint32_t& words)
{
    if (batch < 0 || uint64_t(batch) > (UINT32_MAX - kMlaUsedMetadataPrefixWords - kMlaWindowMetadataTailWords) /
                                           kMlaUsedMetadataSequenceWords) {
        return false;
    }
    words = kMlaUsedMetadataPrefixWords + kMlaUsedMetadataSequenceWords * uint32_t(batch) + kMlaWindowMetadataTailWords;
    return true;
}
bool BuildWindowBsndMetadata(const MlaUsedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype,
                             uint32_t* output, uint32_t outputWords, const MlaUsedInputs& inputs,
                             const MlaWindowConfig& window);
bool BuildWindowTndMetadata(const MlaUsedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype,
                            uint32_t* output, uint32_t outputWords, const MlaUsedInputs& inputs,
                            const MlaWindowConfig& window);
bool BuildWindowMixedMetadata(const MlaUsedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype,
                              uint32_t* output, uint32_t outputWords, const MlaUsedInputs& inputs,
                              const MlaWindowConfig& window, bool queryTotalFromPrefix = false);
} // namespace flash_attn::mla

// Private BSND PA window protocol; pinned sources/licenses: tiling/mla_window.cpp.
namespace flash_attn::mla {
constexpr uint32_t kMlaWindowPagedBsndMetadataVersion = 15;
constexpr uint32_t kMlaMetadataWindowPagedBsnd = 15;
constexpr uint32_t kMlaWindowPagedTndMetadataVersion = 16;
constexpr uint32_t kMlaMetadataWindowPagedTnd = 16;
constexpr uint32_t kMlaWindowPagedMetadataTailWords = 3;
struct MlaWindowPagedMetadataArgs {
    MlaPagedMetadataArgs input;
    MlaWindowConfig window;
};
static_assert(std::is_standard_layout_v<MlaWindowPagedMetadataArgs> &&
              std::is_trivially_copyable_v<MlaWindowPagedMetadataArgs>);
inline bool WindowPagedMetadataWords(int64_t batch, uint32_t& words)
{
    if (batch < 0 || uint64_t(batch) > (UINT32_MAX - kMlaPagedMetadataPrefixWords - kMlaWindowPagedMetadataTailWords) /
                                           kMlaPagedMetadataSequenceWords) {
        return false;
    }
    words = kMlaPagedMetadataPrefixWords + uint32_t(batch) * kMlaPagedMetadataSequenceWords +
            kMlaWindowPagedMetadataTailWords;
    return true;
}
bool BuildWindowPagedBsndMetadata(const MlaPagedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype,
                                  uint32_t* output, uint32_t outputWords, const MlaPagedInputs& inputs,
                                  const MlaWindowConfig& window);
bool BuildWindowPagedTndMetadata(const MlaPagedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype,
                                 uint32_t* output, uint32_t outputWords, const MlaPagedInputs& inputs,
                                 const MlaWindowConfig& window, bool fromQueryPrefix = false);
} // namespace flash_attn::mla
