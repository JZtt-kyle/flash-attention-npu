// Nonpaged window protocols; source/license: mla_window.cpp and LICENSE.cann / LICENSE.fa3.
#pragma once
#include "mla_used_metadata.h"
#include "mla_window.h"
#include "mla_window_params.h"

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
