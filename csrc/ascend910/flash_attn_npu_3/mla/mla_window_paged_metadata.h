// Private BSND PA window protocol; pinned sources/licenses: mla_window.cpp.
#pragma once
#include "mla_paged_metadata.h"
#include "mla_paged_metadata_args.h"
#include "mla_window.h"
#include "mla_window_params.h"

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
