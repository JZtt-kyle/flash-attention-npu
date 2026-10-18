// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
// mixed native partition provenance: mla_auto_split_mixed_plan.h.
// Shared CPU/AICPU producer and independent CPU/AICore consumer protocol.
#pragma once
#include "mla_auto_split_bsnd_metadata.h"

namespace flash_attn::mla {
constexpr uint32_t kMlaAutoSplitMixedVersion = 22;
constexpr uint32_t kMlaSoftcapAutoSplitMixedVersion = 39;
// Same tail schema as v18, distinct identity and mixed allocation geometry.
inline bool AutoSplitMixedMetadataWords(int64_t batch, uint32_t& words)
{
    if (batch < 0 || uint64_t(batch) > (UINT32_MAX - kMlaAutoSplitFixedWords) / 10)
        return false;
    words = kMlaAutoSplitFixedWords + 10 * uint32_t(batch);
    return true;
}
struct MlaAutoSplitMixedContext {
    MlaUsedMetadataContext used;
    MlaWindowKernelParams window;
    uint32_t hasSoftcap = 0;
};
struct MlaAutoSplitMixedMetadataArgs {
    MlaUsedMetadataArgs input;
    MlaWindowConfig window;
    bool queryTotalFromPrefix = false;
    uint32_t hasSoftcap = 0;
};
bool BuildAutoSplitMixedMetadata(const MlaUsedShape&, const MlaPlatformInfo&, uint32_t dtype, uint32_t* output,
                                 uint32_t words, const MlaUsedInputs&, const MlaWindowConfig&,
                                 bool queryTotalFromPrefix = false, uint32_t hasSoftcap = 0);
} // namespace flash_attn::mla
