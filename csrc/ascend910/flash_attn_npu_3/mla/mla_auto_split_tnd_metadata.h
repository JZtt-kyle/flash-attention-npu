// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
// TND native partition provenance: mla_auto_split_tnd_plan.h.
// Local wire protocol only; no public/production automatic TND path yet.
#pragma once
#include "mla_auto_split_bsnd_metadata.h"

namespace flash_attn::mla {
constexpr uint32_t kMlaAutoSplitTndVersion = 20;
// Same tail schema as v18, distinct identity and TND allocation geometry.
inline bool AutoSplitTndMetadataWords(int64_t batch, uint32_t& words)
{
    if (batch < 0 || uint64_t(batch) > (UINT32_MAX - kMlaAutoSplitFixedWords) / 10)
        return false;
    words = kMlaAutoSplitFixedWords + 10 * uint32_t(batch);
    return true;
}
struct MlaAutoSplitTndContext {
    MlaUsedMetadataContext used;
    MlaWindowKernelParams window;
};
struct MlaAutoSplitTndMetadataArgs {
    MlaUsedMetadataArgs input;
    MlaWindowConfig window;
};
bool BuildAutoSplitTndMetadata(const MlaUsedShape&, const MlaPlatformInfo&, uint32_t dtype, uint32_t* output,
                               uint32_t words, const MlaUsedInputs&, const MlaWindowConfig&);
} // namespace flash_attn::mla
