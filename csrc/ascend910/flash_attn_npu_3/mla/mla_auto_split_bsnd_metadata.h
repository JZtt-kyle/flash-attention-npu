// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
// Native partition provenance: mla_auto_split_kv_plan.h. This is a local wire
// protocol, not a CANN ABI; the production adapter defines the FA3 capacity.
#pragma once
#include "mla_used_metadata.h"
#include "mla_window.h"
#include "mla_window_params.h"

namespace flash_attn::mla {
constexpr uint32_t kMlaAutoSplitBsndVersion = 18;
static_assert(kMlaMaxAic == 26, "Changing endpoint capacity requires a new wire version");
// v5 geometry/lengths remain independently validated. The tail contains window
// (3), native usedCoreNum (1), AIC endpoints (3*26), FD scalars (4), FD arrays
// (6*26+2*52), then B+1 low/high M prefixes. No addresses, costs or CPU objects.
constexpr uint32_t kMlaAutoSplitTailPrefix = 346;
constexpr uint32_t kMlaAutoSplitFixedWords = 758;
inline bool AutoSplitBsndMetadataWords(int64_t batch, uint32_t& words)
{
    if (batch < 0 || uint64_t(batch) > (UINT32_MAX - kMlaAutoSplitFixedWords) / 10)
        return false;
    words = kMlaAutoSplitFixedWords + 10 * uint32_t(batch);
    return true;
}
struct MlaAutoSplitBsndContext {
    MlaUsedMetadataContext used;
    MlaWindowKernelParams window;
};
struct MlaAutoSplitBsndMetadataArgs {
    MlaUsedMetadataArgs input;
    MlaWindowConfig window;
};
// Static public allocation capacity; native per-M partitions remain variable.
constexpr uint32_t kMlaAutoSplitCapacity = kMlaMaxAic;
bool BuildAutoSplitBsndMetadata(const MlaUsedShape&, const MlaPlatformInfo&, uint32_t dtype, uint32_t* output,
                                uint32_t words, const MlaUsedInputs&, const MlaWindowConfig&);
} // namespace flash_attn::mla
