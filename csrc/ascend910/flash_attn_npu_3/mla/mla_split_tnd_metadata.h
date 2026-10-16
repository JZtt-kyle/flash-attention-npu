// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
// TND explicit-split adapter for the pinned CANN/FA3 plan in mla_split_kv_plan.h.
#pragma once
#include "mla_used_metadata.h"
#include "mla_window.h"
#include "mla_window_params.h"

namespace flash_attn::mla {
struct MlaSplitTndContext {
    MlaUsedMetadataContext used;
    MlaWindowKernelParams window;
    uint32_t splits;
    uint32_t hasSoftcap = 0;
};
constexpr uint32_t kMlaSplitTndVersion = 19;
constexpr uint32_t kMlaSoftcapSplitTndVersion = 35;
// v5 allocation/used prefix and spans, then window(kind,left,right), splits,
// and B+1 low/high uint64 compact M prefixes. No addresses or CPU objects.
inline bool SplitTndMetadataWords(int64_t batch, uint32_t& words)
{
    if (batch < 0 || uint64_t(batch) > (UINT32_MAX - 416) / 10)
        return false;
    words = 416 + 10 * uint32_t(batch);
    return true;
}
struct MlaSplitTndMetadataArgs {
    MlaUsedMetadataArgs input;
    MlaWindowConfig window;
    int64_t splits;
    uint32_t hasSoftcap = 0;
};
bool BuildSplitTndMetadata(const MlaUsedShape&, const MlaPlatformInfo&, uint32_t dtype, uint32_t* output,
                           uint32_t words, const MlaUsedInputs&, const MlaWindowConfig&, int64_t splits,
                           uint32_t hasSoftcap = 0);
} // namespace flash_attn::mla
