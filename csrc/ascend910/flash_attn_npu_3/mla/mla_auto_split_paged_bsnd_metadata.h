// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
// PA source geometry and native AIC/FD provenance: mla_auto_split_paged_bsnd_plan.h.
#pragma once
#include "mla_paged_metadata.h"
#include "mla_paged_metadata_args.h"
#include "mla_auto_split_bsnd_metadata.h"

namespace flash_attn::mla {
constexpr uint32_t kMlaAutoSplitPagedBsndVersion = 24;
// Same record fields; bind the enabled feature, never the positive cap value.
constexpr uint32_t kMlaSoftcapAutoSplitPagedBsndVersion = 42;
constexpr uint32_t kMlaAutoSplitPagedBsndFixedWords = 756;
// v6 logical PA snapshot + native AIC/FD/Vector tail + B+1 uint64 M prefixes.
// Physical pages, cache addresses, strides and scale remain invocation state.
inline bool AutoSplitPagedBsndMetadataWords(int64_t batch, uint32_t& words)
{
    if (batch < 0 || uint64_t(batch) > (UINT32_MAX - kMlaAutoSplitPagedBsndFixedWords) / 7)
        return false;
    words = kMlaAutoSplitPagedBsndFixedWords + 7 * uint32_t(batch);
    return true;
}
struct MlaAutoSplitPagedBsndContext {
    MlaPagedMetadataContext paged;
    MlaWindowKernelParams window;
    uint32_t hasSoftcap = 0;
};
struct MlaAutoSplitPagedBsndMetadataArgs {
    MlaPagedMetadataArgs input;
    MlaWindowConfig window;
    uint32_t hasSoftcap = 0;
};
// False leaves output untouched for unsafe output/overlap. True writes a
// complete success or zero-payload failure record, including empty inputs.
bool BuildAutoSplitPagedBsndMetadata(const MlaPagedShape&, const MlaPlatformInfo&, uint32_t dtype, uint32_t* output,
                                     uint32_t words, const MlaPagedInputs&, const MlaWindowConfig&,
                                     uint32_t hasSoftcap = 0);
} // namespace flash_attn::mla
