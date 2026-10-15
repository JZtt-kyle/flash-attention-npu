// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
// PA source geometry and native AIC/FD provenance: mla_auto_split_paged_tnd_plan.h.
#pragma once
#include "mla_paged_metadata.h"
#include "mla_paged_metadata_args.h"
#include "mla_auto_split_bsnd_metadata.h"

namespace flash_attn::mla {
constexpr uint32_t kMlaAutoSplitPagedTndVersion = 26;
constexpr uint32_t kMlaAutoSplitPagedTndFixedWords = 756;
// v26: v6 compact Q allocation/used PA snapshot + native AIC/FD/Vector tail + B+1 uint64 M prefixes.
// Physical pages, cache addresses, strides and scale remain invocation state.
inline bool AutoSplitPagedTndMetadataWords(int64_t batch, uint32_t& words)
{
    if (batch < 0 || uint64_t(batch) > (UINT32_MAX - kMlaAutoSplitPagedTndFixedWords) / 7)
        return false;
    words = kMlaAutoSplitPagedTndFixedWords + 7 * uint32_t(batch);
    return true;
}
struct MlaAutoSplitPagedTndContext {
    MlaPagedMetadataContext paged;
    MlaWindowKernelParams window;
};
struct MlaAutoSplitPagedTndMetadataArgs {
    MlaPagedMetadataArgs input;
    MlaWindowConfig window;
    bool queryTotalFromPrefix = false;
};
// False leaves output untouched for unsafe output/overlap. True writes a
// complete success or zero-payload failure record, including empty inputs.
bool BuildAutoSplitPagedTndMetadata(const MlaPagedShape&, const MlaPlatformInfo&, uint32_t dtype, uint32_t* output,
                                    uint32_t words, const MlaPagedInputs&, const MlaWindowConfig&,
                                    bool queryTotalFromPrefix = false);
} // namespace flash_attn::mla
