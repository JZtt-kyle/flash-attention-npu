// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
// TND PA explicit-split protocol; source attribution in the plan header.
#pragma once
#include "mla_paged_metadata.h"
#include "mla_paged_metadata_args.h"
#include "mla_window.h"
#include "mla_window_params.h"

namespace flash_attn::mla {
constexpr uint32_t kMlaSplitPagedTndVersion = 25;
// Same fields; bind softcap activation independently of the positive cap value.
constexpr uint32_t kMlaSoftcapSplitPagedTndVersion = 44;
struct MlaSplitPagedTndContext {
    MlaPagedMetadataContext paged;
    MlaWindowKernelParams window;
    uint32_t splits;
    uint32_t hasSoftcap = 0;
};
// v6 logical PA snapshot, window(kind,left,right), S, B+1 uint64 M prefixes.
// Does not bind page IDs, physical cache, addresses, strides or scale.
inline bool SplitPagedTndMetadataWords(int64_t batch, uint32_t& words)
{
    if (batch < 0 || uint64_t(batch) > (UINT32_MAX - 414) / 7)
        return false;
    words = 414 + 7 * uint32_t(batch);
    return true;
}
struct MlaSplitPagedTndMetadataArgs {
    MlaPagedMetadataArgs input;
    MlaWindowConfig window;
    int64_t splits;
    bool queryTotalFromPrefix = false;
    uint32_t hasSoftcap = 0;
};
// False leaves output untouched (unsafe descriptor/overlap); true writes a
// complete record, including a zero-payload failure record. No device reads.
bool BuildSplitPagedTndMetadata(const MlaPagedShape&, const MlaPlatformInfo&, uint32_t dtype, uint32_t* output,
                                uint32_t words, const MlaPagedInputs&, const MlaWindowConfig&, int64_t splits,
                                bool queryTotalFromPrefix = false, uint32_t hasSoftcap = 0);
} // namespace flash_attn::mla
