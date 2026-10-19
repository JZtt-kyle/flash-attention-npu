// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
#pragma once
#include "mla_sequence.h"
#include "../tiling/mla_window.h"
#include "../mla_params.h"
#include "mla_paged.h"

// BSND explicit-split adapter for the pinned CANN/FA3 plan in tiling/mla_split_plan.h.
namespace flash_attn::mla {
struct MlaSplitBsndContext {
    MlaUsedMetadataContext used;
    MlaWindowKernelParams window;
    uint32_t splits;
    uint32_t hasSoftcap = 0;
};
constexpr uint32_t kMlaSplitBsndVersion = 17;
constexpr uint32_t kMlaSoftcapSplitBsndVersion = 32;
// v5 allocation/used prefix and spans, then window(kind,left,right), splits,
// and B+1 low/high uint64 compact M prefixes. No addresses or CPU objects.
inline bool SplitBsndMetadataWords(int64_t batch, uint32_t& words)
{
    if (batch < 0 || uint64_t(batch) > (UINT32_MAX - 416) / 10)
        return false;
    words = 416 + 10 * uint32_t(batch);
    return true;
}
struct MlaSplitBsndMetadataArgs {
    MlaUsedMetadataArgs input;
    MlaWindowConfig window;
    int64_t splits;
    uint32_t hasSoftcap = 0;
};
bool BuildSplitBsndMetadata(const MlaUsedShape&, const MlaPlatformInfo&, uint32_t dtype, uint32_t* output,
                            uint32_t words, const MlaUsedInputs&, const MlaWindowConfig&, int64_t splits,
                            uint32_t hasSoftcap = 0);
} // namespace flash_attn::mla

// TND explicit-split adapter for the pinned CANN/FA3 plan in tiling/mla_split_plan.h.
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

// Mixed TND Q / BSND KV explicit-split adapter for the pinned CANN/FA3 plan in tiling/mla_split_plan.h.
namespace flash_attn::mla {
struct MlaSplitMixedContext {
    MlaUsedMetadataContext used;
    MlaWindowKernelParams window;
    uint32_t splits;
    uint32_t hasSoftcap = 0;
};
constexpr uint32_t kMlaSplitMixedVersion = 21;
constexpr uint32_t kMlaSoftcapSplitMixedVersion = 38;
// v5 allocation/used prefix and spans, then window(kind,left,right), splits,
// and B+1 low/high uint64 compact M prefixes. No addresses or CPU objects.
inline bool SplitMixedMetadataWords(int64_t batch, uint32_t& words)
{
    if (batch < 0 || uint64_t(batch) > (UINT32_MAX - 416) / 10)
        return false;
    words = 416 + 10 * uint32_t(batch);
    return true;
}
struct MlaSplitMixedMetadataArgs {
    MlaUsedMetadataArgs input;
    MlaWindowConfig window;
    int64_t splits;
    bool queryTotalFromPrefix = false;
    uint32_t hasSoftcap = 0;
};
bool BuildSplitMixedMetadata(const MlaUsedShape&, const MlaPlatformInfo&, uint32_t dtype, uint32_t* output,
                             uint32_t words, const MlaUsedInputs&, const MlaWindowConfig&, int64_t splits,
                             bool queryTotalFromPrefix = false, uint32_t hasSoftcap = 0);
} // namespace flash_attn::mla

// BSND PA explicit-split protocol; source attribution in the plan header.
namespace flash_attn::mla {
constexpr uint32_t kMlaSplitPagedBsndVersion = 23;
// Same record fields; bind the enabled feature, never the positive cap value.
constexpr uint32_t kMlaSoftcapSplitPagedBsndVersion = 41;
struct MlaSplitPagedBsndContext {
    MlaPagedMetadataContext paged;
    MlaWindowKernelParams window;
    uint32_t splits;
    uint32_t hasSoftcap = 0;
};
// v6 logical PA snapshot, window(kind,left,right), S, B+1 uint64 M prefixes.
// Does not bind page IDs, physical cache, addresses, strides or scale.
inline bool SplitPagedBsndMetadataWords(int64_t batch, uint32_t& words)
{
    if (batch < 0 || uint64_t(batch) > (UINT32_MAX - 414) / 7)
        return false;
    words = 414 + 7 * uint32_t(batch);
    return true;
}
struct MlaSplitPagedBsndMetadataArgs {
    MlaPagedMetadataArgs input;
    MlaWindowConfig window;
    int64_t splits;
    uint32_t hasSoftcap = 0;
};
// False leaves output untouched (unsafe descriptor/overlap); true writes a
// complete record, including a zero-payload failure record. No device reads.
bool BuildSplitPagedBsndMetadata(const MlaPagedShape&, const MlaPlatformInfo&, uint32_t dtype, uint32_t* output,
                                 uint32_t words, const MlaPagedInputs&, const MlaWindowConfig&, int64_t splits,
                                 uint32_t hasSoftcap = 0);
} // namespace flash_attn::mla

// TND PA explicit-split protocol; source attribution in the plan header.
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
