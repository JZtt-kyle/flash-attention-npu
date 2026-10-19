// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
#pragma once
#include "mla_sequence.h"
#include "../tiling/mla_window.h"
#include "../mla_params.h"
#include "mla_paged.h"

// Native partition provenance: tiling/mla_auto_split_plan.h. This is a local wire
// protocol, not a CANN ABI; the production adapter defines the FA3 capacity.
namespace flash_attn::mla {
constexpr uint32_t kMlaAutoSplitBsndVersion = 18;
constexpr uint32_t kMlaSoftcapAutoSplitBsndVersion = 33;
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
    uint32_t hasSoftcap = 0;
};
struct MlaAutoSplitBsndMetadataArgs {
    MlaUsedMetadataArgs input;
    MlaWindowConfig window;
    uint32_t hasSoftcap = 0;
};
// Static public allocation capacity; native per-M partitions remain variable.
constexpr uint32_t kMlaAutoSplitCapacity = kMlaMaxAic;
bool BuildAutoSplitBsndMetadata(const MlaUsedShape&, const MlaPlatformInfo&, uint32_t dtype, uint32_t* output,
                                uint32_t words, const MlaUsedInputs&, const MlaWindowConfig&, uint32_t hasSoftcap = 0);
} // namespace flash_attn::mla

// TND native partition provenance: tiling/mla_auto_split_plan.h.
// Layout-specific protocol shared by the production AICPU and AICore stages.
namespace flash_attn::mla {
constexpr uint32_t kMlaAutoSplitTndVersion = 20;
constexpr uint32_t kMlaSoftcapAutoSplitTndVersion = 36;
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
    uint32_t hasSoftcap = 0;
};
struct MlaAutoSplitTndMetadataArgs {
    MlaUsedMetadataArgs input;
    MlaWindowConfig window;
    uint32_t hasSoftcap = 0;
};
bool BuildAutoSplitTndMetadata(const MlaUsedShape&, const MlaPlatformInfo&, uint32_t dtype, uint32_t* output,
                               uint32_t words, const MlaUsedInputs&, const MlaWindowConfig&, uint32_t hasSoftcap = 0);
} // namespace flash_attn::mla

// mixed native partition provenance: tiling/mla_auto_split_plan.h.
// Shared CPU/AICPU producer and independent CPU/AICore consumer protocol.
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

// PA source geometry and native AIC/FD provenance: tiling/mla_auto_split_plan.h.
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

// PA source geometry and native AIC/FD provenance: tiling/mla_auto_split_plan.h.
namespace flash_attn::mla {
constexpr uint32_t kMlaAutoSplitPagedTndVersion = 26;
// Same fields; bind softcap activation independently of the positive cap value.
constexpr uint32_t kMlaSoftcapAutoSplitPagedTndVersion = 45;
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
    uint32_t hasSoftcap = 0;
};
struct MlaAutoSplitPagedTndMetadataArgs {
    MlaPagedMetadataArgs input;
    MlaWindowConfig window;
    bool queryTotalFromPrefix = false;
    uint32_t hasSoftcap = 0;
};
// False leaves output untouched for unsafe output/overlap. True writes a
// complete success or zero-payload failure record, including empty inputs.
bool BuildAutoSplitPagedTndMetadata(const MlaPagedShape&, const MlaPlatformInfo&, uint32_t dtype, uint32_t* output,
                                    uint32_t words, const MlaPagedInputs&, const MlaWindowConfig&,
                                    bool queryTotalFromPrefix = false, uint32_t hasSoftcap = 0);
} // namespace flash_attn::mla
