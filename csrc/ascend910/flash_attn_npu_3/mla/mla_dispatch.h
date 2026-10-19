#pragma once

#include "acl/acl.h"
#include "mla_params.h"
#include "mla_metadata.h"
#include "mla_layout.h"
#include "mla_int8_schedule.h"

// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.

// Private host launch interface; no device algorithm headers in flash_api.cpp.
namespace flash_attn::mla {
struct MlaLaunchArgs {
    aclrtStream stream;
    uint64_t ffts;
    uint8_t *qc, *qr, *kv, *kr, *output, *lse, *workspace, *metadata, *cacheSeqlens;
    MlaDenseMetadataContext context;
    float scale;
    bool hasKvLengths = false;
    MlaBsndInputLayouts layouts{};
    bool hasStrides = false;
};

aclError LaunchDenseFp16(const MlaLaunchArgs& args);
aclError LaunchDenseBf16(const MlaLaunchArgs& args);

struct MlaTndLaunchArgs {
    aclrtStream stream;
    uint64_t ffts;
    uint8_t *qc, *qr, *kv, *kr, *output, *lse, *workspace, *metadata, *cuQuery, *cuKv;
    MlaTndMetadataContext context;
    float scale;
    MlaBsndInputLayouts layouts{};
    bool hasStrides = false;
};
aclError LaunchTndFp16(const MlaTndLaunchArgs& args);
aclError LaunchTndBf16(const MlaTndLaunchArgs& args);

struct MlaTndBsndLaunchArgs {
    aclrtStream stream;
    uint64_t ffts;
    uint8_t *qc, *qr, *kv, *kr, *output, *lse, *workspace, *metadata, *cuQuery, *cacheSeqlens;
    MlaTndBsndMetadataContext context;
    float scale;
    MlaBsndInputLayouts layouts{};
    bool hasStrides = false;
};
aclError LaunchTndBsndFp16(const MlaTndBsndLaunchArgs& args);
aclError LaunchTndBsndBf16(const MlaTndBsndLaunchArgs& args);

struct MlaUsedLaunchArgs {
    aclrtStream stream;
    uint64_t ffts;
    uint8_t *qc, *qr, *kv, *kr, *output, *lse, *workspace, *metadata;
    MlaUsedMetadataContext context;
    MlaUsedMetadataInputs inputs;
    float scale;
    float softcap = 0.0f;
    MlaBsndInputLayouts layouts{};
    bool hasStrides = false;
    bool causal = false;
    bool isWindow = false;
    MlaWindowKernelParams window{};
};
aclError LaunchBsndUsedFp16(const MlaUsedLaunchArgs& args);
aclError LaunchBsndUsedBf16(const MlaUsedLaunchArgs& args);
aclError LaunchTndUsedFp16(const MlaUsedLaunchArgs& args);
aclError LaunchTndUsedBf16(const MlaUsedLaunchArgs& args);

// Bisheng 9.0's ASC launcher does not expose a scheduling-mode argument.
// The link wrapper changes only the launch inside this thread-local scope.
// Ordinary FA3 calls pass through unchanged; a different compiler path fails
// explicitly instead of silently launching MLA in NORMAL_MODE.
struct MlaBatchLaunchState {
    uint32_t calls = 0;
    aclError error = ACL_SUCCESS;
};
extern thread_local MlaBatchLaunchState* activeMlaLaunch;

class MlaBatchLaunchScope {
  public:
    MlaBatchLaunchScope() : previous(activeMlaLaunch)
    {
        activeMlaLaunch = &state;
    }
    ~MlaBatchLaunchScope()
    {
        activeMlaLaunch = previous;
    }
    MlaBatchLaunchScope(const MlaBatchLaunchScope&) = delete;
    MlaBatchLaunchScope& operator=(const MlaBatchLaunchScope&) = delete;
    aclError Result() const
    {
        return state.calls == 1 ? state.error : ACL_ERROR_INVALID_PARAM;
    }

  private:
    MlaBatchLaunchState state;
    MlaBatchLaunchState* previous;
};
} // namespace flash_attn::mla

// Private production PA launch interface. No public entry is enabled here.
namespace flash_attn::mla {
struct MlaPagedLaunchArgs {
    aclrtStream stream;
    uint64_t ffts;
    uint8_t *qc, *qr, *output, *lse, *workspace, *metadata;
    MlaPagedMetadataContext context;
    MlaPagedMetadataInputs inputs;
    MlaPagedCacheInputs cache;
    MlaBsndInputLayouts layouts;
    float scale;
    bool causal = false;
    bool isWindow = false;
    MlaWindowKernelParams window;
};
aclError LaunchPagedBsndFp16(const MlaPagedLaunchArgs& args);
aclError LaunchPagedBsndBf16(const MlaPagedLaunchArgs& args);
aclError LaunchPagedTndFp16(const MlaPagedLaunchArgs& args);
aclError LaunchPagedTndBf16(const MlaPagedLaunchArgs& args);
struct MlaSplitBsndLaunchArgs {
    MlaUsedLaunchArgs used;
    uint32_t splits;
    uint8_t *accum, *accumLse;
};
aclError LaunchSplitBsndFp16(const MlaSplitBsndLaunchArgs&);
aclError LaunchSplitBsndBf16(const MlaSplitBsndLaunchArgs&);
struct MlaSplitTndLaunchArgs {
    MlaUsedLaunchArgs used;
    uint32_t splits;
    uint8_t *accum, *accumLse;
};
aclError LaunchSplitTndFp16(const MlaSplitTndLaunchArgs&);
aclError LaunchSplitTndBf16(const MlaSplitTndLaunchArgs&);
struct MlaSplitMixedLaunchArgs {
    MlaUsedLaunchArgs used;
    uint32_t splits;
    uint8_t *accum, *accumLse;
};
aclError LaunchSplitMixedFp16(const MlaSplitMixedLaunchArgs&);
aclError LaunchSplitMixedBf16(const MlaSplitMixedLaunchArgs&);
struct MlaSplitPagedBsndLaunchArgs {
    MlaPagedLaunchArgs paged;
    uint32_t splits;
    uint8_t *accum, *accumLse;
    float softcap = 0.0f;
};
aclError LaunchSplitPagedBsndFp16(const MlaSplitPagedBsndLaunchArgs&);
aclError LaunchSplitPagedBsndBf16(const MlaSplitPagedBsndLaunchArgs&);
using MlaSplitPagedTndLaunchArgs = MlaSplitPagedBsndLaunchArgs;
aclError LaunchSplitPagedTndFp16(const MlaSplitPagedTndLaunchArgs&);
aclError LaunchSplitPagedTndBf16(const MlaSplitPagedTndLaunchArgs&);
aclError LaunchAutoSplitBsndFp16(const MlaSplitBsndLaunchArgs&);
aclError LaunchAutoSplitBsndBf16(const MlaSplitBsndLaunchArgs&);
aclError LaunchAutoSplitTndFp16(const MlaSplitTndLaunchArgs&);
aclError LaunchAutoSplitTndBf16(const MlaSplitTndLaunchArgs&);
aclError LaunchAutoSplitMixedFp16(const MlaSplitMixedLaunchArgs&);
aclError LaunchAutoSplitMixedBf16(const MlaSplitMixedLaunchArgs&);
aclError LaunchAutoSplitPagedBsndFp16(const MlaSplitPagedBsndLaunchArgs&);
aclError LaunchAutoSplitPagedBsndBf16(const MlaSplitPagedBsndLaunchArgs&);
aclError LaunchAutoSplitPagedTndFp16(const MlaSplitPagedBsndLaunchArgs&);
aclError LaunchAutoSplitPagedTndBf16(const MlaSplitPagedBsndLaunchArgs&);
struct MlaPrefillBsndLaunchArgs {
    aclrtStream stream;
    uint64_t ffts;
    uint64_t q, k, v, out, lse, accum, accumLse, workspace, metadata, usedQuery, usedKv;
    MlaPrefillBsndContext context;
    MlaPrefillParams params;
    uint64_t cuQuery = 0, cuKv = 0;
    MlaPagedCacheInputs cache{};
};
aclError LaunchPrefillBsndFp16(const MlaPrefillBsndLaunchArgs&);
aclError LaunchPrefillBsndBf16(const MlaPrefillBsndLaunchArgs&);
struct MlaGqaLaunchArgs {
    aclrtStream stream;
    uint64_t ffts, q, qv, k, v, out, lse, accum, accumLse, workspace, metadata;
    uint64_t cuQuery, cuKv, usedQuery, usedKv;
    MlaGqaContext context;
    MlaBsndInputLayouts inputs;
    MlaPagedCacheInputs cache;
    float scale, softcap;
};
aclError LaunchGqaFp16(const MlaGqaLaunchArgs&);
aclError LaunchGqaBf16(const MlaGqaLaunchArgs&);
} // namespace flash_attn::mla

namespace flash_attn::mla::Int8 {
struct LaunchArgs {
    aclrtStream stream;
    uint64_t ffts, q, qv, k, v, qs, ks, vs, table, lengths, metadata, mask, out, lse, accum, accumLse, workspace;
    Config config;
    MlaPagedCacheLayout cache;
    float scale;
    uint64_t prefix = 0;
    uint32_t totalQuery = 0;
};
aclError Launch(const LaunchArgs&);
} // namespace flash_attn::mla::Int8
