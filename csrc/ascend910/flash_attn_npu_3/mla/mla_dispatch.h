#pragma once
#include <type_traits>

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

struct MlaTndLaunchArgs {
    aclrtStream stream;
    uint64_t ffts;
    uint8_t *qc, *qr, *kv, *kr, *output, *lse, *workspace, *metadata, *cuQuery, *cuKv;
    MlaTndMetadataContext context;
    float scale;
    MlaBsndInputLayouts layouts{};
    bool hasStrides = false;
};

struct MlaTndBsndLaunchArgs {
    aclrtStream stream;
    uint64_t ffts;
    uint8_t *qc, *qr, *kv, *kr, *output, *lse, *workspace, *metadata, *cuQuery, *cacheSeqlens;
    MlaTndBsndMetadataContext context;
    float scale;
    MlaBsndInputLayouts layouts{};
    bool hasStrides = false;
};

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
    uint64_t checked = 0;
};

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

// Private PA launch interface used by the host forward adapters.
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
    uint64_t checked = 0;
};
struct MlaSplitLaunchArgs {
    MlaUsedLaunchArgs used;
    uint32_t splits;
    uint8_t *accum, *accumLse;
};
struct MlaSplitPagedLaunchArgs {
    MlaPagedLaunchArgs paged;
    uint32_t splits;
    uint8_t *accum, *accumLse;
    float softcap = 0.0f;
};
struct MlaPrefillBsndLaunchArgs {
    aclrtStream stream;
    uint64_t ffts;
    uint64_t q, k, v, out, lse, accum, accumLse, workspace, metadata, usedQuery, usedKv;
    MlaPrefillBsndContext context;
    MlaPrefillParams params;
    uint64_t cuQuery = 0, cuKv = 0;
    MlaPagedCacheInputs cache{};
    uint64_t checked = 0;
};
struct MlaGqaLaunchArgs {
    aclrtStream stream;
    uint64_t ffts, q, qv, k, v, out, lse, accum, accumLse, workspace, metadata;
    uint64_t cuQuery, cuKv, usedQuery, usedKv;
    MlaGqaContext context;
    MlaBsndInputLayouts inputs;
    MlaPagedCacheInputs cache;
    float scale, softcap;
};
template <typename DType>
aclError LaunchDense(const MlaLaunchArgs& args);
template <typename DType>
aclError LaunchTnd(const MlaTndLaunchArgs& args);
template <typename DType>
aclError LaunchTndBsnd(const MlaTndBsndLaunchArgs& args);
template <typename DType>
aclError LaunchBsndUsed(const MlaUsedLaunchArgs& args);
template <typename DType>
aclError LaunchTndUsed(const MlaUsedLaunchArgs& args);
template <typename DType, bool IsTnd>
aclError LaunchPaged(const MlaPagedLaunchArgs& args);
template <typename DType, bool IsTnd, bool IsMixed>
aclError LaunchSplit(const MlaSplitLaunchArgs&);
template <typename DType, bool IsTnd>
aclError LaunchSplitPaged(const MlaSplitPagedLaunchArgs&);
template <typename DType, bool IsTnd, bool IsMixed>
aclError LaunchAutoSplit(const MlaSplitLaunchArgs&);
template <typename DType, bool IsTnd>
aclError LaunchAutoSplitPaged(const MlaSplitPagedLaunchArgs&);
template <typename DType>
aclError LaunchPrefillBsnd(const MlaPrefillBsndLaunchArgs&);
template <typename DType>
aclError LaunchGqa(const MlaGqaLaunchArgs&);

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
