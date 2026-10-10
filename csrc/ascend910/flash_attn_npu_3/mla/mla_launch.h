// Private host launch interface; no device algorithm headers in flash_api.cpp.
#pragma once
#include "acl/acl.h"
#include "mla_dense_params.h"
#include "mla_metadata.h"
#include "mla_tnd_metadata.h"
#include "mla_tnd_bsnd_metadata.h"
#include "mla_used_metadata.h"
#include "mla_bsnd_layout.h"

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
};
aclError LaunchTndFp16(const MlaTndLaunchArgs& args);
aclError LaunchTndBf16(const MlaTndLaunchArgs& args);

struct MlaTndBsndLaunchArgs {
    aclrtStream stream;
    uint64_t ffts;
    uint8_t *qc, *qr, *kv, *kr, *output, *lse, *workspace, *metadata, *cuQuery, *cacheSeqlens;
    MlaTndBsndMetadataContext context;
    float scale;
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
    MlaBsndInputLayouts layouts{};
    bool hasStrides = false;
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
