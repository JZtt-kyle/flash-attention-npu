// Private host launch interface; no device algorithm headers in flash_api.cpp.
#pragma once
#include "acl/acl.h"
#include "mla_dense_params.h"
#include "mla_metadata.h"

namespace flash_attn::mla {
struct MlaLaunchArgs {
    aclrtStream stream;
    uint64_t ffts;
    uint8_t *qc, *qr, *kv, *kr, *output, *lse, *workspace, *metadata;
    MlaDenseMetadataContext context;
    float scale;
};

aclError LaunchDenseFp16(const MlaLaunchArgs& args);
aclError LaunchDenseBf16(const MlaLaunchArgs& args);

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
