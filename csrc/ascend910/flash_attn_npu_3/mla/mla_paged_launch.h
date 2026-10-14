// Private production PA launch interface. No public entry is enabled here.
#pragma once
#include "mla_launch.h"
#include "mla_paged_metadata.h"
#include "mla_paged_inputs.h"
#include "mla_window_params.h"

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
} // namespace flash_attn::mla
