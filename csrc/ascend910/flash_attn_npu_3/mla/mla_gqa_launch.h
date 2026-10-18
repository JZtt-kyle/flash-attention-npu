#pragma once
#include "mla_launch.h"
#include "mla_gqa_metadata.h"
#include "mla_paged_inputs.h"
namespace flash_attn::mla {
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
