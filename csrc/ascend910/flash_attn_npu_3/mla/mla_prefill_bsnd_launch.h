#pragma once
#include "mla_launch.h"
#include "mla_prefill_bsnd_metadata.h"
namespace flash_attn::mla {
struct MlaPrefillBsndLaunchArgs {
    aclrtStream stream;
    uint64_t ffts;
    uint64_t q, k, v, out, lse, accum, accumLse, workspace, metadata, usedQuery, usedKv;
    MlaPrefillBsndContext context;
    MlaPrefillParams params;
};
aclError LaunchPrefillBsndFp16(const MlaPrefillBsndLaunchArgs&);
aclError LaunchPrefillBsndBf16(const MlaPrefillBsndLaunchArgs&);
} // namespace flash_attn::mla
