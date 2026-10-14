#pragma once
#include "mla_launch.h"
#include "mla_split_bsnd_metadata.h"
namespace flash_attn::mla {
struct MlaSplitBsndLaunchArgs {
    MlaUsedLaunchArgs used;
    uint32_t splits;
    uint8_t *accum, *accumLse;
};
aclError LaunchSplitBsndFp16(const MlaSplitBsndLaunchArgs&);
aclError LaunchSplitBsndBf16(const MlaSplitBsndLaunchArgs&);
} // namespace flash_attn::mla
