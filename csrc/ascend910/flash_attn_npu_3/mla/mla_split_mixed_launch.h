#pragma once
#include "mla_launch.h"
#include "mla_split_mixed_metadata.h"
namespace flash_attn::mla {
struct MlaSplitMixedLaunchArgs {
    MlaUsedLaunchArgs used;
    uint32_t splits;
    uint8_t *accum, *accumLse;
};
aclError LaunchSplitMixedFp16(const MlaSplitMixedLaunchArgs&);
aclError LaunchSplitMixedBf16(const MlaSplitMixedLaunchArgs&);
} // namespace flash_attn::mla
