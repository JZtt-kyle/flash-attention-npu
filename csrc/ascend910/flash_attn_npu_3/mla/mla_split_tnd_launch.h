#pragma once
#include "mla_launch.h"
#include "mla_split_tnd_metadata.h"
namespace flash_attn::mla {
struct MlaSplitTndLaunchArgs {
    MlaUsedLaunchArgs used;
    uint32_t splits;
    uint8_t *accum, *accumLse;
};
aclError LaunchSplitTndFp16(const MlaSplitTndLaunchArgs&);
aclError LaunchSplitTndBf16(const MlaSplitTndLaunchArgs&);
} // namespace flash_attn::mla
