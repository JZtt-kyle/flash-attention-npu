// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
#pragma once
#include "mla_split_mixed_launch.h"
#include "mla_auto_split_mixed_metadata.h"
namespace flash_attn::mla {
aclError LaunchAutoSplitMixedFp16(const MlaSplitMixedLaunchArgs&);
aclError LaunchAutoSplitMixedBf16(const MlaSplitMixedLaunchArgs&);
} // namespace flash_attn::mla
