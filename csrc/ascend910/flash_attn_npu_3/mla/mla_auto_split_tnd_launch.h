// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
#pragma once
#include "mla_split_tnd_launch.h"
#include "mla_auto_split_tnd_metadata.h"
namespace flash_attn::mla {
aclError LaunchAutoSplitTndFp16(const MlaSplitTndLaunchArgs&);
aclError LaunchAutoSplitTndBf16(const MlaSplitTndLaunchArgs&);
} // namespace flash_attn::mla
