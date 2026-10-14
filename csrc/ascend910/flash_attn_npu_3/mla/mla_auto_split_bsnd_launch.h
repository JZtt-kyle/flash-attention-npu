// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
#pragma once
#include "mla_split_bsnd_launch.h"
#include "mla_auto_split_bsnd_metadata.h"
namespace flash_attn::mla {
aclError LaunchAutoSplitBsndFp16(const MlaSplitBsndLaunchArgs&);
aclError LaunchAutoSplitBsndBf16(const MlaSplitBsndLaunchArgs&);
} // namespace flash_attn::mla
