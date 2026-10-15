// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
#pragma once
#include "mla_split_paged_bsnd_launch.h"
#include "mla_split_paged_tnd_metadata.h"
namespace flash_attn::mla {
using MlaSplitPagedTndLaunchArgs = MlaSplitPagedBsndLaunchArgs;
aclError LaunchSplitPagedTndFp16(const MlaSplitPagedTndLaunchArgs&);
aclError LaunchSplitPagedTndBf16(const MlaSplitPagedTndLaunchArgs&);
} // namespace flash_attn::mla
