// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
#pragma once
#include "mla_split_paged_bsnd_launch.h"
#include "mla_auto_split_paged_bsnd_metadata.h"
namespace flash_attn::mla {
aclError LaunchAutoSplitPagedBsndFp16(const MlaSplitPagedBsndLaunchArgs&);
aclError LaunchAutoSplitPagedBsndBf16(const MlaSplitPagedBsndLaunchArgs&);
} // namespace flash_attn::mla
