// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
#pragma once
#include "mla_paged_launch.h"
#include "mla_split_paged_bsnd_metadata.h"
namespace flash_attn::mla {
struct MlaSplitPagedBsndLaunchArgs {
    MlaPagedLaunchArgs paged;
    uint32_t splits;
    uint8_t *accum, *accumLse;
};
aclError LaunchSplitPagedBsndFp16(const MlaSplitPagedBsndLaunchArgs&);
aclError LaunchSplitPagedBsndBf16(const MlaSplitPagedBsndLaunchArgs&);
} // namespace flash_attn::mla
