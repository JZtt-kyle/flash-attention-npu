// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
#define MLA_SPLIT_TYPE bfloat16_t
#define MLA_SPLIT_PARTIAL mla_split_paged_bsnd_partial_bf16
#define MLA_SPLIT_MERGE mla_split_paged_bsnd_merge_bf16
#define MLA_SPLIT_LAUNCH LaunchSplitPagedBsndBf16
#include "mla_split_paged_bsnd_launch_impl.hpp"
