/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */
// Generated MLA instantiation. CANN source copyrights and licenses
// are retained in the shared implementation and mla/LICENSE.cann.
#define MLA_SPLIT_TYPE bfloat16_t
#define MLA_SPLIT_PARTIAL mla_split_paged_bsnd_partial_bf16
#define MLA_SPLIT_MERGE mla_split_paged_bsnd_merge_bf16
#define MLA_SPLIT_LAUNCH LaunchSplitPagedBsndBf16
#include "../mla/device/mla_split_paged_bsnd_launch_impl.hpp"
