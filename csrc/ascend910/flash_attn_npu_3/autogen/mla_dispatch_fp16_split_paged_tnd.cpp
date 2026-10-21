/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */
// Generated MLA instantiation. CANN source copyrights and licenses
// are retained in the shared implementation and mla/LICENSE.cann.
#define MLA_SPLIT_TYPE half
#define MLA_SPLIT_PARTIAL mla_split_paged_tnd_partial_fp16
#define MLA_SPLIT_MERGE mla_split_paged_tnd_merge_fp16
#define MLA_SPLIT_LAUNCH LaunchSplitPagedTndFp16
#define MLA_SPLIT_PAGED_TND 1
#define MLA_SPLIT_INIT mla_split_paged_tnd_init_fp16
#include "../mla/mla_split_dispatch_impl.hpp"
