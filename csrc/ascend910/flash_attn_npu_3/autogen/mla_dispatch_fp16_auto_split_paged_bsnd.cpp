/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */
// Generated MLA instantiation. CANN source copyrights and licenses
// are retained in the shared implementation and mla/LICENSE.cann.
#define MLA_AUTO_TYPE half
#define MLA_AUTO_INIT mla_auto_split_paged_bsnd_init_fp16
#define MLA_AUTO_PARTIAL mla_auto_split_paged_bsnd_partial_fp16
#define MLA_AUTO_MERGE mla_auto_split_paged_bsnd_merge_fp16
#define MLA_AUTO_LAUNCH LaunchAutoSplitPagedBsndFp16
#include "../mla/device/mla_auto_split_paged_bsnd_launch_impl.hpp"
