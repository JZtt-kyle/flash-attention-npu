/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */
// Generated MLA instantiation. CANN source copyrights and licenses
// are retained in the shared implementation and mla/LICENSE.cann.
#define MLA_AUTO_TYPE half
#define MLA_AUTO_INIT mla_auto_split_mixed_init_fp16
#define MLA_AUTO_PARTIAL mla_auto_split_mixed_partial_fp16
#define MLA_AUTO_MERGE mla_auto_split_mixed_merge_fp16
#define MLA_AUTO_LAUNCH LaunchAutoSplitMixedFp16
#define MLA_AUTO_PAGED 0
#define MLA_AUTO_TND 1
#define MLA_AUTO_MIXED 1
#include "../mla/mla_auto_split_dispatch_impl.hpp"
