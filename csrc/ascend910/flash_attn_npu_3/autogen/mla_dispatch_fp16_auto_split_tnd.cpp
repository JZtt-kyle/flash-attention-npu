/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */
// Generated MLA instantiation. CANN source copyrights and licenses
// are retained in the shared implementation and mla/LICENSE.cann.
#define MLA_AUTO_TYPE half
#define MLA_AUTO_INIT mla_auto_split_tnd_init_fp16
#define MLA_AUTO_PARTIAL mla_auto_split_tnd_partial_fp16
#define MLA_AUTO_MERGE mla_auto_split_tnd_merge_fp16
#define MLA_AUTO_LAUNCH LaunchAutoSplitTndFp16
#define MLA_AUTO_TND 1
#define MLA_AUTO_MIXED 0
#include "../mla/device/mla_auto_split_launch_impl.hpp"
