/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */
// Generated MLA instantiation. CANN source copyrights and licenses
// are retained in the shared implementation and mla/LICENSE.cann.
#define MLA_SPLIT_TYPE half
#define MLA_SPLIT_PARTIAL mla_split_tnd_partial_fp16
#define MLA_SPLIT_MERGE mla_split_tnd_merge_fp16
#define MLA_SPLIT_LAUNCH LaunchSplitTndFp16
#include "../mla/device/mla_split_tnd_launch_impl.hpp"
