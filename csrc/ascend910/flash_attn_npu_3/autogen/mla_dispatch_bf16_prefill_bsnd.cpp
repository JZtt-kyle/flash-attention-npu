/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */
// Generated MLA instantiation. CANN source copyrights and licenses
// are retained in the shared implementation and mla/LICENSE.cann.
#define MLA_PREFILL_TYPE bfloat16_t
#define MLA_PREFILL_INIT MlaPrefillTndBf16Init
#define MLA_PREFILL_ATTENTION MlaPrefillBsndBf16Attention
#define MLA_PREFILL_MERGE MlaPrefillBsndBf16Merge
#define MLA_PREFILL_LAUNCH LaunchPrefillBsndBf16
#include "../mla/device/mla_prefill_bsnd_launch_impl.hpp"
