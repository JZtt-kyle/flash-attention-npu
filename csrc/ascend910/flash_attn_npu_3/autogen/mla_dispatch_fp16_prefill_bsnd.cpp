/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */
// Generated MLA instantiation. CANN source copyrights and licenses
// are retained in the shared implementation and mla/LICENSE.cann.
#define MLA_PREFILL_TYPE half
#define MLA_PREFILL_INIT MlaPrefillTndFp16Init
#define MLA_PREFILL_ATTENTION MlaPrefillBsndFp16Attention
#define MLA_PREFILL_MERGE MlaPrefillBsndFp16Merge
#define MLA_PREFILL_LAUNCH LaunchPrefillBsndFp16
#include "../mla/mla_prefill_dispatch_impl.hpp"
