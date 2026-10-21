/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */
// Generated MLA template instantiations; implementation licenses are
// retained in the shared headers and mla/LICENSE.cann.
#include "../mla/mla_auto_split_dispatch_impl.hpp"

namespace flash_attn::mla {
template aclError LaunchAutoSplit<bfloat16_t, false, false>(const MlaSplitBsndLaunchArgs&);
template aclError LaunchAutoSplit<bfloat16_t, true, false>(const MlaSplitTndLaunchArgs&);
template aclError LaunchAutoSplit<bfloat16_t, true, true>(const MlaSplitMixedLaunchArgs&);
template aclError LaunchAutoSplitPaged<bfloat16_t, false>(const MlaSplitPagedBsndLaunchArgs&);
template aclError LaunchAutoSplitPaged<bfloat16_t, true>(const MlaSplitPagedBsndLaunchArgs&);
} // namespace flash_attn::mla
