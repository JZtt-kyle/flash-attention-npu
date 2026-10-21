/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */
// Generated MLA template instantiations; implementation licenses are
// retained in the shared headers and mla/LICENSE.cann.
#include "../mla/mla_split_dispatch_impl.hpp"

namespace flash_attn::mla {
template aclError LaunchSplitBsnd<bfloat16_t>(const MlaSplitBsndLaunchArgs&);
template aclError LaunchSplitTnd<bfloat16_t, false>(const MlaSplitTndLaunchArgs&);
template aclError LaunchSplitTnd<bfloat16_t, true>(const MlaSplitMixedLaunchArgs&);
template aclError LaunchSplitPagedBsnd<bfloat16_t>(const MlaSplitPagedBsndLaunchArgs&);
template aclError LaunchSplitPagedTnd<bfloat16_t>(const MlaSplitPagedTndLaunchArgs&);
} // namespace flash_attn::mla
