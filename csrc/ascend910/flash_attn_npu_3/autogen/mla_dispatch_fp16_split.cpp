/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */
// Generated MLA template instantiations; implementation licenses are
// retained in the shared headers and mla/LICENSE.cann.
#include "../mla/mla_split_dispatch_impl.hpp"

namespace flash_attn::mla {
template aclError LaunchSplitBsnd<half>(const MlaSplitBsndLaunchArgs&);
template aclError LaunchSplitTnd<half, false>(const MlaSplitTndLaunchArgs&);
template aclError LaunchSplitTnd<half, true>(const MlaSplitMixedLaunchArgs&);
template aclError LaunchSplitPagedBsnd<half>(const MlaSplitPagedBsndLaunchArgs&);
template aclError LaunchSplitPagedTnd<half>(const MlaSplitPagedTndLaunchArgs&);
} // namespace flash_attn::mla
