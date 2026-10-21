/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */
// Generated MLA template instantiations; implementation licenses are
// retained in the shared headers and mla/LICENSE.cann.
#include "../mla/mla_split_dispatch_impl.hpp"

namespace flash_attn::mla {
template aclError LaunchSplit<half, false, false>(const MlaSplitLaunchArgs&);
template aclError LaunchSplit<half, true, false>(const MlaSplitLaunchArgs&);
template aclError LaunchSplit<half, true, true>(const MlaSplitLaunchArgs&);
template aclError LaunchSplitPaged<half, false>(const MlaSplitPagedLaunchArgs&);
template aclError LaunchSplitPaged<half, true>(const MlaSplitPagedLaunchArgs&);
} // namespace flash_attn::mla
