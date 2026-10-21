/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */
// Generated MLA template instantiations; implementation licenses are
// retained in the shared headers and mla/LICENSE.cann.
#include "../mla/mla_dispatch_impl.hpp"

namespace flash_attn::mla {
template aclError LaunchDense<half>(const MlaLaunchArgs&);
template aclError LaunchTnd<half>(const MlaTndLaunchArgs&);
template aclError LaunchTndBsnd<half>(const MlaTndBsndLaunchArgs&);
template aclError LaunchBsndUsed<half>(const MlaUsedLaunchArgs&);
template aclError LaunchTndUsed<half>(const MlaUsedLaunchArgs&);
template aclError LaunchPaged<half, false>(const MlaPagedLaunchArgs&);
template aclError LaunchPaged<half, true>(const MlaPagedLaunchArgs&);
} // namespace flash_attn::mla
