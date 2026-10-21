/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */
// Generated MLA template instantiations; implementation licenses are
// retained in the shared headers and mla/LICENSE.cann.
#include "../mla/mla_dispatch_impl.hpp"

namespace flash_attn::mla {
template aclError LaunchDense<bfloat16_t>(const MlaLaunchArgs&);
template aclError LaunchTnd<bfloat16_t>(const MlaTndLaunchArgs&);
template aclError LaunchTndBsnd<bfloat16_t>(const MlaTndBsndLaunchArgs&);
template aclError LaunchBsndUsed<bfloat16_t>(const MlaUsedLaunchArgs&);
template aclError LaunchTndUsed<bfloat16_t>(const MlaUsedLaunchArgs&);
template aclError LaunchPaged<bfloat16_t, false>(const MlaPagedLaunchArgs&);
template aclError LaunchPaged<bfloat16_t, true>(const MlaPagedLaunchArgs&);
} // namespace flash_attn::mla
