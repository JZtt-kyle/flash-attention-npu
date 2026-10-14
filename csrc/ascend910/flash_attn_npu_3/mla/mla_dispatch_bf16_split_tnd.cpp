// BSND explicit-split production instantiation; implementation retains source licenses.
#define MLA_SPLIT_TYPE bfloat16_t
#define MLA_SPLIT_PARTIAL mla_split_tnd_partial_bf16
#define MLA_SPLIT_MERGE mla_split_tnd_merge_bf16
#define MLA_SPLIT_LAUNCH LaunchSplitTndBf16
#include "mla_split_tnd_launch_impl.hpp"
