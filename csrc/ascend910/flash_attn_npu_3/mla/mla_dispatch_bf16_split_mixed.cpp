// Mixed explicit-split production instantiation; implementation retains source licenses.
#define MLA_SPLIT_MIXED 1
#define MLA_SPLIT_TYPE bfloat16_t
#define MLA_SPLIT_PARTIAL mla_split_mixed_partial_bf16
#define MLA_SPLIT_MERGE mla_split_mixed_merge_bf16
#define MLA_SPLIT_LAUNCH LaunchSplitMixedBf16
#include "mla_split_tnd_launch_impl.hpp"
