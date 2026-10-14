// BSND explicit-split production instantiation; implementation retains source licenses.
#define MLA_SPLIT_TYPE bfloat16_t
#define MLA_SPLIT_PARTIAL mla_split_bsnd_partial_bf16
#define MLA_SPLIT_MERGE mla_split_bsnd_merge_bf16
#define MLA_SPLIT_LAUNCH LaunchSplitBsndBf16
#include "mla_split_bsnd_launch_impl.hpp"
