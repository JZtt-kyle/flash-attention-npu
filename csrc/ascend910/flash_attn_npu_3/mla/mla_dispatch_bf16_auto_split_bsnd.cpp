// Native automatic split instantiation; implementation retains source licenses.
#define MLA_AUTO_TYPE bfloat16_t
#define MLA_AUTO_INIT mla_auto_split_bsnd_init_bf16
#define MLA_AUTO_PARTIAL mla_auto_split_bsnd_partial_bf16
#define MLA_AUTO_MERGE mla_auto_split_bsnd_merge_bf16
#define MLA_AUTO_LAUNCH LaunchAutoSplitBsndBf16
#include "mla_auto_split_bsnd_launch_impl.hpp"
