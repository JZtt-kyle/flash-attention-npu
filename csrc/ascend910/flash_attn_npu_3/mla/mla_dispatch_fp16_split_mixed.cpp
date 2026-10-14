// Mixed explicit-split production instantiation; implementation retains source licenses.
#define MLA_SPLIT_MIXED 1
#define MLA_SPLIT_TYPE half
#define MLA_SPLIT_PARTIAL mla_split_mixed_partial_fp16
#define MLA_SPLIT_MERGE mla_split_mixed_merge_fp16
#define MLA_SPLIT_LAUNCH LaunchSplitMixedFp16
#include "mla_split_tnd_launch_impl.hpp"
