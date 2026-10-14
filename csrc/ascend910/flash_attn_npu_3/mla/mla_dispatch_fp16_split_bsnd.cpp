// BSND explicit-split production instantiation; implementation retains source licenses.
#define MLA_SPLIT_TYPE half
#define MLA_SPLIT_PARTIAL mla_split_bsnd_partial_fp16
#define MLA_SPLIT_MERGE mla_split_bsnd_merge_fp16
#define MLA_SPLIT_LAUNCH LaunchSplitBsndFp16
#include "mla_split_bsnd_launch_impl.hpp"
