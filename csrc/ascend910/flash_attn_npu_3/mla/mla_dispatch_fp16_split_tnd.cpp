// BSND explicit-split production instantiation; implementation retains source licenses.
#define MLA_SPLIT_TYPE half
#define MLA_SPLIT_PARTIAL mla_split_tnd_partial_fp16
#define MLA_SPLIT_MERGE mla_split_tnd_merge_fp16
#define MLA_SPLIT_LAUNCH LaunchSplitTndFp16
#include "mla_split_tnd_launch_impl.hpp"
