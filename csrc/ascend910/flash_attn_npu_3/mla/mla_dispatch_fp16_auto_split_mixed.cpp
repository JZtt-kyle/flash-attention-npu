// Native automatic split instantiation; implementation retains source licenses.
#define MLA_AUTO_TYPE half
#define MLA_AUTO_INIT mla_auto_split_mixed_init_fp16
#define MLA_AUTO_PARTIAL mla_auto_split_mixed_partial_fp16
#define MLA_AUTO_MERGE mla_auto_split_mixed_merge_fp16
#define MLA_AUTO_LAUNCH LaunchAutoSplitMixedFp16
#include "mla_auto_split_mixed_launch_impl.hpp"
