// Native automatic split instantiation; implementation retains source licenses.
#define MLA_AUTO_TYPE half
#define MLA_AUTO_INIT mla_auto_split_tnd_init_fp16
#define MLA_AUTO_PARTIAL mla_auto_split_tnd_partial_fp16
#define MLA_AUTO_MERGE mla_auto_split_tnd_merge_fp16
#define MLA_AUTO_LAUNCH LaunchAutoSplitTndFp16
#include "mla_auto_split_tnd_launch_impl.hpp"
