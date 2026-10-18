#define MLA_GQA_TYPE bfloat16_t
#define MLA_GQA_INIT MlaGqaBf16Init
#define MLA_GQA_ATTENTION MlaGqaBf16Attention
#define MLA_GQA_MERGE MlaGqaBf16Merge
#define MLA_GQA_LAUNCH LaunchGqaBf16
#include "mla_gqa_launch_impl.hpp"
