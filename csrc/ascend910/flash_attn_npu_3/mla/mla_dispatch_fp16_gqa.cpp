#define MLA_GQA_TYPE half
#define MLA_GQA_INIT MlaGqaFp16Init
#define MLA_GQA_ATTENTION MlaGqaFp16Attention
#define MLA_GQA_MERGE MlaGqaFp16Merge
#define MLA_GQA_LAUNCH LaunchGqaFp16
#include "mla_gqa_launch_impl.hpp"
