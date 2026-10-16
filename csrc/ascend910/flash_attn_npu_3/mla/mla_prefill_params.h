// Independent 192/128 numerical launch data. Not a serialized metadata ABI.
#pragma once
#include "mla_bsnd_layout.h"
#include "mla_window_params.h"
namespace flash_attn::mla {
struct MlaPrefillParams {
    uint32_t batch, sq, sk, heads, kvHeads, mBase, nBase, cores;
    float scale;
    MlaWindowKernelParams window;
    MlaBsndTensorLayout q, k, v;
    float softcap = 0.0f;
};
// A validated fragment of one S1G block for one KV head. The caller owns
// partitioning and bounds; begin/end are S2 tile indices, rows exclude padding.
struct MlaPrefillTask {
    uint32_t batch, kvHead, rowStart, rows, queryLength, kvLength, begin, end;
};
} // namespace flash_attn::mla
