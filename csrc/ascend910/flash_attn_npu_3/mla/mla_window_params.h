// Device-only window parameters after the checked CPU NormalizeWindow step.
// No metadata ABI or public entry is defined by this standalone component.
#pragma once
#include <cstdint>

namespace flash_attn::mla {
struct MlaWindowKernelParams {
    // Same values as MlaWindowKind: None=0, Causal=1, Local=2.
    uint32_t kind = 0;
    int64_t left = 0, right = 0;
};
} // namespace flash_attn::mla
