// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
// BSND 192/128 adapter for ca18aee's fia_tiling_nonquant.cpp and SplitCore.
#pragma once
#include "mla_prefill_params.h"
#include "mla_metadata.h"

#if defined(__NPU_ARCH__) && defined(__aicore__)
#define MLA_PREFILL_INLINE __aicore__ inline
#else
#define MLA_PREFILL_INLINE inline
#endif
namespace flash_attn::mla {
constexpr uint32_t kMlaPrefillBsndVersion = 27;
// 20 header words, B pairs of effective lengths, the existing 346-word
// native partition tail, and B*Hkv+1 uint64 M prefixes. No device addresses.
struct MlaPrefillBsndContext {
    uint32_t batch, sq, sk, heads, kvHeads, splits, dtype, cores, usedFlags;
    MlaWindowKernelParams window;
    uint64_t workspaceBytes;
};
struct MlaPrefillBsndLayout {
    uint32_t words, m, n, capacity;
    uint64_t groups, scratch, outOffset, sumOffset, maxOffset, bytes;
};
MLA_PREFILL_INLINE bool PrefillBsndLayout(const MlaPrefillBsndContext& c, MlaPrefillBsndLayout& out)
{
    if (!c.heads || !c.kvHeads || c.heads % c.kvHeads || c.splits > 256 || !c.cores || c.cores > 26 ||
        c.usedFlags > 3 || c.window.kind > 2 || c.window.left < -1 || c.window.right < -1 ||
        c.window.left > INT32_MAX || c.window.right > INT32_MAX || c.sq > kMlaMaxQueryLength ||
        c.sk > kMlaMaxKvLength || c.heads > kMlaMaxQueryHeads || uint64_t(c.sq) * c.heads > kMlaMaxDenseRows ||
        uint64_t(c.batch) * c.sq > kMlaMaxBatchRows / c.heads ||
        uint64_t(c.batch) * c.kvHeads > (UINT32_MAX - 758) / 10 ||
        uint64_t(c.batch) * (2 + uint64_t(2) * c.kvHeads) > UINT32_MAX - 368)
        return false;
    const uint32_t g = c.heads / c.kvHeads;
    MlaPrefillBsndLayout r{};
    r.words = 368 + c.batch * (2 + 2 * c.kvHeads);
    // Exact non-PA BSND CalcInnerSize/CalcMBaseSize rules from fixed CANN.
    r.n = c.sq <= 16 ? (g < 5 ? 8192 : g < 10 ? 4096 : 2048) : 512;
    if (c.window.kind && r.n > 1024)
        r.n = 1024;
    if (r.n > c.sk)
        r.n = c.sk;
    if (!r.n)
        r.n = 32; // Empty-capacity validation launch has no numeric task.
    const uint32_t aligned = (r.n + 31) / 32 * 32;
    r.m = aligned <= 512 ? 512 : aligned <= 1024 ? 256 : aligned <= 2048 ? 128 : aligned <= 4096 ? 64 : 32;
    r.capacity = c.splits ? c.splits : c.cores;
    const uint64_t publicCapacity = c.splits == 0 ? 26 : c.splits;
    const uint64_t strideRows = uint64_t(c.batch ? c.batch : 1) * (c.sq ? c.sq : 1) * c.heads;
    if (strideRows > uint64_t(INT64_MAX) / 512 / publicCapacity)
        return false;
    if (c.batch && c.sq && c.sk) {
        r.groups = uint64_t(c.batch) * c.kvHeads * ((uint64_t(c.sq) * g + r.m - 1) / r.m);
        r.scratch = uint64_t(c.cores) * (12 * uint64_t(r.m) * aligned + 16 * uint64_t(r.m) * 128);
        r.outOffset = r.scratch;
        const uint64_t rows = r.groups * r.capacity * r.m;
        if (rows > (uint64_t(INT64_MAX) - r.scratch) / (144 * 4))
            return false;
        r.sumOffset = r.outOffset + rows * 128 * 4;
        r.maxOffset = r.sumOffset + rows * 8 * 4;
        r.bytes = c.splits == 1 ? r.scratch : r.maxOffset + rows * 8 * 4;
    }
    out = r;
    return true;
}
struct MlaPrefillBsndMetadataArgs {
    MlaPrefillBsndContext context;
    uint64_t usedQuery, usedKv, output;
    uint32_t words;
};
// Unsafe output/overlap returns false without writes. Other failures publish
// a nonzero status and zero payload; successful snapshots own their lengths.
bool BuildPrefillBsndMetadata(const MlaPrefillBsndMetadataArgs&);
} // namespace flash_attn::mla
#undef MLA_PREFILL_INLINE
