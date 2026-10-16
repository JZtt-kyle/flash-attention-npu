// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
// BSND/TND 192/128 adapter for ca18aee's fia_tiling_nonquant.cpp and SplitCore.
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
constexpr uint32_t kMlaPrefillTndVersion = 28;
constexpr uint32_t kMlaPrefillMixedVersion = 29;
constexpr uint32_t kMlaPrefillPagedBsndVersion = 30;
constexpr uint32_t kMlaPrefillPagedTndVersion = 31;
// v27 has 20 header words and B pairs of effective lengths; v28 has 24 header
// words and B allocation/used spans. v29 shares those spans; v30 has a 24-word
// header and B effective-length pairs. All retain the 346-word native partition
// tail and B*Hkv+1 uint64 M prefixes, without device addresses. v31 combines
// v29 allocation spans with page size in header word 21 (no TND KV total).
struct MlaPrefillBsndContext {
    uint32_t batch, sq, sk, heads, kvHeads, splits, dtype, cores, usedFlags;
    MlaWindowKernelParams window;
    uint64_t workspaceBytes;
    // Layout: 0 = v27 BSND, 1 = v28 TND, 2 = v29 TND Q / BSND KV.
    // v29 retains v28 spans, but KV spans describe rectangular cache slots.
    uint32_t tnd = 0, totalQ = 0, totalK = 0;
    // v30/v31: BSND/TND Q and independent 192/128 ND paged KV.
    uint32_t pageSize = 0;
};
struct MlaPrefillBsndLayout {
    uint32_t words, m, n, capacity;
    uint64_t groups, scratch, outOffset, sumOffset, maxOffset, bytes;
};
MLA_PREFILL_INLINE uint32_t PrefillSequenceWord(const MlaPrefillBsndContext& c, uint32_t b)
{
    return c.tnd ? 24 + 8 * b : (c.pageSize ? 24 : 20) + 2 * b;
}
MLA_PREFILL_INLINE uint32_t PrefillLengthWord(const MlaPrefillBsndContext& c, uint32_t b)
{
    return PrefillSequenceWord(c, b) + (c.tnd ? 6 : 0);
}
MLA_PREFILL_INLINE uint32_t PrefillTailWord(const MlaPrefillBsndContext& c)
{
    return PrefillSequenceWord(c, c.batch);
}
MLA_PREFILL_INLINE bool PrefillBsndLayout(const MlaPrefillBsndContext& c, MlaPrefillBsndLayout& out)
{
    if (!c.heads || !c.kvHeads || c.heads % c.kvHeads || c.splits > 256 || !c.cores || c.cores > 26 || c.tnd > 2 ||
        (c.pageSize && (c.tnd == 1 || c.pageSize < 16 || c.pageSize > 1024 || c.pageSize % 16 || c.sk % c.pageSize)) ||
        c.totalQ > INT32_MAX || c.totalK > INT32_MAX || (c.tnd && uint64_t(c.totalQ) * c.heads > kMlaMaxBatchRows) ||
        ((!c.tnd && c.totalQ) || (c.tnd != 1 && c.totalK)) || c.usedFlags > 3 || c.window.kind > 2 ||
        c.window.left < -1 || c.window.right < -1 || c.window.left > INT32_MAX || c.window.right > INT32_MAX ||
        c.sq > kMlaMaxQueryLength || c.sk > kMlaMaxKvLength || c.heads > kMlaMaxQueryHeads ||
        uint64_t(c.sq) * c.heads > kMlaMaxDenseRows ||
        (!c.tnd && uint64_t(c.batch) * c.sq > kMlaMaxBatchRows / c.heads) ||
        uint64_t(c.batch) * c.kvHeads > (UINT32_MAX - 758) / 10 ||
        uint64_t(c.batch) * ((c.tnd ? 8 : 2) + uint64_t(2) * c.kvHeads) > UINT32_MAX - 372)
        return false;
    const uint32_t g = c.heads / c.kvHeads;
    MlaPrefillBsndLayout r{};
    r.words = PrefillTailWord(c) + 348 + 2 * c.batch * c.kvHeads;
    // Exact CalcInnerSize/CalcMBaseSize rules from fixed CANN, including PA alignment.
    r.n = !c.tnd && c.sq <= 16 ? (g < 5 ? 8192 : g < 10 ? 4096 : 2048) : 512;
    if (c.window.kind && r.n > 1024)
        r.n = 1024;
    if (c.pageSize && r.n > c.pageSize && r.n % c.pageSize)
        r.n = r.n / c.pageSize * c.pageSize;
    if (r.n > c.sk)
        r.n = c.sk;
    if (!r.n)
        r.n = 32; // Empty-capacity validation launch has no numeric task.
    const uint32_t aligned = (r.n + 31) / 32 * 32;
    r.m = aligned <= 512 ? 512 : aligned <= 1024 ? 256 : aligned <= 2048 ? 128 : aligned <= 4096 ? 64 : 32;
    r.capacity = c.splits ? c.splits : c.cores;
    const uint64_t publicCapacity = c.splits == 0 ? 26 : c.splits;
    const uint64_t strideRows =
        (c.tnd ? uint64_t(c.totalQ ? c.totalQ : 1) : uint64_t(c.batch ? c.batch : 1) * (c.sq ? c.sq : 1)) * c.heads;
    if (strideRows > uint64_t(INT64_MAX) / 512 / publicCapacity)
        return false;
    if (c.batch && c.sq && c.sk && (!c.tnd || (c.totalQ && (c.tnd == 2 || c.totalK)))) {
        r.groups = uint64_t(c.batch) * c.kvHeads * ((uint64_t(c.sq) * g + r.m - 1) / r.m);
        if (c.tnd) {
            const uint64_t compact = (uint64_t(c.totalQ) * g / r.m + c.batch) * c.kvHeads;
            if (compact < r.groups)
                r.groups = compact;
        }
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
    uint64_t cuQuery = 0, cuKv = 0;
    // Public mixed metadata receives only the device Q prefix, not a Q tensor.
    uint32_t inferTotalQ = 0;
};
// Unsafe output/overlap returns false without writes. Other failures publish
// a nonzero status and zero payload; successful snapshots own their lengths.
bool BuildPrefillBsndMetadata(const MlaPrefillBsndMetadataArgs&);
} // namespace flash_attn::mla
#undef MLA_PREFILL_INLINE
