// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
// GQA scheduling adapter; fixed source/provenance in mla_gqa_schedule.h.
#pragma once
#include "mla_prefill_bsnd_metadata.h"

#if defined(__NPU_ARCH__) && defined(__aicore__)
#define MLA_GQA_INLINE __aicore__ inline
#else
#define MLA_GQA_INLINE inline
#endif
namespace flash_attn::mla {
constexpr uint32_t kMlaGqaVersion = 47;
// Physical tensor heads and logical scheduling groups are deliberately distinct.
// Layout: BSND=0, TND Q/KV=1, mixed=2, BSND PA=3, TND PA=4.
struct MlaGqaContext {
    uint32_t batch, sq, sk, heads, kvHeads, splits, dtype;
    uint32_t physicalCores, margin, packed, layout, totalQ, totalK, pageSize, usedFlags, valueDim, hasSoftcap;
    MlaWindowKernelParams window;
    uint64_t workspaceBytes = 0;
};
struct MlaGqaLayout {
    uint32_t words, m, n, cores, headGroups, g, capacity;
    uint64_t groups, scratch, outOffset, sumOffset, maxOffset, bytes;
};
MLA_GQA_INLINE bool GqaTnd(const MlaGqaContext& c)
{
    return c.layout == 1 || c.layout == 2 || c.layout == 4;
}
MLA_GQA_INLINE bool GqaEmptyKv(const MlaGqaContext& c)
{
    return !c.sk || (c.layout == 1 && !c.totalK);
}
MLA_GQA_INLINE uint32_t GqaTail(const MlaGqaContext& c)
{
    return 32 + 8 * c.batch;
}
MLA_GQA_INLINE bool GqaLayout(const MlaGqaContext& c, MlaGqaLayout& out)
{
    if (!c.kvHeads || !c.heads || c.heads % c.kvHeads || c.packed > 1 || (c.packed && c.heads == c.kvHeads) ||
        !c.physicalCores || c.physicalCores > 26 || c.margin >= c.physicalCores || c.layout > 4 ||
        (c.valueDim != 128 && c.valueDim != 512) || (c.valueDim == 512 && c.kvHeads != 1) ||
        ((c.layout >= 3) != bool(c.pageSize)))
        return false;
    MlaGqaLayout l{};
    l.cores = c.physicalCores - c.margin;
    l.g = c.packed ? c.heads / c.kvHeads : 1;
    l.headGroups = c.heads / l.g;
    // Reuse the existing static geometry, overflow checks and fixed generic
    // tiling rules. Compressed attention retains the source's 256x512 ceiling.
    MlaPrefillBsndContext p{c.batch,      c.sq,     c.sk,       c.heads,
                            l.headGroups, c.splits, c.dtype,    l.cores,
                            c.usedFlags,  c.window, 0,          GqaTnd(c) ? (c.layout == 1 ? 1U : 2U) : 0U,
                            c.totalQ,     c.totalK, c.pageSize, c.hasSoftcap};
    MlaPrefillBsndLayout base{};
    if (!PrefillBsndLayout(p, base) || uint64_t(c.batch) * (8 + uint64_t(2) * l.headGroups) > UINT32_MAX - 380)
        return false;
    l.m = c.valueDim == 512 ? 256 : base.m;
    l.n = c.valueDim == 512 ? (c.sk < 512 ? c.sk : 512) : base.n;
    if (c.valueDim == 512 && c.pageSize && l.n > c.pageSize && l.n % c.pageSize)
        l.n = l.n / c.pageSize * c.pageSize;
    if (!l.n)
        l.n = 32;
    l.words = GqaTail(c) + 348 + 2 * c.batch * l.headGroups;
    l.capacity = c.splits ? c.splits : l.cores;
    l.groups = uint64_t(c.batch) * l.headGroups * ((uint64_t(c.sq) * l.g + l.m - 1) / l.m);
    if (GqaTnd(c)) {
        const uint64_t compact = (uint64_t(c.totalQ) * l.g / l.m + c.batch) * l.headGroups;
        if (compact < l.groups)
            l.groups = compact;
        if (!c.totalQ)
            l.groups = 0;
    }
    if (l.groups && !GqaEmptyKv(c)) {
        l.scratch = uint64_t(l.cores) *
                    (12 * uint64_t(l.m) * ((l.n + 31) / 32 * 32) + 16 * uint64_t(l.m) * c.valueDim + 16 * l.m);
        l.outOffset = l.scratch;
        const uint64_t rows = l.groups * l.capacity * l.m;
        if (rows > (uint64_t(INT64_MAX) - l.scratch) / ((c.valueDim + 16) * 4))
            return false;
        l.sumOffset = l.outOffset + rows * c.valueDim * 4;
        l.maxOffset = l.sumOffset + rows * 8 * 4;
        l.bytes = l.maxOffset + rows * 8 * 4;
    }
    out = l;
    return true;
}
MLA_GQA_INLINE uint32_t GqaHeader(const MlaGqaContext& c, const MlaGqaLayout& l, uint32_t i)
{
    const uint32_t fields[32] = {kMlaMetadataMagic,
                                 kMlaGqaVersion,
                                 kMlaGqaVersion,
                                 l.words,
                                 c.dtype,
                                 0,
                                 c.batch,
                                 c.sq,
                                 c.sk,
                                 c.heads,
                                 c.kvHeads,
                                 c.splits,
                                 c.physicalCores,
                                 c.margin,
                                 c.packed,
                                 c.layout,
                                 c.totalQ,
                                 c.totalK,
                                 c.pageSize,
                                 c.usedFlags,
                                 c.valueDim,
                                 c.hasSoftcap,
                                 c.window.kind,
                                 uint32_t(c.window.left),
                                 uint32_t(c.window.right),
                                 l.m,
                                 l.n,
                                 l.cores,
                                 l.headGroups,
                                 0,
                                 0,
                                 0};
    return fields[i];
}
struct MlaGqaMetadataArgs {
    MlaGqaContext context;
    uint64_t cuQuery, cuKv, usedQuery, usedKv, output;
    uint32_t words, inferTotalQ = 0;
};
bool BuildGqaMetadata(const MlaGqaMetadataArgs&);
} // namespace flash_attn::mla
#undef MLA_GQA_INLINE
