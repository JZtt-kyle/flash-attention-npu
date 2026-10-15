// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
#pragma once
#include "mla_prefill_bsnd_metadata.h"
#include "mla_auto_split_bsnd_consumer.h"
#include "mla_split_bsnd_consumer.h"
#if defined(__NPU_ARCH__)
#define MLA_PREFILL_INLINE __aicore__ inline
#else
#define MLA_PREFILL_INLINE inline
#endif
namespace flash_attn::mla {
template <typename Reader>
struct PrefillHeaderView {
    const Reader& read;
    // v27 combines generator/schedule status at word 5; reserved is word 19.
    MLA_PREFILL_INLINE uint32_t operator()(uint32_t i) const
    {
        return i == 6 ? 0 : i == 7 ? read(19) : read(i);
    }
};
// Translate only scheduling coordinates (b,kvHead) into bN2. Tensor addresses
// remain true BSND throughout the numerical and merge kernels.
template <typename Reader>
struct PrefillPartitionView {
    const Reader& read;
    const MlaPrefillBsndContext& c;
    MLA_PREFILL_INLINE uint32_t operator()(uint32_t i) const
    {
        const uint32_t virtualBatch = c.batch * c.kvHeads;
        const uint32_t base = kMlaUsedMetadataPrefixWords + 8 * virtualBatch;
        if (i >= base)
            return read(20 + 2 * c.batch + i - base);
        const uint32_t slot = i - kMlaUsedMetadataPrefixWords;
        return read(20 + 2 * ((slot / 8) / c.kvHeads) + (slot % 8 == 7));
    }
};
template <typename Reader, typename Arrays>
MLA_PREFILL_INLINE MlaMetadataReadStatus ReadPrefillBsndMetadata(const Reader& read, uint32_t words,
                                                                 const MlaPrefillBsndContext& c, const Arrays& arrays,
                                                                 MlaAutoSplitBsndSchedule& output)
{
    using S = MlaMetadataReadStatus;
    MlaPrefillBsndLayout l{};
    if (!PrefillBsndLayout(c, l) || words != l.words)
        return S::InvalidBuffer;
    const auto status = ReadMetadataHeader(PrefillHeaderView<Reader>{read}, words, l.words, 27, 27, c.dtype);
    if (status != S::Success)
        return status;
    const uint32_t fields[] = {c.batch,
                               c.sq,
                               c.sk,
                               c.heads,
                               c.kvHeads,
                               c.splits,
                               c.cores,
                               l.m,
                               l.n,
                               c.usedFlags,
                               c.window.kind,
                               uint32_t(c.window.left),
                               uint32_t(c.window.right),
                               0};
    for (uint32_t i = 0; i < 14; ++i)
        if (read(6 + i) != fields[i])
            return S::ShapeMismatch;
    if (c.workspaceBytes < l.bytes)
        return S::InvalidWorkspace;
    for (uint32_t b = 0; b < c.batch; ++b) {
        const uint32_t q = c.usedFlags & 1 ? arrays(0, b) : c.sq;
        const uint32_t k = c.usedFlags & 2 ? arrays(1, b) : c.sk;
        if (q > c.sq || k > c.sk || read(20 + 2 * b) != q || read(21 + 2 * b) != k)
            return S::InvalidUsedLengths;
    }
    const uint32_t tail = 20 + 2 * c.batch;
    if (read(tail) != c.window.kind || int32_t(read(tail + 1)) != c.window.left ||
        int32_t(read(tail + 2)) != c.window.right)
        return S::ShapeMismatch;
    MlaUsedKernelSchedule used{};
    auto& t = used.tiling;
    t.batch = c.batch * c.kvHeads;
    t.sq = c.sq;
    t.sk = c.sk;
    t.heads = c.heads / c.kvHeads;
    t.mBaseSize = l.m;
    t.s2BaseSize = l.n;
    t.blockDim = c.cores;
    MlaUsedMetadataContext context{};
    context.batch = t.batch;
    context.heads = t.heads;
    context.aicNum = c.cores;
    context.query.maxLength = c.sq;
    context.kv.maxLength = c.sk;
    return ReadAutoSplitPartitions<false, false, true>(PrefillPartitionView<Reader>{read, c}, context, c.window, used,
                                                       output);
}
template <typename Reader>
MLA_PREFILL_INLINE bool PrefillBsndTask(const Reader& read, const MlaPrefillBsndContext& c,
                                        const MlaAutoSplitBsndSchedule& s, uint64_t group, uint32_t split,
                                        MlaAutoSplitBsndTask& t)
{
    PrefillPartitionView<Reader> view{read, c};
    if (!AutoSplitBsndTask(view, s, c.window, group, c.splits ? 0 : split, t))
        return false;
    if (c.splits) {
        const auto& a = s.used.tiling;
        const uint32_t b = t.batch / c.kvHeads;
        t.split = split;
        t.splitCount = c.splits;
        SplitBsndFragment(read(20 + 2 * b), read(21 + 2 * b), a.heads, t.m, a.mBaseSize, a.s2BaseSize, c.window,
                          c.splits, split, t.s2Begin, t.s2End);
    }
    return true;
}
} // namespace flash_attn::mla
#undef MLA_PREFILL_INLINE
