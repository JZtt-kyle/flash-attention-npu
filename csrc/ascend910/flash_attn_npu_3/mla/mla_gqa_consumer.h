// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
#pragma once
#include "mla_gqa_metadata.h"
#include "mla_prefill_bsnd_consumer.h"
#if defined(__NPU_ARCH__)
#define MLA_GQA_INLINE __aicore__ inline
#else
#define MLA_GQA_INLINE inline
#endif
namespace flash_attn::mla {
template <typename Reader>
struct GqaAllocationView {
    const Reader& read;
    MLA_GQA_INLINE uint32_t operator()(uint32_t i) const
    {
        return read(32 + i - kMlaUsedMetadataPrefixWords);
    }
};
template <typename Reader>
struct GqaPartitionView {
    const Reader& read;
    const MlaGqaContext& c;
    const MlaGqaLayout& l;
    MLA_GQA_INLINE uint32_t operator()(uint32_t i) const
    {
        const uint32_t base = kMlaUsedMetadataPrefixWords + 8 * c.batch * l.headGroups;
        if (i >= base)
            return read(GqaTail(c) + i - base);
        const uint32_t slot = i - kMlaUsedMetadataPrefixWords;
        return read(32 + 8 * ((slot / 8) / l.headGroups) + slot % 8);
    }
};
template <typename Reader, typename Arrays>
MLA_GQA_INLINE MlaMetadataReadStatus ReadGqaMetadataWithLayout(const Reader& read, const MlaGqaContext& c,
                                                               const MlaGqaLayout& l, const Arrays& arrays,
                                                               MlaAutoSplitBsndSchedule& output)
{
    using S = MlaMetadataReadStatus;
    if (read(5))
        return S::InvalidTiling;
#if defined(__NPU_ARCH__)
#pragma unroll
#endif
    for (uint32_t i = 0; i < 32; ++i)
        if (read(i) != GqaHeader(c, l, i))
            return S::ShapeMismatch;
    if (c.workspaceBytes < l.bytes)
        return S::InvalidWorkspace;
    MlaUsedMetadataContext geometry{};
    geometry.batch = c.batch;
    geometry.usedFlags = c.usedFlags;
    const uint64_t q = GqaTnd(c) ? c.totalQ : uint64_t(c.batch) * c.sq;
    const uint64_t k = c.layout == 1 ? c.totalK : uint64_t(c.batch) * c.sk;
    geometry.query = {GqaTnd(c) ? 1U : 0U, {uint32_t(q), uint32_t(q >> 32)}, c.sq};
    geometry.kv = {c.layout == 1 ? 1U : 0U, {uint32_t(k), uint32_t(k >> 32)}, c.sk};
    for (uint32_t axis = 0; axis < 2; ++axis) {
        uint32_t maximum;
        const auto status = ReadUsedMetadataAxis(GqaAllocationView<Reader>{read}, arrays, axis, geometry, maximum);
        if (status != S::Success)
            return status;
    }
    const uint32_t tail = GqaTail(c);
    if (read(tail) != c.window.kind || int32_t(read(tail + 1)) != c.window.left ||
        int32_t(read(tail + 2)) != c.window.right)
        return S::ShapeMismatch;
    MlaUsedKernelSchedule used{};
    auto& t = used.tiling;
    t.batch = c.batch * l.headGroups;
    t.sq = c.sq;
    // Zero KV still has query groups in the coordinate component. Only the
    // initializer executes for empty allocation; the logical view keeps those
    // prefixes checkable without reading any KV or allocating partial storage.
    t.sk = c.sk ? c.sk : 1;
    t.heads = l.g;
    t.mBaseSize = l.m;
    t.s2BaseSize = l.n;
    t.blockDim = l.cores;
    MlaUsedMetadataContext context{};
    context.batch = t.batch;
    context.heads = t.heads;
    context.aicNum = l.cores;
    context.query.maxLength = c.sq;
    context.kv.maxLength = c.sk;
    MlaAutoSplitBsndSchedule schedule{};
    const auto status = ReadAutoSplitPartitions<false, false, true>(GqaPartitionView<Reader>{read, c, l}, context,
                                                                    c.window, used, schedule);
    if (status != S::Success)
        return status;
    if (schedule.groups > l.groups)
        return S::InvalidPrefixes;
    output = schedule;
    return S::Success;
}
// Public reader validates the layout; the device loader has already done
// exactly that check before validating buffer addresses and can reuse it.
template <typename Reader, typename Arrays>
MLA_GQA_INLINE MlaMetadataReadStatus ReadGqaMetadata(const Reader& read, uint32_t words, const MlaGqaContext& c,
                                                     const Arrays& arrays, MlaAutoSplitBsndSchedule& output)
{
    MlaGqaLayout l{};
    if (!GqaLayout(c, l) || words != l.words)
        return MlaMetadataReadStatus::InvalidBuffer;
    return ReadGqaMetadataWithLayout(read, c, l, arrays, output);
}
// Workspace capacity is an address stride, not the number of runnable slots.
// Using it as the task stride pins every unsplit group to core zero whenever
// capacity equals the launch width (the automatic-split default).
MLA_GQA_INLINE uint32_t GqaTaskSlots(const MlaGqaContext& c, const MlaAutoSplitBsndSchedule& s)
{
    return c.splits ? c.splits : (s.maxSplits ? s.maxSplits : 1);
}
template <typename Reader>
MLA_GQA_INLINE bool GqaTask(const Reader& read, const MlaGqaContext& c, const MlaGqaLayout& l,
                            const MlaAutoSplitBsndSchedule& s, uint64_t group, uint32_t split, MlaAutoSplitBsndTask& t)
{
    if (split >= l.capacity ||
        !AutoSplitBsndTask(GqaPartitionView<Reader>{read, c, l}, s, c.window, group, c.splits ? 0 : split, t))
        return false;
    if (c.splits) {
        const uint32_t b = t.batch / l.headGroups;
        t.split = split;
        t.splitCount = c.splits;
        SplitBsndFragment(read(32 + 8 * b + 6), read(32 + 8 * b + 7), l.g, t.m, l.m, l.n, c.window, c.splits, split,
                          t.s2Begin, t.s2End);
    }
    return true;
}
} // namespace flash_attn::mla
#undef MLA_GQA_INLINE
