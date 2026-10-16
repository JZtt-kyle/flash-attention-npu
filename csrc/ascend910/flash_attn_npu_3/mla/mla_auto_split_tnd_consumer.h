// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
// Independent CPU/AICore v20 consumer; never invokes the producer or CPU plan.
#pragma once
#include "mla_auto_split_tnd_metadata.h"
#include "mla_auto_split_bsnd_consumer.h"
#if defined(__NPU_ARCH__)
#define MLA_AUTO_TND_INLINE __aicore__ inline
#else
#define MLA_AUTO_TND_INLINE inline
#endif
namespace flash_attn::mla {
using MlaAutoSplitTndKernelSchedule = MlaAutoSplitBsndSchedule;
using MlaAutoSplitTndKernelMerge = MlaAutoSplitBsndMerge;
using MlaAutoSplitTndKernelRow = MlaAutoSplitBsndRow;
struct MlaAutoSplitTndKernelTask {
    MlaAutoSplitBsndTask native;
    MlaUsedSequence sequence;
};
template <typename Reader, typename Arrays>
MLA_AUTO_TND_INLINE MlaMetadataReadStatus ReadAutoSplitTndMetadata(const Reader& read, uint32_t words,
                                                                   const MlaAutoSplitTndContext& context,
                                                                   const MlaUsedMetadataInputs& inputs,
                                                                   const Arrays& arrays,
                                                                   MlaAutoSplitTndKernelSchedule& output)
{
    return ReadAutoSplitMetadata<true, kMlaAutoSplitTndVersion, true, kMlaSoftcapAutoSplitTndVersion>(
        read, words, context, inputs, arrays, output);
}
// Queries require an unchanged, successfully validated record and context.
// All failures leave output untouched. Effective lengths govern mask geometry;
// allocation prefixes govern physical addresses and preserve unused Q gaps.
template <typename Reader>
MLA_AUTO_TND_INLINE bool AutoSplitTndTask(const Reader& read, const MlaAutoSplitTndKernelSchedule& s,
                                          const MlaWindowKernelParams& window, uint64_t group, uint32_t split,
                                          MlaAutoSplitTndKernelTask& output)
{
    MlaAutoSplitTndKernelTask t{};
    if (!AutoSplitBsndTask(read, s, window, group, split, t.native))
        return false;
    const uint32_t word = kMlaUsedMetadataPrefixWords + 8 * t.native.batch;
    t.sequence = {ReadMetadataWide(read, word),
                  ReadMetadataWide(read, word + 2),
                  read(word + 4),
                  read(word + 5),
                  read(word + 6),
                  read(word + 7)};
    output = t;
    return true;
}
template <typename Reader>
MLA_AUTO_TND_INLINE bool AutoSplitTndMerge(const Reader& read, const MlaAutoSplitTndKernelSchedule& s, uint32_t head,
                                           uint32_t part, MlaAutoSplitTndKernelMerge& output)
{
    return AutoSplitBsndMerge(read, s, head, part, output);
}
MLA_AUTO_TND_INLINE bool AutoSplitTndRow(const MlaAutoSplitTndKernelSchedule& s, const MlaAutoSplitTndKernelTask& task,
                                         uint32_t row, MlaAutoSplitTndKernelRow& output)
{
    const auto& t = task.native;
    if (row >= t.rowEnd - t.rowBegin)
        return false;
    const auto& a = s.used.tiling;
    const uint64_t flat = uint64_t(t.rowBegin) + row, totalQ = UsedMetadataTotal(s.used.query);
    const uint64_t query = task.sequence.queryStart + flat / a.heads, head = flat % a.heads;
    MlaAutoSplitTndKernelRow r{};
    r.out = (query * a.heads + head) * 512 * 2;
    r.lse = (head * totalQ + query) * 4;
    if (s.aicNum > 1) {
        r.accumulatorLse = uint64_t(t.split) * a.heads * totalQ * 4 + r.lse;
        r.accumulatorOut = r.accumulatorLse * 512;
    }
    r.partialOut = r.partialSum = r.partialMax = UINT64_MAX;
    if (t.kind == 1)
        auto_split_detail::Offsets(s, t.slot * a.mBaseSize + row, r.partialOut, r.partialSum, r.partialMax);
    output = r;
    return true;
}
} // namespace flash_attn::mla
#undef MLA_AUTO_TND_INLINE
