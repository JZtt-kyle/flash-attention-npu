// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
// Independent CPU/AICore v26 validation. No producer or CPU plan calls.
#pragma once
#include "mla_auto_split_paged_tnd_metadata.h"
#include "mla_paged_metadata_consumer.h"
#include "mla_auto_split_bsnd_consumer.h"
#if defined(__NPU_ARCH__)
#define MLA_AUTO_PA_INLINE __aicore__ inline
#else
#define MLA_AUTO_PA_INLINE inline
#endif
namespace flash_attn::mla {
static_assert(kMlaPagedMetadataPrefixWords == 408 && kMlaPagedMetadataSequenceWords == 5,
              "Shared native partition queries require the v6 PA span layout");
struct MlaAutoSplitPagedTndSchedule {
    MlaPagedKernelSchedule paged;
    MlaAutoSplitBsndSchedule native;
    uint64_t budgetBytes;
};
template <typename Reader>
struct AutoSplitPagedTndBaseView {
    const Reader& read;
    uint32_t words;
    MLA_AUTO_PA_INLINE uint32_t operator()(uint32_t i) const
    {
        return i == 1 || i == 2 ? kMlaPagedMetadataVersion : i == 3 ? words : read(i);
    }
};

// Revalidate current lengths, cache descriptors and ALL effective pages before
// any write or synchronization, even for unused Q or fully masked fragments.
// Keep record/context/arrays/pages immutable until all queries and uses finish.
template <typename Reader, typename Arrays, typename Pages>
MLA_AUTO_PA_INLINE MlaMetadataReadStatus ReadAutoSplitPagedTndMetadata(
    const Reader& read, uint32_t words, const MlaAutoSplitPagedTndContext& context,
    const MlaPagedMetadataInputs& inputs, const Arrays& arrays, const MlaPagedCacheInputs& cache, const Pages& pages,
    MlaAutoSplitPagedTndSchedule& output)
{
    using Status = MlaMetadataReadStatus;
    const auto& c = context.paged;
    if (c.batch > (UINT32_MAX - kMlaAutoSplitPagedTndFixedWords) / 7)
        return Status::InvalidBuffer;
    auto status = ReadMetadataHeader(read, words, kMlaAutoSplitPagedTndFixedWords + 7 * c.batch,
                                     kMlaAutoSplitPagedTndVersion, kMlaAutoSplitPagedTndVersion, c.dtype);
    if (status != Status::Success)
        return status;
    if (c.query.layout != 1)
        return Status::ShapeMismatch;
    const uint32_t base = kMlaPagedMetadataPrefixWords + 5 * c.batch;
    const auto& w = context.window;
    if (w.kind > 2 || w.left < -1 || w.right < -1 || w.left > int64_t(c.logicalKvCapacity) - 1 ||
        w.right > (c.query.maxLength == 0 ? 0 : int64_t(c.query.maxLength) - 1) || read(base) != w.kind ||
        int32_t(read(base + 1)) != w.left || int32_t(read(base + 2)) != w.right)
        return Status::ShapeMismatch;
    if ((w.kind == 0 && (w.left != int64_t(c.logicalKvCapacity) - 1 || w.right != int64_t(c.query.maxLength) - 1)) ||
        (w.kind == 1 && (w.left != int64_t(c.logicalKvCapacity) - 1 || w.right != 0)) ||
        (w.kind == 2 && c.query.maxLength && c.logicalKvCapacity && (w.left < 0 || w.right < 0)))
        return Status::ShapeMismatch;
    MlaAutoSplitPagedTndSchedule result{};
    status = ReadPagedMetadata(AutoSplitPagedTndBaseView<Reader>{read, base}, base, c, inputs, arrays, cache, pages,
                               result.paged);
    if (status != Status::Success)
        return status;
    const uint64_t rows = uint64_t(c.query.maxLength) * c.heads;
    const uint64_t totalQ = uint64_t(c.query.totalTokens.low) | (uint64_t(c.query.totalTokens.high) << 32);
    // Public automatic results always reserve 26 slices, independently of AIC count.
    if ((totalQ ? totalQ : 1) * c.heads > uint64_t(INT64_MAX) / (26 * 512 * sizeof(float)))
        return Status::InvalidWorkspace;
    const bool active = c.batch && totalQ && c.logicalKvCapacity;
    if (c.blockDim != (active ? c.aicNum : 1))
        return Status::PlatformMismatch;
    const uint32_t alignedN = c.pageSize < 512 && 512 % c.pageSize ? 512 / c.pageSize * c.pageSize : 512;
    const uint32_t capN = c.logicalKvCapacity < alignedN ? c.logicalKvCapacity : alignedN;
    if (active) {
        const uint64_t m = ((rows < 256 ? rows : 256) + 15) / 16 * 16;
        const uint64_t n = (capN + 31) / 32 * 32;
        const uint64_t scratch = 2 * uint64_t(c.aicNum) * (6 * m * n + 8 * m * 512 + 8 * 256);
        const uint64_t partialRows = c.aicNum == 1 ? 0 : 2 * uint64_t(c.aicNum) * 256;
        result.budgetBytes = (scratch + 31) / 32 * 32 + partialRows * 528 * sizeof(float);
        if (c.workspaceBytes < result.budgetBytes || c.libApiBytes > uint64_t(INT64_MAX) - result.budgetBytes)
            return Status::InvalidWorkspace;
    }
    // The logical query layout is compact TND; KV capacity only determines bounds.
    // These fields do not claim physical contiguous KV storage. All physical
    // validation above remains PA-specific and queries return logical ranges.
    MlaUsedMetadataContext logical{};
    logical.batch = c.batch;
    logical.aicNum = c.aicNum;
    logical.query = {1, c.query.totalTokens, c.query.maxLength};
    const uint64_t virtualKv = uint64_t(c.batch) * c.logicalKvCapacity;
    logical.kv = {0, {uint32_t(virtualKv), uint32_t(virtualKv >> 32)}, c.logicalKvCapacity};
    logical.workspaceBytes = c.workspaceBytes;
    MlaUsedKernelSchedule geometry{};
    geometry.query = logical.query;
    geometry.kv = logical.kv;
    auto& a = geometry.tiling;
    a = result.paged.tiling;
    a.sk = c.logicalKvCapacity;
    a.mBaseSize = a.mBaseSize ? a.mBaseSize : 256;
    a.s2BaseSize = a.s2BaseSize ? a.s2BaseSize : capN ? capN : 1;
    const uint64_t m = ((rows < a.mBaseSize ? rows : a.mBaseSize) + 15) / 16 * 16;
    a.mm1ResSize = m * ((a.s2BaseSize + 31) / 32 * 32);
    a.mm2ResSize = m * 512;
    status = ReadAutoSplitPartitions<true, true>(read, logical, w, geometry, result.native);
    if (status != Status::Success)
        return status;
    if (result.native.bytes > result.budgetBytes)
        return Status::InvalidWorkspace;
    output = result;
    return Status::Success;
}
// Queries require a successfully validated, unchanged record and context.
// All failures preserve the destination. Original Q allocation is retained.
template <typename Reader>
MLA_AUTO_PA_INLINE bool AutoSplitPagedTndTask(const Reader& read, const MlaAutoSplitPagedTndSchedule& s,
                                              const MlaWindowKernelParams& w, uint64_t group, uint32_t split,
                                              MlaAutoSplitBsndTask& output)
{
    return AutoSplitBsndTask<true>(read, s.native, w, group, split, output);
}
template <typename Reader>
MLA_AUTO_PA_INLINE bool AutoSplitPagedTndMerge(const Reader& read, const MlaAutoSplitPagedTndSchedule& s, uint32_t head,
                                               uint32_t part, MlaAutoSplitBsndMerge& output)
{
    return AutoSplitBsndMerge<true>(read, s.native, head, part, output);
}
template <typename Reader>
MLA_AUTO_PA_INLINE bool AutoSplitPagedTndRow(const Reader& read, const MlaAutoSplitPagedTndSchedule& s,
                                             const MlaAutoSplitBsndTask& t, uint32_t row, MlaAutoSplitBsndRow& output)
{
    if (row >= t.rowEnd - t.rowBegin)
        return false;
    const auto& a = s.native.used.tiling;
    const uint64_t flat = uint64_t(t.rowBegin) + row;
    const uint64_t query = ReadMetadataWide(read, kMlaPagedMetadataPrefixWords + 5 * t.batch) + flat / a.heads;
    const uint64_t head = flat % a.heads, total = UsedMetadataTotal(s.native.used.query);
    MlaAutoSplitBsndRow r{};
    r.out = (query * a.heads + head) * 512 * 2;
    r.lse = (head * total + query) * 4;
    if (s.native.aicNum > 1) {
        r.accumulatorLse = uint64_t(t.split) * a.heads * total * 4 + r.lse;
        r.accumulatorOut = r.accumulatorLse * 512;
    }
    r.partialOut = r.partialSum = r.partialMax = UINT64_MAX;
    if (t.kind == 1)
        auto_split_detail::Offsets(s.native, t.slot * a.mBaseSize + row, r.partialOut, r.partialSum, r.partialMax);
    output = r;
    return true;
}
} // namespace flash_attn::mla
#undef MLA_AUTO_PA_INLINE
