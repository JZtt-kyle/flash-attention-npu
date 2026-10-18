// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
// Independent CPU/AICore v24 validation. No producer or CPU plan calls.
#pragma once
#include "mla_auto_split_paged_bsnd_metadata.h"
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
struct MlaAutoSplitPagedBsndSchedule {
    MlaPagedKernelSchedule paged;
    MlaAutoSplitBsndSchedule native;
    uint64_t budgetBytes;
};
template <typename Reader>
struct AutoSplitPagedBaseView {
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
MLA_AUTO_PA_INLINE MlaMetadataReadStatus ReadAutoSplitPagedBsndMetadata(
    const Reader& read, uint32_t words, const MlaAutoSplitPagedBsndContext& context,
    const MlaPagedMetadataInputs& inputs, const Arrays& arrays, const MlaPagedCacheInputs& cache, const Pages& pages,
    MlaAutoSplitPagedBsndSchedule& output)
{
    using Status = MlaMetadataReadStatus;
    const auto& c = context.paged;
    if (c.batch > (UINT32_MAX - kMlaAutoSplitPagedBsndFixedWords) / 7)
        return Status::InvalidBuffer;
    if (context.hasSoftcap > 1)
        return Status::ShapeMismatch;
    const uint32_t version = context.hasSoftcap ? kMlaSoftcapAutoSplitPagedBsndVersion : kMlaAutoSplitPagedBsndVersion;
    auto status =
        ReadMetadataHeader(read, words, kMlaAutoSplitPagedBsndFixedWords + 7 * c.batch, version, version, c.dtype);
    if (status != Status::Success)
        return status;
    if (c.query.layout != 0)
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
    MlaAutoSplitPagedBsndSchedule result{};
    status = ReadPagedMetadata(AutoSplitPagedBaseView<Reader>{read, base}, base, c, inputs, arrays, cache, pages,
                               result.paged);
    if (status != Status::Success)
        return status;
    const uint64_t rows = uint64_t(c.query.maxLength) * c.heads;
    const bool active = c.batch && rows && c.logicalKvCapacity;
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
    // The logical query layout is BSND; KV capacity only determines bounds.
    // These fields do not claim physical contiguous KV storage. All physical
    // validation above remains PA-specific and queries return logical ranges.
    MlaUsedMetadataContext logical{};
    logical.batch = c.batch;
    logical.aicNum = c.aicNum;
    logical.query = {0, c.query.totalTokens, c.query.maxLength};
    logical.workspaceBytes = c.workspaceBytes;
    MlaUsedKernelSchedule geometry{};
    auto& a = geometry.tiling;
    a = result.paged.tiling;
    a.sk = c.logicalKvCapacity;
    a.mBaseSize = a.mBaseSize ? a.mBaseSize : 256;
    a.s2BaseSize = a.s2BaseSize ? a.s2BaseSize : capN ? capN : 1;
    const uint64_t m = ((rows < a.mBaseSize ? rows : a.mBaseSize) + 15) / 16 * 16;
    a.mm1ResSize = m * ((a.s2BaseSize + 31) / 32 * 32);
    a.mm2ResSize = m * 512;
    status = ReadAutoSplitPartitions<false, true>(read, logical, w, geometry, result.native);
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
MLA_AUTO_PA_INLINE bool AutoSplitPagedBsndTask(const Reader& read, const MlaAutoSplitPagedBsndSchedule& s,
                                               const MlaWindowKernelParams& w, uint64_t group, uint32_t split,
                                               MlaAutoSplitBsndTask& output)
{
    return AutoSplitBsndTask<true>(read, s.native, w, group, split, output);
}
template <typename Reader>
MLA_AUTO_PA_INLINE bool AutoSplitPagedBsndMerge(const Reader& read, const MlaAutoSplitPagedBsndSchedule& s,
                                                uint32_t head, uint32_t part, MlaAutoSplitBsndMerge& output)
{
    return AutoSplitBsndMerge<true>(read, s.native, head, part, output);
}
MLA_AUTO_PA_INLINE bool AutoSplitPagedBsndRow(const MlaAutoSplitPagedBsndSchedule& s, const MlaAutoSplitBsndTask& t,
                                              uint32_t row, MlaAutoSplitBsndRow& output)
{
    return AutoSplitBsndRow(s.native, t, row, output);
}
} // namespace flash_attn::mla
#undef MLA_AUTO_PA_INLINE
