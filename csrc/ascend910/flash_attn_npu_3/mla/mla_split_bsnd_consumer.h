// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
// Independent CPU/AICore validation of v17. The producer's plan is never trusted.
#pragma once
#include "mla_split_bsnd_metadata.h"
#include "mla_used_metadata_consumer.h"
#if defined(__NPU_ARCH__)
#define MLA_SPLIT_INLINE __aicore__ inline
#else
#define MLA_SPLIT_INLINE inline
#endif
namespace flash_attn::mla {
struct MlaSplitBsndSchedule {
    MlaUsedKernelSchedule used;
    uint64_t groups, tasks, outOffset, sumOffset, maxOffset, bytes;
};
// A schema view reuses v5's complete geometry, lengths, source resource and
// endpoint validation AFTER the real v17 header has been checked.
template <typename Reader>
struct SplitUsedView {
    const Reader& read;
    uint32_t words;
    MLA_SPLIT_INLINE uint32_t operator()(uint32_t i) const
    {
        return i == 1 || i == 2 ? kMlaUsedMetadataVersion : i == 3 ? words : read(i);
    }
};
template <typename Reader, typename Arrays>
MLA_SPLIT_INLINE MlaMetadataReadStatus ReadSplitBsndMetadata(const Reader& read, uint32_t words,
                                                             const MlaSplitBsndContext& context,
                                                             const MlaUsedMetadataInputs& inputs, const Arrays& arrays,
                                                             MlaSplitBsndSchedule& output)
{
    using Status = MlaMetadataReadStatus;
    const auto& c = context.used;
    if (c.batch > (UINT32_MAX - 416) / 10)
        return Status::InvalidBuffer;
    auto status =
        ReadMetadataHeader(read, words, 416 + 10 * c.batch, kMlaSplitBsndVersion, kMlaSplitBsndVersion, c.dtype);
    if (status != Status::Success)
        return status;
    if (c.query.layout || c.kv.layout || context.splits < 2 || context.splits > 256)
        return Status::ShapeMismatch;
    const uint32_t base = kMlaUsedMetadataPrefixWords + kMlaUsedMetadataSequenceWords * c.batch;
    const auto& w = context.window;
    if (w.kind > 2 || w.left < -1 || w.right < -1 || w.left > int64_t(c.kv.maxLength) - 1 ||
        w.right > (c.query.maxLength == 0 ? 0 : int64_t(c.query.maxLength) - 1) || read(base) != w.kind ||
        int32_t(read(base + 1)) != w.left || int32_t(read(base + 2)) != w.right || read(base + 3) != context.splits)
        return Status::ShapeMismatch;
    MlaSplitBsndSchedule result{};
    status = ReadUsedMetadata(SplitUsedView<Reader>{read, base}, base, c, inputs, arrays, result.used);
    if (status != Status::Success)
        return status;
    const auto& s = result.used.tiling;
    const bool empty = !c.batch || !s.sq || !s.sk;
    uint64_t prefix = 0;
    for (uint32_t b = 0; b <= c.batch; ++b) {
        if (ReadMetadataWide(read, base + 4 + 2 * b) != prefix)
            return Status::InvalidEndpoints;
        if (b != c.batch && !empty)
            prefix +=
                (uint64_t(read(kMlaUsedMetadataPrefixWords + 8 * b + 6)) * c.heads + s.mBaseSize - 1) / s.mBaseSize;
    }
    result.groups = prefix;
    result.tasks = prefix * context.splits;
    if (!empty) {
        // The static allocation envelope bounds every actual M prefix. M is
        // 256 or a divisor of capacity rows, as checked by the v5 consumer.
        const uint64_t rows = uint64_t(s.sq) * s.heads;
        const uint64_t partialRows =
            uint64_t(c.batch) * ((rows + s.mBaseSize - 1) / s.mBaseSize) * context.splits * s.mBaseSize;
        const uint64_t reserve =
            2 * uint64_t(c.aicNum) * (6 * uint64_t(s.mm1ResSize) + 8 * uint64_t(s.mm2ResSize) + 8 * s.mBaseSize);
        result.outOffset = (reserve + 31) / 32 * 32;
        if (partialRows > (uint64_t(INT64_MAX) - result.outOffset) / (528 * sizeof(float)))
            return Status::InvalidWorkspace;
        result.sumOffset = result.outOffset + partialRows * 512 * sizeof(float);
        result.maxOffset = result.sumOffset + partialRows * 8 * sizeof(float);
        result.bytes = result.maxOffset + partialRows * 8 * sizeof(float);
        if (result.bytes > c.workspaceBytes)
            return Status::InvalidWorkspace;
    }
    output = result;
    return Status::Success;
}
// Calculate a fragment using original effective Q/KV geometry, never a sliced Q.
MLA_SPLIT_INLINE void SplitBsndFragment(uint32_t q, uint32_t k, uint32_t heads, uint32_t m, uint32_t mBase,
                                        uint32_t nBase, const MlaWindowKernelParams& w, uint32_t splits, uint32_t split,
                                        uint32_t& begin, uint32_t& end)
{
    const uint64_t row = uint64_t(m) * mBase;
    const uint64_t last = row + mBase < uint64_t(q) * heads ? row + mBase : uint64_t(q) * heads;
    int64_t lo = 0, hi = k;
    if (w.kind) {
        hi = int64_t((last - 1) / heads) + k - q + (w.kind == 2 ? w.right : 0) + 1;
        hi = hi < 0 ? 0 : hi > k ? k : hi;
        if (w.kind == 2) {
            lo = int64_t(row / heads) + k - q - w.left;
            lo = lo < 0 ? 0 : lo > k ? k : lo;
        }
    }
    if (lo >= hi) {
        begin = end = 0;
        return;
    }
    const uint32_t first = uint64_t(lo) / nBase, stop = (uint64_t(hi) + nBase - 1) / nBase;
    const uint32_t width = (stop - first + splits - 1) / splits;
    begin = first + split * width;
    begin = begin > stop ? stop : begin;
    end = begin + width > stop ? stop : begin + width;
}
} // namespace flash_attn::mla
#undef MLA_SPLIT_INLINE
