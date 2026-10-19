#pragma once

#include "mla_metadata.h"
#include "mla_metadata_consumer.h"

// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.

// Independent CPU/AICore validation of v17. The producer's plan is never trusted.
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
    const uint32_t version = context.hasSoftcap ? kMlaSoftcapSplitBsndVersion : kMlaSplitBsndVersion;
    auto status = ReadMetadataHeader(read, words, 416 + 10 * c.batch, version, version, c.dtype);
    if (status != Status::Success)
        return status;
    if (c.query.layout || c.kv.layout || context.hasSoftcap > 1 || context.splits < (context.hasSoftcap ? 1U : 2U) ||
        context.splits > 256)
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

// Independent CPU/AICore validation of v19. The producer's plan is never trusted.
#if defined(__NPU_ARCH__)
#define MLA_SPLIT_INLINE __aicore__ inline
#else
#define MLA_SPLIT_INLINE inline
#endif
namespace flash_attn::mla {
struct MlaSplitTndSchedule {
    MlaUsedKernelSchedule used;
    uint64_t groups, tasks, outOffset, sumOffset, maxOffset, bytes;
};
template <typename Reader, typename Arrays>
MLA_SPLIT_INLINE MlaMetadataReadStatus ReadSplitTndMetadata(const Reader& read, uint32_t words,
                                                            const MlaSplitTndContext& context,
                                                            const MlaUsedMetadataInputs& inputs, const Arrays& arrays,
                                                            MlaSplitTndSchedule& output)
{
    using Status = MlaMetadataReadStatus;
    const auto& c = context.used;
    if (c.batch > (UINT32_MAX - 416) / 10)
        return Status::InvalidBuffer;
    if (context.hasSoftcap > 1)
        return Status::ShapeMismatch;
    const uint32_t version = context.hasSoftcap ? kMlaSoftcapSplitTndVersion : kMlaSplitTndVersion;
    auto status = ReadMetadataHeader(read, words, 416 + 10 * c.batch, version, version, c.dtype);
    if (status != Status::Success)
        return status;
    if (c.query.layout != 1 || c.kv.layout != 1 || context.splits < (context.hasSoftcap ? 1 : 2) ||
        context.splits > 256)
        return Status::ShapeMismatch;
    const uint32_t base = kMlaUsedMetadataPrefixWords + kMlaUsedMetadataSequenceWords * c.batch;
    const auto& w = context.window;
    if (w.kind > 2 || w.left < -1 || w.right < -1 || w.left > int64_t(c.kv.maxLength) - 1 ||
        w.right > (c.query.maxLength == 0 ? 0 : int64_t(c.query.maxLength) - 1) || read(base) != w.kind ||
        int32_t(read(base + 1)) != w.left || int32_t(read(base + 2)) != w.right || read(base + 3) != context.splits)
        return Status::ShapeMismatch;
    MlaSplitTndSchedule result{};
    status = ReadUsedMetadata(SplitUsedView<Reader>{read, base}, base, c, inputs, arrays, result.used);
    if (status != Status::Success)
        return status;
    const auto& s = result.used.tiling;
    const uint64_t totalQ = uint64_t(c.query.totalTokens.low) | (uint64_t(c.query.totalTokens.high) << 32);
    const bool empty = !totalQ || (!c.kv.totalTokens.low && !c.kv.totalTokens.high);
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
        // Independently bound every allocation partition, including one tail
        // per nonempty segment. Never substitute B*maxQ for compact totalQ.
        const uint64_t active = c.batch < totalQ ? c.batch : totalQ;
        const uint64_t rows = uint64_t(c.query.maxLength) * c.heads;
        const uint64_t perSample = active * ((rows + s.mBaseSize - 1) / s.mBaseSize);
        const uint64_t padded = (totalQ * c.heads + active * (s.mBaseSize - 1)) / s.mBaseSize;
        const uint64_t bound = perSample < padded ? perSample : padded;
        if (prefix > bound)
            return Status::InvalidEndpoints;
        const uint64_t partialRows = bound * context.splits * s.mBaseSize;
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
// Fragment geometry is layout independent; both paths use original used
// lengths and the same independently checked FA3 ceil-div rule.
} // namespace flash_attn::mla
#undef MLA_SPLIT_INLINE

// Independent CPU/AICore validation of v21. The producer's plan is never trusted.
#if defined(__NPU_ARCH__)
#define MLA_SPLIT_INLINE __aicore__ inline
#else
#define MLA_SPLIT_INLINE inline
#endif
namespace flash_attn::mla {
struct MlaSplitMixedSchedule {
    MlaUsedKernelSchedule used;
    uint64_t groups, tasks, outOffset, sumOffset, maxOffset, bytes;
};
template <typename Reader, typename Arrays>
MLA_SPLIT_INLINE MlaMetadataReadStatus ReadSplitMixedMetadata(const Reader& read, uint32_t words,
                                                              const MlaSplitMixedContext& context,
                                                              const MlaUsedMetadataInputs& inputs, const Arrays& arrays,
                                                              MlaSplitMixedSchedule& output)
{
    using Status = MlaMetadataReadStatus;
    const auto& c = context.used;
    if (c.batch > (UINT32_MAX - 416) / 10)
        return Status::InvalidBuffer;
    if (context.hasSoftcap > 1)
        return Status::ShapeMismatch;
    const uint32_t version = context.hasSoftcap ? kMlaSoftcapSplitMixedVersion : kMlaSplitMixedVersion;
    auto status = ReadMetadataHeader(read, words, 416 + 10 * c.batch, version, version, c.dtype);
    if (status != Status::Success)
        return status;
    if (c.query.layout != 1 || c.kv.layout != 0 || context.splits < (context.hasSoftcap ? 1 : 2) ||
        context.splits > 256)
        return Status::ShapeMismatch;
    const uint32_t base = kMlaUsedMetadataPrefixWords + kMlaUsedMetadataSequenceWords * c.batch;
    const auto& w = context.window;
    if (w.kind > 2 || w.left < -1 || w.right < -1 || w.left > int64_t(c.kv.maxLength) - 1 ||
        w.right > (c.query.maxLength == 0 ? 0 : int64_t(c.query.maxLength) - 1) || read(base) != w.kind ||
        int32_t(read(base + 1)) != w.left || int32_t(read(base + 2)) != w.right || read(base + 3) != context.splits)
        return Status::ShapeMismatch;
    MlaSplitMixedSchedule result{};
    status = ReadUsedMetadata(SplitUsedView<Reader>{read, base}, base, c, inputs, arrays, result.used);
    if (status != Status::Success)
        return status;
    const auto& s = result.used.tiling;
    const uint64_t totalQ = uint64_t(c.query.totalTokens.low) | (uint64_t(c.query.totalTokens.high) << 32);
    // Check the FP32 accumulator even when Q/KV is empty or all used lengths
    // are zero. TND Q output/accumulator strides depend on allocation totalQ.
    const uint64_t strideQ = totalQ ? totalQ : 1;
    if (strideQ * c.heads > uint64_t(INT64_MAX) / (uint64_t(context.splits) * 512 * sizeof(float)))
        return Status::InvalidWorkspace;
    const bool empty = !totalQ || !c.batch || !c.kv.maxLength;
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
        // Independently bound every allocation partition, including one tail
        // per nonempty segment. Never substitute B*maxQ for compact totalQ.
        const uint64_t active = c.batch < totalQ ? c.batch : totalQ;
        const uint64_t rows = uint64_t(c.query.maxLength) * c.heads;
        const uint64_t perSample = active * ((rows + s.mBaseSize - 1) / s.mBaseSize);
        const uint64_t padded = (totalQ * c.heads + active * (s.mBaseSize - 1)) / s.mBaseSize;
        const uint64_t bound = perSample < padded ? perSample : padded;
        if (prefix > bound)
            return Status::InvalidEndpoints;
        const uint64_t partialRows = bound * context.splits * s.mBaseSize;
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
// Fragment geometry is layout independent; both paths use original used
// lengths and the same independently checked FA3 ceil-div rule.
} // namespace flash_attn::mla
#undef MLA_SPLIT_INLINE

// Independent CPU/AICore v23 validation and queries. No producer plan calls.
#if defined(__NPU_ARCH__)
#define MLA_SPLIT_PA_INLINE __aicore__ inline
#else
#define MLA_SPLIT_PA_INLINE inline
#endif
namespace flash_attn::mla {
struct MlaSplitPagedBsndSchedule {
    MlaPagedKernelSchedule paged;
    uint32_t mBase, nBase, splits;
    uint64_t groups, tasks, outOffset, sumOffset, maxOffset, bytes, budgetBytes;
};
template <typename Reader>
struct SplitPagedView {
    const Reader& read;
    uint32_t words;
    MLA_SPLIT_PA_INLINE uint32_t operator()(uint32_t i) const
    {
        return i == 1 || i == 2 ? kMlaPagedMetadataVersion : i == 3 ? words : read(i);
    }
};

// Revalidate the complete current cache view and ALL effective pages, even
// for zero Q-used or wholly masked fragments. Neither padding nor page IDs
// are serialized. Call before writes/synchronization; buffers stay immutable
// until all queries/launches complete. This validates safety, not provenance.
template <typename Reader, typename Arrays, typename Pages>
MLA_SPLIT_PA_INLINE MlaMetadataReadStatus ReadSplitPagedBsndMetadata(
    const Reader& read, uint32_t words, const MlaSplitPagedBsndContext& context, const MlaPagedMetadataInputs& inputs,
    const Arrays& arrays, const MlaPagedCacheInputs& cache, const Pages& pages, MlaSplitPagedBsndSchedule& output)
{
    using Status = MlaMetadataReadStatus;
    const auto& c = context.paged;
    if (c.batch > (UINT32_MAX - 414) / 7)
        return Status::InvalidBuffer;
    if (context.hasSoftcap > 1)
        return Status::ShapeMismatch;
    const uint32_t version = context.hasSoftcap ? kMlaSoftcapSplitPagedBsndVersion : kMlaSplitPagedBsndVersion;
    auto status = ReadMetadataHeader(read, words, 414 + 7 * c.batch, version, version, c.dtype);
    if (status != Status::Success)
        return status;
    if (c.query.layout != 0 || context.splits < (context.hasSoftcap ? 1 : 2) || context.splits > 256)
        return Status::ShapeMismatch;
    const uint32_t base = kMlaPagedMetadataPrefixWords + kMlaPagedMetadataSequenceWords * c.batch;
    const auto& w = context.window;
    if (w.kind > 2 || w.left < -1 || w.right < -1 || w.left > int64_t(c.logicalKvCapacity) - 1 ||
        w.right > (c.query.maxLength == 0 ? 0 : int64_t(c.query.maxLength) - 1) || read(base) != w.kind ||
        int32_t(read(base + 1)) != w.left || int32_t(read(base + 2)) != w.right || read(base + 3) != context.splits)
        return Status::ShapeMismatch;
    MlaSplitPagedBsndSchedule result{};
    status = ReadPagedMetadata(SplitPagedView<Reader>{read, base}, base, c, inputs, arrays, cache, pages, result.paged);
    if (status != Status::Success)
        return status;
    const auto& s = result.paged.tiling;
    const uint64_t rows = uint64_t(c.query.maxLength) * c.heads;
    // Empty tensor strides still multiply max(dim,1). This is independent
    // of the partial workspace and must be checked even at C=0 or Q-used=0.
    const uint64_t strideRows = uint64_t(c.batch ? c.batch : 1) * (c.query.maxLength ? c.query.maxLength : 1) * c.heads;
    const uint64_t elementBytes = c.batch && c.query.maxLength ? sizeof(float) : 1;
    if (strideRows > uint64_t(INT64_MAX) / (uint64_t(context.splits) * 512 * elementBytes))
        return Status::InvalidWorkspace;
    const bool active = c.batch && rows && c.logicalKvCapacity;
    if (c.blockDim != (active ? c.aicNum : 1))
        return Status::PlatformMismatch;
    const uint32_t alignedBase = c.pageSize < 512 && 512 % c.pageSize ? 512 / c.pageSize * c.pageSize : 512;
    const uint32_t capBase = c.logicalKvCapacity < alignedBase ? c.logicalKvCapacity : alignedBase;
    result.mBase = s.mBaseSize ? s.mBaseSize : 256;
    result.nBase = s.s2BaseSize ? s.s2BaseSize : capBase ? capBase : 1;
    result.splits = context.splits;
    if (active) {
        // Static budget covers every valid PA M/N choice and every used
        // length; actual compact slots below are bounded independently.
        const uint64_t maxM = ((rows < 256 ? rows : 256) + 15) / 16 * 16;
        const uint64_t maxN = (capBase + 31) / 32 * 32;
        const uint64_t reserve = 2 * uint64_t(c.aicNum) * (6 * maxM * maxN + 8 * maxM * 512 + 8 * 256);
        const uint64_t slots = uint64_t(c.batch) * ((rows + 255) / 256) * context.splits * 256;
        const uint64_t start = (reserve + 31) / 32 * 32;
        if (slots > (uint64_t(INT64_MAX) - start) / (528 * sizeof(float)))
            return Status::InvalidWorkspace;
        result.budgetBytes = start + slots * 528 * sizeof(float);
        if (c.workspaceBytes < result.budgetBytes || c.libApiBytes > uint64_t(INT64_MAX) - result.budgetBytes)
            return Status::InvalidWorkspace;
        const uint64_t m = ((rows < result.mBase ? rows : result.mBase) + 15) / 16 * 16;
        const uint64_t n = (result.nBase + 31) / 32 * 32;
        const uint64_t actualReserve = 2 * uint64_t(c.aicNum) * (6 * m * n + 8 * m * 512 + 8 * result.mBase);
        const uint64_t partialRows =
            uint64_t(c.batch) * ((rows + result.mBase - 1) / result.mBase) * context.splits * result.mBase;
        result.outOffset = (actualReserve + 31) / 32 * 32;
        result.sumOffset = result.outOffset + partialRows * 512 * sizeof(float);
        result.maxOffset = result.sumOffset + partialRows * 8 * sizeof(float);
        result.bytes = result.maxOffset + partialRows * 8 * sizeof(float);
        if (result.bytes > result.budgetBytes)
            return Status::InvalidWorkspace;
    }
    uint64_t prefix = 0;
    for (uint32_t b = 0; b <= c.batch; ++b) {
        if (ReadMetadataWide(read, base + 4 + 2 * b) != prefix)
            return Status::InvalidEndpoints;
        if (b != c.batch && active)
            prefix +=
                (uint64_t(read(kMlaPagedMetadataPrefixWords + 5 * b + 3)) * c.heads + result.mBase - 1) / result.mBase;
    }
    result.groups = prefix;
    result.tasks = prefix * context.splits;
    output = result;
    return Status::Success;
}

struct MlaSplitPagedBsndTask {
    uint64_t task, group, iteration;
    uint32_t batch, m, split, owner, rowBegin, rowEnd, s2Begin, s2End;
    uint64_t outOffset, sumOffset, maxOffset;
};
// Query only a successfully validated, unchanged record/context. Independent
// binary search handles zero-length samples and uint64 task IDs without an
// O(number of tasks) allocation. Failure leaves the destination unchanged.
template <typename Reader>
MLA_SPLIT_PA_INLINE bool SplitPagedBsndTask(const Reader& read, const MlaSplitPagedBsndContext& c,
                                            const MlaSplitPagedBsndSchedule& s, uint64_t index,
                                            MlaSplitPagedBsndTask& output)
{
    if (index >= s.tasks)
        return false;
    MlaSplitPagedBsndTask t{};
    t.task = index;
    t.group = index / s.splits;
    t.split = index % s.splits;
    t.owner = index % c.paged.blockDim;
    t.iteration = index / c.paged.blockDim;
    const uint32_t prefix = kMlaPagedMetadataPrefixWords + 5 * c.paged.batch + 4;
    uint32_t lo = 0, hi = c.paged.batch;
    while (lo < hi) {
        const uint32_t mid = lo + (hi - lo) / 2;
        if (ReadMetadataWide(read, prefix + 2 * (mid + 1)) <= t.group)
            lo = mid + 1;
        else
            hi = mid;
    }
    t.batch = lo;
    t.m = t.group - ReadMetadataWide(read, prefix + 2 * lo);
    const uint32_t span = kMlaPagedMetadataPrefixWords + 5 * lo;
    const uint32_t q = read(span + 3), k = read(span + 4);
    t.rowBegin = uint64_t(t.m) * s.mBase;
    const uint64_t last = uint64_t(t.rowBegin) + s.mBase, rows = uint64_t(q) * c.paged.heads;
    t.rowEnd = last < rows ? last : rows;
    SplitBsndFragment(q, k, c.paged.heads, t.m, s.mBase, s.nBase, c.window, s.splits, t.split, t.s2Begin, t.s2End);
    const uint64_t slot = index * s.mBase;
    t.outOffset = s.outOffset + slot * 512 * sizeof(float);
    t.sumOffset = s.sumOffset + slot * 8 * sizeof(float);
    t.maxOffset = s.maxOffset + slot * 8 * sizeof(float);
    output = t;
    return true;
}
} // namespace flash_attn::mla
#undef MLA_SPLIT_PA_INLINE

// Independent CPU/AICore v25 validation and queries. No producer plan calls.
#if defined(__NPU_ARCH__)
#define MLA_SPLIT_PA_INLINE __aicore__ inline
#else
#define MLA_SPLIT_PA_INLINE inline
#endif
namespace flash_attn::mla {
struct MlaSplitPagedTndSchedule {
    MlaPagedKernelSchedule paged;
    uint32_t mBase, nBase, splits;
    uint64_t groups, tasks, outOffset, sumOffset, maxOffset, bytes, budgetBytes;
};
template <typename Reader>
struct SplitPagedTndView {
    const Reader& read;
    uint32_t words;
    MLA_SPLIT_PA_INLINE uint32_t operator()(uint32_t i) const
    {
        return i == 1 || i == 2 ? kMlaPagedMetadataVersion : i == 3 ? words : read(i);
    }
};

// Revalidate the complete current cache view and ALL effective pages, even
// for zero Q-used or wholly masked fragments. Neither padding nor page IDs
// are serialized. Call before writes/synchronization; buffers stay immutable
// until all queries/launches complete. This validates safety, not provenance.
template <typename Reader, typename Arrays, typename Pages>
MLA_SPLIT_PA_INLINE MlaMetadataReadStatus ReadSplitPagedTndMetadata(
    const Reader& read, uint32_t words, const MlaSplitPagedTndContext& context, const MlaPagedMetadataInputs& inputs,
    const Arrays& arrays, const MlaPagedCacheInputs& cache, const Pages& pages, MlaSplitPagedTndSchedule& output)
{
    using Status = MlaMetadataReadStatus;
    const auto& c = context.paged;
    if (c.batch > (UINT32_MAX - 414) / 7)
        return Status::InvalidBuffer;
    if (context.hasSoftcap > 1)
        return Status::ShapeMismatch;
    const uint32_t version = context.hasSoftcap ? kMlaSoftcapSplitPagedTndVersion : kMlaSplitPagedTndVersion;
    auto status = ReadMetadataHeader(read, words, 414 + 7 * c.batch, version, version, c.dtype);
    if (status != Status::Success)
        return status;
    if (c.query.layout != 1 || context.splits < (context.hasSoftcap ? 1 : 2) || context.splits > 256)
        return Status::ShapeMismatch;
    if (uint64_t(c.aivNum) != 2 * uint64_t(c.aicNum))
        return Status::PlatformMismatch;
    const uint32_t base = kMlaPagedMetadataPrefixWords + kMlaPagedMetadataSequenceWords * c.batch;
    const auto& w = context.window;
    if (w.kind > 2 || w.left < -1 || w.right < -1 || w.left > int64_t(c.logicalKvCapacity) - 1 ||
        w.right > (c.query.maxLength == 0 ? 0 : int64_t(c.query.maxLength) - 1) || read(base) != w.kind ||
        int32_t(read(base + 1)) != w.left || int32_t(read(base + 2)) != w.right || read(base + 3) != context.splits)
        return Status::ShapeMismatch;
    MlaSplitPagedTndSchedule result{};
    status =
        ReadPagedMetadata(SplitPagedTndView<Reader>{read, base}, base, c, inputs, arrays, cache, pages, result.paged);
    if (status != Status::Success)
        return status;
    const auto& s = result.paged.tiling;
    const uint64_t rows = uint64_t(c.query.maxLength) * c.heads;
    const uint64_t totalQ = uint64_t(c.query.totalTokens.low) | (uint64_t(c.query.totalTokens.high) << 32);
    // ReadPagedMetadata already checked int32 totalQ and the true Q allocation.
    const uint64_t strideRows = (totalQ ? totalQ : 1) * c.heads;
    if (strideRows > uint64_t(INT64_MAX) / (uint64_t(context.splits) * 512 * sizeof(float)))
        return Status::InvalidWorkspace;
    const bool active = c.batch && totalQ && c.logicalKvCapacity;
    if (c.blockDim != (active ? c.aicNum : 1))
        return Status::PlatformMismatch;
    const uint32_t alignedBase = c.pageSize < 512 && 512 % c.pageSize ? 512 / c.pageSize * c.pageSize : 512;
    const uint32_t capBase = c.logicalKvCapacity < alignedBase ? c.logicalKvCapacity : alignedBase;
    result.mBase = s.mBaseSize ? s.mBaseSize : 256;
    result.nBase = s.s2BaseSize ? s.s2BaseSize : capBase ? capBase : 1;
    result.splits = context.splits;
    if (active) {
        // Static budget covers every valid PA M/N choice and every used
        // length; actual compact slots below are bounded independently.
        const uint64_t maxM = ((rows < 256 ? rows : 256) + 15) / 16 * 16;
        const uint64_t maxN = (capBase + 31) / 32 * 32;
        const uint64_t reserve = 2 * uint64_t(c.aicNum) * (6 * maxM * maxN + 8 * maxM * 512 + 8 * 256);
        const uint64_t nonempty = c.batch < totalQ ? c.batch : totalQ;
        const uint64_t slots = (totalQ * c.heads + nonempty * 255) * context.splits;
        const uint64_t start = (reserve + 31) / 32 * 32;
        if (slots > (uint64_t(INT64_MAX) - start) / (528 * sizeof(float)))
            return Status::InvalidWorkspace;
        result.budgetBytes = start + slots * 528 * sizeof(float);
        if (c.workspaceBytes < result.budgetBytes || c.libApiBytes > uint64_t(INT64_MAX) - result.budgetBytes)
            return Status::InvalidWorkspace;
        const uint64_t m = ((rows < result.mBase ? rows : result.mBase) + 15) / 16 * 16;
        const uint64_t n = (result.nBase + 31) / 32 * 32;
        const uint64_t actualReserve = 2 * uint64_t(c.aicNum) * (6 * m * n + 8 * m * 512 + 8 * result.mBase);
        const uint64_t perSample = nonempty * ((rows + result.mBase - 1) / result.mBase);
        const uint64_t padded = (totalQ * c.heads + nonempty * (result.mBase - 1)) / result.mBase;
        const uint64_t groups = perSample < padded ? perSample : padded;
        const uint64_t partialRows = groups * context.splits * result.mBase;
        result.outOffset = (actualReserve + 31) / 32 * 32;
        if (partialRows > (uint64_t(INT64_MAX) - result.outOffset) / (528 * sizeof(float)))
            return Status::InvalidWorkspace;
        result.sumOffset = result.outOffset + partialRows * 512 * sizeof(float);
        result.maxOffset = result.sumOffset + partialRows * 8 * sizeof(float);
        result.bytes = result.maxOffset + partialRows * 8 * sizeof(float);
        if (result.bytes > result.budgetBytes)
            return Status::InvalidWorkspace;
    }
    uint64_t prefix = 0;
    for (uint32_t b = 0; b <= c.batch; ++b) {
        if (ReadMetadataWide(read, base + 4 + 2 * b) != prefix)
            return Status::InvalidEndpoints;
        if (b != c.batch && active)
            prefix +=
                (uint64_t(read(kMlaPagedMetadataPrefixWords + 5 * b + 3)) * c.heads + result.mBase - 1) / result.mBase;
    }
    if (active) {
        const uint64_t nonempty = c.batch < totalQ ? c.batch : totalQ;
        const uint64_t perSample = nonempty * ((rows + result.mBase - 1) / result.mBase);
        const uint64_t padded = (totalQ * c.heads + nonempty * (result.mBase - 1)) / result.mBase;
        if (prefix > (perSample < padded ? perSample : padded))
            return Status::InvalidEndpoints;
    }
    result.groups = prefix;
    result.tasks = prefix * context.splits;
    output = result;
    return Status::Success;
}

struct MlaSplitPagedTndTask {
    uint64_t task, group, iteration;
    uint32_t batch, m, split, owner, rowBegin, rowEnd, s2Begin, s2End;
    uint64_t outOffset, sumOffset, maxOffset;
    uint64_t queryStart;
    uint32_t queryCapacity, queryLength, kvLength;
};
// Query only a successfully validated, unchanged record/context. Independent
// binary search handles zero-length samples and uint64 task IDs without an
// O(number of tasks) allocation. Failure leaves the destination unchanged.
template <typename Reader>
MLA_SPLIT_PA_INLINE bool SplitPagedTndTask(const Reader& read, const MlaSplitPagedTndContext& c,
                                           const MlaSplitPagedTndSchedule& s, uint64_t index,
                                           MlaSplitPagedTndTask& output)
{
    if (index >= s.tasks)
        return false;
    MlaSplitPagedTndTask t{};
    t.task = index;
    t.group = index / s.splits;
    t.split = index % s.splits;
    t.owner = index % c.paged.blockDim;
    t.iteration = index / c.paged.blockDim;
    const uint32_t prefix = kMlaPagedMetadataPrefixWords + 5 * c.paged.batch + 4;
    uint32_t lo = 0, hi = c.paged.batch;
    while (lo < hi) {
        const uint32_t mid = lo + (hi - lo) / 2;
        if (ReadMetadataWide(read, prefix + 2 * (mid + 1)) <= t.group)
            lo = mid + 1;
        else
            hi = mid;
    }
    t.batch = lo;
    t.m = t.group - ReadMetadataWide(read, prefix + 2 * lo);
    const uint32_t span = kMlaPagedMetadataPrefixWords + 5 * lo;
    const uint32_t q = read(span + 3), k = read(span + 4);
    t.queryStart = ReadMetadataWide(read, span);
    t.queryCapacity = read(span + 2);
    t.queryLength = q;
    t.kvLength = k;
    t.rowBegin = uint64_t(t.m) * s.mBase;
    const uint64_t last = uint64_t(t.rowBegin) + s.mBase, rows = uint64_t(q) * c.paged.heads;
    t.rowEnd = last < rows ? last : rows;
    SplitBsndFragment(q, k, c.paged.heads, t.m, s.mBase, s.nBase, c.window, s.splits, t.split, t.s2Begin, t.s2End);
    const uint64_t slot = index * s.mBase;
    t.outOffset = s.outOffset + slot * 512 * sizeof(float);
    t.sumOffset = s.sumOffset + slot * 8 * sizeof(float);
    t.maxOffset = s.maxOffset + slot * 8 * sizeof(float);
    output = t;
    return true;
}
struct MlaSplitPagedTndMerge {
    uint64_t group, iteration;
    uint32_t batch, m, owner, rowBegin, rowEnd;
    uint64_t outOffset, sumOffset, maxOffset;
};
template <typename Reader>
MLA_SPLIT_PA_INLINE bool SplitPagedTndMerge(const Reader& read, const MlaSplitPagedTndContext& c,
                                            const MlaSplitPagedTndSchedule& s, uint64_t index,
                                            MlaSplitPagedTndMerge& output)
{
    if (index >= s.groups)
        return false;
    MlaSplitPagedTndTask task{};
    if (!SplitPagedTndTask(read, c, s, index * s.splits, task))
        return false;
    const uint32_t vectors = c.paged.blockDim * 2;
    output = {index,         index / vectors, task.batch,     task.m,         uint32_t(index % vectors),
              task.rowBegin, task.rowEnd,     task.outOffset, task.sumOffset, task.maxOffset};
    return true;
}
struct MlaSplitPagedTndRow {
    uint64_t out, lse, accumulatorOut, accumulatorLse, partialOut, partialSum, partialMax;
};
template <typename Reader>
MLA_SPLIT_PA_INLINE bool SplitPagedTndRow(const Reader& read, const MlaSplitPagedTndContext& c,
                                          const MlaSplitPagedTndSchedule& s, uint64_t index, uint32_t row,
                                          MlaSplitPagedTndRow& output)
{
    MlaSplitPagedTndTask task{};
    if (!SplitPagedTndTask(read, c, s, index, task) || row >= task.rowEnd - task.rowBegin)
        return false;
    const uint64_t flat = uint64_t(task.rowBegin) + row;
    const uint64_t query = task.queryStart + flat / c.paged.heads, head = flat % c.paged.heads;
    const uint64_t total = uint64_t(c.paged.query.totalTokens.low) | (uint64_t(c.paged.query.totalTokens.high) << 32);
    const uint64_t accum = (uint64_t(task.split) * c.paged.heads + head) * total + query;
    output = {(query * c.paged.heads + head) * 512 * 2,
              (head * total + query) * sizeof(float),
              accum * 512 * sizeof(float),
              accum * sizeof(float),
              task.outOffset + row * 512 * sizeof(float),
              task.sumOffset + row * 8 * sizeof(float),
              task.maxOffset + row * 8 * sizeof(float)};
    return true;
}
} // namespace flash_attn::mla
#undef MLA_SPLIT_PA_INLINE

// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.

// Independent v18/v20/v22 safety validation and shared v24 partition checks. No producer, CPU plan or SplitCore call.
// Validates coverage/addressing, not the optimality/authenticity of a partition.
#if defined(__NPU_ARCH__)
#define MLA_AUTO_INLINE __aicore__ inline
#else
#define MLA_AUTO_INLINE inline
#endif
namespace flash_attn::mla {
struct MlaAutoSplitBsndSchedule {
    MlaUsedKernelSchedule used;
    uint32_t aicNum, usedCoreNum, fdHeads, maxSplits, vectors;
    uint32_t bEnd[26], mEnd[26], nEnd[26];
    uint64_t groups, partialPrefix[27], outOffset, sumOffset, maxOffset, bytes;
};
struct MlaAutoSplitBsndTask {
    uint64_t group, slot, outOffset, sumOffset, maxOffset;
    uint32_t batch, m, split, splitCount, owner, fdHead, kind; // Direct=0, Partial=1, Empty=2
    uint32_t rowBegin, rowEnd, s2Begin, s2End, kvBegin, kvEnd;
};
struct MlaAutoSplitBsndMerge {
    uint64_t group, firstSlot, outOffset, sumOffset, maxOffset;
    uint32_t batch, m, head, part, owner, splitCount, rowBegin, rowEnd;
};
struct MlaAutoSplitBsndRow {
    uint64_t out, lse, accumulatorOut, accumulatorLse, partialOut, partialSum, partialMax;
};
namespace auto_split_detail {
template <bool IsPaged = false>
MLA_AUTO_INLINE uint32_t Base(uint32_t batch)
{
    return IsPaged ? 408 + 5 * batch : kMlaUsedMetadataPrefixWords + 8 * batch;
}
template <bool IsPaged, typename Reader>
MLA_AUTO_INLINE uint32_t QueryLength(const Reader& read, uint32_t batch)
{
    return read(IsPaged ? 408 + 5 * batch + 3 : kMlaUsedMetadataPrefixWords + 8 * batch + 6);
}
template <bool IsPaged, typename Reader>
MLA_AUTO_INLINE uint32_t KvLength(const Reader& read, uint32_t batch)
{
    return read(IsPaged ? 408 + 5 * batch + 4 : kMlaUsedMetadataPrefixWords + 8 * batch + 7);
}
MLA_AUTO_INLINE bool Less(uint32_t b, uint32_t m, uint32_t n, uint32_t z, uint32_t j, uint32_t k)
{
    return b < z || (b == z && (m < j || (m == j && n < k)));
}
template <typename Reader>
struct UsedView {
    const Reader& read;
    uint32_t words;
    MLA_AUTO_INLINE uint32_t operator()(uint32_t i) const
    {
        return i == 1 || i == 2 ? kMlaUsedMetadataVersion : i == 3 ? words : read(i);
    }
};
// The caller has already checked the span and M index. All products are wide.
template <bool IsPaged, typename Reader>
MLA_AUTO_INLINE void Geometry(const Reader& read, const MlaAutoSplitBsndSchedule& s, const MlaWindowKernelParams& w,
                              MlaAutoSplitBsndTask& t)
{
    const auto& a = s.used.tiling;
    const uint32_t q = QueryLength<IsPaged>(read, t.batch), k = KvLength<IsPaged>(read, t.batch);
    const uint64_t row = uint64_t(t.m) * a.mBaseSize, rows = uint64_t(q) * a.heads;
    t.rowBegin = row;
    t.rowEnd = row + a.mBaseSize < rows ? row + a.mBaseSize : rows;
    int64_t lo = 0, hi = k;
    if (w.kind) {
        hi = int64_t((t.rowEnd - 1) / a.heads) + k - q + (w.kind == 2 ? w.right : 0) + 1;
        hi = hi < 0 ? 0 : hi > k ? k : hi;
        if (w.kind == 2) {
            lo = int64_t(row / a.heads) + k - q - w.left;
            lo = lo < 0 ? 0 : lo > k ? k : lo;
        }
    }
    if (lo >= hi)
        lo = hi = 0;
    t.kvBegin = lo;
    t.kvEnd = hi;
    t.s2Begin = uint64_t(lo) / a.s2BaseSize;
    t.s2End = (uint64_t(hi) + a.s2BaseSize - 1) / a.s2BaseSize;
}
// Adjacent exclusive endpoints form disjoint intervals. Intersect them with
// this M's visible interval; do not uniformly divide by maxSplits.
MLA_AUTO_INLINE uint32_t Parts(const MlaAutoSplitBsndSchedule& s, MlaAutoSplitBsndTask& t, uint32_t owner = UINT32_MAX,
                               uint32_t* ownerSplit = nullptr)
{
    const uint32_t first = t.s2Begin, stop = t.s2End;
    uint32_t count = 0;
    for (uint32_t core = 0; core < s.usedCoreNum; ++core) {
        uint32_t lo = first, hi = stop;
        if (core) {
            if (Less(t.batch, t.m, UINT32_MAX, s.bEnd[core - 1], s.mEnd[core - 1], 0))
                continue;
            if (t.batch == s.bEnd[core - 1] && t.m == s.mEnd[core - 1])
                lo = lo > s.nEnd[core - 1] ? lo : s.nEnd[core - 1];
        }
        if (Less(s.bEnd[core], s.mEnd[core], UINT32_MAX, t.batch, t.m, 0))
            continue;
        if (t.batch == s.bEnd[core] && t.m == s.mEnd[core])
            hi = hi < s.nEnd[core] ? hi : s.nEnd[core];
        if (lo < hi) {
            if (ownerSplit && core == owner)
                *ownerSplit = count;
            if (count == t.split) {
                t.owner = core;
                t.s2Begin = lo;
                t.s2End = hi;
            }
            ++count;
        }
    }
    return count;
}
MLA_AUTO_INLINE void Offsets(const MlaAutoSplitBsndSchedule& s, uint64_t row, uint64_t& o, uint64_t& l, uint64_t& m)
{
    o = s.outOffset + row * 512 * 4;
    l = s.sumOffset + row * 8 * 4;
    m = s.maxOffset + row * 8 * 4;
}
} // namespace auto_split_detail

// Geometry and invocation descriptors must already be independently validated.
// This common checker consumes a logical used-span view plus the native tail;
// it never calls a producer or recomputes the source's optimal partition.
template <bool IsTnd, bool IsPaged = false, bool Prefill = false, typename Reader>
MLA_AUTO_INLINE MlaMetadataReadStatus ReadAutoSplitPartitions(const Reader& read, const MlaUsedMetadataContext& c,
                                                              const MlaWindowKernelParams& w,
                                                              const MlaUsedKernelSchedule& used,
                                                              MlaAutoSplitBsndSchedule& output)
{
    using Status = MlaMetadataReadStatus;
    using namespace auto_split_detail;
    const uint32_t base = Base<IsPaged>(c.batch), prefixWord = base + kMlaAutoSplitTailPrefix;
    MlaAutoSplitBsndSchedule s{};
    s.used = used;
    auto& a = s.used.tiling;
    const uint64_t totalQ = UsedMetadataTotal(c.query);
    const bool empty = IsTnd ? !totalQ || !UsedMetadataTotal(c.kv) : !c.batch || !a.sq || !a.sk;
    if (c.aicNum > 1 && !Prefill) {
        // FP32 accumulator storage and contiguous strides are independent of
        // the much smaller FD workspace. Used lengths may all be zero while
        // the allocation capacity still overflows. Empty tensors retain strides.
        const uint64_t rows =
            IsTnd ? (totalQ ? totalQ : 1) * a.heads : uint64_t(c.batch ? c.batch : 1) * (a.sq ? a.sq : 1) * a.heads;
        if (rows > uint64_t(INT64_MAX) / 512 / c.aicNum || (totalQ && rows > uint64_t(INT64_MAX) / 2048 / c.aicNum))
            return Status::InvalidWorkspace;
    }
    if (!empty && a.blockDim != c.aicNum)
        return Status::PlatformMismatch;
    s.aicNum = c.aicNum;
    s.usedCoreNum = read(base + 3);
    s.fdHeads = read(base + 82);
    s.maxSplits = read(base + 83);
    s.vectors = read(base + 84);
    if (!s.usedCoreNum || s.usedCoreNum > c.aicNum || s.fdHeads > c.aicNum || s.maxSplits > c.aicNum ||
        s.vectors > s.usedCoreNum * 2 || read(base + 85) != 8)
        return Status::InvalidTiling;
    uint64_t prefix = 0;
    uint32_t lastBatch = 0, lastM = 0, lastN = 0;
    bool live = false;
    for (uint32_t b = 0; b <= c.batch; ++b) {
        if (ReadMetadataWide(read, prefixWord + 2 * b) != prefix)
            return Status::InvalidPrefixes;
        if (b == c.batch || empty)
            continue;
        const uint64_t rows = uint64_t(QueryLength<IsPaged>(read, b)) * a.heads;
        const uint32_t count = (rows + a.mBaseSize - 1) / a.mBaseSize, k = KvLength<IsPaged>(read, b);
        prefix += count;
        // Right-bottom causal/local bands always include the last row's last
        // key when both lengths are nonzero. This bounds all earlier tiles
        // without enumerating a potentially enormous number of M blocks.
        if (count && k) {
            live = true;
            lastBatch = b;
            lastM = count - 1;
            lastN = (k + a.s2BaseSize - 1) / a.s2BaseSize - 1;
        }
    }
    if constexpr (IsTnd) {
        if (!empty) {
            // Each nonempty allocation has its own M tail. Compact totalQ,
            // not B*maxQ, bounds every valid allocation/used partition.
            const uint64_t active = c.batch < totalQ ? c.batch : totalQ;
            const uint64_t rows = uint64_t(c.query.maxLength) * a.heads;
            const uint64_t perSample = active * ((rows + a.mBaseSize - 1) / a.mBaseSize);
            const uint64_t padded = (totalQ * a.heads + active * (a.mBaseSize - 1)) / a.mBaseSize;
            if (prefix > (perSample < padded ? perSample : padded))
                return Status::InvalidPrefixes;
        }
    }
    s.groups = prefix;
    for (uint32_t core = 0; core < 26; ++core) {
        const uint32_t b = read(base + 4 + core), m = read(base + 30 + core), n = read(base + 56 + core);
        if (core >= s.usedCoreNum) {
            if (b || m || n)
                return Status::InvalidEndpoints;
            continue;
        }
        if (b > c.batch || (b == c.batch && (m || n)))
            return Status::InvalidEndpoints;
        if (b < c.batch) {
            const uint64_t rows = uint64_t(QueryLength<IsPaged>(read, b)) * a.heads;
            const uint32_t count = empty ? 0 : (rows + a.mBaseSize - 1) / a.mBaseSize;
            if (m >= (count ? count : 1) || (!count && n))
                return Status::InvalidEndpoints;
            if (n) {
                MlaAutoSplitBsndTask t{};
                t.batch = b;
                t.m = m;
                Geometry<IsPaged>(read, s, w, t);
                // Native cost partitioning can stop at/before the first
                // visible tile of a local window. That empty prefix belongs
                // to neither an FD fragment nor persistent storage.
                if (n > t.s2End)
                    return Status::InvalidEndpoints;
            }
        }
        if (core && !Less(s.bEnd[core - 1], s.mEnd[core - 1], s.nEnd[core - 1], b, m, n))
            return Status::InvalidEndpoints;
        s.bEnd[core] = b;
        s.mEnd[core] = m;
        s.nEnd[core] = n;
    }
    const uint32_t last = s.usedCoreNum - 1;
    if ((live && !Less(lastBatch, lastM, lastN, s.bEnd[last], s.mEnd[last], s.nEnd[last])) ||
        (!live && s.usedCoreNum != 1))
        return Status::InvalidEndpoints;
    // Every cross-core M must contain an interior endpoint. Reconstruct the
    // sorted FD list from those endpoints, then check all stored bookkeeping.
    uint32_t heads = 0, maxSplits = 0;
    uint32_t previousB = UINT32_MAX, previousM = UINT32_MAX;
    uint32_t fdParts[27]{};
    for (uint32_t core = 0; core + 1 < s.usedCoreNum; ++core) {
        if (!s.nEnd[core] || (s.bEnd[core] == previousB && s.mEnd[core] == previousM))
            continue;
        MlaAutoSplitBsndTask t{};
        t.batch = s.bEnd[core];
        t.m = s.mEnd[core];
        Geometry<IsPaged>(read, s, w, t);
        if (s.nEnd[core] <= t.s2Begin || s.nEnd[core] == t.s2End)
            continue;
        const uint32_t count = Parts(s, t), rows = t.rowEnd - t.rowBegin;
        if (count < 2 || heads >= s.fdHeads || read(base + 86 + heads) != t.batch || read(base + 112 + heads) != t.m ||
            read(base + 138 + heads) != count || read(base + 190 + heads) != (rows + 7) / 8 ||
            read(base + 216 + heads) != (rows - 1) % 8 + 1)
            return Status::InvalidEndpoints;
        s.partialPrefix[heads + 1] = s.partialPrefix[heads] + count;
        fdParts[heads + 1] = fdParts[heads] + (rows + 7) / 8;
        maxSplits = maxSplits > count ? maxSplits : count;
        previousB = t.batch;
        previousM = t.m;
        ++heads;
    }
    if (heads != s.fdHeads || maxSplits != s.maxSplits || s.partialPrefix[heads] > uint64_t(2) * c.aicNum ||
        (heads == 0 ? s.vectors != 0 : s.vectors == 0))
        return Status::InvalidEndpoints;
    for (uint32_t i = 0; i < 26; ++i) {
        if (i >= heads && (read(base + 86 + i) || read(base + 112 + i) || read(base + 138 + i) ||
                           read(base + 190 + i) || read(base + 216 + i)))
            return Status::InvalidEndpoints;
        uint32_t startSplit = 0;
        if (i && i < s.usedCoreNum && s.nEnd[i - 1]) {
            MlaAutoSplitBsndTask t{};
            t.batch = s.bEnd[i - 1];
            t.m = s.mEnd[i - 1];
            Geometry<IsPaged>(read, s, w, t);
            if constexpr (Prefill) {
                // The same fragment traversal supplies its inverse owner ->
                // split mapping; do not enumerate every split and rescan all
                // cores just to validate this one ordinal.
                Parts(s, t, i, &startSplit);
            } else {
                const uint32_t first = t.s2Begin, stop = t.s2End, count = Parts(s, t);
                for (uint32_t split = 0; split < count; ++split) {
                    t.split = split;
                    t.s2Begin = first;
                    t.s2End = stop;
                    Parts(s, t);
                    if (t.owner == i)
                        startSplit = split;
                }
            }
        }
        if (read(base + 164 + i) != startSplit)
            return Status::InvalidEndpoints;
    }
    uint32_t end = 0;
    for (uint32_t i = 0; i < 52; ++i) {
        const uint32_t h = read(base + 242 + i), part = read(base + 294 + i);
        if (i >= s.vectors) {
            if (h || part)
                return Status::InvalidEndpoints;
        } else {
            if (h >= heads || part >= fdParts[h + 1] - fdParts[h] || fdParts[h] + part + 1 <= end)
                return Status::InvalidEndpoints;
            end = fdParts[h] + part + 1;
        }
    }
    if (end != fdParts[heads])
        return Status::InvalidEndpoints;
    if (!empty && !Prefill) {
        const uint64_t rows = c.aicNum == 1 ? 0 : uint64_t(2) * c.aicNum * a.mBaseSize;
        const uint64_t reserve =
            2 * uint64_t(c.aicNum) * (6 * uint64_t(a.mm1ResSize) + 8 * uint64_t(a.mm2ResSize) + 8 * a.mBaseSize);
        s.outOffset = (reserve + 31) / 32 * 32;
        s.sumOffset = s.outOffset + rows * 512 * 4;
        s.maxOffset = s.sumOffset + rows * 8 * 4;
        s.bytes = s.maxOffset + rows * 8 * 4;
        if (s.bytes > c.workspaceBytes)
            return Status::InvalidWorkspace;
    }
    output = s;
    return Status::Success;
}

template <bool IsTnd, uint32_t Version, bool IsKvTnd = IsTnd, uint32_t SoftcapVersion = kMlaSoftcapAutoSplitBsndVersion,
          typename Reader, typename Arrays, typename Context>
MLA_AUTO_INLINE MlaMetadataReadStatus ReadAutoSplitMetadata(const Reader& read, uint32_t words, const Context& context,
                                                            const MlaUsedMetadataInputs& inputs, const Arrays& arrays,
                                                            MlaAutoSplitBsndSchedule& output)
{
    using Status = MlaMetadataReadStatus;
    using namespace auto_split_detail;
    const auto& c = context.used;
    if (c.batch > (UINT32_MAX - kMlaAutoSplitFixedWords) / 10)
        return Status::InvalidBuffer;
    uint32_t version = Version;
    if constexpr (Version == kMlaAutoSplitBsndVersion || IsTnd) {
        if (context.hasSoftcap > 1)
            return Status::ShapeMismatch;
        if (context.hasSoftcap)
            version = SoftcapVersion;
    }
    auto status = ReadMetadataHeader(read, words, kMlaAutoSplitFixedWords + 10 * c.batch, version, version, c.dtype);
    if (status != Status::Success)
        return status;
    if (c.query.layout != uint32_t(IsTnd) || c.kv.layout != uint32_t(IsKvTnd))
        return Status::ShapeMismatch;
    const uint32_t base = Base(c.batch);
    const auto& w = context.window;
    if (w.kind > 2 || w.left < -1 || w.right < -1 || w.left > int64_t(c.kv.maxLength) - 1 ||
        w.right > (c.query.maxLength == 0 ? 0 : int64_t(c.query.maxLength) - 1) || read(base) != w.kind ||
        int32_t(read(base + 1)) != w.left || int32_t(read(base + 2)) != w.right)
        return Status::ShapeMismatch;
    if ((w.kind == 0 && (w.left != int64_t(c.kv.maxLength) - 1 || w.right != int64_t(c.query.maxLength) - 1)) ||
        (w.kind == 1 && (w.left != int64_t(c.kv.maxLength) - 1 || w.right != 0)) ||
        (w.kind == 2 && c.query.maxLength && c.kv.maxLength && (w.left < 0 || w.right < 0)))
        return Status::ShapeMismatch;
    MlaAutoSplitBsndSchedule s{};
    status = ReadUsedMetadata(UsedView<Reader>{read, base}, base, c, inputs, arrays, s.used);
    if (status != Status::Success)
        return status;
    return ReadAutoSplitPartitions<IsTnd>(read, c, w, s.used, output);
}

// Keep the established v18 entry point and BSND wire identity unchanged.
template <typename Reader, typename Arrays>
MLA_AUTO_INLINE MlaMetadataReadStatus ReadAutoSplitBsndMetadata(const Reader& read, uint32_t words,
                                                                const MlaAutoSplitBsndContext& context,
                                                                const MlaUsedMetadataInputs& inputs,
                                                                const Arrays& arrays, MlaAutoSplitBsndSchedule& output)
{
    return ReadAutoSplitMetadata<false, kMlaAutoSplitBsndVersion>(read, words, context, inputs, arrays, output);
}

// Queries require an unchanged, successfully validated record and context.
// Like the CPU plan, every failure leaves the caller's output untouched.
template <bool IsPaged = false, typename Reader>
MLA_AUTO_INLINE bool AutoSplitBsndTask(const Reader& read, const MlaAutoSplitBsndSchedule& s,
                                       const MlaWindowKernelParams& window, uint64_t group, uint32_t split,
                                       MlaAutoSplitBsndTask& output)
{
    using namespace auto_split_detail;
    if (group >= s.groups)
        return false;
    const auto& a = s.used.tiling;
    const uint32_t base = Base<IsPaged>(a.batch), prefix = base + kMlaAutoSplitTailPrefix;
    uint32_t lo = 0, hi = a.batch;
    while (lo < hi) {
        const uint32_t mid = lo + (hi - lo) / 2;
        if (ReadMetadataWide(read, prefix + 2 * (mid + 1)) <= group)
            lo = mid + 1;
        else
            hi = mid;
    }
    MlaAutoSplitBsndTask t{};
    t.group = group;
    t.slot = UINT64_MAX;
    t.fdHead = UINT32_MAX;
    t.batch = lo;
    t.m = group - ReadMetadataWide(read, prefix + 2 * lo);
    t.split = split;
    Geometry<IsPaged>(read, s, window, t);
    const bool empty = t.s2Begin == t.s2End;
    const uint32_t count = empty ? 1 : Parts(s, t);
    if (split >= count)
        return false;
    t.splitCount = count;
    t.kind = empty ? 2 : count > 1 ? 1 : 0;
    if (empty) {
        t.owner = group % s.aicNum;
    } else {
        const uint32_t begin = t.s2Begin * a.s2BaseSize, end = t.s2End * a.s2BaseSize;
        t.kvBegin = t.kvBegin > begin ? t.kvBegin : begin;
        t.kvEnd = t.kvEnd < end ? t.kvEnd : end;
    }
    if (t.kind == 1) {
        for (uint32_t h = 0; h < s.fdHeads; ++h)
            if (read(base + 86 + h) == t.batch && read(base + 112 + h) == t.m)
                t.fdHead = h;
        if (t.fdHead == UINT32_MAX)
            return false;
        t.slot = s.partialPrefix[t.fdHead] + split;
        Offsets(s, t.slot * a.mBaseSize, t.outOffset, t.sumOffset, t.maxOffset);
    }
    output = t;
    return true;
}
template <bool IsPaged = false, typename Reader>
MLA_AUTO_INLINE bool AutoSplitBsndMerge(const Reader& read, const MlaAutoSplitBsndSchedule& s, uint32_t head,
                                        uint32_t part, MlaAutoSplitBsndMerge& output)
{
    using namespace auto_split_detail;
    const auto& a = s.used.tiling;
    const uint32_t base = Base<IsPaged>(a.batch);
    if (head >= s.fdHeads || part >= read(base + 190 + head))
        return false;
    MlaAutoSplitBsndMerge r{};
    r.head = head;
    r.part = part;
    r.batch = read(base + 86 + head);
    r.m = read(base + 112 + head);
    r.group = ReadMetadataWide(read, base + kMlaAutoSplitTailPrefix + 2 * r.batch) + r.m;
    r.firstSlot = s.partialPrefix[head];
    r.splitCount = read(base + 138 + head);
    for (uint32_t i = 0; i < s.vectors; ++i) {
        const uint32_t h = read(base + 242 + i), p = read(base + 294 + i);
        if (head < h || (head == h && part <= p)) {
            r.owner = i;
            break;
        }
    }
    r.rowBegin = uint64_t(r.m) * a.mBaseSize + part * 8;
    const uint64_t end = uint64_t(QueryLength<IsPaged>(read, r.batch)) * a.heads;
    const uint64_t blockEnd = uint64_t(r.m + 1) * a.mBaseSize;
    const uint64_t limit = end < blockEnd ? end : blockEnd;
    r.rowEnd = uint64_t(r.rowBegin) + 8 < limit ? uint64_t(r.rowBegin) + 8 : limit;
    Offsets(s, r.firstSlot * a.mBaseSize + part * 8, r.outOffset, r.sumOffset, r.maxOffset);
    output = r;
    return true;
}
MLA_AUTO_INLINE bool AutoSplitBsndRow(const MlaAutoSplitBsndSchedule& s, const MlaAutoSplitBsndTask& t, uint32_t row,
                                      MlaAutoSplitBsndRow& output)
{
    if (row >= t.rowEnd - t.rowBegin)
        return false;
    const auto& a = s.used.tiling;
    const uint64_t flat = uint64_t(t.rowBegin) + row;
    MlaAutoSplitBsndRow r{};
    r.out = (uint64_t(t.batch) * a.sq * a.heads + flat) * 512 * 2;
    r.lse = ((uint64_t(t.batch) * a.heads + flat % a.heads) * a.sq + flat / a.heads) * 4;
    if (s.aicNum > 1) {
        r.accumulatorLse = uint64_t(t.split) * a.batch * a.heads * a.sq * 4 + r.lse;
        r.accumulatorOut = r.accumulatorLse * 512;
    }
    r.partialOut = r.partialSum = r.partialMax = UINT64_MAX;
    if (t.kind == 1)
        auto_split_detail::Offsets(s, t.slot * a.mBaseSize + row, r.partialOut, r.partialSum, r.partialMax);
    output = r;
    return true;
}
} // namespace flash_attn::mla
#undef MLA_AUTO_INLINE

// Independent CPU/AICore v20 consumer; never invokes the producer or CPU plan.
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

// Independent CPU/AICore v24 validation. No producer or CPU plan calls.
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

// Independent CPU/AICore v26 validation. No producer or CPU plan calls.
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
    if (context.hasSoftcap > 1)
        return Status::ShapeMismatch;
    const uint32_t version = context.hasSoftcap ? kMlaSoftcapAutoSplitPagedTndVersion : kMlaAutoSplitPagedTndVersion;
    auto status =
        ReadMetadataHeader(read, words, kMlaAutoSplitPagedTndFixedWords + 7 * c.batch, version, version, c.dtype);
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

// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.

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
// Translate only scheduling coordinates (b,kvHead) into bN2. The launcher
// retains the real tensor layout and rebases TND allocation starts per task.
template <typename Reader>
struct PrefillPartitionView {
    const Reader& read;
    const MlaPrefillBsndContext& c;
    MLA_PREFILL_INLINE uint32_t operator()(uint32_t i) const
    {
        const uint32_t virtualBatch = c.batch * c.kvHeads;
        const uint32_t base = kMlaUsedMetadataPrefixWords + 8 * virtualBatch;
        if (i >= base)
            return read(PrefillTailWord(c) + i - base);
        const uint32_t slot = i - kMlaUsedMetadataPrefixWords;
        return read(PrefillLengthWord(c, (slot / 8) / c.kvHeads) + (slot % 8 == 7));
    }
};
template <typename Reader>
struct PrefillAllocationView {
    const Reader& read;
    MLA_PREFILL_INLINE uint32_t operator()(uint32_t i) const
    {
        return read(24 + i - kMlaUsedMetadataPrefixWords);
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
    const uint32_t version =
        c.hasSoftcap ? (c.pageSize   ? (c.tnd ? kMlaSoftcapPrefillPagedTndVersion : kMlaSoftcapPrefillPagedBsndVersion)
                        : c.tnd == 2 ? kMlaSoftcapPrefillMixedVersion
                        : c.tnd == 1 ? kMlaSoftcapPrefillTndVersion
                                     : kMlaSoftcapPrefillBsndVersion)
        : c.pageSize ? (c.tnd ? 31 : 30)
        : c.tnd == 2 ? 29
        : c.tnd      ? 28
                     : 27;
    const auto status = ReadMetadataHeader(PrefillHeaderView<Reader>{read}, words, l.words, version, version, c.dtype);
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
    if (c.pageSize && !c.tnd && (read(20) != c.pageSize || read(21) || read(22) || read(23)))
        return S::ShapeMismatch;
    if (c.tnd) {
        if (read(20) != c.totalQ || read(21) != (c.pageSize ? c.pageSize : c.totalK) || read(22) || read(23))
            return S::ShapeMismatch;
        MlaUsedMetadataContext geometry{};
        geometry.batch = c.batch;
        geometry.usedFlags = c.usedFlags;
        geometry.query = {1, {c.totalQ, 0}, c.sq};
        const uint64_t totalK = c.tnd == 1 ? c.totalK : uint64_t(c.batch) * c.sk;
        geometry.kv = {c.tnd == 1 ? 1U : 0U, {uint32_t(totalK), uint32_t(totalK >> 32)}, c.sk};
        for (uint32_t axis = 0; axis < 2; ++axis) {
            uint32_t maximum;
            const auto checked =
                ReadUsedMetadataAxis(PrefillAllocationView<Reader>{read}, arrays, axis, geometry, maximum);
            if (checked != S::Success)
                return checked;
        }
    } else
        for (uint32_t b = 0; b < c.batch; ++b) {
            const uint32_t q = c.usedFlags & 1 ? arrays(0, b) : c.sq;
            const uint32_t k = c.usedFlags & 2 ? arrays(1, b) : c.sk;
            if (q > c.sq || k > c.sk || read(PrefillLengthWord(c, b)) != q || read(PrefillLengthWord(c, b) + 1) != k)
                return S::InvalidUsedLengths;
        }
    const uint32_t tail = PrefillTailWord(c);
    if (read(tail) != c.window.kind || int32_t(read(tail + 1)) != c.window.left ||
        int32_t(read(tail + 2)) != c.window.right)
        return S::ShapeMismatch;
    MlaUsedKernelSchedule used{};
    auto& t = used.tiling;
    t.batch = c.batch * c.kvHeads;
    t.sq = c.tnd && !c.totalQ ? 0 : c.sq;
    t.sk = c.tnd == 1 && !c.totalK ? 0 : c.sk;
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
// Page IDs are invocation data: validate every effective page, including
// batches with no Q work, before any output write or synchronization.
template <typename Reader, typename Pages>
MLA_PREFILL_INLINE MlaMetadataReadStatus ReadPrefillPagedCache(const Reader& read, const MlaPrefillBsndContext& c,
                                                               const MlaPagedCacheInputs& cache, const Pages& pages)
{
    using S = MlaMetadataReadStatus;
    if (!c.pageSize)
        return S::ShapeMismatch;
    MlaPagedMetadataContext geometry{};
    geometry.batch = c.batch;
    geometry.pageSize = c.pageSize;
    geometry.logicalKvCapacity = c.sk;
    const auto status = CheckPagedCacheDescriptor(geometry, cache, c.kvHeads, 192, 128);
    if (status != S::Success)
        return status;
    for (uint32_t b = 0; b < c.batch; ++b) {
        const uint32_t length = read(PrefillLengthWord(c, b) + 1);
        if (length > c.sk)
            return S::InvalidCacheLengths;
        const uint32_t count = (length + c.pageSize - 1) / c.pageSize;
        for (uint32_t i = 0; i < count; ++i) {
            const int32_t id = int32_t(pages(uint64_t(b) * cache.layout.tableRowStride + i));
            if (id < 0 || uint64_t(id) >= cache.layout.physicalPages)
                return S::InvalidPageTable;
        }
    }
    return S::Success;
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
        SplitBsndFragment(read(PrefillLengthWord(c, b)), read(PrefillLengthWord(c, b) + 1), a.heads, t.m, a.mBaseSize,
                          a.s2BaseSize, c.window, c.splits, split, t.s2Begin, t.s2End);
    }
    return true;
}
} // namespace flash_attn::mla
#undef MLA_PREFILL_INLINE

// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.

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
