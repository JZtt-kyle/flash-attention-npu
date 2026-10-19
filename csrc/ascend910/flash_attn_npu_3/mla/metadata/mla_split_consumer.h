// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
#pragma once
#include "mla_split.h"
#include "mla_sequence_consumer.h"
#include "mla_paged_consumer.h"

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
