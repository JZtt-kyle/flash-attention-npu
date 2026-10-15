// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
// Independent v18/v20/v22 safety validation and shared v24 partition checks. No producer, CPU plan or SplitCore call.
// Validates coverage/addressing, not the optimality/authenticity of a partition.
#pragma once
#include "mla_auto_split_bsnd_metadata.h"
#include "mla_used_metadata_consumer.h"
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

template <bool IsTnd, uint32_t Version, bool IsKvTnd = IsTnd, typename Reader, typename Arrays, typename Context>
MLA_AUTO_INLINE MlaMetadataReadStatus ReadAutoSplitMetadata(const Reader& read, uint32_t words, const Context& context,
                                                            const MlaUsedMetadataInputs& inputs, const Arrays& arrays,
                                                            MlaAutoSplitBsndSchedule& output)
{
    using Status = MlaMetadataReadStatus;
    using namespace auto_split_detail;
    const auto& c = context.used;
    if (c.batch > (UINT32_MAX - kMlaAutoSplitFixedWords) / 10)
        return Status::InvalidBuffer;
    auto status = ReadMetadataHeader(read, words, kMlaAutoSplitFixedWords + 10 * c.batch, Version, Version, c.dtype);
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
