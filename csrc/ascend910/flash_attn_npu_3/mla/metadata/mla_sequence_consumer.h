#pragma once
#include "mla_sequence.h"
#include "mla_consumer.h"
#include "mla_mask.h"

// v2 BSND length snapshot validation. Schedule/resource checks and their CANN
// provenance are shared with metadata/mla_consumer.h. No attention launch here.
#if defined(__NPU_ARCH__)
#define MLA_BSND_INLINE __aicore__ inline
#else
#define MLA_BSND_INLINE inline
#endif

namespace flash_attn::mla {

template <typename Reader, typename LengthReader>
struct BsndMetadataLengths {
    const Reader& record;
    const LengthReader& current;
    bool explicitLengths;

    MLA_BSND_INLINE bool Check(uint32_t batch, uint32_t capacity) const
    {
        for (uint32_t b = 0; b < batch; ++b) {
            const uint32_t snapshot = record(kMlaBsndKvLengthMetadataPrefixWords + b);
            // Unsigned comparison also rejects negative int32 lengths.
            if (snapshot > capacity || snapshot != (explicitLengths ? current(b) : capacity)) {
                return false;
            }
        }
        return true;
    }

    MLA_BSND_INLINE bool Covers(uint64_t end, uint64_t tasks, uint32_t mBlocks, uint32_t batch) const
    {
        if (end == tasks) {
            return true;
        }
        // The pinned source may omit a zero-KV suffix. It must start at a
        // sample boundary and every omitted sample must be empty; checking
        // just the last sample would allow a hole with nonzero KV work.
        if (end % mBlocks != 0) {
            return false;
        }
        for (uint32_t b = end / mBlocks; b < batch; ++b) {
            if (record(kMlaBsndKvLengthMetadataPrefixWords + b) != 0) {
                return false;
            }
        }
        return true;
    }
};

// Both readers and the static context belong to this call. The caller validates
// their addresses/alignment and keeps buffers immutable until all cores finish.
// Missing current lengths means full capacity, not "trust the snapshot".
// Success returns geometry only: a short-length schedule cannot drive MlaDense.
template <typename Reader, typename LengthReader>
MLA_BSND_INLINE MlaMetadataReadStatus ReadBsndKvLengthMetadata(const Reader& read, uint32_t words,
                                                               const MlaDenseMetadataContext& context,
                                                               const LengthReader& current, bool explicitLengths,
                                                               uint64_t lengthCount, MlaDenseKernelSchedule& output)
{
    using Status = MlaMetadataReadStatus;
    if (context.batch > UINT32_MAX - kMlaBsndKvLengthMetadataPrefixWords) {
        return Status::InvalidBuffer;
    }
    const uint32_t expected = kMlaBsndKvLengthMetadataPrefixWords + context.batch;
    const auto status = ReadMetadataHeader(read, words, expected, kMlaBsndKvLengthMetadataVersion,
                                           kMlaMetadataBsndKvLengths, context.dtype);
    if (status != Status::Success) {
        return status;
    }
    if (lengthCount != (explicitLengths ? uint64_t(context.batch) : 0)) {
        return Status::InvalidCacheLengths;
    }
    return ReadMetadataSchedule(read, context,
                                BsndMetadataLengths<Reader, LengthReader>{read, current, explicitLengths}, output);
}
} // namespace flash_attn::mla
#undef MLA_BSND_INLINE

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// The upstream license is reproduced in mla/LICENSE.cann.
// Bounds and variable-M endpoint rules follow tiling/mla_sequence.cpp and its
// pinned CANN sources. This validates the wire/addressing contract, not the
// optimality or authenticity of a split. No host scheduler runs on the device.
#if defined(__NPU_ARCH__)
#define MLA_TND_INLINE __aicore__ inline
#else
#define MLA_TND_INLINE inline
#endif

namespace flash_attn::mla {

// A distinct type prevents passing TND batch coordinates to a BSND Kernel.
// tiling.sq/sk are observed maxima, never per-sample lengths or tensor strides.
struct MlaTndKernelSchedule {
    MlaDenseKernelSchedule tiling;
    uint32_t totalQuery, totalKv;
};

template <typename Reader, typename QueryReader, typename KvReader>
MLA_TND_INLINE MlaMetadataReadStatus ReadTndMetadata(const Reader& read, uint32_t words, const MlaTndMetadataContext& c,
                                                     const QueryReader& query, uint64_t queryCount, const KvReader& kv,
                                                     uint64_t kvCount, MlaTndKernelSchedule& output)
{
    using Status = MlaMetadataReadStatus;
    if (c.batch > (UINT32_MAX - kMlaTndMetadataPrefixWords) / kMlaTndMetadataSequenceWords) {
        return Status::InvalidBuffer;
    }
    auto status = ReadMetadataHeader(read, words, kMlaTndMetadataPrefixWords + kMlaTndMetadataSequenceWords * c.batch,
                                     kMlaTndMetadataVersion, kMlaMetadataCompactTnd, c.dtype);
    if (status != Status::Success) {
        return status;
    }
    if (queryCount != uint64_t(c.batch) + 1 || kvCount != uint64_t(c.batch) + 1) {
        return Status::InvalidPrefixes;
    }
#define MLA_TND_WORD(field) read(offsetof(MlaTndMetadataPrefix, field) / sizeof(uint32_t))
    if (c.heads == 0 || c.heads > kMlaMaxQueryHeads || c.totalQuery > INT32_MAX || c.totalKv > INT32_MAX ||
        c.maxQueryLength > kMlaMaxQueryLength || c.maxKvLength > kMlaMaxKvLength ||
        uint64_t(c.maxQueryLength) * c.heads > kMlaMaxDenseRows ||
        uint64_t(c.totalQuery) * c.heads > kMlaMaxBatchRows || MLA_TND_WORD(totalQuery) != c.totalQuery ||
        MLA_TND_WORD(totalKv) != c.totalKv || MLA_TND_WORD(maxQueryLength) != c.maxQueryLength ||
        MLA_TND_WORD(maxKvLength) != c.maxKvLength || MLA_TND_WORD(schedule.bSize) != c.batch ||
        MLA_TND_WORD(schedule.gSize) != c.heads || MLA_TND_WORD(schedule.n2Size) != 1 ||
        MLA_TND_WORD(schedule.headDim) != 512 || MLA_TND_WORD(schedule.headDimRope) != 64) {
        return Status::ShapeMismatch;
    }
    // Compare both complete current prefixes to contiguous snapshot spans.
    // Bounds before subtraction reject negative int32 values and wrapping sums.
    if (query(0) != 0 || kv(0) != 0) {
        return Status::InvalidPrefixes;
    }
    uint32_t qEnd = 0, kEnd = 0, maxQ = 0, maxK = 0;
    for (uint32_t b = 0; b < c.batch; ++b) {
        const uint32_t word = kMlaTndMetadataPrefixWords + kMlaTndMetadataSequenceWords * b;
        const uint32_t q = read(word + 2), k = read(word + 3);
        if (read(word) != qEnd || read(word + 1) != kEnd || q > c.maxQueryLength || k > c.maxKvLength ||
            q > c.totalQuery - qEnd || k > c.totalKv - kEnd) {
            return Status::InvalidPrefixes;
        }
        qEnd += q;
        kEnd += k;
        if (query(b + 1) != qEnd || kv(b + 1) != kEnd) {
            return Status::InvalidPrefixes;
        }
        maxQ = q > maxQ ? q : maxQ;
        maxK = k > maxK ? k : maxK;
    }
    if (qEnd != c.totalQuery || kEnd != c.totalKv) {
        return Status::InvalidPrefixes;
    }
    if (MLA_TND_WORD(schedule.s1Size) != maxQ || MLA_TND_WORD(schedule.s2Size) != maxK) {
        return Status::ShapeMismatch;
    }
    MlaTndKernelSchedule decoded{};
    decoded.totalQuery = c.totalQuery;
    decoded.totalKv = c.totalKv;
    auto& s = decoded.tiling;
    s.batch = c.batch;
    s.sq = maxQ;
    s.sk = maxK;
    s.heads = c.heads;
    status = ReadMetadataPlatform(read, c, s);
    if (status != Status::Success) {
        return status;
    }
    s.mBaseSize = MLA_TND_WORD(schedule.mBaseSize);
    s.s2BaseSize = MLA_TND_WORD(schedule.s2BaseSize);
    if (c.batch == 0 || c.totalQuery == 0 || c.totalKv == 0) {
        status = ReadEmptyMetadataSchedule(read, c, s);
        if (status == Status::Success) {
            output = decoded;
        }
        return status;
    }
    const uint64_t rows = uint64_t(maxQ) * c.heads;
    if (s.mBaseSize < 128 || s.mBaseSize > 256 || (s.mBaseSize != 256 && rows % s.mBaseSize != 0) ||
        s.s2BaseSize != (maxK < 512 ? maxK : 512) || MLA_TND_WORD(schedule.numOfFdHead) != 0 ||
        MLA_TND_WORD(schedule.maxS2SplitNum) != 0 || MLA_TND_WORD(schedule.usedVecNumOfFd) != 0) {
        return Status::InvalidTiling;
    }
    // Per-sample M/S2 counts, not a rectangular B * maxQ * maxKV bound.
    uint64_t tiles = 0;
    for (uint32_t b = 0; b < c.batch; ++b) {
        const uint32_t word = kMlaTndMetadataPrefixWords + kMlaTndMetadataSequenceWords * b;
        const uint64_t m = (uint64_t(read(word + 2)) * c.heads + s.mBaseSize - 1) / s.mBaseSize;
        tiles += m * ((read(word + 3) + s.s2BaseSize - 1) / s.s2BaseSize);
        if (tiles > UINT32_MAX) {
            return Status::InvalidEndpoints;
        }
    }
    uint32_t previousBatch = 0, previousM = 0;
    for (uint32_t i = 0; i < s.usedCoreNum; ++i) {
        const uint32_t b = read(offsetof(MlaTndMetadataPrefix, schedule.bN2End) / sizeof(uint32_t) + i);
        const uint32_t m = read(offsetof(MlaTndMetadataPrefix, schedule.gS1End) / sizeof(uint32_t) + i);
        const uint32_t n = read(offsetof(MlaTndMetadataPrefix, schedule.s2End) / sizeof(uint32_t) + i);
        if (b > c.batch || n != 0 || b < previousBatch || (b == previousBatch && m <= previousM)) {
            return Status::InvalidEndpoints;
        }
        uint32_t mBlocks = 0;
        if (b < c.batch) {
            const uint32_t q = read(kMlaTndMetadataPrefixWords + kMlaTndMetadataSequenceWords * b + 2);
            mBlocks = (uint64_t(q) * c.heads + s.mBaseSize - 1) / s.mBaseSize;
        }
        // Preserve source batch coordinates across empty Q samples. They may
        // advance lexicographically while the count of physical M blocks stays 0.
        if ((b == c.batch && m != 0) || (b < c.batch && m >= (mBlocks == 0 ? 1 : mBlocks))) {
            return Status::InvalidEndpoints;
        }
        s.bN2End[i] = previousBatch = b;
        s.gS1End[i] = previousM = m;
        s.s2End[i] = n;
    }
    for (uint32_t b = previousBatch; b < c.batch; ++b) {
        const uint32_t word = kMlaTndMetadataPrefixWords + kMlaTndMetadataSequenceWords * b;
        if (read(word + 2) != 0 && read(word + 3) != 0) {
            return Status::InvalidEndpoints;
        }
    }
    status = ReadMetadataWorkspace(read, c, s, rows);
    if (status == Status::Success) {
        output = decoded;
    }
    return status;
#undef MLA_TND_WORD
}
} // namespace flash_attn::mla
#undef MLA_TND_INLINE

// Source bounds and variable-M endpoints: tiling/mla_sequence.cpp and its pinned
// CANN sources; upstream license: LICENSE.cann. As with the compact decoder,
// this checks addressing/resources, not split optimality or authenticity.
#if defined(__NPU_ARCH__)
#define MLA_MIXED_INLINE __aicore__ inline
#else
#define MLA_MIXED_INLINE inline
#endif

namespace flash_attn::mla {

// Distinct from compact TND and BSND schedules. sq is the observed Q maximum;
// sk is physical KV capacity, never the observed maximum or a used length.
struct MlaTndBsndKernelSchedule {
    MlaDenseKernelSchedule tiling;
    uint32_t totalQuery;
    uint64_t totalKvCapacity;
};

template <typename Reader, typename QueryReader, typename KvReader>
MLA_MIXED_INLINE MlaMetadataReadStatus ReadTndBsndMetadata(const Reader& read, uint32_t words,
                                                           const MlaTndBsndMetadataContext& c, const QueryReader& query,
                                                           uint64_t queryCount, const KvReader& kv, uint64_t kvCount,
                                                           MlaTndBsndKernelSchedule& output)
{
    using Status = MlaMetadataReadStatus;
    if (c.batch > (UINT32_MAX - kMlaTndBsndMetadataPrefixWords) / kMlaTndBsndMetadataSequenceWords) {
        return Status::InvalidBuffer;
    }
    auto status =
        ReadMetadataHeader(read, words, kMlaTndBsndMetadataPrefixWords + kMlaTndBsndMetadataSequenceWords * c.batch,
                           kMlaTndBsndMetadataVersion, kMlaMetadataTndBsnd, c.dtype);
    if (status != Status::Success) {
        return status;
    }
    if (queryCount != uint64_t(c.batch) + 1) {
        return Status::InvalidPrefixes;
    }
    if (kvCount != c.batch) {
        return Status::InvalidCacheLengths;
    }
#define MLA_MIXED_WORD(field) read(offsetof(MlaTndBsndMetadataPrefix, field) / sizeof(uint32_t))
    const uint64_t totalKvCapacity = uint64_t(c.batch) * c.kvCapacity;
    if (c.heads == 0 || c.heads > kMlaMaxQueryHeads || c.totalQuery > INT32_MAX ||
        c.maxQueryLength > kMlaMaxQueryLength || c.kvCapacity > kMlaMaxKvLength ||
        uint64_t(c.maxQueryLength) * c.heads > kMlaMaxDenseRows ||
        uint64_t(c.totalQuery) * c.heads > kMlaMaxBatchRows || MLA_MIXED_WORD(totalQuery) != c.totalQuery ||
        MLA_MIXED_WORD(maxQueryLength) != c.maxQueryLength || MLA_MIXED_WORD(kvCapacity) != c.kvCapacity ||
        ReadMetadataWide(read, offsetof(MlaTndBsndMetadataPrefix, totalKvCapacity) / sizeof(uint32_t)) !=
            totalKvCapacity ||
        MLA_MIXED_WORD(schedule.bSize) != c.batch || MLA_MIXED_WORD(schedule.gSize) != c.heads ||
        MLA_MIXED_WORD(schedule.n2Size) != 1 || MLA_MIXED_WORD(schedule.headDim) != 512 ||
        MLA_MIXED_WORD(schedule.headDimRope) != 64 || MLA_MIXED_WORD(schedule.s2Size) != c.kvCapacity) {
        return Status::ShapeMismatch;
    }
    // B=0 still requires Q=[0]. KV is noncumulative; no KV prefix is read.
    if (query(0) != 0) {
        return Status::InvalidPrefixes;
    }
    uint32_t qEnd = 0, maxQ = 0;
    for (uint32_t b = 0; b < c.batch; ++b) {
        const uint32_t word = kMlaTndBsndMetadataPrefixWords + kMlaTndBsndMetadataSequenceWords * b;
        const uint32_t q = read(word + 3), k = read(word + 4);
        if (read(word) != qEnd || q > c.maxQueryLength || q > c.totalQuery - qEnd) {
            return Status::InvalidPrefixes;
        }
        qEnd += q;
        if (query(b + 1) != qEnd) {
            return Status::InvalidPrefixes;
        }
        if (ReadMetadataWide(read, word + 1) != uint64_t(b) * c.kvCapacity || k > c.kvCapacity || kv(b) != k) {
            return Status::InvalidCacheLengths;
        }
        maxQ = q > maxQ ? q : maxQ;
    }
    if (qEnd != c.totalQuery) {
        return Status::InvalidPrefixes;
    }
    if (MLA_MIXED_WORD(schedule.s1Size) != maxQ) {
        return Status::ShapeMismatch;
    }
    MlaTndBsndKernelSchedule decoded{};
    decoded.totalQuery = c.totalQuery;
    decoded.totalKvCapacity = totalKvCapacity;
    auto& s = decoded.tiling;
    s.batch = c.batch;
    s.sq = maxQ;
    s.sk = c.kvCapacity;
    s.heads = c.heads;
    status = ReadMetadataPlatform(read, c, s);
    if (status != Status::Success) {
        return status;
    }
    s.mBaseSize = MLA_MIXED_WORD(schedule.mBaseSize);
    s.s2BaseSize = MLA_MIXED_WORD(schedule.s2BaseSize);
    // All-zero used KV in nonempty physical capacity must retain a complete
    // schedule/workspace. Only physical emptiness uses the empty protocol.
    if (c.batch == 0 || c.totalQuery == 0 || c.kvCapacity == 0) {
        status = ReadEmptyMetadataSchedule(read, c, s);
        if (status == Status::Success) {
            output = decoded;
        }
        return status;
    }
    const uint64_t rows = uint64_t(maxQ) * c.heads;
    if (s.mBaseSize < 128 || s.mBaseSize > 256 || (s.mBaseSize != 256 && rows % s.mBaseSize != 0) ||
        s.s2BaseSize != (c.kvCapacity < 512 ? c.kvCapacity : 512) || MLA_MIXED_WORD(schedule.numOfFdHead) != 0 ||
        MLA_MIXED_WORD(schedule.maxS2SplitNum) != 0 || MLA_MIXED_WORD(schedule.usedVecNumOfFd) != 0) {
        return Status::InvalidTiling;
    }
    uint64_t tiles = 0;
    for (uint32_t b = 0; b < c.batch; ++b) {
        const uint32_t word = kMlaTndBsndMetadataPrefixWords + kMlaTndBsndMetadataSequenceWords * b;
        const uint64_t m = (uint64_t(read(word + 3)) * c.heads + s.mBaseSize - 1) / s.mBaseSize;
        tiles += m * ((read(word + 4) + s.s2BaseSize - 1) / s.s2BaseSize);
        if (tiles > UINT32_MAX) {
            return Status::InvalidEndpoints;
        }
    }
    uint32_t previousBatch = 0, previousM = 0;
    for (uint32_t i = 0; i < s.usedCoreNum; ++i) {
        const uint32_t b = read(offsetof(MlaTndBsndMetadataPrefix, schedule.bN2End) / sizeof(uint32_t) + i);
        const uint32_t m = read(offsetof(MlaTndBsndMetadataPrefix, schedule.gS1End) / sizeof(uint32_t) + i);
        const uint32_t n = read(offsetof(MlaTndBsndMetadataPrefix, schedule.s2End) / sizeof(uint32_t) + i);
        if (b > c.batch || n != 0 || b < previousBatch || (b == previousBatch && m <= previousM)) {
            return Status::InvalidEndpoints;
        }
        uint32_t mBlocks = 0;
        if (b < c.batch) {
            const uint32_t q = read(kMlaTndBsndMetadataPrefixWords + kMlaTndBsndMetadataSequenceWords * b + 3);
            mBlocks = (uint64_t(q) * c.heads + s.mBaseSize - 1) / s.mBaseSize;
        }
        // Preserve source coordinates when advancing across a sample with no Q.
        if ((b == c.batch && m != 0) || (b < c.batch && m >= (mBlocks == 0 ? 1 : mBlocks))) {
            return Status::InvalidEndpoints;
        }
        s.bN2End[i] = previousBatch = b;
        s.gS1End[i] = previousM = m;
        s.s2End[i] = n;
    }
    // Every omitted sample, including a partially omitted last sample, must
    // have zero cost. An aggregate or final-sample-only check would miss holes.
    for (uint32_t b = previousBatch; b < c.batch; ++b) {
        const uint32_t word = kMlaTndBsndMetadataPrefixWords + kMlaTndBsndMetadataSequenceWords * b;
        if (read(word + 3) != 0 && read(word + 4) != 0) {
            return Status::InvalidEndpoints;
        }
    }
    status = ReadMetadataWorkspace(read, c, s, rows);
    if (status == Status::Success) {
        output = decoded;
    }
    return status;
#undef MLA_MIXED_WORD
}
} // namespace flash_attn::mla
#undef MLA_MIXED_INLINE

// Source geometry/variable-M endpoints: tiling/mla_sequence.cpp and its pinned
// CANN sources; upstream license: LICENSE.cann. Validate addressing/resources,
// not split optimality or authenticity (maxCost is diagnostic only).
#if defined(__NPU_ARCH__)
#define MLA_USED_INLINE __aicore__ inline
#else
#define MLA_USED_INLINE inline
#endif

namespace flash_attn::mla {

// Allocation geometry is distinct from effective lengths in the v5 spans.
// tiling.sq/sk retain allocation maxima, including when every used value is 0.
struct MlaUsedKernelSchedule {
    MlaDenseKernelSchedule tiling;
    MlaMetadataAllocation query, kv;
    uint32_t usedFlags;
};
static_assert(sizeof(MlaUsedKernelSchedule) == 98 * sizeof(uint32_t));

MLA_USED_INLINE uint64_t UsedMetadataTotal(const MlaMetadataAllocation& axis)
{
    return uint64_t(axis.totalTokens.low) | (uint64_t(axis.totalTokens.high) << 32);
}

MLA_USED_INLINE bool ValidUsedMetadataAddress(uint64_t address, uint64_t count)
{
    return (address != 0 || count == 0) && address % 4 == 0 && count <= (UINT64_MAX - address) / 4;
}

template <typename Reader, typename Arrays>
MLA_USED_INLINE MlaMetadataReadStatus ReadUsedMetadataAxis(const Reader& read, const Arrays& arrays, uint32_t axis,
                                                           const MlaUsedMetadataContext& c, uint32_t& maximum)
{
    using Status = MlaMetadataReadStatus;
    const auto& allocation = axis == 0 ? c.query : c.kv;
    const uint64_t total = UsedMetadataTotal(allocation);
    const bool tnd = allocation.layout == 1, used = (c.usedFlags & (1U << axis)) != 0;
    if (tnd && arrays(axis, 0) != 0) {
        return Status::InvalidPrefixes;
    }
    uint64_t start = 0;
    maximum = tnd ? 0 : allocation.maxLength;
    for (uint32_t b = 0; b < c.batch; ++b) {
        const uint32_t word = kMlaUsedMetadataPrefixWords + kMlaUsedMetadataSequenceWords * b;
        const uint64_t end = tnd ? uint64_t(arrays(axis, b + 1)) : start + allocation.maxLength;
        if (end < start || end > total || end - start > allocation.maxLength ||
            ReadMetadataWide(read, word + 2 * axis) != start || read(word + 4 + axis) != end - start) {
            return Status::InvalidPrefixes;
        }
        const uint32_t capacity = end - start;
        const uint32_t length = used ? arrays(2 + axis, b) : capacity;
        if (length > capacity || read(word + 6 + axis) != length) {
            return Status::InvalidUsedLengths;
        }
        maximum = capacity > maximum ? capacity : maximum;
        start = end;
    }
    return start == total ? Status::Success : Status::InvalidPrefixes;
}

// Readers return uint32 bits, so negative int32 lengths exceed their validated
// nonnegative capacity before subtraction or indexing. Check every descriptor
// before reading any length, and never write output until all checks succeed.
template <bool IsCausal = false, bool IsTndCausal = false, bool IsMixedCausal = false, bool IsWindow = false,
          bool IsTndWindow = false, bool IsMixedWindow = false, typename Reader, typename Arrays>
MLA_USED_INLINE MlaMetadataReadStatus ReadUsedMetadata(const Reader& read, uint32_t words,
                                                       const MlaUsedMetadataContext& c,
                                                       const MlaUsedMetadataInputs& inputs, const Arrays& arrays,
                                                       MlaUsedKernelSchedule& output, MlaWindowKernelParams window = {})
{
    using Status = MlaMetadataReadStatus;
    constexpr uint32_t tail = IsWindow ? kMlaWindowMetadataTailWords : 0;
    static_assert(!IsTndWindow || IsWindow);
    static_assert(!IsMixedWindow || (IsWindow && !IsTndWindow));
    static_assert(!IsWindow || (!IsCausal && !IsTndCausal && !IsMixedCausal));
    if (c.batch > (UINT32_MAX - kMlaUsedMetadataPrefixWords - tail) / kMlaUsedMetadataSequenceWords) {
        return Status::InvalidBuffer;
    }
    static_assert(!IsTndCausal || IsCausal);
    static_assert(!IsMixedCausal || (IsCausal && !IsTndCausal));
    // The launching kernel fixes its layout and protocol before reading GM.
    constexpr auto causalVersion = IsMixedCausal ? kMlaCausalMixedMetadataVersion
                                   : IsTndCausal ? kMlaCausalTndMetadataVersion
                                                 : kMlaCausalBsndMetadataVersion;
    constexpr auto causalKind = IsMixedCausal ? kMlaMetadataCausalMixed
                                : IsTndCausal ? kMlaMetadataCausalTnd
                                              : kMlaMetadataCausalBsnd;
    auto status =
        ReadMetadataHeader(read, words, kMlaUsedMetadataPrefixWords + kMlaUsedMetadataSequenceWords * c.batch + tail,
                           IsWindow   ? (IsMixedWindow ? kMlaWindowMixedMetadataVersion
                                         : IsTndWindow ? kMlaWindowTndMetadataVersion
                                                       : kMlaWindowBsndMetadataVersion)
                           : IsCausal ? causalVersion
                                      : kMlaUsedMetadataVersion,
                           IsWindow   ? (IsMixedWindow ? kMlaMetadataWindowMixed
                                         : IsTndWindow ? kMlaMetadataWindowTnd
                                                       : kMlaMetadataWindowBsnd)
                           : IsCausal ? causalKind
                                      : kMlaMetadataUsed,
                           c.dtype);
    if (status != Status::Success) {
        return status;
    }
    if constexpr (IsWindow) {
        // Match the invocation's host-normalized identity before any geometry,
        // empty shortcut, synchronization or write. Old records never reach here.
        if (window.kind > 2 || window.left < -1 || window.right < -1 || window.left > int64_t(c.kv.maxLength) - 1 ||
            window.right > (c.query.maxLength == 0 ? 0 : int64_t(c.query.maxLength) - 1) ||
            read(words - 3) != window.kind || int64_t(int32_t(read(words - 2))) != window.left ||
            int64_t(int32_t(read(words - 1))) != window.right) {
            return Status::ShapeMismatch;
        }
    }
#define MLA_USED_WORD(field) read(offsetof(MlaUsedMetadataPrefix, field) / sizeof(uint32_t))
    const uint64_t totalQ = UsedMetadataTotal(c.query);
    if ((IsCausal || IsWindow) &&
        (c.query.layout != uint32_t(IsTndCausal || IsMixedCausal || IsTndWindow || IsMixedWindow) ||
         c.kv.layout != uint32_t(IsTndCausal || IsTndWindow))) {
        return Status::ShapeMismatch;
    }
    if (c.query.layout > 1 || c.kv.layout > 1 || (c.query.layout == 0 && c.kv.layout != 0) || c.usedFlags > 3 ||
        c.heads == 0 || c.heads > kMlaMaxQueryHeads || c.query.maxLength > kMlaMaxQueryLength ||
        c.kv.maxLength > kMlaMaxKvLength || uint64_t(c.query.maxLength) * c.heads > kMlaMaxDenseRows ||
        totalQ > kMlaMaxBatchRows / c.heads || MLA_USED_WORD(usedFlags) != c.usedFlags ||
        MLA_USED_WORD(schedule.bSize) != c.batch || MLA_USED_WORD(schedule.gSize) != c.heads ||
        MLA_USED_WORD(schedule.n2Size) != 1 || MLA_USED_WORD(schedule.headDim) != 512 ||
        MLA_USED_WORD(schedule.headDimRope) != 64) {
        return Status::ShapeMismatch;
    }
    for (uint32_t axis = 0; axis < 2; ++axis) {
        const auto& allocation = axis == 0 ? c.query : c.kv;
        const auto& prefix = axis == 0 ? inputs.cuQuery : inputs.cuKv;
        const auto& used = axis == 0 ? inputs.usedQuery : inputs.usedKv;
        const uint32_t word = offsetof(MlaUsedMetadataPrefix, query) / sizeof(uint32_t) + 4 * axis;
        const uint64_t total = UsedMetadataTotal(allocation);
        const bool tnd = allocation.layout == 1;
        if ((tnd ? total > INT32_MAX : total != uint64_t(c.batch) * allocation.maxLength) ||
            read(word) != allocation.layout || ReadMetadataWide(read, word + 1) != total ||
            read(word + 3) != allocation.maxLength) {
            return Status::ShapeMismatch;
        }
        if (!ValidUsedMetadataAddress(prefix.address, prefix.count) ||
            (tnd ? prefix.address == 0 || prefix.count != uint64_t(c.batch) + 1
                 : prefix.address != 0 || prefix.count != 0)) {
            return Status::InvalidPrefixes;
        }
        if (!ValidUsedMetadataAddress(used.address, used.count) ||
            bool(c.usedFlags & (1U << axis)) != (used.address != 0) ||
            used.count != (used.address != 0 ? c.batch : 0)) {
            return Status::InvalidUsedLengths;
        }
    }
    uint32_t maximum[2];
    for (uint32_t axis = 0; axis < 2; ++axis) {
        status = ReadUsedMetadataAxis(read, arrays, axis, c, maximum[axis]);
        if (status != Status::Success) {
            return status;
        }
    }
    if (MLA_USED_WORD(schedule.s1Size) != maximum[0] || MLA_USED_WORD(schedule.s2Size) != maximum[1]) {
        return Status::ShapeMismatch;
    }
    MlaUsedKernelSchedule decoded{};
    decoded.query = c.query;
    decoded.kv = c.kv;
    decoded.usedFlags = c.usedFlags;
    auto& s = decoded.tiling;
    s.batch = c.batch;
    s.sq = maximum[0];
    s.sk = maximum[1];
    s.heads = c.heads;
    status = ReadMetadataPlatform(read, c, s);
    if (status != Status::Success) {
        return status;
    }
    s.mBaseSize = MLA_USED_WORD(schedule.mBaseSize);
    s.s2BaseSize = MLA_USED_WORD(schedule.s2BaseSize);
    if (c.batch == 0 || maximum[0] == 0 || maximum[1] == 0) {
        status = ReadEmptyMetadataSchedule(read, c, s);
        if (status == Status::Success) {
            output = decoded;
        }
        return status;
    }
    const uint64_t rows = uint64_t(maximum[0]) * c.heads;
    if (s.mBaseSize < 128 || s.mBaseSize > 256 || (s.mBaseSize != 256 && rows % s.mBaseSize != 0) ||
        s.s2BaseSize != (maximum[1] < 512 ? maximum[1] : 512) || MLA_USED_WORD(schedule.numOfFdHead) != 0 ||
        MLA_USED_WORD(schedule.maxS2SplitNum) != 0 || MLA_USED_WORD(schedule.usedVecNumOfFd) != 0 ||
        MLA_USED_WORD(schedule.gS1BaseSizeOfFd) != 8) {
        return Status::InvalidTiling;
    }
    // Non-FD generators zero inactive bookkeeping and unused endpoint slots.
    constexpr uint32_t fdBegin = offsetof(MlaUsedMetadataPrefix, schedule.bN2IdxOfFdHead) / sizeof(uint32_t);
    constexpr uint32_t fdEnd = offsetof(MlaUsedMetadataPrefix, schedule.workspace) / sizeof(uint32_t);
    if constexpr (IsCausal || IsWindow) {
        // All words lie in the validated fixed prefix. Independent loads avoid
        // a scalar branch after every GM read while retaining every zero check.
        uint32_t bits0 = 0, bits1 = 0, bits2 = 0, bits3 = 0;
        uint32_t bits4 = 0, bits5 = 0, bits6 = 0, bits7 = 0;
        uint32_t word = fdBegin;
        for (; word + 7 < fdEnd; word += 8) {
            bits0 |= read(word);
            bits1 |= read(word + 1);
            bits2 |= read(word + 2);
            bits3 |= read(word + 3);
            bits4 |= read(word + 4);
            bits5 |= read(word + 5);
            bits6 |= read(word + 6);
            bits7 |= read(word + 7);
        }
        for (; word < fdEnd; ++word) {
            bits0 |= read(word);
        }
        if ((bits0 | bits1 | bits2 | bits3 | bits4 | bits5 | bits6 | bits7) != 0) {
            return Status::InvalidTiling;
        }
    } else {
        for (uint32_t word = fdBegin; word < fdEnd; ++word) {
            if (read(word) != 0) {
                return Status::InvalidTiling;
            }
        }
    }
    uint64_t tiles = 0;
    for (uint32_t b = 0; b < c.batch; ++b) {
        const uint32_t word = kMlaUsedMetadataPrefixWords + kMlaUsedMetadataSequenceWords * b;
        const uint64_t m = (uint64_t(read(word + 6)) * c.heads + s.mBaseSize - 1) / s.mBaseSize;
        tiles += m * ((read(word + 7) + s.s2BaseSize - 1) / s.s2BaseSize);
        if (tiles > UINT32_MAX) {
            return Status::InvalidEndpoints;
        }
    }
    uint32_t previousBatch = 0, previousM = 0;
    for (uint32_t i = 0; i < kMlaMaxAic; ++i) {
        const uint32_t b = read(offsetof(MlaUsedMetadataPrefix, schedule.bN2End) / sizeof(uint32_t) + i);
        const uint32_t m = read(offsetof(MlaUsedMetadataPrefix, schedule.gS1End) / sizeof(uint32_t) + i);
        const uint32_t n = read(offsetof(MlaUsedMetadataPrefix, schedule.s2End) / sizeof(uint32_t) + i);
        if (i >= s.usedCoreNum) {
            if (b != 0 || m != 0 || n != 0) {
                return Status::InvalidEndpoints;
            }
            continue;
        }
        if (b > c.batch || n != 0 || b < previousBatch || (b == previousBatch && m <= previousM)) {
            return Status::InvalidEndpoints;
        }
        uint32_t mBlocks = 0;
        if (b < c.batch) {
            const uint32_t q = read(kMlaUsedMetadataPrefixWords + kMlaUsedMetadataSequenceWords * b + 6);
            mBlocks = (uint64_t(q) * c.heads + s.mBaseSize - 1) / s.mBaseSize;
        }
        if ((b == c.batch && m != 0) || (b < c.batch && m >= (mBlocks == 0 ? 1 : mBlocks))) {
            return Status::InvalidEndpoints;
        }
        s.bN2End[i] = previousBatch = b;
        s.gS1End[i] = previousM = m;
        s.s2End[i] = n;
    }
    // Causal schedules own every M block, including the zero-cost suffix.
    if ((IsCausal || IsWindow) && (previousBatch != c.batch || previousM != 0)) {
        return Status::InvalidEndpoints;
    }
    // Check the whole omitted suffix, including a partially omitted sample.
    for (uint32_t b = previousBatch; b < c.batch; ++b) {
        const uint32_t word = kMlaUsedMetadataPrefixWords + kMlaUsedMetadataSequenceWords * b;
        if (read(word + 6) != 0 && read(word + 7) != 0) {
            return Status::InvalidEndpoints;
        }
    }
    if (ReadMetadataWide(read, offsetof(MlaUsedMetadataPrefix, schedule.workspace.fdAccumOutSize) / 4) !=
            uint64_t(c.aicNum) * 2 * s.mBaseSize * 512 ||
        ReadMetadataWide(read, offsetof(MlaUsedMetadataPrefix, schedule.workspace.fdLogSumExpSize) / 4) !=
            uint64_t(c.aicNum) * 2 * 2 * s.mBaseSize * 8) {
        return Status::InvalidWorkspace;
    }
    status = ReadMetadataWorkspace(read, c, s, rows);
    if (status == Status::Success) {
        output = decoded;
    }
    return status;
#undef MLA_USED_WORD
}
} // namespace flash_attn::mla
#undef MLA_USED_INLINE
