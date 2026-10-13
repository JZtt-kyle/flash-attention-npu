/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Source geometry/variable-M endpoints: mla_tnd_schedule.cpp and its pinned
// CANN sources; upstream license: LICENSE.cann. Validate addressing/resources,
// not split optimality or authenticity (maxCost is diagnostic only).
#pragma once
#include "mla_used_metadata.h"
#include "mla_causal_metadata.h"
#include "mla_used_metadata_args.h"
#include "mla_metadata_consumer.h"

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
template <bool IsCausal = false, bool IsTndCausal = false, bool IsMixedCausal = false, typename Reader, typename Arrays>
MLA_USED_INLINE MlaMetadataReadStatus ReadUsedMetadata(const Reader& read, uint32_t words,
                                                       const MlaUsedMetadataContext& c,
                                                       const MlaUsedMetadataInputs& inputs, const Arrays& arrays,
                                                       MlaUsedKernelSchedule& output)
{
    using Status = MlaMetadataReadStatus;
    if (c.batch > (UINT32_MAX - kMlaUsedMetadataPrefixWords) / kMlaUsedMetadataSequenceWords) {
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
    auto status = ReadMetadataHeader(read, words, kMlaUsedMetadataPrefixWords + kMlaUsedMetadataSequenceWords * c.batch,
                                     IsCausal ? causalVersion : kMlaUsedMetadataVersion,
                                     IsCausal ? causalKind : kMlaMetadataUsed, c.dtype);
    if (status != Status::Success) {
        return status;
    }
#define MLA_USED_WORD(field) read(offsetof(MlaUsedMetadataPrefix, field) / sizeof(uint32_t))
    const uint64_t totalQ = UsedMetadataTotal(c.query);
    if (IsCausal &&
        (c.query.layout != uint32_t(IsTndCausal || IsMixedCausal) || c.kv.layout != uint32_t(IsTndCausal))) {
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
    if constexpr (IsCausal) {
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
    if (IsCausal && (previousBatch != c.batch || previousM != 0)) {
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
