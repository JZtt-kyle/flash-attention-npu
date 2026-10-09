/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// The upstream license is reproduced in LICENSE.cann next to this file.
// Bounds and variable-M endpoint rules follow mla_tnd_schedule.cpp and its
// pinned CANN sources. This validates the wire/addressing contract, not the
// optimality or authenticity of a split. No host scheduler runs on the device.
#pragma once
#include "mla_tnd_metadata.h"
#include "mla_metadata_consumer.h"

#if defined(__NPU_ARCH__)
#define MLA_TND_INLINE __aicore__ inline
#else
#define MLA_TND_INLINE inline
#endif

namespace flash_attn::mla {

struct MlaTndMetadataContext {
    uint32_t dtype, batch, totalQuery, totalKv, heads, maxQueryLength, maxKvLength;
    uint32_t aicNum, aivNum, blockDim;
    uint64_t workspaceBytes, libApiBytes;
    MlaLocalMemory localMemory;
};

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
