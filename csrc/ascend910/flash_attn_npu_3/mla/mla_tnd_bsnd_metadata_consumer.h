/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Source bounds and variable-M endpoints: mla_tnd_schedule.cpp and its pinned
// CANN sources; upstream license: LICENSE.cann. As with the compact decoder,
// this checks addressing/resources, not split optimality or authenticity.
#pragma once
#include "mla_tnd_bsnd_metadata.h"
#include "mla_metadata_consumer.h"

#if defined(__NPU_ARCH__)
#define MLA_MIXED_INLINE __aicore__ inline
#else
#define MLA_MIXED_INLINE inline
#endif

namespace flash_attn::mla {

struct MlaTndBsndMetadataContext {
    uint32_t dtype, batch, totalQuery, heads, maxQueryLength, kvCapacity;
    uint32_t aicNum, aivNum, blockDim;
    uint64_t workspaceBytes, libApiBytes;
    MlaLocalMemory localMemory;
};

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
