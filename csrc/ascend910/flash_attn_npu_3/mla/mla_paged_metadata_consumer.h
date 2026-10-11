/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Source geometry/endpoints: mla_tnd_schedule.cpp and its pinned CANN sources.
// Page addressing: mla_paged_layout.cpp (LICENSE.cann). This checks compatible
// geometry/resources and every current effective page, not schedule optimality.
#pragma once
#include "mla_paged_metadata.h"
#include "mla_paged_metadata_args.h"
#include "mla_paged_layout.h"
#include "mla_metadata_consumer.h"

#if defined(__NPU_ARCH__)
#define MLA_PAGED_INLINE __aicore__ inline
#else
#define MLA_PAGED_INLINE inline
#endif

namespace flash_attn::mla {

struct MlaPagedMetadataInputs {
    MlaPagedMetadataArray cuQuery, usedQuery, kvLengths;
};

// Invocation-only addresses/counts are relative to data_ptr(), after storage
// offset. The host builds layout with BuildPagedCacheLayout and validates
// dtype/device. The consumer independently checks its spans and geometry.
// table counts int32 entries; k/v count FP16/BF16 elements, including padding.
struct MlaPagedCacheInputs {
    MlaPagedCacheLayout layout;
    MlaPagedMetadataArray table, k, v;
};
static_assert(std::is_standard_layout_v<MlaPagedCacheInputs> && std::is_trivially_copyable_v<MlaPagedCacheInputs>);

struct MlaPagedKernelSchedule {
    MlaDenseKernelSchedule tiling;
    MlaPagedMetadataQuery query;
    uint32_t usedFlags, pageSize, logicalKvCapacity;
};
static_assert(sizeof(MlaPagedKernelSchedule) == 96 * sizeof(uint32_t));

MLA_PAGED_INLINE bool ValidPagedMetadataAddress(uint64_t address, uint64_t count, uint32_t bytes = 4)
{
    return (address != 0 || count == 0) && address % bytes == 0 && count <= (UINT64_MAX - address) / bytes;
}

// Validate the full physical view, including pages unused by this invocation.
// Hkv=1, so headStride never contributes; singleton/empty strides stay unused.
MLA_PAGED_INLINE bool ValidPagedCacheTensor(const MlaBsndTensorLayout& x, const MlaPagedMetadataArray& buffer,
                                            uint64_t pages, uint64_t pageSize, uint64_t dim)
{
    if (x.batchStride > INT64_MAX || x.rowStride > INT64_MAX || x.headStride > INT64_MAX ||
        !ValidPagedMetadataAddress(buffer.address, buffer.count, 2)) {
        return false;
    }
    uint64_t span = 0;
    if (pages != 0) {
        constexpr uint64_t maxElements = INT64_MAX / 2;
        if (x.rowStride > (maxElements - dim) / (pageSize - 1)) {
            return false;
        }
        span = dim + (pageSize - 1) * x.rowStride;
        if (pages > 1 && x.batchStride > (maxElements - span) / (pages - 1)) {
            return false;
        }
        span += (pages - 1) * x.batchStride;
    }
    return x.spanElements == span && buffer.count >= span;
}

MLA_PAGED_INLINE MlaMetadataReadStatus CheckPagedCacheDescriptor(const MlaPagedMetadataContext& c,
                                                                 const MlaPagedCacheInputs& input)
{
    using Status = MlaMetadataReadStatus;
    const auto& p = input.layout;
    if (p.batchSize != c.batch || p.pageSize != c.pageSize || p.logicalCapacity != c.logicalKvCapacity ||
        p.maxPages != c.logicalKvCapacity / c.pageSize || p.physicalPages > uint64_t(INT32_MAX) + 1) {
        return Status::ShapeMismatch;
    }
    if (p.tableRowStride > INT64_MAX || !ValidPagedMetadataAddress(input.table.address, input.table.count)) {
        return Status::InvalidPageTable;
    }
    uint64_t span = 0;
    if (p.batchSize != 0 && p.maxPages != 0) {
        constexpr uint64_t maxElements = INT64_MAX / 4;
        if (p.batchSize > 1 && p.tableRowStride > (maxElements - p.maxPages) / (p.batchSize - 1)) {
            return Status::InvalidPageTable;
        }
        span = (p.batchSize - 1) * p.tableRowStride + p.maxPages;
    }
    if (p.tableSpanElements != span || input.table.count < span) {
        return Status::InvalidPageTable;
    }
    if (!ValidPagedCacheTensor(p.k, input.k, p.physicalPages, p.pageSize, 64) ||
        !ValidPagedCacheTensor(p.v, input.v, p.physicalPages, p.pageSize, 512)) {
        return Status::InvalidBuffer;
    }
    return Status::Success;
}

template <typename Reader, typename Arrays, typename Pages>
MLA_PAGED_INLINE MlaMetadataReadStatus ReadPagedMetadata(const Reader& read, uint32_t words,
                                                         const MlaPagedMetadataContext& c,
                                                         const MlaPagedMetadataInputs& inputs, const Arrays& arrays,
                                                         const MlaPagedCacheInputs& cache, const Pages& pages,
                                                         MlaPagedKernelSchedule& output)
{
    using Status = MlaMetadataReadStatus;
    if (c.batch > (UINT32_MAX - kMlaPagedMetadataPrefixWords) / kMlaPagedMetadataSequenceWords) {
        return Status::InvalidBuffer;
    }
    auto status =
        ReadMetadataHeader(read, words, kMlaPagedMetadataPrefixWords + kMlaPagedMetadataSequenceWords * c.batch,
                           kMlaPagedMetadataVersion, kMlaMetadataPaged, c.dtype);
    if (status != Status::Success) {
        return status;
    }
#define MLA_PAGED_WORD(field) read(offsetof(MlaPagedMetadataPrefix, field) / sizeof(uint32_t))
    const uint64_t totalQ = uint64_t(c.query.totalTokens.low) | (uint64_t(c.query.totalTokens.high) << 32);
    const bool tnd = c.query.layout == 1, used = c.usedFlags != 0;
    if (c.query.layout > 1 || c.usedFlags > 1 || c.heads == 0 || c.heads > kMlaMaxQueryHeads ||
        c.query.maxLength > kMlaMaxQueryLength || uint64_t(c.query.maxLength) * c.heads > kMlaMaxDenseRows ||
        totalQ > kMlaMaxBatchRows / c.heads ||
        (tnd ? totalQ > INT32_MAX : totalQ != uint64_t(c.batch) * c.query.maxLength) || c.pageSize < 16 ||
        c.pageSize > 1024 || c.pageSize % 16 != 0 || c.logicalKvCapacity > kMlaMaxKvLength ||
        c.logicalKvCapacity % c.pageSize != 0 || MLA_PAGED_WORD(query.layout) != c.query.layout ||
        ReadMetadataWide(read, offsetof(MlaPagedMetadataPrefix, query.totalTokens) / 4) != totalQ ||
        MLA_PAGED_WORD(query.maxLength) != c.query.maxLength || MLA_PAGED_WORD(usedFlags) != c.usedFlags ||
        MLA_PAGED_WORD(pageSize) != c.pageSize || MLA_PAGED_WORD(logicalKvCapacity) != c.logicalKvCapacity ||
        MLA_PAGED_WORD(schedule.bSize) != c.batch || MLA_PAGED_WORD(schedule.gSize) != c.heads ||
        MLA_PAGED_WORD(schedule.n2Size) != 1 || MLA_PAGED_WORD(schedule.headDim) != 512 ||
        MLA_PAGED_WORD(schedule.headDimRope) != 64) {
        return Status::ShapeMismatch;
    }
    const auto& prefix = inputs.cuQuery;
    const auto& uq = inputs.usedQuery;
    const auto& kv = inputs.kvLengths;
    if (!ValidPagedMetadataAddress(prefix.address, prefix.count) ||
        (tnd ? prefix.address == 0 || prefix.count != uint64_t(c.batch) + 1
             : prefix.address != 0 || prefix.count != 0)) {
        return Status::InvalidPrefixes;
    }
    if (!ValidPagedMetadataAddress(uq.address, uq.count) || used != (uq.address != 0) ||
        uq.count != (used ? c.batch : 0)) {
        return Status::InvalidUsedLengths;
    }
    if (!ValidPagedMetadataAddress(kv.address, kv.count) || kv.count != c.batch) {
        return Status::InvalidCacheLengths;
    }
    status = CheckPagedCacheDescriptor(c, cache);
    if (status != Status::Success) {
        return status;
    }
    // All descriptors are checked before any dynamic array or page is read.
    if (tnd && arrays(0, 0) != 0) {
        return Status::InvalidPrefixes;
    }
    uint64_t start = 0;
    uint32_t maximumQ = tnd ? 0 : c.query.maxLength, maximumKv = 0;
    for (uint32_t b = 0; b < c.batch; ++b) {
        const uint32_t word = kMlaPagedMetadataPrefixWords + kMlaPagedMetadataSequenceWords * b;
        const uint64_t end = tnd ? uint64_t(arrays(0, b + 1)) : start + c.query.maxLength;
        if (end < start || end > totalQ || end - start > c.query.maxLength || ReadMetadataWide(read, word) != start ||
            read(word + 2) != end - start) {
            return Status::InvalidPrefixes;
        }
        const uint32_t capacity = end - start;
        const uint32_t qLength = used ? arrays(1, b) : capacity;
        if (qLength > capacity || read(word + 3) != qLength) {
            return Status::InvalidUsedLengths;
        }
        const uint32_t length = arrays(2, b);
        if (length > c.logicalKvCapacity || read(word + 4) != length) {
            return Status::InvalidCacheLengths;
        }
        maximumQ = capacity > maximumQ ? capacity : maximumQ;
        maximumKv = length > maximumKv ? length : maximumKv;
        start = end;
    }
    if (start != totalQ) {
        return Status::InvalidPrefixes;
    }
    const uint32_t sourceS2 = (maximumKv + c.pageSize - 1) / c.pageSize * c.pageSize;
    if (MLA_PAGED_WORD(schedule.s1Size) != maximumQ || MLA_PAGED_WORD(schedule.s2Size) != sourceS2) {
        return Status::ShapeMismatch;
    }
    // KV remains non-cumulative for both Q layouts. Never skip an effective
    // page because Q allocation/used is zero; never read table padding.
    for (uint32_t b = 0; b < c.batch; ++b) {
        const uint32_t length = read(kMlaPagedMetadataPrefixWords + kMlaPagedMetadataSequenceWords * b + 4);
        const uint32_t count = (length + c.pageSize - 1) / c.pageSize;
        for (uint32_t j = 0; j < count; ++j) {
            const uint32_t page = pages(uint64_t(b) * cache.layout.tableRowStride + j);
            if (page > INT32_MAX || uint64_t(page) >= cache.layout.physicalPages) {
                return Status::InvalidPage;
            }
        }
    }
    MlaPagedKernelSchedule decoded{};
    decoded.query = c.query;
    decoded.usedFlags = c.usedFlags;
    decoded.pageSize = c.pageSize;
    decoded.logicalKvCapacity = c.logicalKvCapacity;
    auto& s = decoded.tiling;
    s.batch = c.batch;
    s.sq = maximumQ;
    s.sk = sourceS2;
    s.heads = c.heads;
    status = ReadMetadataPlatform(read, c, s);
    if (status != Status::Success) {
        return status;
    }
    s.mBaseSize = MLA_PAGED_WORD(schedule.mBaseSize);
    s.s2BaseSize = MLA_PAGED_WORD(schedule.s2BaseSize);
    if (c.batch == 0 || maximumQ == 0 || sourceS2 == 0) {
        status = ReadEmptyMetadataSchedule(read, c, s);
        if (status == Status::Success) {
            output = decoded;
        }
        return status;
    }
    const uint64_t rows = uint64_t(maximumQ) * c.heads;
    const uint32_t alignedBase = c.pageSize < 512 && 512 % c.pageSize != 0 ? 512 / c.pageSize * c.pageSize : 512;
    if (s.mBaseSize < 128 || s.mBaseSize > 256 || (s.mBaseSize != 256 && rows % s.mBaseSize != 0) ||
        s.s2BaseSize != (sourceS2 < alignedBase ? sourceS2 : alignedBase) ||
        MLA_PAGED_WORD(schedule.numOfFdHead) != 0 || MLA_PAGED_WORD(schedule.maxS2SplitNum) != 0 ||
        MLA_PAGED_WORD(schedule.usedVecNumOfFd) != 0 || MLA_PAGED_WORD(schedule.gS1BaseSizeOfFd) != 8) {
        return Status::InvalidTiling;
    }
    // Non-FD generators zero inactive bookkeeping and unused endpoint slots.
    for (uint32_t word = offsetof(MlaPagedMetadataPrefix, schedule.bN2IdxOfFdHead) / sizeof(uint32_t);
         word < offsetof(MlaPagedMetadataPrefix, schedule.workspace) / sizeof(uint32_t); ++word) {
        if (read(word) != 0) {
            return Status::InvalidTiling;
        }
    }
    uint64_t tiles = 0;
    for (uint32_t b = 0; b < c.batch; ++b) {
        const uint32_t word = kMlaPagedMetadataPrefixWords + kMlaPagedMetadataSequenceWords * b;
        const uint64_t m = (uint64_t(read(word + 3)) * c.heads + s.mBaseSize - 1) / s.mBaseSize;
        tiles += m * ((read(word + 4) + s.s2BaseSize - 1) / s.s2BaseSize);
        if (tiles > UINT32_MAX) {
            return Status::InvalidEndpoints;
        }
    }
    uint32_t previousBatch = 0, previousM = 0;
    for (uint32_t i = 0; i < kMlaMaxAic; ++i) {
        const uint32_t b = read(offsetof(MlaPagedMetadataPrefix, schedule.bN2End) / sizeof(uint32_t) + i);
        const uint32_t m = read(offsetof(MlaPagedMetadataPrefix, schedule.gS1End) / sizeof(uint32_t) + i);
        const uint32_t n = read(offsetof(MlaPagedMetadataPrefix, schedule.s2End) / sizeof(uint32_t) + i);
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
            const uint32_t q = read(kMlaPagedMetadataPrefixWords + kMlaPagedMetadataSequenceWords * b + 3);
            mBlocks = (uint64_t(q) * c.heads + s.mBaseSize - 1) / s.mBaseSize;
        }
        if ((b == c.batch && m != 0) || (b < c.batch && m >= (mBlocks == 0 ? 1 : mBlocks))) {
            return Status::InvalidEndpoints;
        }
        s.bN2End[i] = previousBatch = b;
        s.gS1End[i] = previousM = m;
        s.s2End[i] = n;
    }
    // Check the whole omitted suffix, including a partially omitted sample.
    for (uint32_t b = previousBatch; b < c.batch; ++b) {
        const uint32_t word = kMlaPagedMetadataPrefixWords + kMlaPagedMetadataSequenceWords * b;
        if (read(word + 3) != 0 && read(word + 4) != 0) {
            return Status::InvalidEndpoints;
        }
    }
    if (ReadMetadataWide(read, offsetof(MlaPagedMetadataPrefix, schedule.workspace.fdAccumOutSize) / 4) !=
            uint64_t(c.aicNum) * 2 * s.mBaseSize * 512 ||
        ReadMetadataWide(read, offsetof(MlaPagedMetadataPrefix, schedule.workspace.fdLogSumExpSize) / 4) !=
            uint64_t(c.aicNum) * 2 * 2 * s.mBaseSize * 8) {
        return Status::InvalidWorkspace;
    }
    status = ReadMetadataWorkspace(read, c, s, rows);
    if (status == Status::Success) {
        output = decoded;
    }
    return status;
#undef MLA_PAGED_WORD
}
} // namespace flash_attn::mla
#undef MLA_PAGED_INLINE
