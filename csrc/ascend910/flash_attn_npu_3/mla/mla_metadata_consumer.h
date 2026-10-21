/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "mla_metadata.h"
#include "mla_params.h"
#include "mla_layout.h"

// The upstream license is reproduced in mla/LICENSE.cann.

// Source bounds/workspace: CANN ops-transformer ca18aee610729ca5270a44e2d84c751bc687fbf0,
// fia_tiling_nonquant_mla.cpp and fia_kernel_nonquant_mla.h, as mapped in mla_tiling.cpp.
// Validation/decoding of the private BSND metadata, shared by CPU tests and
// the device consumer. Scheduling remains in the CANN-derived mla_tiling.cpp;
// the formulas below only check the Kernel's addressing/resource contract.

#if defined(__NPU_ARCH__)
#define MLA_METADATA_INLINE __aicore__ inline
#else
#define MLA_METADATA_INLINE inline
#endif

namespace flash_attn::mla {

enum class MlaMetadataReadStatus : uint32_t {
    Success = 0,
    InvalidBuffer,
    InvalidHeader,
    GenerationFailed,
    DtypeMismatch,
    ShapeMismatch,
    PlatformMismatch,
    InvalidTiling,
    InvalidEndpoints,
    InvalidWorkspace,
    InsufficientLocalMemory,
    InvalidCacheLengths,
    InvalidPrefixes,
    InvalidUsedLengths,
    InvalidPageTable,
    InvalidPage,
};

template <typename Reader>
MLA_METADATA_INLINE uint64_t ReadMetadataWide(const Reader& read, uint32_t word)
{
    return uint64_t(read(word)) | (uint64_t(read(word + 1)) << 32);
}

template <typename Reader>
MLA_METADATA_INLINE MlaMetadataReadStatus ReadMetadataHeader(const Reader& read, uint32_t words, uint32_t expectedWords,
                                                             uint32_t version, uint32_t kind, uint32_t dtype)
{
    using Status = MlaMetadataReadStatus;
#define MLA_WORD(field) read(offsetof(MlaDenseMetadata, field) / sizeof(uint32_t))
    if (words != expectedWords) {
        return Status::InvalidBuffer;
    }
    if (MLA_WORD(magic) != kMlaMetadataMagic || MLA_WORD(version) != version || MLA_WORD(kind) != kind ||
        MLA_WORD(wordCount) != expectedWords || MLA_WORD(reserved) != 0) {
        return Status::InvalidHeader;
    }
    if (MLA_WORD(status) != 0 || MLA_WORD(scheduleStatus) != 0) {
        return Status::GenerationFailed;
    }
    if ((dtype != uint32_t(MlaMetadataDtype::Fp16) && dtype != uint32_t(MlaMetadataDtype::Bf16)) ||
        MLA_WORD(dtype) != dtype) {
        return Status::DtypeMismatch;
    }
    return Status::Success;
#undef MLA_WORD
}

// Shared wire/platform/resource validation for BSND and compact TND.
template <typename Reader, typename Context>
MLA_METADATA_INLINE MlaMetadataReadStatus ReadMetadataPlatform(const Reader& read, const Context& c,
                                                               MlaDenseKernelSchedule& s)
{
    using Status = MlaMetadataReadStatus;
#define MLA_WORD(field) read(offsetof(MlaDenseMetadata, field) / sizeof(uint32_t))
#define MLA_WIDE(field) ReadMetadataWide(read, offsetof(MlaDenseMetadata, field) / sizeof(uint32_t))
    s.usedCoreNum = MLA_WORD(schedule.usedCoreNum);
    s.blockDim = MLA_WORD(schedule.blockDim);
    s.cvRatio = MLA_WORD(schedule.cvRatio);
    if (c.aicNum == 0 || c.aicNum > kMlaMaxAic || c.aivNum != 2 * c.aicNum || MLA_WORD(schedule.aicNum) != c.aicNum ||
        MLA_WORD(schedule.aivNum) != c.aivNum || s.cvRatio != 2 || s.usedCoreNum == 0 || s.usedCoreNum > c.aicNum ||
        s.blockDim != c.blockDim || s.blockDim < s.usedCoreNum || s.blockDim > c.aicNum ||
        MLA_WORD(schedule.scheduleMode) != 1) {
        return Status::PlatformMismatch;
    }
    return Status::Success;
#undef MLA_WORD
#undef MLA_WIDE
}

// Shared wire/platform/resource validation for BSND and compact TND.
template <typename Reader, typename Context>
MLA_METADATA_INLINE MlaMetadataReadStatus ReadEmptyMetadataSchedule(const Reader& read, const Context& c,
                                                                    MlaDenseKernelSchedule& s)
{
    using Status = MlaMetadataReadStatus;
#define MLA_WORD(field) read(offsetof(MlaDenseMetadata, field) / sizeof(uint32_t))
#define MLA_WIDE(field) ReadMetadataWide(read, offsetof(MlaDenseMetadata, field) / sizeof(uint32_t))
    // A global empty record has one validation core and no attention state.
    // Check every unused payload word so a nonempty schedule cannot be
    // reinterpreted as an empty one by changing its shape alone.
    if (s.usedCoreNum != 1 || s.mBaseSize != 0 || s.s2BaseSize != 0 || c.workspaceBytes != 0 || c.libApiBytes != 0) {
        return Status::InvalidTiling;
    }
    for (uint32_t word = offsetof(MlaDenseMetadata, schedule.bN2End) / sizeof(uint32_t); word < kMlaDenseMetadataWords;
         ++word) {
        if (read(word) != 0) {
            return Status::InvalidTiling;
        }
    }
    return Status::Success;
#undef MLA_WORD
#undef MLA_WIDE
}

// Shared wire/platform/resource validation for BSND and compact TND.
template <typename Reader, typename Context>
MLA_METADATA_INLINE MlaMetadataReadStatus ReadMetadataWorkspace(const Reader& read, const Context& c,
                                                                MlaDenseKernelSchedule& s, uint64_t rows)
{
    using Status = MlaMetadataReadStatus;
#define MLA_WORD(field) read(offsetof(MlaDenseMetadata, field) / sizeof(uint32_t))
#define MLA_WIDE(field) ReadMetadataWide(read, offsetof(MlaDenseMetadata, field) / sizeof(uint32_t))
    const uint64_t m = ((rows < s.mBaseSize ? rows : s.mBaseSize) + 15) / 16 * 16;
    const uint64_t r1 = m * ((s.s2BaseSize + 31) / 32 * 32), r2 = m * 512;
    const uint64_t p = 2 * s.blockDim * r1 * 4, pv = p + 2 * s.blockDim * r1 * 2;
    const uint64_t vec2 = pv + 2 * s.blockDim * r2 * 4, kernel = vec2 + 2 * s.blockDim * r2 * 4;
    const uint64_t reserve = uint64_t(2) * c.aicNum * (6 * r1 + 8 * r2 + 8 * s.mBaseSize);
    if (MLA_WIDE(schedule.workspace.mm1ResSize) != r1 || MLA_WIDE(schedule.workspace.mm2ResSize) != r2 ||
        MLA_WIDE(schedule.workspace.qkOffsetBytes) != 0 || MLA_WIDE(schedule.workspace.pOffsetBytes) != p ||
        MLA_WIDE(schedule.workspace.pvOffsetBytes) != pv || MLA_WIDE(schedule.workspace.vec2OffsetBytes) != vec2 ||
        MLA_WIDE(schedule.workspace.kernelBytes) != kernel || MLA_WIDE(schedule.workspace.userBytes) != reserve ||
        kernel > reserve || c.workspaceBytes < reserve || MLA_WIDE(schedule.workspace.libApiBytes) != c.libApiBytes ||
        c.libApiBytes > UINT64_MAX - reserve || MLA_WIDE(schedule.workspace.totalBytes) != c.libApiBytes + reserve) {
        return Status::InvalidWorkspace;
    }
    s.mm1ResSize = r1;
    s.mm2ResSize = r2;
    if (MLA_WIDE(schedule.requiredLocalMemory.l1Bytes) != 516096 || c.localMemory.l1Bytes < 516096 ||
        MLA_WIDE(schedule.requiredLocalMemory.ubBytes) != 174080 || c.localMemory.ubBytes < 174080 ||
        MLA_WIDE(schedule.requiredLocalMemory.l0aBytes) != 65536 || c.localMemory.l0aBytes < 65536 ||
        MLA_WIDE(schedule.requiredLocalMemory.l0bBytes) != 65536 || c.localMemory.l0bBytes < 65536 ||
        MLA_WIDE(schedule.requiredLocalMemory.l0cBytes) != 131072 || c.localMemory.l0cBytes < 131072) {
        return Status::InsufficientLocalMemory;
    }
    return Status::Success;
#undef MLA_WORD
#undef MLA_WIDE
}

struct DenseMetadataLengths {
    MLA_METADATA_INLINE bool Check(uint32_t, uint32_t) const
    {
        return true;
    }
    MLA_METADATA_INLINE bool Covers(uint64_t end, uint64_t tasks, uint32_t, uint32_t) const
    {
        return end == tasks;
    }
};

// Call only after checking the protocol header. Both BSND versions use the same
// fixed schedule encoding and resource bounds. Lengths supplies only the
// snapshot check and endpoint coverage rule, never alternative tiling formulas.
// Output is unchanged on failure. This checks compatibility and memory safety,
// not authenticity: maxCost and inactive FD bookkeeping do not drive the Kernel.
template <typename Reader, typename Lengths>
MLA_METADATA_INLINE MlaMetadataReadStatus ReadMetadataSchedule(const Reader& read, const MlaDenseMetadataContext& c,
                                                               const Lengths& lengths, MlaDenseKernelSchedule& output)
{
    using Status = MlaMetadataReadStatus;
#define MLA_WORD(field) read(offsetof(MlaDenseMetadata, field) / sizeof(uint32_t))
#define MLA_WIDE(field) ReadMetadataWide(read, offsetof(MlaDenseMetadata, field) / sizeof(uint32_t))
    const uint64_t rows = uint64_t(c.sq) * c.heads;
    if (c.heads == 0 || rows > kMlaMaxDenseRows || c.sq > kMlaMaxQueryLength || c.heads > kMlaMaxQueryHeads ||
        uint64_t(c.batch) * rows > kMlaMaxBatchRows || c.sk > kMlaMaxKvLength || MLA_WORD(schedule.bSize) != c.batch ||
        MLA_WORD(schedule.s1Size) != c.sq || MLA_WORD(schedule.gSize) != c.heads || MLA_WORD(schedule.s2Size) != c.sk ||
        MLA_WORD(schedule.n2Size) != 1 || MLA_WORD(schedule.headDim) != 512 || MLA_WORD(schedule.headDimRope) != 64) {
        return Status::ShapeMismatch;
    }
    // Even global empty capacities must validate every stored/current length.
    if (!lengths.Check(c.batch, c.sk)) {
        return Status::InvalidCacheLengths;
    }
    MlaDenseKernelSchedule s{};
    s.batch = c.batch;
    s.sq = c.sq;
    s.heads = c.heads;
    s.sk = c.sk;
    const auto platformStatus = ReadMetadataPlatform(read, c, s);
    if (platformStatus != Status::Success) {
        return platformStatus;
    }
    s.mBaseSize = MLA_WORD(schedule.mBaseSize);
    s.s2BaseSize = MLA_WORD(schedule.s2BaseSize);
    if (c.batch == 0 || c.sq == 0 || c.sk == 0) {
        const auto emptyStatus = ReadEmptyMetadataSchedule(read, c, s);
        if (emptyStatus != Status::Success) {
            return emptyStatus;
        }
        output = s;
        return Status::Success;
    }
    const uint32_t s2Base = c.sk < 512 ? c.sk : 512;
    // CalcMBaseSize may choose any divisor in [128,256], including odd values.
    // A 256-row base also permits a final partial block.
    if (s.mBaseSize < 128 || s.mBaseSize > 256 || (s.mBaseSize != 256 && rows % s.mBaseSize != 0) ||
        s.s2BaseSize != s2Base || MLA_WORD(schedule.numOfFdHead) != 0 || MLA_WORD(schedule.maxS2SplitNum) != 0 ||
        MLA_WORD(schedule.usedVecNumOfFd) != 0) {
        return Status::InvalidTiling;
    }
    const uint32_t mBlocks = (rows + s.mBaseSize - 1) / s.mBaseSize;
    const uint64_t tasks = uint64_t(c.batch) * mBlocks;
    if (tasks * ((c.sk + s2Base - 1) / s2Base) > UINT32_MAX) {
        return Status::InvalidEndpoints;
    }
    uint64_t previous = 0;
    for (uint32_t i = 0; i < s.usedCoreNum; ++i) {
        s.bN2End[i] = read(offsetof(MlaDenseMetadata, schedule.bN2End) / sizeof(uint32_t) + i);
        s.gS1End[i] = read(offsetof(MlaDenseMetadata, schedule.gS1End) / sizeof(uint32_t) + i);
        s.s2End[i] = read(offsetof(MlaDenseMetadata, schedule.s2End) / sizeof(uint32_t) + i);
        const uint64_t end = uint64_t(s.bN2End[i]) * mBlocks + s.gS1End[i];
        if (s.bN2End[i] > c.batch || s.gS1End[i] >= mBlocks || s.s2End[i] != 0 || end <= previous || end > tasks) {
            return Status::InvalidEndpoints;
        }
        previous = end;
    }
    if (!lengths.Covers(previous, tasks, mBlocks, c.batch)) {
        return Status::InvalidEndpoints;
    }
    const auto resourceStatus = ReadMetadataWorkspace(read, c, s, rows);
    if (resourceStatus != Status::Success) {
        return resourceStatus;
    }
    output = s;
    return Status::Success;
#undef MLA_WORD
#undef MLA_WIDE
}

// Reader fetches one aligned 32-bit word. Check size/header before the payload;
// use wide arithmetic until every narrowing has been checked.
template <typename Reader>
MLA_METADATA_INLINE MlaMetadataReadStatus ReadDenseMetadata(const Reader& read, uint32_t words,
                                                            const MlaDenseMetadataContext& c,
                                                            MlaDenseKernelSchedule& output)
{
    const auto status = ReadMetadataHeader(read, words, kMlaDenseMetadataWords, kMlaMetadataVersion,
                                           kMlaMetadataDenseSharedKV, c.dtype);
    if (status != MlaMetadataReadStatus::Success) {
        return status;
    }
    return ReadMetadataSchedule(read, c, DenseMetadataLengths{}, output);
}
} // namespace flash_attn::mla
#undef MLA_METADATA_INLINE

// v2 BSND length snapshot validation. Schedule/resource checks and their CANN
// provenance are shared with mla_metadata_consumer.h. No attention launch here.
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

// The upstream license is reproduced in mla/LICENSE.cann.
// Bounds and variable-M endpoint rules follow mla_tiling.cpp and its
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

// Source bounds and variable-M endpoints: mla_tiling.cpp and its pinned
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

// Source geometry/variable-M endpoints: mla_tiling.cpp and its pinned
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

// Source geometry/endpoints: mla_tiling.cpp and its pinned CANN sources.
// Page addressing: mla_layout.cpp (LICENSE.cann). This checks compatible
// geometry/resources and every current effective page, not schedule optimality.

#if defined(__NPU_ARCH__)
#define MLA_PAGED_INLINE __aicore__ inline
#else
#define MLA_PAGED_INLINE inline
#endif

namespace flash_attn::mla {

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
// Default Hkv=1 preserves compressed MLA; prefill passes its actual KV heads.
// Singleton/empty strides stay unused.
MLA_PAGED_INLINE bool ValidPagedCacheTensor(const MlaBsndTensorLayout& x, const MlaPagedMetadataArray& buffer,
                                            uint64_t pages, uint64_t pageSize, uint64_t dim, uint64_t heads = 1)
{
    if (!heads || !pageSize || x.batchStride > INT64_MAX || x.rowStride > INT64_MAX || x.headStride > INT64_MAX ||
        !ValidPagedMetadataAddress(buffer.address, buffer.count, 2)) {
        return false;
    }
    uint64_t span = 0;
    if (pages != 0) {
        constexpr uint64_t maxElements = INT64_MAX / 2;
        if (heads > 1 && x.headStride > (maxElements - dim) / (heads - 1))
            return false;
        dim += (heads - 1) * x.headStride;
        if (pageSize > 1 && x.rowStride > (maxElements - dim) / (pageSize - 1)) {
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
                                                                 const MlaPagedCacheInputs& input, uint64_t heads = 1,
                                                                 uint64_t keyDim = 64, uint64_t valueDim = 512)
{
    using Status = MlaMetadataReadStatus;
    const auto& p = input.layout;
    if (!c.pageSize || p.batchSize != c.batch || p.pageSize != c.pageSize || p.logicalCapacity != c.logicalKvCapacity ||
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
    if (!ValidPagedCacheTensor(p.k, input.k, p.physicalPages, p.pageSize, keyDim, heads) ||
        !ValidPagedCacheTensor(p.v, input.v, p.physicalPages, p.pageSize, valueDim, heads)) {
        return Status::InvalidBuffer;
    }
    return Status::Success;
}

template <bool IsCausal = false, bool IsWindow = false, typename Reader, typename Arrays, typename Pages>
MLA_PAGED_INLINE MlaMetadataReadStatus ReadPagedMetadata(const Reader& read, uint32_t words,
                                                         const MlaPagedMetadataContext& c,
                                                         const MlaPagedMetadataInputs& inputs, const Arrays& arrays,
                                                         const MlaPagedCacheInputs& cache, const Pages& pages,
                                                         MlaPagedKernelSchedule& output,
                                                         MlaWindowKernelParams window = {})
{
    using Status = MlaMetadataReadStatus;
    constexpr uint32_t tail = IsWindow ? kMlaWindowPagedMetadataTailWords : 0;
    static_assert(!IsWindow || !IsCausal);
    if (c.batch > (UINT32_MAX - kMlaPagedMetadataPrefixWords - tail) / kMlaPagedMetadataSequenceWords) {
        return Status::InvalidBuffer;
    }
    auto status = ReadMetadataHeader(
        read, words, kMlaPagedMetadataPrefixWords + kMlaPagedMetadataSequenceWords * c.batch + tail,
        IsWindow   ? (c.query.layout == 1 ? kMlaWindowPagedTndMetadataVersion : kMlaWindowPagedBsndMetadataVersion)
        : IsCausal ? (c.query.layout == 1 ? kMlaCausalPagedTndMetadataVersion : kMlaCausalPagedBsndMetadataVersion)
                   : kMlaPagedMetadataVersion,
        IsWindow   ? (c.query.layout == 1 ? kMlaMetadataWindowPagedTnd : kMlaMetadataWindowPagedBsnd)
        : IsCausal ? (c.query.layout == 1 ? kMlaMetadataCausalPagedTnd : kMlaMetadataCausalPagedBsnd)
                   : kMlaMetadataPaged,
        c.dtype);
    if (status != Status::Success) {
        return status;
    }
    if constexpr (IsWindow) {
        // Invocation identity is checked before geometry, lengths, pages or empty shortcuts.
        if (c.query.layout > 1 || window.kind > 2 || window.left < -1 || window.right < -1 ||
            window.left > int64_t(c.logicalKvCapacity) - 1 ||
            window.right > (c.query.maxLength == 0 ? 0 : int64_t(c.query.maxLength) - 1) ||
            read(words - 3) != window.kind || int64_t(int32_t(read(words - 2))) != window.left ||
            int64_t(int32_t(read(words - 1))) != window.right) {
            return Status::ShapeMismatch;
        }
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
        auto emptyContext = c;
        if (c.batch != 0 && maximumQ != 0 && c.logicalKvCapacity != 0) {
            // All current KV lengths were checked above and are zero. The
            // host may already have reserved workspace for nonempty static
            // capacity; no part of that allocation is consumed. Still require
            // the entire record's empty schedule/workspace payload to be zero.
            emptyContext.workspaceBytes = 0;
            emptyContext.libApiBytes = 0;
        }
        status = ReadEmptyMetadataSchedule(read, emptyContext, s);
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
    // Causal owns every M block, including zero-cost/all-masked rows.
    if ((IsCausal || IsWindow) && (previousBatch != c.batch || previousM != 0)) {
        return Status::InvalidEndpoints;
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
