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

// Source bounds/workspace: CANN ops-transformer ca18aee610729ca5270a44e2d84c751bc687fbf0,
// fia_tiling_nonquant_mla.cpp and fia_kernel_nonquant_mla.h, as mapped in mla_schedule.cpp.
// Validation/decoding of the private v1 dense metadata, shared by CPU tests and
// the device consumer. Scheduling remains in the CANN-derived mla_schedule.cpp;
// the formulas below only check the Kernel's addressing/resource contract.
#pragma once
#include "mla_metadata.h"
#include "mla_dense_params.h"

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
};

template <typename Reader>
MLA_METADATA_INLINE uint64_t ReadMetadataWide(const Reader& read, uint32_t word)
{
    return uint64_t(read(word)) | (uint64_t(read(word + 1)) << 32);
}

// Reader fetches one aligned 32-bit word. Check the size/header before touching
// the payload; use only wide arithmetic until every narrowing has been checked.
// Output is unchanged on failure. This checks compatibility and memory safety,
// not authenticity: maxCost and inactive FD bookkeeping do not drive the Kernel.
template <typename Reader>
MLA_METADATA_INLINE MlaMetadataReadStatus ReadDenseMetadata(const Reader& read, uint32_t words,
                                                            const MlaDenseMetadataContext& c,
                                                            MlaDenseKernelSchedule& output)
{
    using Status = MlaMetadataReadStatus;
#define MLA_WORD(field) read(offsetof(MlaDenseMetadata, field) / sizeof(uint32_t))
#define MLA_WIDE(field) ReadMetadataWide(read, offsetof(MlaDenseMetadata, field) / sizeof(uint32_t))
    if (words != kMlaDenseMetadataWords) {
        return Status::InvalidBuffer;
    }
    if (MLA_WORD(magic) != kMlaMetadataMagic || MLA_WORD(version) != kMlaMetadataVersion ||
        MLA_WORD(kind) != kMlaMetadataDenseSharedKV || MLA_WORD(wordCount) != kMlaDenseMetadataWords ||
        MLA_WORD(reserved) != 0) {
        return Status::InvalidHeader;
    }
    if (MLA_WORD(status) != 0 || MLA_WORD(scheduleStatus) != 0) {
        return Status::GenerationFailed;
    }
    if ((c.dtype != uint32_t(MlaMetadataDtype::Fp16) && c.dtype != uint32_t(MlaMetadataDtype::Bf16)) ||
        MLA_WORD(dtype) != c.dtype) {
        return Status::DtypeMismatch;
    }
    const uint64_t rows = uint64_t(c.sq) * c.heads;
    if (c.batch == 0 || rows == 0 || rows > kMlaMaxDenseRows || c.sq > kMlaMaxQueryLength ||
        c.heads > kMlaMaxQueryHeads || uint64_t(c.batch) * rows > kMlaMaxBatchRows || c.sk == 0 || c.sk > 1024 ||
        MLA_WORD(schedule.bSize) != c.batch || MLA_WORD(schedule.s1Size) != c.sq ||
        MLA_WORD(schedule.gSize) != c.heads || MLA_WORD(schedule.s2Size) != c.sk || MLA_WORD(schedule.n2Size) != 1 ||
        MLA_WORD(schedule.headDim) != 512 || MLA_WORD(schedule.headDimRope) != 64) {
        return Status::ShapeMismatch;
    }
    MlaDenseKernelSchedule s{};
    s.batch = c.batch;
    s.sq = c.sq;
    s.heads = c.heads;
    s.sk = c.sk;
    s.usedCoreNum = MLA_WORD(schedule.usedCoreNum);
    s.blockDim = MLA_WORD(schedule.blockDim);
    s.cvRatio = MLA_WORD(schedule.cvRatio);
    if (c.aicNum == 0 || c.aicNum > kMlaMaxAic || c.aivNum != 2 * c.aicNum || MLA_WORD(schedule.aicNum) != c.aicNum ||
        MLA_WORD(schedule.aivNum) != c.aivNum || s.cvRatio != 2 || s.usedCoreNum == 0 || s.usedCoreNum > c.aicNum ||
        s.blockDim != c.blockDim || s.blockDim < s.usedCoreNum || s.blockDim > c.aicNum ||
        MLA_WORD(schedule.scheduleMode) != 1) {
        return Status::PlatformMismatch;
    }
    s.mBaseSize = MLA_WORD(schedule.mBaseSize);
    s.s2BaseSize = MLA_WORD(schedule.s2BaseSize);
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
    if (previous != tasks) {
        return Status::InvalidEndpoints;
    }
    const uint64_t m = ((rows < s.mBaseSize ? rows : s.mBaseSize) + 15) / 16 * 16;
    const uint64_t r1 = m * ((s2Base + 31) / 32 * 32), r2 = m * 512;
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
    output = s;
    return Status::Success;
#undef MLA_WORD
#undef MLA_WIDE
}
} // namespace flash_attn::mla
#undef MLA_METADATA_INLINE
