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

#pragma once

#include <array>
#include <cstdint>

namespace flash_attn::mla {

// Source: CANN ops-transformer ca18aee610729ca5270a44e2d84c751bc687fbf0,
// attention/fused_infer_attention_score/op_host/fused_infer_attention_score_tiling.h.
// This is an endpoint capacity, not a hard-coded device core count.
constexpr uint32_t kMlaMaxAic = 26;

// Preserve the source's uint32 S1G ceil-div, dense query stride (Hq*512)
// and LSE DMA stride ((Sq-1)*sizeof(float)) without wrapping any field.
constexpr uint32_t kMlaMaxDenseRows = UINT32_MAX - 255;
constexpr uint32_t kMlaMaxQueryHeads = UINT32_MAX / 512;
constexpr uint32_t kMlaMaxQueryLength = UINT32_MAX / sizeof(float) + 1;
constexpr uint64_t kMlaMaxBatchRows = INT64_MAX / (512 * 2);
// Validated dense KV capacity; larger sequences remain explicitly unsupported.
constexpr uint32_t kMlaMaxKvLength = 16384;

struct MlaDenseShape {
    int64_t batch;
    int64_t queryLength;
    int64_t queryHeads;
    int64_t kvLength;
};

struct MlaLocalMemory {
    uint64_t l1Bytes;
    uint64_t ubBytes;
    uint64_t l0aBytes;
    uint64_t l0bBytes;
    uint64_t l0cBytes;
};

struct MlaPlatformInfo {
    uint32_t aicNum;
    uint32_t aivNum;
    MlaLocalMemory localMemory;
    uint64_t libApiWorkspaceBytes;
    // Populate entry c-1 with PlatformAscendC::CalcTschBlockDim(c*ratio, c, c*ratio).
    // Passing platform results keeps the same computation usable by host/AICPU
    // without a gert dependency or an assumed relationship between cores and blockDim.
    std::array<uint32_t, kMlaMaxAic> blockDims;
};

struct MlaFlashDecodeSchedule {
    uint32_t numOfFdHead;
    uint32_t maxS2SplitNum;
    uint32_t usedVecNumOfFd;
    uint32_t gS1BaseSizeOfFd;
    std::array<uint32_t, kMlaMaxAic> bN2IdxOfFdHead;
    std::array<uint32_t, kMlaMaxAic> gS1IdxOfFdHead;
    std::array<uint32_t, kMlaMaxAic> s2SplitNumOfFdHead;
    std::array<uint32_t, kMlaMaxAic> s2SplitStartIdxOfCore;
    std::array<uint32_t, kMlaMaxAic> gS1SplitNumOfFdHead;
    std::array<uint32_t, kMlaMaxAic> gS1LastPartSizeOfFdHead;
    std::array<uint32_t, kMlaMaxAic * 2> gS1IdxEndOfFdHead;
    std::array<uint32_t, kMlaMaxAic * 2> gS1IdxEndOfFdHeadSplit;
};

struct MlaWorkspace {
    // Source tiling fields: elements per core per slot, NOT bytes.
    uint64_t mm1ResSize;
    uint64_t mm2ResSize;
    // Source fills these even when FD is disabled; no FD workspace is allocated.
    uint64_t fdAccumOutSize;
    uint64_t fdLogSumExpSize;
    // Actual kernel segments use launch blockDim, while the host budget uses aicNum.
    uint64_t qkOffsetBytes;
    uint64_t pOffsetBytes;
    uint64_t pvOffsetBytes;
    uint64_t vec2OffsetBytes;
    uint64_t kernelBytes;
    uint64_t userBytes;
    uint64_t libApiBytes;
    uint64_t totalBytes;
};

struct MlaSchedule {
    uint32_t bSize;
    uint32_t n2Size;
    uint32_t gSize;
    uint32_t s1Size;
    uint32_t s2Size;
    uint32_t headDim;
    uint32_t headDimRope;
    uint32_t mBaseSize;
    uint32_t s2BaseSize;
    uint32_t aicNum;
    uint32_t aivNum;
    uint32_t cvRatio;
    uint32_t usedCoreNum;
    uint32_t blockDim;
    uint32_t scheduleMode; // CANN ScheduleMode::BATCH_MODE = 1.
    // Exclusive ends in (batch*Hkv, S1G block, S2 block) coordinates.
    // Core 0 starts at (0,0,0); subsequent cores start at the previous endpoint.
    std::array<uint32_t, kMlaMaxAic> bN2End;
    std::array<uint32_t, kMlaMaxAic> gS1End;
    std::array<uint32_t, kMlaMaxAic> s2End;
    int64_t maxCost;
    MlaFlashDecodeSchedule fd;
    MlaWorkspace workspace;
    MlaLocalMemory requiredLocalMemory;
};

enum class MlaScheduleStatus {
    Success,
    UnsupportedShape,
    FieldOverflow,
    InvalidPlatform,
    InsufficientLocalMemory,
    WorkspaceOverflow,
    UnexpectedFlashDecode,
    InvalidSplit,
    AllocationFailure,
};

// Pure CPU scheduling for contiguous BSND, shared 512+64 KV, Hkv=1,
// Sq*Hq>0 within the address/field bounds above, 1<=Sk<=kMlaMaxKvLength, B>0,
// no mask/length arrays, num_splits=1.
// FP16 and BF16 have identical scheduling. This is not a public forward or
// metadata entry point; callers must separately validate tensor/layout/options.
// No tensor addresses or scale are stored. On failure, output is unchanged.
MlaScheduleStatus BuildDenseSchedule(const MlaDenseShape& shape, const MlaPlatformInfo& platform, MlaSchedule& output);

} // namespace flash_attn::mla
