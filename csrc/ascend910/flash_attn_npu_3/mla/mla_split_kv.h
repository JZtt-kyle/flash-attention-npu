/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Sources: CANN ca18aee610729ca5270a44e2d84c751bc687fbf0 SplitCore and
// fia_tiling_nonquant_mla.cpp; FA3 d15f1531a460ba456f41b01a774f33ab2db8febf
// hopper/flash_api.cpp and block.h. See LICENSE.cann and LICENSE.fa3.
#pragma once

#include "mla_window.h"

namespace flash_attn::mla {

// Independent CPU protocol, not a production metadata record or a device launch.
// Lengths are already parsed allocation/used lengths; no pointers are retained.
struct MlaNativeSplitKvSchedule {
    uint32_t usedCoreNum;
    std::array<uint32_t, kMlaMaxAic> bN2End, gS1End, s2End;
    int64_t maxCost, totalCost;
    uint64_t totalMBlocks, totalS2Blocks;
    MlaFlashDecodeSchedule fd;
};

// Preserve native variable-count FD partitions and ALL reduction bookkeeping.
// Zero-cost rows can be outside native endpoints and require output initialization.
// Logical tiles do not imply validated device resources. FD reduction M tile = 8.
MlaCausalStatus BuildNativeSplitKvSchedule(const MlaCausalScheduleInput& input, const MlaWindowConfig& window,
                                           MlaNativeSplitKvSchedule& output);

// FA3 <=0 requests backend auto, 1 disables splitting, 2..256 is explicit.
// autoSplits is a backend-selected upper bound (1 if native fd.maxS2SplitNum=0),
// not FA3's CUDA occupancy heuristic. Ignored for positive requests.
MlaCausalStatus ResolveSplitKvRequest(int64_t requested, int64_t autoSplits, uint32_t& output);

struct MlaSplitKvBlock {
    MlaWindowBlock whole;
    uint32_t s2Begin, s2End, kvBegin, kvEnd;
    int64_t cost;
};

// Uniform explicit splits of the validated mask-dependent S2 interval, using
// FA3's ceil-div rule. numSplits is resolved, 1..256. Empty splits are [end,end),
// with zero cost; boundary tiles still need the original row mask. This is
// separate from native auto's cost-balanced endpoints, not a reinterpretation.
MlaCausalStatus BuildSplitKvBlock(const MlaCausalBlockInput& input, const MlaWindowConfig& window, int64_t numSplits,
                                  int64_t splitIndex, MlaSplitKvBlock& output);

struct MlaSplitKvWorkspaceInput {
    int64_t mBaseSize, s2BaseSize, maxQueryRows;
    int64_t aicNum, blockDim;
    bool flashDecode;
    uint64_t libApiBytes;
};
struct MlaSplitKvWorkspace {
    MlaWorkspace attention;
    // Offsets from the user workspace, before libApiBytes is added to the budget.
    // Source FD stores unnormalized FP32 O and separate sum/max (8 floats/row).
    uint64_t fdAccumOffsetBytes, fdSumOffsetBytes, fdMaxOffsetBytes;
};

// Source host budget uses aicNum; kernel segments use blockDim. FD capacity is
// two tasks per physical AIC. Preserves the extra host aMLA accounting. This is
// not the public FP32 O/LSE accumulator, and does not initialize any buffer.
MlaScheduleStatus BuildSplitKvWorkspace(const MlaSplitKvWorkspaceInput& input, MlaSplitKvWorkspace& output);

enum class MlaAccumulatorQueryLayout {
    Bsnd,
    Tnd
};
struct MlaAccumulatorInput {
    MlaAccumulatorQueryLayout layout;
    int64_t batch, queryCapacity, totalQuery, queryHeads, numSplits;
};
struct MlaAccumulatorTensor {
    uint32_t rank;
    std::array<uint64_t, 5> shape, strides; // contiguous ELEMENT strides, unused entries zero
    uint64_t elements, bytes;
};
struct MlaAccumulatorLayout {
    MlaAccumulatorTensor out, lse;
};

// FA3 allocation capacity, not used rows: BSND [S,B,H,Q,512]/[S,B,H,Q],
// TND [S,H,T,512]/[S,H,T]. FP32 for S>1; S=1 has no accumulator storage.
// BSND requires totalQuery=B*queryCapacity; TND totalQuery<=INT32_MAX, with
// prefix/max/used relationships validated by the existing length parser.
MlaScheduleStatus BuildSplitAccumulatorLayout(const MlaAccumulatorInput& input, MlaAccumulatorLayout& output);

// Checked FP32 row offsets in bytes; TND uses absolute allocation-token query
// and batch=0. S=1 or out-of-range coordinates reject. Failure is atomic for
// this and every builder in this header; callers never receive partial output.
MlaScheduleStatus GetSplitAccumulatorOffsets(const MlaAccumulatorInput& input, int64_t split, int64_t batch,
                                             int64_t head, int64_t query, uint64_t& outBytes, uint64_t& lseBytes);

} // namespace flash_attn::mla
