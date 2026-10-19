/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Source: CANN ops-transformer ca18aee610729ca5270a44e2d84c751bc687fbf0,
// attention/common/op_host/split_core.cpp::CalcS2Range / CalcNextTokenLeftUp.
// The upstream license is reproduced in mla/LICENSE.cann.

#pragma once
#include "mla_schedule.h"
#include <vector>

namespace flash_attn::mla {

struct MlaCausalBlockInput {
    // Effective lengths of ONE sample, not allocation capacities or batch maxima.
    // Layout/address/used validation belongs to the caller. Rows use S1G order.
    int64_t queryLength;
    int64_t kvLength;
    int64_t queryHeads;
    int64_t mBaseSize;
    int64_t s2BaseSize;
    int64_t mBlock;
};

struct MlaCausalBlock {
    int64_t nextTokenLeftUp; // Sk - Sq; key j is visible iff j <= query i + this value.
    uint32_t rowBegin;
    uint32_t rowEnd;
    uint32_t queryBegin;
    uint32_t queryEnd;
    uint32_t kvBegin;
    uint32_t kvEnd;
    uint32_t s2Begin;
    uint32_t s2End;
    uint32_t leadingEmptyRows;
    int64_t cost;
    int64_t normalS2Cost;
    int64_t lastS2Cost;
};

enum class MlaCausalStatus {
    Success,
    UnsupportedShape,
    FieldOverflow,
    InvalidTile,
    InvalidBlock,
    InvalidPlatform,
    InvalidSplit,
    AllocationFailure,
};

// CPU-only geometry for shared 512+64 KV, Hkv=1, right-down causal, num_splits=1.
// Reuses the pinned source's S2 block range. All intervals are half-open and
// relative to one sample. kvBegin/kvEnd are the union of visible logical keys;
// boundary S2 blocks still need a per-row causal mask in the attention kernel.
// leadingEmptyRows counts flattened S1G rows with NO visible key, including
// all rows when Sk=0. It is not a count of query tokens or unused allocation.
// Sq=0 accepts only mBlock=0 and produces empty intervals. Otherwise mBlock
// must name a real M block. Logical tiles are 1..256 by 1..512; accepting a
// size here does not validate device resources or enable a production kernel.
// Uses the existing per-sample MLA shape bounds. Output is unchanged on failure.
// No tensor, metadata format, core partition, workspace or public entry is created.
MlaCausalStatus BuildCausalBlock(const MlaCausalBlockInput& input, MlaCausalBlock& output);

struct MlaCausalSequence {
    int64_t queryLength, kvLength;
};

struct MlaCausalScheduleInput {
    // Owned effective lengths, obtained after the existing layout/used parsers.
    // No allocation prefix, page ID, address or batch maximum is a mask length.
    std::vector<MlaCausalSequence> sequences;
    int64_t queryHeads, mBaseSize, s2BaseSize;
    int64_t aicNum, cvRatio;
};

struct MlaCausalSchedule {
    uint32_t usedCoreNum;
    std::array<uint32_t, kMlaMaxAic> bN2End, gS1End, s2End;
    std::array<int64_t, kMlaMaxAic> coreCost;
    int64_t maxCost, totalCost;
    uint64_t totalMBlocks, totalS2Blocks;
};

// Independent CPU partition at caller-selected logical tile sizes. Retain the
// pinned SplitCore candidate, round partial S2 endpoints up to whole M blocks,
// coalesce duplicates and recompute each core's mask-dependent cost. The last
// endpoint is always (B,0,0), including trailing zero-cost output blocks.
// Every effective Q row has one owner, even if it sees no key. Empty work uses
// one zero-cost core. Unused array entries are zero; output is unchanged on
// failure. No FD, workspace, platform resource validation or production launch.
MlaCausalStatus BuildCausalSchedule(const MlaCausalScheduleInput& input, MlaCausalSchedule& output);

} // namespace flash_attn::mla
