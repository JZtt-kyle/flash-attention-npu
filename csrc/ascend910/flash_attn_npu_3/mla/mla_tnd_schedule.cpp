/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Derived from CANN ops-transformer ca18aee610729ca5270a44e2d84c751bc687fbf0,
// attention/fused_infer_attention_score/op_host/fused_infer_attention_score_tiling_info_parser.cpp:
// GetS1Size, GetMaxActualSeq, ProcessKVActualSeqLoop; and op_host/arch22/fia_tiling_nonquant_mla.cpp:
// CalcInnerSize, CalcMBaseSize, CreateSplitInput, CalcMmResSize, CalcNormalWorkspaceSize.
// The upstream license is reproduced in LICENSE.cann next to this file.
// Changes: FA3 int32 prefix validation and owned token spans; numerical int64
// cumulative tails for the existing SplitCore; whole-M endpoint mapping for
// num_splits=1 with variable Q lengths. No production/device entry changes.

#include "mla_tnd_schedule.h"
#include "detail/cann_split_core.h"

#include <algorithm>
#include <limits>
#include <new>
#include <numeric>
#include <utility>

namespace flash_attn::mla {
namespace {

MlaScheduleStatus Parse(const MlaTndShape& shape, const int32_t* cuQuery, uint64_t queryCount, const int32_t* cuKv,
                        uint64_t kvCount, MlaTndLengths& result)
{
    for (int64_t size :
         {shape.batch, shape.totalQuery, shape.totalKv, shape.queryHeads, shape.maxQueryLength, shape.maxKvLength}) {
        if (size < 0) {
            return MlaScheduleStatus::UnsupportedShape;
        }
        if (uint64_t(size) > UINT32_MAX) {
            return MlaScheduleStatus::FieldOverflow;
        }
    }
    if (shape.queryHeads == 0 || shape.maxKvLength > kMlaMaxKvLength) {
        return MlaScheduleStatus::UnsupportedShape;
    }
    if (shape.totalQuery > INT32_MAX || shape.totalKv > INT32_MAX || shape.queryHeads > kMlaMaxQueryHeads ||
        shape.maxQueryLength > kMlaMaxQueryLength ||
        uint64_t(shape.maxQueryLength) * shape.queryHeads > kMlaMaxDenseRows ||
        uint64_t(shape.totalQuery) * shape.queryHeads > kMlaMaxBatchRows) {
        return MlaScheduleStatus::FieldOverflow;
    }
    const uint64_t count = uint64_t(shape.batch) + 1;
    if (cuQuery == nullptr || cuKv == nullptr || queryCount != count || kvCount != count) {
        return MlaScheduleStatus::UnsupportedShape;
    }
    if (cuQuery[0] != 0 || cuKv[0] != 0) {
        return MlaScheduleStatus::UnsupportedShape;
    }
    uint32_t maxQ = 0, maxKv = 0;
    for (uint64_t i = 1; i < count; ++i) {
        // Check before subtraction/narrowing, including INT32_MIN and negative tails.
        if (cuQuery[i] < cuQuery[i - 1] || cuKv[i] < cuKv[i - 1]) {
            return MlaScheduleStatus::UnsupportedShape;
        }
        const uint32_t q = uint32_t(cuQuery[i]) - uint32_t(cuQuery[i - 1]);
        const uint32_t kv = uint32_t(cuKv[i]) - uint32_t(cuKv[i - 1]);
        if (q > shape.maxQueryLength || kv > shape.maxKvLength) {
            return MlaScheduleStatus::UnsupportedShape;
        }
        maxQ = std::max(maxQ, q);
        maxKv = std::max(maxKv, kv);
    }
    if (cuQuery[shape.batch] != shape.totalQuery || cuKv[shape.batch] != shape.totalKv) {
        return MlaScheduleStatus::UnsupportedShape;
    }
    result.maxQueryLength = maxQ;
    result.maxKvLength = maxKv;
    result.sequences.reserve(shape.batch);
    for (uint64_t b = 0; b < uint64_t(shape.batch); ++b) {
        result.sequences.push_back({uint32_t(cuQuery[b]), uint32_t(cuKv[b]), uint32_t(cuQuery[b + 1] - cuQuery[b]),
                                    uint32_t(cuKv[b + 1] - cuKv[b])});
    }
    return MlaScheduleStatus::Success;
}

uint64_t Align(uint64_t value, uint64_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

bool ValidSplit(const detail::SplitResult& split, const detail::BaseInfo& base, const std::vector<uint32_t>& blocks,
                uint32_t aicNum)
{
    if (split.usedCoreNum == 0 || split.usedCoreNum > aicNum) {
        return false;
    }
    std::pair<uint32_t, uint32_t> previous{0, 0};
    for (uint32_t i = 0; i < split.usedCoreNum; ++i) {
        const auto b = split.bN2End[i], m = split.gS1End[i];
        if (b > base.bSize || split.s2End[i] != 0 || (b == base.bSize ? m != 0 : m >= std::max(blocks[b], 1U))) {
            return false;
        }
        const std::pair<uint32_t, uint32_t> end{b, m};
        if (end <= previous) {
            return false;
        }
        previous = end;
    }
    // As in the source, compute endpoints may omit a zero-cost suffix.
    // A future device consumer must initialize zero-KV outputs separately.
    for (uint32_t b = previous.first; b < base.bSize; ++b) {
        if (detail::GetS1SeqSize(b, base) != 0 && detail::GetS2SeqSize(b, base) != 0) {
            return false;
        }
    }
    return true;
}

void SplitWithoutFlashDecode(const detail::BaseInfo& base, const detail::SplitParam& param,
                             const std::vector<uint32_t>& blocks, detail::SplitResult& split)
{
    const uint32_t cores = split.bN2End.size();
    detail::SplitCore(cores, base, param, split);
    if (split.numOfFdHead == 0) {
        return;
    }
    detail::SplitContext context(base, param);
    detail::CalcSplitInfo(context);
    // Prefix sums over variable-sized batches, never B times a common M count.
    std::vector<uint64_t> rowPrefix(uint64_t(base.bSize) + 1, 0);
    std::vector<int64_t> costPrefix(uint64_t(base.bSize) + 1, 0), firstCost(base.bSize, 0);
    for (uint32_t b = 0; b < base.bSize; ++b) {
        rowPrefix[b + 1] = rowPrefix[b] + blocks[b];
        costPrefix[b + 1] = costPrefix[b];
        if (blocks[b] != 0) {
            detail::BatchCache batch{};
            detail::CalcBatchCache(b, context, batch);
            detail::S1GCache first{}, last{};
            detail::CalcS1GCache(0, context, batch, first);
            detail::CalcS1GCache(blocks[b] - 1, context, batch, last);
            firstCost[b] = first.s1GCost;
            costPrefix[b + 1] += int64_t(blocks[b] - 1) * first.s1GCost + last.s1GCost;
        }
    }
    detail::SplitResult whole{cores, split.vecCubeRatio};
    uint64_t previousRow = 0;
    int64_t previousCost = 0;
    for (uint32_t i = 0; i < split.usedCoreNum; ++i) {
        uint32_t b = split.bN2End[i], m = split.gS1End[i] + (split.s2End[i] != 0);
        if (b < base.bSize && m != 0 && m == blocks[b]) {
            ++b;
            m = 0;
        }
        const uint64_t end = rowPrefix[b] + m;
        if (end > previousRow) {
            const int64_t cost = costPrefix[b] + (b < base.bSize ? int64_t(m) * firstCost[b] : 0);
            const uint32_t core = whole.usedCoreNum++;
            whole.bN2End[core] = b;
            whole.gS1End[core] = m;
            whole.maxCost = std::max(whole.maxCost, cost - previousCost);
            previousRow = end;
            previousCost = cost;
        }
    }
    split = std::move(whole);
}

MlaScheduleStatus Calculate(const MlaTndShape& shape, const MlaPlatformInfo& platform, MlaTndSchedule& result)
{
    const auto& lengths = result.lengths;
    auto& s = result.schedule;
    // Reuse existing platform/local-memory validation and static geometry.
    // B=0 deliberately avoids the rectangular B*maxQ address/tile bound: TND
    // has only totalQuery rows. Parse already checked its true address bound.
    auto status = BuildDenseSchedule({0, lengths.maxQueryLength, shape.queryHeads, lengths.maxKvLength}, platform, s);
    if (status != MlaScheduleStatus::Success) {
        return status;
    }
    s.bSize = uint32_t(shape.batch);
    if (shape.batch == 0 || shape.totalQuery == 0 || shape.totalKv == 0) {
        return MlaScheduleStatus::Success;
    }
    // Preserve ProcessKVActualSeqLoop: equality is over cumulative tails,
    // NOT the per-sample differences (even for uniform compact TND).
    const uint32_t firstTail = lengths.sequences[0].kvLength;
    bool sameTails = true;
    for (const auto& seq : lengths.sequences) {
        sameTails = sameTails && seq.kvStart + seq.kvLength == firstTail;
    }
    const uint64_t rows = uint64_t(s.s1Size) * s.gSize;
    s.mBaseSize = 256;
    if (s.bSize % s.aicNum != 0 && sameTails) {
        const uint64_t lcm = uint64_t(s.bSize / std::gcd(s.bSize, s.aicNum)) * s.aicNum;
        if (lcm > UINT32_MAX) {
            return MlaScheduleStatus::FieldOverflow;
        }
        const uint32_t parts = lcm / s.bSize;
        if (rows % parts == 0 && rows / parts >= 128 && rows / parts <= 256) {
            s.mBaseSize = rows / parts;
        }
    }
    s.s2BaseSize = std::min(s.s2Size, 512U);
    s.requiredLocalMemory = {7 * 128 * 288 * 2, (s.cvRatio == 1 ? 174U : 170U) * 1024, 65536, 65536, 131072};
    std::vector<uint32_t> blocks;
    blocks.reserve(s.bSize);
    uint64_t totalTiles = 0;
    for (const auto& seq : lengths.sequences) {
        const uint64_t m = (uint64_t(seq.queryLength) * s.gSize + s.mBaseSize - 1) / s.mBaseSize;
        totalTiles += m * ((seq.kvLength + s.s2BaseSize - 1) / s.s2BaseSize);
        if (totalTiles > UINT32_MAX) {
            return MlaScheduleStatus::FieldOverflow;
        }
        blocks.push_back(m);
    }
    auto& w = s.workspace;
    const uint64_t m = Align(std::min(rows, uint64_t(s.mBaseSize)), 16);
    w.mm1ResSize = m * Align(s.s2BaseSize, 32);
    w.mm2ResSize = m * s.headDim;
    w.fdAccumOutSize = uint64_t(s.aicNum) * 2 * s.mBaseSize * s.headDim;
    w.fdLogSumExpSize = uint64_t(s.aicNum) * 2 * 2 * s.mBaseSize * (32 / sizeof(float));
    w.userBytes = uint64_t(2) * s.aicNum * (6 * w.mm1ResSize + 8 * w.mm2ResSize + 8 * s.mBaseSize);
    w.libApiBytes = platform.libApiWorkspaceBytes;
    if (w.libApiBytes > UINT64_MAX - w.userBytes) {
        return MlaScheduleStatus::WorkspaceOverflow;
    }
    w.totalBytes = w.userBytes + w.libApiBytes;
    detail::BaseInfo base{};
    base.bSize = s.bSize;
    base.n2Size = 1;
    base.gSize = s.gSize;
    base.s1Size = s.s1Size;
    base.s2Size = s.s2Size;
    base.isAccumSeqS1 = base.isAccumSeqS2 = true;
    base.actualLenQDims = base.actualLenKvDims = s.bSize;
    base.actualSeqS1Size.reserve(s.bSize);
    base.actualSeqS2Size.reserve(s.bSize);
    for (const auto& seq : lengths.sequences) {
        base.actualSeqS1Size.push_back(int64_t(seq.queryStart) + seq.queryLength);
        base.actualSeqS2Size.push_back(int64_t(seq.kvStart) + seq.kvLength);
    }
    detail::SplitParam param{};
    param.mBaseSize = s.mBaseSize;
    param.s2BaseSize = s.s2BaseSize;
    detail::SplitResult split{s.aicNum, s.cvRatio};
    SplitWithoutFlashDecode(base, param, blocks, split);
    if (split.numOfFdHead != 0 || split.maxS2SplitNum != 0 || split.usedVecNumOfFd != 0) {
        return MlaScheduleStatus::UnexpectedFlashDecode;
    }
    if (!ValidSplit(split, base, blocks, s.aicNum)) {
        return MlaScheduleStatus::InvalidSplit;
    }
    s.usedCoreNum = split.usedCoreNum;
    s.blockDim = platform.blockDims[s.usedCoreNum - 1];
    if (s.blockDim < s.usedCoreNum) {
        return MlaScheduleStatus::InvalidPlatform;
    }
    s.maxCost = split.maxCost;
    std::copy(split.bN2End.begin(), split.bN2End.end(), s.bN2End.begin());
    std::copy(split.gS1End.begin(), split.gS1End.end(), s.gS1End.begin());
    std::copy(split.s2End.begin(), split.s2End.end(), s.s2End.begin());
    // Native non-FD SplitCore leaves all FD vectors zero, as does whole-M mapping.
    s.fd.gS1BaseSizeOfFd = param.gS1BaseSizeOfFd;
    const uint64_t slots = uint64_t(2) * s.blockDim;
    w.pOffsetBytes = slots * w.mm1ResSize * 4;
    w.pvOffsetBytes = w.pOffsetBytes + slots * w.mm1ResSize * 2;
    w.vec2OffsetBytes = w.pvOffsetBytes + slots * w.mm2ResSize * 4;
    w.kernelBytes = w.vec2OffsetBytes + slots * w.mm2ResSize * 4;
    return w.kernelBytes <= w.userBytes ? MlaScheduleStatus::Success : MlaScheduleStatus::WorkspaceOverflow;
}
} // namespace

MlaScheduleStatus ParseTndLengths(const MlaTndShape& shape, const int32_t* cuQuery, uint64_t queryCount,
                                  const int32_t* cuKv, uint64_t kvCount, MlaTndLengths& output)
{
    try {
        MlaTndLengths result;
        auto status = Parse(shape, cuQuery, queryCount, cuKv, kvCount, result);
        if (status == MlaScheduleStatus::Success) {
            output = std::move(result);
        }
        return status;
    } catch (const std::bad_alloc&) {
        return MlaScheduleStatus::AllocationFailure;
    }
}

MlaScheduleStatus BuildTndSchedule(const MlaTndShape& shape, const MlaPlatformInfo& platform, const int32_t* cuQuery,
                                   uint64_t queryCount, const int32_t* cuKv, uint64_t kvCount, MlaTndSchedule& output)
{
    try {
        MlaTndSchedule result;
        auto status = Parse(shape, cuQuery, queryCount, cuKv, kvCount, result.lengths);
        if (status == MlaScheduleStatus::Success) {
            status = Calculate(shape, platform, result);
        }
        if (status == MlaScheduleStatus::Success) {
            output = std::move(result);
        }
        return status;
    } catch (const std::bad_alloc&) {
        return MlaScheduleStatus::AllocationFailure;
    }
}
} // namespace flash_attn::mla
