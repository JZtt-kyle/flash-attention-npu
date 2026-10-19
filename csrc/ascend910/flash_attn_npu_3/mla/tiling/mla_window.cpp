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
// attention/common/op_host/split_core.cpp (band range/cost/SplitCore), adapted
// through the existing causal complete-M endpoint algorithm. LICENSE.cann applies.
// FA3 normalization follows Dao-AILab/flash-attention
// d15f1531a460ba456f41b01a774f33ab2db8febf hopper/flash_api.cpp.
// Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar,
// Pradeep Ramani, Tri Dao. BSD 3-Clause license reproduced in LICENSE.fa3.
// Changes: checked bounds, explicit empty rows, no tensor/runtime dependency.
#include "mla_window.h"
#include "../detail/cann_split_core.h"
#include <algorithm>
#include <new>

namespace flash_attn::mla {

MlaCausalStatus NormalizeWindow(const MlaWindowConfig& in, MlaWindow& output)
{
    if (in.maxQueryLength < 0 || in.maxKvLength < 0) {
        return MlaCausalStatus::UnsupportedShape;
    }
    if (uint64_t(in.maxQueryLength) > kMlaMaxQueryLength || uint64_t(in.maxKvLength) > UINT32_MAX) {
        return MlaCausalStatus::FieldOverflow;
    }
    if (in.maxKvLength > kMlaMaxKvLength) {
        return MlaCausalStatus::UnsupportedShape;
    }
    int64_t left = in.windowLeft >= in.maxKvLength - 1 ? -1 : in.windowLeft;
    int64_t right = in.windowRight >= in.maxQueryLength - 1 ? -1 : in.windowRight;
    const bool causal = in.causal && !(in.maxQueryLength == 1 && left == -1 && right == -1);
    if (causal) {
        right = 0;
    }
    // FA3 forward passes these host-normalized values to int parameters in
    // set_params_fprop; metadata retains int64. Reject lossy negative narrowing
    // instead of claiming the two paths have the same mask in that domain.
    if (left < INT32_MIN || right < INT32_MIN) {
        return MlaCausalStatus::FieldOverflow;
    }
    MlaWindow result{};
    result.kind = left < 0 && right == 0    ? MlaWindowKind::Causal
                  : left >= 0 || right >= 0 ? MlaWindowKind::Local
                                            : MlaWindowKind::None;
    result.left = left < 0 ? in.maxKvLength - 1 : left;
    result.right = right < 0 ? in.maxQueryLength - 1 : right;
    output = result;
    return MlaCausalStatus::Success;
}

namespace {
MlaCausalStatus ValidateSequence(const MlaCausalBlockInput& in, const MlaWindowConfig& config)
{
    MlaCausalBlock block{};
    const auto status = BuildCausalBlock(in, block);
    if (status != MlaCausalStatus::Success) {
        return status;
    }
    return in.queryLength <= config.maxQueryLength && in.kvLength <= config.maxKvLength
               ? MlaCausalStatus::Success
               : MlaCausalStatus::UnsupportedShape;
}

detail::BaseInfo WindowBase(uint32_t heads, const MlaWindow& window)
{
    detail::BaseInfo base{};
    base.n2Size = 1;
    base.gSize = heads;
    base.attenMaskFlag = window.kind != MlaWindowKind::None;
    base.sparseMode = int32_t(window.kind == MlaWindowKind::Local    ? detail::SparseMode::BAND
                              : window.kind == MlaWindowKind::Causal ? detail::SparseMode::RIGHT_DOWN_CAUSAL
                                                                     : detail::SparseMode::DEFAULT_MASK);
    base.preToken = window.kind == MlaWindowKind::Causal ? INT32_MAX : window.left;
    base.nextToken = window.kind == MlaWindowKind::Causal ? 0 : window.right;
    return base;
}

MlaCausalStatus Calculate(const MlaCausalScheduleInput& in, const MlaWindowConfig& config, const MlaWindow& window,
                          MlaCausalSchedule& result)
{
    auto status = ValidateSequence({0, 0, in.queryHeads, in.mBaseSize, in.s2BaseSize, 0}, config);
    if (status != MlaCausalStatus::Success) {
        return status;
    }
    if (in.aicNum < 1 || in.aicNum > kMlaMaxAic || (in.cvRatio != 1 && in.cvRatio != 2)) {
        return MlaCausalStatus::InvalidPlatform;
    }
    if (in.sequences.size() > UINT32_MAX) {
        return MlaCausalStatus::FieldOverflow;
    }
    // Check before the source's uint32 products, sums and ceil-div operations.
    // Use the same conservative rectangular tile bound as existing schedulers.
    uint64_t rows = 0, tiles = 0;
    for (const auto& seq : in.sequences) {
        status =
            ValidateSequence({seq.queryLength, seq.kvLength, in.queryHeads, in.mBaseSize, in.s2BaseSize, 0}, config);
        if (status != MlaCausalStatus::Success) {
            return status;
        }
        const uint64_t count = uint64_t(seq.queryLength) * in.queryHeads;
        rows += count;
        tiles += ((count + in.mBaseSize - 1) / in.mBaseSize) * ((seq.kvLength + in.s2BaseSize - 1) / in.s2BaseSize);
        if (rows > kMlaMaxBatchRows || tiles > UINT32_MAX) {
            return MlaCausalStatus::FieldOverflow;
        }
    }
    auto base = WindowBase(in.queryHeads, window);
    base.bSize = in.sequences.size();
    base.actualLenQDims = base.actualLenKvDims = base.bSize;
    base.actualSeqS1Size.reserve(in.sequences.size());
    base.actualSeqS2Size.reserve(in.sequences.size());
    std::vector<uint64_t> prefix;
    prefix.reserve(in.sequences.size() + 1);
    prefix.push_back(0);
    for (const auto& seq : in.sequences) {
        base.actualSeqS1Size.push_back(seq.queryLength);
        base.actualSeqS2Size.push_back(seq.kvLength);
        prefix.push_back(prefix.back() + (uint64_t(seq.queryLength) * in.queryHeads + in.mBaseSize - 1) / in.mBaseSize);
    }
    result.totalMBlocks = prefix.back();
    detail::SplitParam param{};
    param.mBaseSize = in.mBaseSize;
    param.s2BaseSize = in.s2BaseSize;
    detail::SplitContext context(base, param);
    detail::CalcSplitInfo(context);
    detail::CalcCostInfo(context);
    result.totalCost = context.costInfo.totalCost;
    result.totalS2Blocks = context.costInfo.totalBlockNum;
    detail::SplitResult split{uint32_t(in.aicNum), uint32_t(in.cvRatio)};
    detail::SplitCore(in.aicNum, context, split);
    if (split.usedCoreNum == 0 || split.usedCoreNum > uint64_t(in.aicNum)) {
        return MlaCausalStatus::InvalidSplit;
    }
    uint64_t previous = 0;
    uint32_t costBatch = 0, costM = 0;
    int64_t assignedCost = 0;
    for (uint32_t c = 0; c < split.usedCoreNum; ++c) {
        uint32_t b = split.bN2End[c];
        uint64_t m = uint64_t(split.gS1End[c]) + (split.s2End[c] != 0);
        if (b > base.bSize || (b == base.bSize && m != 0) || (b < base.bSize && m > prefix[b + 1] - prefix[b])) {
            return MlaCausalStatus::InvalidSplit;
        }
        // Include all trailing zero-cost query blocks: their O/LSE need owners.
        if (c + 1 == split.usedCoreNum) {
            b = base.bSize;
            m = 0;
        }
        const uint64_t end = prefix[b] + m;
        if (end < previous) {
            return MlaCausalStatus::InvalidSplit;
        }
        if (end == previous && result.totalMBlocks != 0) {
            continue;
        }
        // Canonicalize a whole-M endpoint past completed/empty samples.
        while (b < base.bSize && m == prefix[b + 1] - prefix[b]) {
            ++b;
            m = 0;
        }
        int64_t cost = 0;
        while (costBatch < b || (costBatch == b && costM < m)) {
            const uint64_t limit = costBatch < b ? prefix[costBatch + 1] - prefix[costBatch] : m;
            if (in.sequences[costBatch].kvLength != 0) {
                detail::BatchCache batch{};
                detail::CalcBatchCache(costBatch, context, batch);
                for (; costM < limit; ++costM) {
                    detail::S1GCache block{};
                    detail::CalcS1GCache(costM, context, batch, block);
                    cost += block.s1GCost;
                }
            }
            if (costBatch == b) {
                costM = m;
                break;
            }
            ++costBatch;
            costM = 0;
        }
        const uint32_t core = result.usedCoreNum++;
        result.bN2End[core] = b;
        result.gS1End[core] = m;
        result.coreCost[core] = cost;
        result.maxCost = std::max(result.maxCost, cost);
        assignedCost += cost;
        previous = end;
    }
    return result.usedCoreNum != 0 && previous == result.totalMBlocks && assignedCost == result.totalCost
               ? MlaCausalStatus::Success
               : MlaCausalStatus::InvalidSplit;
}
} // namespace

MlaCausalStatus BuildWindowBlock(const MlaCausalBlockInput& in, const MlaWindowConfig& config, MlaWindowBlock& output)
{
    MlaWindow window{};
    auto status = NormalizeWindow(config, window);
    if (status != MlaCausalStatus::Success) {
        return status;
    }
    MlaWindowBlock result{};
    status = BuildCausalBlock(in, result);
    if (status != MlaCausalStatus::Success) {
        return status;
    }
    if (in.queryLength > config.maxQueryLength || in.kvLength > config.maxKvLength) {
        return MlaCausalStatus::UnsupportedShape;
    }
    if (window.kind == MlaWindowKind::Causal) {
        result.preTokenLeftUp = INT32_MAX;
        output = result;
        return MlaCausalStatus::Success;
    }
    auto base = WindowBase(in.queryHeads, window);
    detail::SplitParam param{};
    param.mBaseSize = in.mBaseSize;
    param.s2BaseSize = in.s2BaseSize;
    detail::BatchCache batch{};
    batch.s1Size = in.queryLength;
    batch.s2Size = in.kvLength;
    result.preTokenLeftUp = batch.preTokenLeftUp = detail::CalcPreTokenLeftUp(batch.s1Size, batch.s2Size, base);
    result.nextTokenLeftUp = batch.nextTokenLeftUp = detail::CalcNextTokenLeftUp(batch.s1Size, batch.s2Size, base);
    result.kvBegin = result.kvEnd = result.leadingEmptyRows = 0;
    result.cost = result.normalS2Cost = result.lastS2Cost = 0;
    if (result.rowEnd > result.rowBegin) {
        if (window.kind == MlaWindowKind::None) {
            result.kvEnd = in.kvLength;
            result.leadingEmptyRows = in.kvLength == 0 ? result.rowEnd - result.rowBegin : 0;
        } else {
            const int64_t first = std::max(int64_t(0), int64_t(result.queryBegin) - result.preTokenLeftUp);
            const int64_t end = std::min(in.kvLength, int64_t(result.queryEnd) + result.nextTokenLeftUp);
            if (first < end) {
                result.kvBegin = first;
                result.kvEnd = end;
            }
            const int64_t emptyQueries =
                in.kvLength == 0 ? in.queryLength : std::clamp(-result.nextTokenLeftUp, int64_t(0), in.queryLength);
            result.leadingEmptyRows = std::clamp(emptyQueries * in.queryHeads - result.rowBegin, int64_t(0),
                                                 int64_t(result.rowEnd - result.rowBegin));
        }
    }
    const auto range = detail::CalcS2Range(in.mBlock, base, param, batch);
    result.s2Begin = range.first;
    result.s2End = range.second;
    if (result.s2End > result.s2Begin) {
        result.normalS2Cost = detail::CalcCost(result.rowEnd - result.rowBegin, in.s2BaseSize);
        const uint32_t lastSize =
            std::min(uint64_t(in.s2BaseSize), uint64_t(in.kvLength) - uint64_t(result.s2End - 1) * in.s2BaseSize);
        result.lastS2Cost = detail::CalcCost(result.rowEnd - result.rowBegin, lastSize);
        result.cost = int64_t(result.s2End - result.s2Begin - 1) * result.normalS2Cost + result.lastS2Cost;
    }
    output = result;
    return MlaCausalStatus::Success;
}

MlaCausalStatus BuildWindowSchedule(const MlaCausalScheduleInput& input, const MlaWindowConfig& config,
                                    MlaCausalSchedule& output)
{
    MlaWindow window{};
    auto status = NormalizeWindow(config, window);
    if (status != MlaCausalStatus::Success) {
        return status;
    }
    if (window.kind == MlaWindowKind::Causal) {
        for (const auto& seq : input.sequences) {
            status = ValidateSequence(
                {seq.queryLength, seq.kvLength, input.queryHeads, input.mBaseSize, input.s2BaseSize, 0}, config);
            if (status != MlaCausalStatus::Success) {
                return status;
            }
        }
        return BuildCausalSchedule(input, output);
    }
    try {
        MlaCausalSchedule result{};
        status = Calculate(input, config, window, result);
        if (status == MlaCausalStatus::Success) {
            output = result;
        }
        return status;
    } catch (const std::bad_alloc&) {
        return MlaCausalStatus::AllocationFailure;
    }
}
} // namespace flash_attn::mla
