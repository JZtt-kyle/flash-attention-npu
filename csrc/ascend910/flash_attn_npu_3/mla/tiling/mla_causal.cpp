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
// attention/common/op_host/split_core.cpp: CalcS2Range, CalcNextTokenLeftUp,
// CalcS1GCache, SplitCore. The upstream license is reproduced in LICENSE.cann.
// Changes: checked effective S1G geometry, explicit empty-row ownership, and
// FA3 num_splits=1 whole-M endpoints with causal costs recomputed per block.
#include "mla_causal.h"
#include "../metadata/mla_mask.h"
#include "../detail/cann_split_core.h"
#include <algorithm>
#include <new>

namespace flash_attn::mla {
namespace {

MlaCausalStatus Validate(int64_t q, int64_t k, int64_t heads, int64_t m, int64_t n)
{
    for (int64_t size : {q, k, heads}) {
        if (size < 0) {
            return MlaCausalStatus::UnsupportedShape;
        }
        if (uint64_t(size) > UINT32_MAX) {
            return MlaCausalStatus::FieldOverflow;
        }
    }
    if (heads == 0 || k > kMlaMaxKvLength) {
        return MlaCausalStatus::UnsupportedShape;
    }
    if (q > kMlaMaxQueryLength || heads > kMlaMaxQueryHeads || uint64_t(q) * heads > kMlaMaxDenseRows) {
        return MlaCausalStatus::FieldOverflow;
    }
    if (m <= 0 || m > 256 || n <= 0 || n > 512) {
        return MlaCausalStatus::InvalidTile;
    }
    return MlaCausalStatus::Success;
}

detail::BaseInfo CausalBase(uint32_t heads)
{
    detail::BaseInfo base{};
    base.n2Size = 1;
    base.gSize = heads;
    base.attenMaskFlag = true;
    base.sparseMode = int32_t(detail::SparseMode::RIGHT_DOWN_CAUSAL);
    // Unlimited left context, with ample headroom for the source subtraction.
    base.preToken = INT32_MAX;
    return base;
}

MlaCausalStatus Calculate(const MlaCausalScheduleInput& in, MlaCausalSchedule& result)
{
    auto status = Validate(0, 0, in.queryHeads, in.mBaseSize, in.s2BaseSize);
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
        status = Validate(seq.queryLength, seq.kvLength, in.queryHeads, in.mBaseSize, in.s2BaseSize);
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
    auto base = CausalBase(in.queryHeads);
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

MlaCausalStatus BuildCausalBlock(const MlaCausalBlockInput& in, MlaCausalBlock& output)
{
    const auto status = Validate(in.queryLength, in.kvLength, in.queryHeads, in.mBaseSize, in.s2BaseSize);
    if (status != MlaCausalStatus::Success) {
        return status;
    }
    const uint64_t rows = uint64_t(in.queryLength) * in.queryHeads;
    const uint64_t blocks = (rows + in.mBaseSize - 1) / in.mBaseSize;
    if (in.mBlock < 0 || uint64_t(in.mBlock) >= std::max(blocks, uint64_t(1))) {
        return MlaCausalStatus::InvalidBlock;
    }
    MlaCausalBlock result{};
    auto base = CausalBase(in.queryHeads);
    detail::SplitParam param{};
    param.mBaseSize = in.mBaseSize;
    param.s2BaseSize = in.s2BaseSize;
    detail::BatchCache batch{};
    batch.s1Size = in.queryLength;
    batch.s2Size = in.kvLength;
    batch.preTokenLeftUp = base.preToken;
    batch.nextTokenLeftUp = detail::CalcNextTokenLeftUp(batch.s1Size, batch.s2Size, base);
    result.nextTokenLeftUp = batch.nextTokenLeftUp;
    result.rowBegin = uint64_t(in.mBlock) * in.mBaseSize;
    result.rowEnd = std::min(uint64_t(result.rowBegin) + in.mBaseSize, rows);
    if (rows != 0) {
        result.queryBegin = result.rowBegin / in.queryHeads;
        result.queryEnd = (result.rowEnd - 1) / in.queryHeads + 1;
        result.kvEnd = std::max(int64_t(0), int64_t(result.queryEnd) + result.nextTokenLeftUp);
        const int64_t emptyEnd = std::max(in.queryLength - in.kvLength, int64_t(0)) * in.queryHeads;
        result.leadingEmptyRows =
            std::clamp(emptyEnd - result.rowBegin, int64_t(0), int64_t(result.rowEnd - result.rowBegin));
    }
    const auto range = detail::CalcS2Range(in.mBlock, base, param, batch);
    result.s2Begin = range.first;
    result.s2End = range.second;
    if (result.s2End != 0) {
        result.normalS2Cost = detail::CalcCost(result.rowEnd - result.rowBegin, in.s2BaseSize);
        const uint32_t lastSize =
            std::min(uint64_t(in.s2BaseSize), uint64_t(in.kvLength) - uint64_t(result.s2End - 1) * in.s2BaseSize);
        result.lastS2Cost = detail::CalcCost(result.rowEnd - result.rowBegin, lastSize);
        result.cost = int64_t(result.s2End - 1) * result.normalS2Cost + result.lastS2Cost;
    }
    output = result;
    return MlaCausalStatus::Success;
}

MlaCausalStatus BuildCausalSchedule(const MlaCausalScheduleInput& input, MlaCausalSchedule& output)
{
    try {
        MlaCausalSchedule result{};
        const auto status = Calculate(input, result);
        if (status == MlaCausalStatus::Success) {
            output = result;
        }
        return status;
    } catch (const std::bad_alloc&) {
        return MlaCausalStatus::AllocationFailure;
    }
}

bool BuildCausalFullSchedule(const MlaUsedShape& shape, const MlaPlatformInfo& platform, MlaCausalFullSchedule& output)
{
    if (shape.batch <= 0 || shape.query.maxLength <= 0 || shape.kv.maxLength <= 0 ||
        shape.query.layout != MlaSequenceLayout::Bsnd || shape.kv.layout != MlaSequenceLayout::Bsnd) {
        return false;
    }
    try {
        MlaUsedSchedule prepared;
        if (PrepareUsedSchedule(shape, platform, {}, prepared) != MlaScheduleStatus::Success) {
            return false;
        }
        const auto& s = prepared.schedule;
        MlaCausalScheduleInput input{{}, shape.queryHeads, s.mBaseSize, s.s2BaseSize, s.aicNum, s.cvRatio};
        input.sequences.assign(shape.batch, {shape.query.maxLength, shape.kv.maxLength});
        MlaCausalSchedule schedule{};
        if (BuildCausalSchedule(input, schedule) != MlaCausalStatus::Success) {
            return false;
        }
        MlaCausalFullSchedule full{};
        full.enabled = 1;
        full.batch = shape.batch;
        full.queryCapacity = shape.query.maxLength;
        full.kvCapacity = shape.kv.maxLength;
        full.heads = shape.queryHeads;
        full.aicNum = s.aicNum;
        full.cvRatio = s.cvRatio;
        full.mBaseSize = s.mBaseSize;
        full.s2BaseSize = s.s2BaseSize;
        full.usedCoreNum = schedule.usedCoreNum;
        full.bN2End = schedule.bN2End;
        full.gS1End = schedule.gS1End;
        full.s2End = schedule.s2End;
        full.maxCost = schedule.maxCost;
        output = full;
        return true;
    } catch (const std::bad_alloc&) {
        return false;
    }
}
} // namespace flash_attn::mla
