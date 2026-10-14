// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
// Fixed CANN/FA3 provenance and licenses are listed in mla_split_kv_plan.h.
// Task ownership and persistent storage are local explicit-split adaptation;
// native auto-FD's variable-count scheduling is not reinterpreted here.
#include "mla_split_kv_plan.h"
#include <algorithm>
#include <new>
#include <stdexcept>
#include <utility>

namespace flash_attn::mla {
namespace {
bool Multiply(uint64_t a, uint64_t b, uint64_t& output)
{
    if (b && a > uint64_t(INT64_MAX) / b) {
        return false;
    }
    output = a * b;
    return true;
}
bool Add(uint64_t a, uint64_t b, uint64_t& output)
{
    if (b > uint64_t(INT64_MAX) || a > uint64_t(INT64_MAX) - b) {
        return false;
    }
    output = a + b;
    return true;
}
MlaAccumulatorInput AccumulatorInput(const MlaSplitKvPlanConfig& c)
{
    return {MlaAccumulatorQueryLayout::Bsnd, c.batch,      c.queryCapacity,
            c.batch * c.queryCapacity,       c.queryHeads, c.numSplits};
}
} // namespace

MlaScheduleStatus BuildSplitKvPlanLayout(const MlaSplitKvPlanConfig& c, MlaSplitKvPlanLayout& output)
{
    if (c.batch < 0 || c.queryCapacity < 0 || c.kvCapacity < 0 || c.queryHeads < 1 || c.mBaseSize < 1 ||
        c.mBaseSize > 256 || c.s2BaseSize < 1 || c.s2BaseSize > 512) {
        return MlaScheduleStatus::UnsupportedShape;
    }
    if (c.numSplits < 2 || c.numSplits > 256) {
        return MlaScheduleStatus::InvalidSplit;
    }
    if (c.aicNum < 1 || c.aicNum > kMlaMaxAic || c.blockDim < 1 || c.blockDim > c.aicNum ||
        (c.cvRatio != 1 && c.cvRatio != 2)) {
        return MlaScheduleStatus::InvalidPlatform;
    }
    if (uint64_t(c.batch) > UINT32_MAX || c.queryCapacity > kMlaMaxQueryLength || c.kvCapacity > kMlaMaxKvLength ||
        c.queryHeads > kMlaMaxQueryHeads) {
        return MlaScheduleStatus::FieldOverflow;
    }
    const uint64_t rows = uint64_t(c.queryCapacity) * c.queryHeads;
    if (rows > kMlaMaxDenseRows || (rows && uint64_t(c.batch) > kMlaMaxBatchRows / rows)) {
        return MlaScheduleStatus::FieldOverflow;
    }
    MlaSplitKvPlanLayout result{};
    auto status = BuildSplitAccumulatorLayout(AccumulatorInput(c), result.accumulator);
    if (status != MlaScheduleStatus::Success) {
        return status;
    }
    if (c.libApiBytes > uint64_t(INT64_MAX)) {
        return MlaScheduleStatus::WorkspaceOverflow;
    }
    result.emptyCapacity = c.batch == 0 || rows == 0 || c.kvCapacity == 0;
    if (!result.emptyCapacity) {
        const uint64_t perSample = (rows + c.mBaseSize - 1) / c.mBaseSize;
        uint64_t outBytes, statsBytes;
        if (!Multiply(c.batch, perSample, result.maxGroups) ||
            !Multiply(result.maxGroups, c.numSplits, result.maxTasks) ||
            !Multiply(result.maxTasks, c.mBaseSize, result.partialRows) ||
            !Multiply(result.partialRows, 512 * sizeof(float), outBytes) ||
            !Multiply(result.partialRows, 8 * sizeof(float), statsBytes)) {
            return MlaScheduleStatus::WorkspaceOverflow;
        }
        status = BuildSplitKvWorkspace({c.mBaseSize, c.s2BaseSize, int64_t(rows), c.aicNum, c.blockDim, false, 0},
                                       result.scratch);
        if (status != MlaScheduleStatus::Success) {
            return status;
        }
        // Logical M can be odd. Native aMLA host reserve is not necessarily
        // 32-byte aligned, while the persistent FP32 statistics require it.
        result.outOffsetBytes = (result.scratch.attention.userBytes + 31) / 32 * 32;
        if (!Add(result.outOffsetBytes, outBytes, result.sumOffsetBytes) ||
            !Add(result.sumOffsetBytes, statsBytes, result.maxOffsetBytes) ||
            !Add(result.maxOffsetBytes, statsBytes, result.userBytes) ||
            !Add(result.userBytes, c.libApiBytes, result.totalBytes)) {
            return MlaScheduleStatus::WorkspaceOverflow;
        }
    }
    output = result;
    return MlaScheduleStatus::Success;
}

MlaScheduleStatus BuildSplitKvPlan(const MlaSplitKvPlanConfig& config, const MlaWindowConfig& window,
                                   const std::vector<MlaCausalSequence>& lengths, MlaSplitKvPlan& output)
{
    MlaSplitKvPlan result;
    auto status = BuildSplitKvPlanLayout(config, result.layout_);
    if (status != MlaScheduleStatus::Success) {
        return status;
    }
    MlaWindow normalized{};
    if (window.maxQueryLength != config.queryCapacity || window.maxKvLength != config.kvCapacity ||
        NormalizeWindow(window, normalized) != MlaCausalStatus::Success || lengths.size() != uint64_t(config.batch)) {
        return MlaScheduleStatus::UnsupportedShape;
    }
    for (const auto& seq : lengths) {
        if (seq.queryLength < 0 || seq.queryLength > config.queryCapacity || seq.kvLength < 0 ||
            seq.kvLength > config.kvCapacity) {
            return MlaScheduleStatus::UnsupportedShape;
        }
    }
    try {
        result.config_ = config;
        result.window_ = window;
        result.lengths_ = lengths;
        result.prefix_.reserve(lengths.size() + 1);
        result.prefix_.push_back(0);
        for (const auto& seq : lengths) {
            const uint64_t rows = uint64_t(seq.queryLength) * config.queryHeads;
            const uint64_t groups = result.layout_.emptyCapacity ? 0 : (rows + config.mBaseSize - 1) / config.mBaseSize;
            result.prefix_.push_back(result.prefix_.back() + groups);
        }
        output = std::move(result);
        return MlaScheduleStatus::Success;
    } catch (const std::bad_alloc&) {
        return MlaScheduleStatus::AllocationFailure;
    } catch (const std::length_error&) {
        return MlaScheduleStatus::AllocationFailure;
    }
}

MlaScheduleStatus MlaSplitKvPlan::Merge(uint64_t index, MlaSplitKvPlannedMerge& output) const
{
    if (index >= Groups()) {
        return MlaScheduleStatus::InvalidSplit;
    }
    const auto& c = config_;
    const uint32_t batch = uint32_t(std::upper_bound(prefix_.begin(), prefix_.end(), index) - prefix_.begin() - 1);
    const uint64_t m = index - prefix_[batch];
    const uint64_t row = m * c.mBaseSize;
    const uint64_t slot = index * c.numSplits * c.mBaseSize;
    const uint32_t vectors = c.blockDim * c.cvRatio;
    MlaSplitKvPlannedMerge result{};
    result.group = index;
    result.iteration = index / vectors;
    result.batch = batch;
    result.m = m;
    result.owner = index % vectors;
    result.rowBegin = row;
    result.rowEnd = std::min(row + c.mBaseSize, uint64_t(lengths_[batch].queryLength) * c.queryHeads);
    result.outOffsetBytes = layout_.outOffsetBytes + slot * 512 * sizeof(float);
    result.sumOffsetBytes = layout_.sumOffsetBytes + slot * 8 * sizeof(float);
    result.maxOffsetBytes = layout_.maxOffsetBytes + slot * 8 * sizeof(float);
    output = result;
    return MlaScheduleStatus::Success;
}

MlaScheduleStatus MlaSplitKvPlan::Task(uint64_t index, MlaSplitKvPlannedTask& output) const
{
    if (index >= Tasks()) {
        return MlaScheduleStatus::InvalidSplit;
    }
    const auto& c = config_;
    MlaSplitKvPlannedMerge merge{};
    Merge(index / c.numSplits, merge);
    const auto& seq = lengths_[merge.batch];
    MlaSplitKvPlannedTask result{};
    result.task = index;
    result.group = merge.group;
    result.iteration = index / c.blockDim;
    result.owner = index % c.blockDim;
    result.batch = merge.batch;
    result.m = merge.m;
    result.split = index % c.numSplits;
    if (BuildSplitKvBlock({seq.queryLength, seq.kvLength, c.queryHeads, c.mBaseSize, c.s2BaseSize, merge.m}, window_,
                          c.numSplits, result.split, result.fragment) != MlaCausalStatus::Success) {
        return MlaScheduleStatus::InvalidSplit;
    }
    const uint64_t slot = uint64_t(result.split) * c.mBaseSize;
    result.outOffsetBytes = merge.outOffsetBytes + slot * 512 * sizeof(float);
    result.sumOffsetBytes = merge.sumOffsetBytes + slot * 8 * sizeof(float);
    result.maxOffsetBytes = merge.maxOffsetBytes + slot * 8 * sizeof(float);
    output = result;
    return MlaScheduleStatus::Success;
}

MlaScheduleStatus MlaSplitKvPlan::Row(uint64_t task, uint32_t row, MlaSplitKvPlannedRow& output) const
{
    MlaSplitKvPlannedTask t{};
    auto status = Task(task, t);
    if (status != MlaScheduleStatus::Success) {
        return status;
    }
    if (row >= t.fragment.whole.rowEnd - t.fragment.whole.rowBegin) {
        return MlaScheduleStatus::InvalidSplit;
    }
    const auto& c = config_;
    const uint64_t flat = uint64_t(t.fragment.whole.rowBegin) + row;
    const uint64_t query = flat / c.queryHeads, head = flat % c.queryHeads;
    MlaSplitKvPlannedRow result{};
    status = GetSplitAccumulatorOffsets(AccumulatorInput(c), t.split, t.batch, head, query, result.accumulatorOut,
                                        result.accumulatorLse);
    if (status != MlaScheduleStatus::Success) {
        return status;
    }
    result.out = (uint64_t(t.batch) * c.queryCapacity * c.queryHeads + flat) * 512 * 2;
    result.lse = ((uint64_t(t.batch) * c.queryHeads + head) * c.queryCapacity + query) * sizeof(float);
    result.partialOut = t.outOffsetBytes + row * 512 * sizeof(float);
    result.partialSum = t.sumOffsetBytes + row * 8 * sizeof(float);
    result.partialMax = t.maxOffsetBytes + row * 8 * sizeof(float);
    output = result;
    return MlaScheduleStatus::Success;
}
} // namespace flash_attn::mla
