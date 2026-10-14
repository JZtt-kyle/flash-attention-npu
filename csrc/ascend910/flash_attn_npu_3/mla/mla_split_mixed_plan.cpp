// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
// Fixed CANN/FA3 provenance and licenses: mla_split_mixed_plan.h.
// Compact Q / rectangular KV allocation bounds and addressing are local adapters.
// Reuses the explicit range, workspace and accumulator arithmetic; KV origins
// remain 64-bit batch*capacity even beyond the TND prefix range.
#include "mla_split_mixed_plan.h"
#include <algorithm>
#include <new>
#include <stdexcept>
#include <utility>

namespace flash_attn::mla {
namespace {
bool Multiply(uint64_t a, uint64_t b, uint64_t& output)
{
    if (b && a > uint64_t(INT64_MAX) / b)
        return false;
    output = a * b;
    return true;
}
bool Add(uint64_t a, uint64_t b, uint64_t& output)
{
    if (b > uint64_t(INT64_MAX) || a > uint64_t(INT64_MAX) - b)
        return false;
    output = a + b;
    return true;
}
MlaAccumulatorInput AccumulatorInput(const MlaSplitMixedPlanConfig& c)
{
    const auto& s = c.shape;
    return {MlaAccumulatorQueryLayout::Tnd, s.batch, s.maxQueryLength, s.totalQuery, s.queryHeads, c.numSplits};
}
} // namespace

MlaScheduleStatus BuildSplitMixedPlanLayout(const MlaSplitMixedPlanConfig& c, MlaSplitKvPlanLayout& output)
{
    const auto& s = c.shape;
    if (s.batch < 0 || s.totalQuery < 0 || s.maxQueryLength < 0 || s.kvCapacity < 0 || s.queryHeads < 1 ||
        c.mBaseSize < 1 || c.mBaseSize > 256 || c.s2BaseSize < 1 || c.s2BaseSize > 512) {
        return MlaScheduleStatus::UnsupportedShape;
    }
    if (c.numSplits < 2 || c.numSplits > 256)
        return MlaScheduleStatus::InvalidSplit;
    if (c.aicNum < 1 || c.aicNum > kMlaMaxAic || c.blockDim < 1 || c.blockDim > c.aicNum ||
        (c.cvRatio != 1 && c.cvRatio != 2))
        return MlaScheduleStatus::InvalidPlatform;
    if (uint64_t(s.batch) > UINT32_MAX || s.totalQuery > INT32_MAX || s.maxQueryLength > kMlaMaxQueryLength ||
        s.kvCapacity > kMlaMaxKvLength || s.queryHeads > kMlaMaxQueryHeads) {
        return MlaScheduleStatus::FieldOverflow;
    }
    const uint64_t rows = uint64_t(s.maxQueryLength) * s.queryHeads;
    const uint64_t totalRows = uint64_t(s.totalQuery) * s.queryHeads;
    if (rows > kMlaMaxDenseRows || totalRows > kMlaMaxBatchRows)
        return MlaScheduleStatus::FieldOverflow;
    if (uint64_t(s.totalQuery) > uint64_t(s.batch) * s.maxQueryLength)
        return MlaScheduleStatus::UnsupportedShape;
    MlaSplitKvPlanLayout result{};
    auto status = BuildSplitAccumulatorLayout(AccumulatorInput(c), result.accumulator);
    if (status != MlaScheduleStatus::Success)
        return status;
    if (c.libApiBytes > uint64_t(INT64_MAX))
        return MlaScheduleStatus::WorkspaceOverflow;
    result.emptyCapacity = s.totalQuery == 0 || s.batch == 0 || s.kvCapacity == 0;
    if (!result.emptyCapacity) {
        const uint64_t active = std::min(uint64_t(s.batch), uint64_t(s.totalQuery));
        // Each positive segment contributes at most M-1 padding rows. This
        // bound remains valid with arbitrary used lengths and zero segments.
        const uint64_t perSample = (rows + c.mBaseSize - 1) / c.mBaseSize;
        result.maxGroups = std::min(active * perSample, (totalRows + active * (c.mBaseSize - 1)) / c.mBaseSize);
        uint64_t outBytes, statsBytes;
        if (!Multiply(result.maxGroups, c.numSplits, result.maxTasks) ||
            !Multiply(result.maxTasks, c.mBaseSize, result.partialRows) ||
            !Multiply(result.partialRows, 512 * sizeof(float), outBytes) ||
            !Multiply(result.partialRows, 8 * sizeof(float), statsBytes))
            return MlaScheduleStatus::WorkspaceOverflow;
        status = BuildSplitKvWorkspace({c.mBaseSize, c.s2BaseSize, int64_t(rows), c.aicNum, c.blockDim, false, 0},
                                       result.scratch);
        if (status != MlaScheduleStatus::Success)
            return status;
        result.outOffsetBytes = (result.scratch.attention.userBytes + 31) / 32 * 32;
        if (!Add(result.outOffsetBytes, outBytes, result.sumOffsetBytes) ||
            !Add(result.sumOffsetBytes, statsBytes, result.maxOffsetBytes) ||
            !Add(result.maxOffsetBytes, statsBytes, result.userBytes) ||
            !Add(result.userBytes, c.libApiBytes, result.totalBytes))
            return MlaScheduleStatus::WorkspaceOverflow;
    }
    output = result;
    return MlaScheduleStatus::Success;
}

MlaScheduleStatus BuildSplitMixedPlan(const MlaSplitMixedPlanConfig& c, const MlaWindowConfig& window,
                                      const MlaUsedInputs& inputs, MlaSplitMixedPlan& output)
{
    MlaSplitMixedPlan result;
    auto status = BuildSplitMixedPlanLayout(c, result.layout_);
    if (status != MlaScheduleStatus::Success)
        return status;
    const auto& s = c.shape;
    MlaWindow normalized{};
    if (window.maxQueryLength != s.maxQueryLength || window.maxKvLength != s.kvCapacity ||
        NormalizeWindow(window, normalized) != MlaCausalStatus::Success)
        return MlaScheduleStatus::UnsupportedShape;
    try {
        MlaUsedLengths lengths;
        status = ParseUsedLengths({s.batch,
                                   s.queryHeads,
                                   {MlaSequenceLayout::Tnd, s.totalQuery, s.maxQueryLength},
                                   {MlaSequenceLayout::Bsnd, s.batch * s.kvCapacity, s.kvCapacity}},
                                  inputs, lengths);
        if (status != MlaScheduleStatus::Success)
            return status;
        result.config_ = c;
        result.window_ = window;
        result.sequences_ = std::move(lengths.sequences);
        result.prefix_.reserve(result.sequences_.size() + 1);
        result.prefix_.push_back(0);
        for (const auto& seq : result.sequences_) {
            const uint64_t rows = uint64_t(seq.queryLength) * s.queryHeads;
            const uint64_t groups = result.layout_.emptyCapacity ? 0 : (rows + c.mBaseSize - 1) / c.mBaseSize;
            result.prefix_.push_back(result.prefix_.back() + groups);
        }
        if (result.Groups() > result.layout_.maxGroups)
            return MlaScheduleStatus::WorkspaceOverflow;
        output = std::move(result);
        return MlaScheduleStatus::Success;
    } catch (const std::bad_alloc&) {
        return MlaScheduleStatus::AllocationFailure;
    } catch (const std::length_error&) {
        return MlaScheduleStatus::AllocationFailure;
    }
}

MlaScheduleStatus MlaSplitMixedPlan::Merge(uint64_t index, MlaSplitKvPlannedMerge& output) const
{
    if (index >= Groups())
        return MlaScheduleStatus::InvalidSplit;
    const auto& c = config_;
    const uint32_t batch = uint32_t(std::upper_bound(prefix_.begin(), prefix_.end(), index) - prefix_.begin() - 1);
    const uint64_t m = index - prefix_[batch], row = m * c.mBaseSize;
    const uint64_t slot = index * c.numSplits * c.mBaseSize;
    const uint32_t vectors = c.blockDim * c.cvRatio;
    MlaSplitKvPlannedMerge result{};
    result.group = index;
    result.iteration = index / vectors;
    result.batch = batch;
    result.m = m;
    result.owner = index % vectors;
    result.rowBegin = row;
    result.rowEnd = std::min(row + c.mBaseSize, uint64_t(sequences_[batch].queryLength) * c.shape.queryHeads);
    result.outOffsetBytes = layout_.outOffsetBytes + slot * 512 * sizeof(float);
    result.sumOffsetBytes = layout_.sumOffsetBytes + slot * 8 * sizeof(float);
    result.maxOffsetBytes = layout_.maxOffsetBytes + slot * 8 * sizeof(float);
    output = result;
    return MlaScheduleStatus::Success;
}

MlaScheduleStatus MlaSplitMixedPlan::Task(uint64_t index, MlaSplitMixedPlannedTask& output) const
{
    if (index >= Tasks())
        return MlaScheduleStatus::InvalidSplit;
    const auto& c = config_;
    MlaSplitKvPlannedMerge merge{};
    Merge(index / c.numSplits, merge);
    MlaSplitMixedPlannedTask result{};
    auto& t = result.partial;
    const auto& seq = sequences_[merge.batch];
    result.sequence = seq;
    t.task = index;
    t.group = merge.group;
    t.iteration = index / c.blockDim;
    t.owner = index % c.blockDim;
    t.batch = merge.batch;
    t.m = merge.m;
    t.split = index % c.numSplits;
    if (BuildSplitKvBlock({seq.queryLength, seq.kvLength, c.shape.queryHeads, c.mBaseSize, c.s2BaseSize, merge.m},
                          window_, c.numSplits, t.split, t.fragment) != MlaCausalStatus::Success) {
        return MlaScheduleStatus::InvalidSplit;
    }
    const uint64_t slot = uint64_t(t.split) * c.mBaseSize;
    t.outOffsetBytes = merge.outOffsetBytes + slot * 512 * sizeof(float);
    t.sumOffsetBytes = merge.sumOffsetBytes + slot * 8 * sizeof(float);
    t.maxOffsetBytes = merge.maxOffsetBytes + slot * 8 * sizeof(float);
    output = result;
    return MlaScheduleStatus::Success;
}

MlaScheduleStatus MlaSplitMixedPlan::Row(uint64_t task, uint32_t row, MlaSplitKvPlannedRow& output) const
{
    MlaSplitMixedPlannedTask taskInfo{};
    auto status = Task(task, taskInfo);
    if (status != MlaScheduleStatus::Success)
        return status;
    const auto& t = taskInfo.partial;
    if (row >= t.fragment.whole.rowEnd - t.fragment.whole.rowBegin)
        return MlaScheduleStatus::InvalidSplit;
    const auto& s = config_.shape;
    const uint64_t flat = uint64_t(t.fragment.whole.rowBegin) + row;
    const uint64_t query = taskInfo.sequence.queryStart + flat / s.queryHeads, head = flat % s.queryHeads;
    MlaSplitKvPlannedRow result{};
    status = GetSplitAccumulatorOffsets(AccumulatorInput(config_), t.split, 0, head, query, result.accumulatorOut,
                                        result.accumulatorLse);
    if (status != MlaScheduleStatus::Success)
        return status;
    result.out = (query * s.queryHeads + head) * 512 * 2;
    result.lse = (head * s.totalQuery + query) * sizeof(float);
    result.partialOut = t.outOffsetBytes + row * 512 * sizeof(float);
    result.partialSum = t.sumOffsetBytes + row * 8 * sizeof(float);
    result.partialMax = t.maxOffsetBytes + row * 8 * sizeof(float);
    output = result;
    return MlaScheduleStatus::Success;
}
} // namespace flash_attn::mla
