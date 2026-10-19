// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
#include "mla_split_plan.h"
#include <algorithm>
#include <new>
#include <stdexcept>
#include <utility>

// Fixed CANN/FA3 provenance and licenses are listed in tiling/mla_split_plan.h.
// Task ownership and persistent storage are local explicit-split adaptation;
// native auto-FD's variable-count scheduling is not reinterpreted here.

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
    if (c.numSplits < 1 || c.numSplits > 256) {
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

// Fixed CANN/FA3 provenance and licenses: tiling/mla_split_plan.h.
// Compact TND allocation bounds and persistent addressing are local adapters.

namespace flash_attn::mla {
namespace {

MlaAccumulatorInput AccumulatorInput(const MlaSplitTndPlanConfig& c)
{
    const auto& s = c.shape;
    return {MlaAccumulatorQueryLayout::Tnd, s.batch, s.maxQueryLength, s.totalQuery, s.queryHeads, c.numSplits};
}
} // namespace

MlaScheduleStatus BuildSplitTndPlanLayout(const MlaSplitTndPlanConfig& c, MlaSplitKvPlanLayout& output)
{
    const auto& s = c.shape;
    if (s.batch < 0 || s.totalQuery < 0 || s.totalKv < 0 || s.maxQueryLength < 0 || s.maxKvLength < 0 ||
        s.queryHeads < 1 || c.mBaseSize < 1 || c.mBaseSize > 256 || c.s2BaseSize < 1 || c.s2BaseSize > 512) {
        return MlaScheduleStatus::UnsupportedShape;
    }
    if (c.numSplits < 1 || c.numSplits > 256)
        return MlaScheduleStatus::InvalidSplit;
    if (c.aicNum < 1 || c.aicNum > kMlaMaxAic || c.blockDim < 1 || c.blockDim > c.aicNum ||
        (c.cvRatio != 1 && c.cvRatio != 2))
        return MlaScheduleStatus::InvalidPlatform;
    if (uint64_t(s.batch) > UINT32_MAX || s.totalQuery > INT32_MAX || s.totalKv > INT32_MAX ||
        s.maxQueryLength > kMlaMaxQueryLength || s.maxKvLength > kMlaMaxKvLength || s.queryHeads > kMlaMaxQueryHeads) {
        return MlaScheduleStatus::FieldOverflow;
    }
    const uint64_t rows = uint64_t(s.maxQueryLength) * s.queryHeads;
    const uint64_t totalRows = uint64_t(s.totalQuery) * s.queryHeads;
    if (rows > kMlaMaxDenseRows || totalRows > kMlaMaxBatchRows)
        return MlaScheduleStatus::FieldOverflow;
    if (uint64_t(s.totalQuery) > uint64_t(s.batch) * s.maxQueryLength ||
        uint64_t(s.totalKv) > uint64_t(s.batch) * s.maxKvLength)
        return MlaScheduleStatus::UnsupportedShape;
    MlaSplitKvPlanLayout result{};
    auto status = BuildSplitAccumulatorLayout(AccumulatorInput(c), result.accumulator);
    if (status != MlaScheduleStatus::Success)
        return status;
    if (c.libApiBytes > uint64_t(INT64_MAX))
        return MlaScheduleStatus::WorkspaceOverflow;
    result.emptyCapacity = s.totalQuery == 0 || s.totalKv == 0;
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

MlaScheduleStatus BuildSplitTndPlan(const MlaSplitTndPlanConfig& c, const MlaWindowConfig& window,
                                    const MlaUsedInputs& inputs, MlaSplitTndPlan& output)
{
    MlaSplitTndPlan result;
    auto status = BuildSplitTndPlanLayout(c, result.layout_);
    if (status != MlaScheduleStatus::Success)
        return status;
    const auto& s = c.shape;
    MlaWindow normalized{};
    if (window.maxQueryLength != s.maxQueryLength || window.maxKvLength != s.maxKvLength ||
        NormalizeWindow(window, normalized) != MlaCausalStatus::Success)
        return MlaScheduleStatus::UnsupportedShape;
    try {
        MlaUsedLengths lengths;
        status = ParseUsedLengths({s.batch,
                                   s.queryHeads,
                                   {MlaSequenceLayout::Tnd, s.totalQuery, s.maxQueryLength},
                                   {MlaSequenceLayout::Tnd, s.totalKv, s.maxKvLength}},
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

MlaScheduleStatus MlaSplitTndPlan::Merge(uint64_t index, MlaSplitKvPlannedMerge& output) const
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

MlaScheduleStatus MlaSplitTndPlan::Task(uint64_t index, MlaSplitTndPlannedTask& output) const
{
    if (index >= Tasks())
        return MlaScheduleStatus::InvalidSplit;
    const auto& c = config_;
    MlaSplitKvPlannedMerge merge{};
    Merge(index / c.numSplits, merge);
    MlaSplitTndPlannedTask result{};
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

MlaScheduleStatus MlaSplitTndPlan::Row(uint64_t task, uint32_t row, MlaSplitKvPlannedRow& output) const
{
    MlaSplitTndPlannedTask taskInfo{};
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

// Fixed CANN/FA3 provenance and licenses: tiling/mla_split_plan.h.
// Compact Q / rectangular KV allocation bounds and addressing are local adapters.
// Reuses the explicit range, workspace and accumulator arithmetic; KV origins
// remain 64-bit batch*capacity even beyond the TND prefix range.

namespace flash_attn::mla {
namespace {

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
    if (c.numSplits < 1 || c.numSplits > 256)
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

// Reuse pinned PA geometry and explicit split arithmetic, without flattening
// physical pages into a fictitious contiguous KV allocation.

namespace flash_attn::mla {
namespace {

MlaSplitKvPlanConfig ExplicitBsndConfig(const MlaPagedShape& s, const MlaPlatformInfo& p, int64_t splits, uint32_t m,
                                        uint32_t n)
{
    const bool active = s.batch && s.query.maxLength && s.logicalKvCapacity;
    return {s.batch,
            s.query.maxLength,
            s.logicalKvCapacity,
            s.queryHeads,
            m,
            n,
            splits,
            p.aicNum,
            active ? p.aicNum : 1,
            2,
            p.libApiWorkspaceBytes};
}
} // namespace

MlaScheduleStatus BuildSplitPagedBsndBudget(const MlaPagedShape& s, const MlaPlatformInfo& p, int64_t splits,
                                            MlaSplitPagedBsndBudget& output)
{
    if (s.query.layout != MlaSequenceLayout::Bsnd || s.pageSize < 16 || s.pageSize > 1024 || s.pageSize % 16 ||
        s.logicalKvCapacity < 0 || s.logicalKvCapacity > kMlaMaxKvLength || s.logicalKvCapacity % s.pageSize ||
        s.batch < 0 || uint64_t(s.batch) > (UINT32_MAX - 414) / 7)
        return MlaScheduleStatus::UnsupportedShape;
    MlaSchedule validation{};
    auto status = BuildDenseSchedule({s.batch, s.query.maxLength, s.queryHeads, 0}, p, validation);
    if (status != MlaScheduleStatus::Success)
        return status;
    if (s.query.totalTokens < 0 || uint64_t(s.query.totalTokens) != uint64_t(s.batch) * s.query.maxLength)
        return MlaScheduleStatus::UnsupportedShape;
    const uint32_t blocks = s.batch && s.query.maxLength && s.logicalKvCapacity ? p.aicNum : 1;
    if (p.aivNum != 2 * p.aicNum)
        return MlaScheduleStatus::InvalidPlatform;
    for (uint32_t i = 0; i < p.aicNum; ++i)
        if (p.blockDims[i] != blocks)
            return MlaScheduleStatus::InvalidPlatform;
    MlaSplitKvPlanLayout bound{};
    status = BuildSplitKvPlanLayout(ExplicitBsndConfig(s, p, splits, 256, PagedSplitCapacityBase(s)), bound);
    if (status != MlaScheduleStatus::Success)
        return status;
    MlaSplitPagedBsndBudget result{};
    const uint64_t rows = uint64_t(s.query.maxLength) * s.queryHeads;
    result.maxGroups = bound.emptyCapacity ? 0 : uint64_t(s.batch) * ((rows + 127) / 128);
    result.maxTasks = result.maxGroups * splits;
    result.partialRows = bound.partialRows;
    result.scratchBytes = bound.outOffsetBytes;
    result.userBytes = bound.userBytes;
    result.totalBytes = bound.totalBytes;
    result.accumulator = bound.accumulator;
    output = result;
    return MlaScheduleStatus::Success;
}

MlaScheduleStatus BuildSplitPagedBsndPlan(const MlaPagedShape& shape, const MlaPlatformInfo& platform,
                                          const MlaPagedInputs& inputs, const MlaWindowConfig& window, int64_t splits,
                                          MlaSplitPagedBsndPlan& output)
{
    try {
        MlaSplitPagedBsndPlan result;
        auto status = BuildSplitPagedBsndBudget(shape, platform, splits, result.budget);
        if (status != MlaScheduleStatus::Success)
            return status;
        status = BuildPagedSchedule(shape, platform, inputs, result.paged);
        if (status != MlaScheduleStatus::Success)
            return status;
        const auto& s = result.paged.schedule;
        result.mBase = s.mBaseSize ? s.mBaseSize : 256;
        result.nBase = s.s2BaseSize ? s.s2BaseSize : PagedSplitCapacityBase(shape);
        std::vector<MlaCausalSequence> lengths;
        lengths.reserve(result.paged.lengths.sequences.size());
        for (const auto& v : result.paged.lengths.sequences)
            lengths.push_back({v.queryLength, v.kvLength});
        status = BuildSplitKvPlan(ExplicitBsndConfig(shape, platform, splits, result.mBase, result.nBase), window,
                                  lengths, result.split);
        if (status != MlaScheduleStatus::Success)
            return status;
        const auto& l = result.split.Layout();
        if (result.split.Groups() > result.budget.maxGroups || l.partialRows > result.budget.partialRows ||
            l.userBytes > result.budget.userBytes)
            return MlaScheduleStatus::WorkspaceOverflow;
        output = std::move(result);
        return MlaScheduleStatus::Success;
    } catch (const std::bad_alloc&) {
        return MlaScheduleStatus::AllocationFailure;
    } catch (const std::length_error&) {
        return MlaScheduleStatus::AllocationFailure;
    }
}
namespace {

MlaSplitMixedPlanConfig ExplicitTndConfig(const MlaPagedShape& s, const MlaPlatformInfo& p, int64_t splits, uint32_t m,
                                          uint32_t n)
{
    const bool active = s.batch && s.query.totalTokens && s.logicalKvCapacity;
    return {{s.batch, s.query.totalTokens, s.queryHeads, s.query.maxLength, s.logicalKvCapacity},
            m,
            n,
            splits,
            p.aicNum,
            active ? p.aicNum : 1,
            2,
            p.libApiWorkspaceBytes};
}
} // namespace

MlaScheduleStatus BuildSplitPagedTndBudget(const MlaPagedShape& s, const MlaPlatformInfo& p, int64_t splits,
                                           MlaSplitPagedTndBudget& output)
{
    if (s.query.layout != MlaSequenceLayout::Tnd || s.pageSize < 16 || s.pageSize > 1024 || s.pageSize % 16 ||
        s.logicalKvCapacity < 0 || s.logicalKvCapacity > kMlaMaxKvLength || s.logicalKvCapacity % s.pageSize ||
        s.batch < 0 || uint64_t(s.batch) > (UINT32_MAX - 414) / 7)
        return MlaScheduleStatus::UnsupportedShape;
    MlaSchedule validation{};
    auto status = BuildDenseSchedule({0, s.query.maxLength, s.queryHeads, 0}, p, validation);
    if (status != MlaScheduleStatus::Success)
        return status;
    const uint32_t blocks = s.batch && s.query.totalTokens && s.logicalKvCapacity ? p.aicNum : 1;
    if (p.aivNum != 2 * p.aicNum)
        return MlaScheduleStatus::InvalidPlatform;
    for (uint32_t i = 0; i < p.aicNum; ++i)
        if (p.blockDims[i] != blocks)
            return MlaScheduleStatus::InvalidPlatform;
    MlaSplitKvPlanLayout bound{};
    status = BuildSplitMixedPlanLayout(ExplicitTndConfig(s, p, splits, 256, PagedSplitCapacityBase(s)), bound);
    if (status != MlaScheduleStatus::Success)
        return status;
    MlaSplitPagedTndBudget result{};
    const uint64_t rows = uint64_t(s.query.maxLength) * s.queryHeads;
    if (!bound.emptyCapacity) {
        const uint64_t totalRows = uint64_t(s.query.totalTokens) * s.queryHeads;
        const uint64_t active = std::min(uint64_t(s.batch), uint64_t(s.query.totalTokens));
        result.maxGroups = std::min(active * ((rows + 127) / 128), (totalRows + active * 127) / 128);
        result.maxTasks = result.maxGroups * splits;
        // A source divisor M<256 can pad unequal TND segments differently.
        // Bound every M in [128,256] by at most 255 extra rows per segment;
        // B*maxQ and a rectangular M=256 slot count are not valid substitutes.
        result.partialRows = (totalRows + active * 255) * splits;
        result.scratchBytes = bound.outOffsetBytes;
        if (result.partialRows > (uint64_t(INT64_MAX) - result.scratchBytes) / (528 * sizeof(float)))
            return MlaScheduleStatus::WorkspaceOverflow;
        result.userBytes = result.scratchBytes + result.partialRows * 528 * sizeof(float);
        if (p.libApiWorkspaceBytes > uint64_t(INT64_MAX) - result.userBytes)
            return MlaScheduleStatus::WorkspaceOverflow;
        result.totalBytes = result.userBytes + p.libApiWorkspaceBytes;
    }
    result.accumulator = bound.accumulator;
    output = result;
    return MlaScheduleStatus::Success;
}

MlaScheduleStatus BuildSplitPagedTndPlan(const MlaPagedShape& shape, const MlaPlatformInfo& platform,
                                         const MlaPagedInputs& inputs, const MlaWindowConfig& window, int64_t splits,
                                         MlaSplitPagedTndPlan& output)
{
    try {
        MlaSplitPagedTndPlan result;
        auto status = BuildSplitPagedTndBudget(shape, platform, splits, result.budget);
        if (status != MlaScheduleStatus::Success)
            return status;
        status = BuildPagedSchedule(shape, platform, inputs, result.paged);
        if (status != MlaScheduleStatus::Success)
            return status;
        const auto& s = result.paged.schedule;
        result.mBase = s.mBaseSize ? s.mBaseSize : 256;
        result.nBase = s.s2BaseSize ? s.s2BaseSize : PagedSplitCapacityBase(shape);
        // Only reuse the mixed plan's logical intervals and compact Q layout.
        // Its virtual KV origins never escape this PA adapter.
        status = BuildSplitMixedPlan(ExplicitTndConfig(shape, platform, splits, result.mBase, result.nBase), window,
                                     {inputs.cuQuery, {}, inputs.usedQuery, inputs.kvLengths}, result.logical_);
        if (status != MlaScheduleStatus::Success)
            return status;
        const auto& l = result.Layout();
        if (result.Groups() > result.budget.maxGroups || l.partialRows > result.budget.partialRows ||
            l.userBytes > result.budget.userBytes)
            return MlaScheduleStatus::WorkspaceOverflow;
        output = std::move(result);
        return MlaScheduleStatus::Success;
    } catch (const std::bad_alloc&) {
        return MlaScheduleStatus::AllocationFailure;
    } catch (const std::length_error&) {
        return MlaScheduleStatus::AllocationFailure;
    }
}
MlaScheduleStatus MlaSplitPagedTndPlan::Task(uint64_t index, MlaSplitPagedTndPlannedTask& output) const
{
    MlaSplitMixedPlannedTask logical{};
    const auto status = logical_.Task(index, logical);
    if (status != MlaScheduleStatus::Success)
        return status;
    output = {logical.partial, logical.sequence.queryStart, logical.sequence.queryCapacity,
              logical.sequence.queryLength, logical.sequence.kvLength};
    return MlaScheduleStatus::Success;
}
} // namespace flash_attn::mla
