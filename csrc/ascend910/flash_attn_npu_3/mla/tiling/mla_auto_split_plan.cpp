// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
#include "mla_auto_split_plan.h"
#include <algorithm>
#include <new>
#include <stdexcept>
#include <tuple>
#include <utility>

// Fixed CANN/FA3 provenance is listed in tiling/mla_auto_split_plan.h.
// Native partition and FD endpoints are retained; persistent addresses and
// zero-cost output ownership are local adaptations, not uniform resplitting.

namespace flash_attn::mla {
namespace {
MlaAccumulatorInput Accumulator(const MlaAutoSplitKvConfig& c)
{
    return {MlaAccumulatorQueryLayout::Bsnd, c.batch,      c.queryCapacity,
            c.batch * c.queryCapacity,       c.queryHeads, c.aicNum};
}
void Offsets(const MlaAutoSplitKvLayout& l, uint64_t row, uint64_t& out, uint64_t& sum, uint64_t& max)
{
    out = l.outOffsetBytes + row * 512 * sizeof(float);
    sum = l.sumOffsetBytes + row * 8 * sizeof(float);
    max = l.maxOffsetBytes + row * 8 * sizeof(float);
}
MlaScheduleStatus Convert(MlaCausalStatus s)
{
    switch (s) {
    case MlaCausalStatus::Success:
        return MlaScheduleStatus::Success;
    case MlaCausalStatus::FieldOverflow:
        return MlaScheduleStatus::FieldOverflow;
    case MlaCausalStatus::InvalidPlatform:
        return MlaScheduleStatus::InvalidPlatform;
    case MlaCausalStatus::AllocationFailure:
        return MlaScheduleStatus::AllocationFailure;
    default:
        return MlaScheduleStatus::InvalidSplit;
    }
}
} // namespace

MlaScheduleStatus BuildAutoSplitKvLayout(const MlaAutoSplitKvConfig& c, MlaAutoSplitKvLayout& output)
{
    if (c.batch < 0 || c.queryCapacity < 0 || c.kvCapacity < 0 || c.queryHeads < 1 || c.mBaseSize < 1 ||
        c.mBaseSize > 256 || c.s2BaseSize < 1 || c.s2BaseSize > 512) {
        return MlaScheduleStatus::UnsupportedShape;
    }
    if (c.aicNum < 1 || c.aicNum > kMlaMaxAic || (c.cvRatio != 1 && c.cvRatio != 2)) {
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
    MlaAutoSplitKvLayout result{};
    auto status = BuildSplitAccumulatorLayout(Accumulator(c), result.accumulator);
    if (status != MlaScheduleStatus::Success)
        return status;
    if (c.libApiBytes > uint64_t(INT64_MAX))
        return MlaScheduleStatus::WorkspaceOverflow;
    result.emptyCapacity = c.batch == 0 || rows == 0 || c.kvCapacity == 0;
    if (!result.emptyCapacity) {
        result.maxGroups = uint64_t(c.batch) * ((rows + c.mBaseSize - 1) / c.mBaseSize);
        // Each native core has at most its first and last M block in FD.
        result.maxPartialSlots = c.aicNum == 1 ? 0 : 2 * c.aicNum;
        result.partialRows = result.maxPartialSlots * c.mBaseSize;
        status = BuildSplitKvWorkspace({c.mBaseSize, c.s2BaseSize, int64_t(rows), c.aicNum, c.aicNum, false, 0},
                                       result.scratch);
        if (status != MlaScheduleStatus::Success)
            return status;
        result.outOffsetBytes = (result.scratch.attention.userBytes + 31) / 32 * 32;
        result.sumOffsetBytes = result.outOffsetBytes + result.partialRows * 512 * sizeof(float);
        result.maxOffsetBytes = result.sumOffsetBytes + result.partialRows * 8 * sizeof(float);
        result.userBytes = result.maxOffsetBytes + result.partialRows * 8 * sizeof(float);
        if (c.libApiBytes > uint64_t(INT64_MAX) - result.userBytes)
            return MlaScheduleStatus::WorkspaceOverflow;
        result.totalBytes = result.userBytes + c.libApiBytes;
    }
    output = result;
    return MlaScheduleStatus::Success;
}

MlaScheduleStatus BuildAutoSplitKvPlan(const MlaAutoSplitKvConfig& c, const MlaWindowConfig& window,
                                       const std::vector<MlaCausalSequence>& lengths, MlaAutoSplitKvPlan& output)
{
    MlaAutoSplitKvPlan result;
    auto status = BuildAutoSplitKvLayout(c, result.layout_);
    if (status != MlaScheduleStatus::Success)
        return status;
    MlaWindow normalized{};
    if (window.maxQueryLength != c.queryCapacity || window.maxKvLength != c.kvCapacity ||
        NormalizeWindow(window, normalized) != MlaCausalStatus::Success || lengths.size() != uint64_t(c.batch)) {
        return MlaScheduleStatus::UnsupportedShape;
    }
    for (const auto& seq : lengths) {
        if (seq.queryLength < 0 || seq.queryLength > c.queryCapacity || seq.kvLength < 0 ||
            seq.kvLength > c.kvCapacity) {
            return MlaScheduleStatus::UnsupportedShape;
        }
    }
    try {
        result.config_ = c;
        result.window_ = window;
        result.lengths_ = lengths;
        result.prefix_.reserve(lengths.size() + 1);
        result.prefix_.push_back(0);
        for (const auto& seq : lengths) {
            const uint64_t rows = uint64_t(seq.queryLength) * c.queryHeads;
            result.prefix_.push_back(result.prefix_.back() +
                                     (result.layout_.emptyCapacity ? 0 : (rows + c.mBaseSize - 1) / c.mBaseSize));
        }
        status = Convert(BuildNativeSplitKvSchedule(
            {lengths, c.queryHeads, c.mBaseSize, c.s2BaseSize, c.aicNum, c.cvRatio}, window, result.native_));
        if (status != MlaScheduleStatus::Success)
            return status;
        const auto& fd = result.native_.fd;
        for (uint32_t head = 0; head < fd.numOfFdHead; ++head) {
            const uint32_t b = fd.bN2IdxOfFdHead[head], m = fd.gS1IdxOfFdHead[head];
            if (b >= lengths.size() || m >= result.prefix_[b + 1] - result.prefix_[b] ||
                fd.s2SplitNumOfFdHead[head] < 2 || fd.s2SplitNumOfFdHead[head] > uint64_t(c.aicNum)) {
                return MlaScheduleStatus::InvalidSplit;
            }
            result.partialPrefix_[head + 1] = result.partialPrefix_[head] + fd.s2SplitNumOfFdHead[head];
        }
        if (result.PartialSlots() > result.layout_.maxPartialSlots)
            return MlaScheduleStatus::InvalidSplit;
        // Verify the bridge at every split head and FD vector endpoint before
        // publishing it. Work is bounded by AIC^2, independent of M capacity.
        for (uint32_t head = 0; head < fd.numOfFdHead; ++head) {
            const uint64_t group = result.prefix_[fd.bN2IdxOfFdHead[head]] + fd.gS1IdxOfFdHead[head];
            for (uint32_t split = 0; split < fd.s2SplitNumOfFdHead[head]; ++split) {
                MlaAutoSplitKvTask task{};
                status = result.Task(group, split, task);
                if (status != MlaScheduleStatus::Success)
                    return status;
            }
            MlaAutoSplitKvMerge merge{};
            for (uint32_t part = 0; part < fd.gS1SplitNumOfFdHead[head]; ++part) {
                status = result.Merge(head, part, merge);
                if (status != MlaScheduleStatus::Success)
                    return status;
            }
        }
        output = std::move(result);
        return MlaScheduleStatus::Success;
    } catch (const std::bad_alloc&) {
        return MlaScheduleStatus::AllocationFailure;
    } catch (const std::length_error&) {
        return MlaScheduleStatus::AllocationFailure;
    }
}

MlaScheduleStatus MlaAutoSplitKvPlan::Task(uint64_t group, uint32_t split, MlaAutoSplitKvTask& output) const
{
    if (group >= Groups())
        return MlaScheduleStatus::InvalidSplit;
    const auto& c = config_;
    MlaAutoSplitKvTask t{};
    t.group = group;
    t.slot = UINT64_MAX;
    t.fdHead = UINT32_MAX;
    t.batch = std::upper_bound(prefix_.begin(), prefix_.end(), group) - prefix_.begin() - 1;
    t.m = group - prefix_[t.batch];
    t.split = split;
    const auto& seq = lengths_[t.batch];
    if (BuildWindowBlock({seq.queryLength, seq.kvLength, c.queryHeads, c.mBaseSize, c.s2BaseSize, t.m}, window_,
                         t.fragment.whole) != MlaCausalStatus::Success)
        return MlaScheduleStatus::InvalidSplit;
    const auto& w = t.fragment.whole;
    if (w.s2Begin == w.s2End) {
        if (split)
            return MlaScheduleStatus::InvalidSplit;
        t.kind = MlaAutoSplitKvTaskKind::Empty;
        t.splitCount = 1;
        t.owner = group % c.aicNum; // separate output initialization traversal
        t.fragment.s2Begin = t.fragment.s2End = w.s2End;
        t.fragment.kvBegin = t.fragment.kvEnd = w.kvEnd;
    } else {
        uint32_t count = 0, cursor = w.s2Begin;
        const auto key = std::make_pair(t.batch, t.m);
        for (uint32_t core = 0; core < native_.usedCoreNum; ++core) {
            uint32_t first = w.s2Begin, last = w.s2End;
            if (core) {
                const auto beginKey = std::make_pair(native_.bN2End[core - 1], native_.gS1End[core - 1]);
                if (key < beginKey)
                    continue;
                if (key == beginKey)
                    first = std::max(first, native_.s2End[core - 1]);
            }
            const auto endKey = std::make_pair(native_.bN2End[core], native_.gS1End[core]);
            if (key > endKey)
                continue;
            if (key == endKey)
                last = std::min(last, native_.s2End[core]);
            if (first >= last)
                continue;
            if (first != cursor)
                return MlaScheduleStatus::InvalidSplit;
            cursor = last;
            if (count == split) {
                t.owner = core;
                t.fragment.s2Begin = first;
                t.fragment.s2End = last;
            }
            ++count;
        }
        if (cursor != w.s2End || split >= count)
            return MlaScheduleStatus::InvalidSplit;
        t.splitCount = count;
        t.kind = count > 1 ? MlaAutoSplitKvTaskKind::Partial : MlaAutoSplitKvTaskKind::Direct;
        const auto& fd = native_.fd;
        for (uint32_t head = 0; head < fd.numOfFdHead; ++head) {
            if (fd.bN2IdxOfFdHead[head] == t.batch && fd.gS1IdxOfFdHead[head] == t.m)
                t.fdHead = head;
        }
        if (count > 1) {
            if (t.fdHead == UINT32_MAX || fd.s2SplitNumOfFdHead[t.fdHead] != count)
                return MlaScheduleStatus::InvalidSplit;
            t.slot = partialPrefix_[t.fdHead] + split;
            Offsets(layout_, t.slot * c.mBaseSize, t.outOffsetBytes, t.sumOffsetBytes, t.maxOffsetBytes);
        } else if (t.fdHead != UINT32_MAX)
            return MlaScheduleStatus::InvalidSplit;
        auto& f = t.fragment;
        f.kvBegin = std::max(uint64_t(w.kvBegin), uint64_t(f.s2Begin) * c.s2BaseSize);
        f.kvEnd = std::min(uint64_t(w.kvEnd), uint64_t(f.s2End) * c.s2BaseSize);
        f.cost =
            int64_t(f.s2End - f.s2Begin - 1) * w.normalS2Cost + (f.s2End == w.s2End ? w.lastS2Cost : w.normalS2Cost);
    }
    output = t;
    return MlaScheduleStatus::Success;
}

MlaScheduleStatus MlaAutoSplitKvPlan::Merge(uint32_t head, uint32_t part, MlaAutoSplitKvMerge& output) const
{
    const auto& fd = native_.fd;
    if (head >= fd.numOfFdHead || part >= fd.gS1SplitNumOfFdHead[head])
        return MlaScheduleStatus::InvalidSplit;
    MlaAutoSplitKvMerge r{};
    r.head = head;
    r.part = part;
    r.batch = fd.bN2IdxOfFdHead[head];
    r.m = fd.gS1IdxOfFdHead[head];
    r.group = prefix_[r.batch] + r.m;
    r.firstSlot = partialPrefix_[head];
    r.splitCount = fd.s2SplitNumOfFdHead[head];
    r.owner = UINT32_MAX;
    const auto key = std::make_pair(head, part);
    for (uint32_t vector = 0; vector < fd.usedVecNumOfFd; ++vector) {
        if (key <= std::make_pair(fd.gS1IdxEndOfFdHead[vector], fd.gS1IdxEndOfFdHeadSplit[vector])) {
            r.owner = vector;
            break;
        }
    }
    if (r.owner == UINT32_MAX)
        return MlaScheduleStatus::InvalidSplit;
    const uint64_t begin = uint64_t(r.m) * config_.mBaseSize;
    const uint64_t end =
        std::min(begin + config_.mBaseSize, uint64_t(lengths_[r.batch].queryLength) * config_.queryHeads);
    r.rowBegin = begin + uint64_t(part) * 8;
    r.rowEnd = std::min(end, uint64_t(r.rowBegin) + 8);
    if (r.rowBegin >= r.rowEnd)
        return MlaScheduleStatus::InvalidSplit;
    Offsets(layout_, r.firstSlot * config_.mBaseSize + uint64_t(part) * 8, r.outOffsetBytes, r.sumOffsetBytes,
            r.maxOffsetBytes);
    output = r;
    return MlaScheduleStatus::Success;
}

MlaScheduleStatus MlaAutoSplitKvPlan::Row(uint64_t group, uint32_t split, uint32_t row, MlaAutoSplitKvRow& output) const
{
    MlaAutoSplitKvTask t{};
    auto status = Task(group, split, t);
    if (status != MlaScheduleStatus::Success)
        return status;
    if (row >= t.fragment.whole.rowEnd - t.fragment.whole.rowBegin)
        return MlaScheduleStatus::InvalidSplit;
    const auto& c = config_;
    const uint64_t flat = uint64_t(t.fragment.whole.rowBegin) + row;
    const uint64_t query = flat / c.queryHeads, head = flat % c.queryHeads;
    MlaAutoSplitKvRow r{};
    r.out = (uint64_t(t.batch) * c.queryCapacity * c.queryHeads + flat) * 512 * 2;
    r.lse = ((uint64_t(t.batch) * c.queryHeads + head) * c.queryCapacity + query) * sizeof(float);
    if (c.aicNum > 1) {
        status =
            GetSplitAccumulatorOffsets(Accumulator(c), split, t.batch, head, query, r.accumulatorOut, r.accumulatorLse);
        if (status != MlaScheduleStatus::Success)
            return status;
    }
    r.partialOut = r.partialSum = r.partialMax = UINT64_MAX;
    if (t.kind == MlaAutoSplitKvTaskKind::Partial) {
        Offsets(layout_, t.slot * c.mBaseSize + row, r.partialOut, r.partialSum, r.partialMax);
    }
    output = r;
    return MlaScheduleStatus::Success;
}
} // namespace flash_attn::mla

// Fixed CANN/FA3 provenance is listed in tiling/mla_auto_split_plan.h.
// Native partition and FD endpoints are retained; persistent addresses and
// zero-cost output ownership are local adaptations, not uniform resplitting.

namespace flash_attn::mla {
namespace {
MlaAccumulatorInput Accumulator(const MlaAutoSplitTndPlanConfig& c)
{
    const auto& shape = c.shape;
    return {MlaAccumulatorQueryLayout::Tnd,
            shape.batch,
            shape.maxQueryLength,
            shape.totalQuery,
            shape.queryHeads,
            c.aicNum};
}

} // namespace

MlaScheduleStatus BuildAutoSplitCompactPlanLayout(const MlaAutoSplitTndPlanConfig& c, MlaSequenceLayout kvLayout,
                                                  MlaAutoSplitKvLayout& output)
{
    const auto& s = c.shape;
    if (kvLayout != MlaSequenceLayout::Tnd && kvLayout != MlaSequenceLayout::Bsnd)
        return MlaScheduleStatus::UnsupportedShape;
    if (s.batch < 0 || s.totalQuery < 0 || s.totalKv < 0 || s.maxQueryLength < 0 || s.maxKvLength < 0 ||
        s.queryHeads < 1 || c.mBaseSize < 1 || c.mBaseSize > 256 || c.s2BaseSize < 1 || c.s2BaseSize > 512) {
        return MlaScheduleStatus::UnsupportedShape;
    }
    if (c.aicNum < 1 || c.aicNum > kMlaMaxAic || (c.cvRatio != 1 && c.cvRatio != 2))
        return MlaScheduleStatus::InvalidPlatform;
    if (uint64_t(s.batch) > UINT32_MAX || s.totalQuery > INT32_MAX ||
        (kvLayout == MlaSequenceLayout::Tnd && s.totalKv > INT32_MAX) || s.maxQueryLength > kMlaMaxQueryLength ||
        s.maxKvLength > kMlaMaxKvLength || s.queryHeads > kMlaMaxQueryHeads) {
        return MlaScheduleStatus::FieldOverflow;
    }
    const uint64_t rows = uint64_t(s.maxQueryLength) * s.queryHeads;
    const uint64_t totalRows = uint64_t(s.totalQuery) * s.queryHeads;
    if (rows > kMlaMaxDenseRows || totalRows > kMlaMaxBatchRows)
        return MlaScheduleStatus::FieldOverflow;
    if (uint64_t(s.totalQuery) > uint64_t(s.batch) * s.maxQueryLength ||
        uint64_t(s.totalKv) > uint64_t(s.batch) * s.maxKvLength ||
        (kvLayout == MlaSequenceLayout::Bsnd && s.totalKv != s.batch * s.maxKvLength))
        return MlaScheduleStatus::UnsupportedShape;
    MlaAutoSplitKvLayout result{};
    auto status = BuildSplitAccumulatorLayout(Accumulator(c), result.accumulator);
    if (status != MlaScheduleStatus::Success)
        return status;
    if (c.libApiBytes > uint64_t(INT64_MAX))
        return MlaScheduleStatus::WorkspaceOverflow;
    result.emptyCapacity = s.totalQuery == 0 || s.totalKv == 0;
    if (!result.emptyCapacity) {
        const uint64_t active = std::min(uint64_t(s.batch), uint64_t(s.totalQuery));
        // Each nonempty allocation contributes at most M-1 padding rows.
        result.maxGroups = std::min(active * ((rows + c.mBaseSize - 1) / c.mBaseSize),
                                    (totalRows + active * (c.mBaseSize - 1)) / c.mBaseSize);
        // Each native core has at most its first and last M block in FD.
        result.maxPartialSlots = c.aicNum == 1 ? 0 : 2 * c.aicNum;
        result.partialRows = result.maxPartialSlots * c.mBaseSize;
        status = BuildSplitKvWorkspace({c.mBaseSize, c.s2BaseSize, int64_t(rows), c.aicNum, c.aicNum, false, 0},
                                       result.scratch);
        if (status != MlaScheduleStatus::Success)
            return status;
        result.outOffsetBytes = (result.scratch.attention.userBytes + 31) / 32 * 32;
        result.sumOffsetBytes = result.outOffsetBytes + result.partialRows * 512 * sizeof(float);
        result.maxOffsetBytes = result.sumOffsetBytes + result.partialRows * 8 * sizeof(float);
        result.userBytes = result.maxOffsetBytes + result.partialRows * 8 * sizeof(float);
        if (c.libApiBytes > uint64_t(INT64_MAX) - result.userBytes)
            return MlaScheduleStatus::WorkspaceOverflow;
        result.totalBytes = result.userBytes + c.libApiBytes;
    }
    output = result;
    return MlaScheduleStatus::Success;
}

MlaScheduleStatus BuildAutoSplitCompactPlan(const MlaAutoSplitTndPlanConfig& c, MlaSequenceLayout kvLayout,
                                            const MlaWindowConfig& window, const MlaUsedInputs& inputs,
                                            MlaAutoSplitTndPlan& output)
{
    MlaAutoSplitTndPlan result;
    auto status = BuildAutoSplitCompactPlanLayout(c, kvLayout, result.layout_);
    if (status != MlaScheduleStatus::Success)
        return status;
    const auto& shape = c.shape;
    MlaWindow normalized{};
    if (window.maxQueryLength != shape.maxQueryLength || window.maxKvLength != shape.maxKvLength ||
        NormalizeWindow(window, normalized) != MlaCausalStatus::Success) {
        return MlaScheduleStatus::UnsupportedShape;
    }
    try {
        MlaUsedLengths parsed;
        status = ParseUsedLengths({shape.batch,
                                   shape.queryHeads,
                                   {MlaSequenceLayout::Tnd, shape.totalQuery, shape.maxQueryLength},
                                   {kvLayout, shape.totalKv, shape.maxKvLength}},
                                  inputs, parsed);
        if (status != MlaScheduleStatus::Success)
            return status;
        result.config_ = c;
        result.window_ = window;
        result.sequences_ = std::move(parsed.sequences);
        std::vector<MlaCausalSequence> lengths;
        lengths.reserve(result.sequences_.size());
        for (const auto& seq : result.sequences_)
            lengths.push_back({seq.queryLength, seq.kvLength});
        result.prefix_.reserve(lengths.size() + 1);
        result.prefix_.push_back(0);
        for (const auto& seq : lengths) {
            const uint64_t rows = uint64_t(seq.queryLength) * c.shape.queryHeads;
            result.prefix_.push_back(result.prefix_.back() +
                                     (result.layout_.emptyCapacity ? 0 : (rows + c.mBaseSize - 1) / c.mBaseSize));
        }
        if (result.Groups() > result.layout_.maxGroups)
            return MlaScheduleStatus::WorkspaceOverflow;
        status = Convert(BuildNativeSplitKvSchedule(
            {lengths, c.shape.queryHeads, c.mBaseSize, c.s2BaseSize, c.aicNum, c.cvRatio}, window, result.native_));
        if (status != MlaScheduleStatus::Success)
            return status;
        const auto& fd = result.native_.fd;
        for (uint32_t head = 0; head < fd.numOfFdHead; ++head) {
            const uint32_t b = fd.bN2IdxOfFdHead[head], m = fd.gS1IdxOfFdHead[head];
            if (b >= lengths.size() || m >= result.prefix_[b + 1] - result.prefix_[b] ||
                fd.s2SplitNumOfFdHead[head] < 2 || fd.s2SplitNumOfFdHead[head] > uint64_t(c.aicNum)) {
                return MlaScheduleStatus::InvalidSplit;
            }
            result.partialPrefix_[head + 1] = result.partialPrefix_[head] + fd.s2SplitNumOfFdHead[head];
        }
        if (result.PartialSlots() > result.layout_.maxPartialSlots)
            return MlaScheduleStatus::InvalidSplit;
        // Verify the bridge at every split head and FD vector endpoint before
        // publishing it. Work is bounded by AIC^2, independent of M capacity.
        for (uint32_t head = 0; head < fd.numOfFdHead; ++head) {
            const uint64_t group = result.prefix_[fd.bN2IdxOfFdHead[head]] + fd.gS1IdxOfFdHead[head];
            for (uint32_t split = 0; split < fd.s2SplitNumOfFdHead[head]; ++split) {
                MlaAutoSplitTndTask task{};
                status = result.Task(group, split, task);
                if (status != MlaScheduleStatus::Success)
                    return status;
            }
            MlaAutoSplitKvMerge merge{};
            for (uint32_t part = 0; part < fd.gS1SplitNumOfFdHead[head]; ++part) {
                status = result.Merge(head, part, merge);
                if (status != MlaScheduleStatus::Success)
                    return status;
            }
        }
        output = std::move(result);
        return MlaScheduleStatus::Success;
    } catch (const std::bad_alloc&) {
        return MlaScheduleStatus::AllocationFailure;
    } catch (const std::length_error&) {
        return MlaScheduleStatus::AllocationFailure;
    }
}

MlaScheduleStatus BuildAutoSplitTndPlanLayout(const MlaAutoSplitTndPlanConfig& c, MlaAutoSplitKvLayout& output)
{
    return BuildAutoSplitCompactPlanLayout(c, MlaSequenceLayout::Tnd, output);
}
MlaScheduleStatus BuildAutoSplitTndPlan(const MlaAutoSplitTndPlanConfig& c, const MlaWindowConfig& window,
                                        const MlaUsedInputs& inputs, MlaAutoSplitTndPlan& output)
{
    return BuildAutoSplitCompactPlan(c, MlaSequenceLayout::Tnd, window, inputs, output);
}

MlaScheduleStatus MlaAutoSplitTndPlan::Task(uint64_t group, uint32_t split, MlaAutoSplitTndTask& output) const
{
    if (group >= Groups())
        return MlaScheduleStatus::InvalidSplit;
    const auto& c = config_;
    MlaAutoSplitTndTask result{};
    auto& t = result.native;
    t.group = group;
    t.slot = UINT64_MAX;
    t.fdHead = UINT32_MAX;
    t.batch = std::upper_bound(prefix_.begin(), prefix_.end(), group) - prefix_.begin() - 1;
    t.m = group - prefix_[t.batch];
    t.split = split;
    const auto& seq = sequences_[t.batch];
    result.sequence = seq;
    if (BuildWindowBlock({seq.queryLength, seq.kvLength, c.shape.queryHeads, c.mBaseSize, c.s2BaseSize, t.m}, window_,
                         t.fragment.whole) != MlaCausalStatus::Success)
        return MlaScheduleStatus::InvalidSplit;
    const auto& w = t.fragment.whole;
    if (w.s2Begin == w.s2End) {
        if (split)
            return MlaScheduleStatus::InvalidSplit;
        t.kind = MlaAutoSplitKvTaskKind::Empty;
        t.splitCount = 1;
        t.owner = group % c.aicNum; // separate output initialization traversal
        t.fragment.s2Begin = t.fragment.s2End = w.s2End;
        t.fragment.kvBegin = t.fragment.kvEnd = w.kvEnd;
    } else {
        uint32_t count = 0, cursor = w.s2Begin;
        const auto key = std::make_pair(t.batch, t.m);
        for (uint32_t core = 0; core < native_.usedCoreNum; ++core) {
            uint32_t first = w.s2Begin, last = w.s2End;
            if (core) {
                const auto beginKey = std::make_pair(native_.bN2End[core - 1], native_.gS1End[core - 1]);
                if (key < beginKey)
                    continue;
                if (key == beginKey)
                    first = std::max(first, native_.s2End[core - 1]);
            }
            const auto endKey = std::make_pair(native_.bN2End[core], native_.gS1End[core]);
            if (key > endKey)
                continue;
            if (key == endKey)
                last = std::min(last, native_.s2End[core]);
            if (first >= last)
                continue;
            if (first != cursor)
                return MlaScheduleStatus::InvalidSplit;
            cursor = last;
            if (count == split) {
                t.owner = core;
                t.fragment.s2Begin = first;
                t.fragment.s2End = last;
            }
            ++count;
        }
        if (cursor != w.s2End || split >= count)
            return MlaScheduleStatus::InvalidSplit;
        t.splitCount = count;
        t.kind = count > 1 ? MlaAutoSplitKvTaskKind::Partial : MlaAutoSplitKvTaskKind::Direct;
        const auto& fd = native_.fd;
        for (uint32_t head = 0; head < fd.numOfFdHead; ++head) {
            if (fd.bN2IdxOfFdHead[head] == t.batch && fd.gS1IdxOfFdHead[head] == t.m)
                t.fdHead = head;
        }
        if (count > 1) {
            if (t.fdHead == UINT32_MAX || fd.s2SplitNumOfFdHead[t.fdHead] != count)
                return MlaScheduleStatus::InvalidSplit;
            t.slot = partialPrefix_[t.fdHead] + split;
            Offsets(layout_, t.slot * c.mBaseSize, t.outOffsetBytes, t.sumOffsetBytes, t.maxOffsetBytes);
        } else if (t.fdHead != UINT32_MAX)
            return MlaScheduleStatus::InvalidSplit;
        auto& f = t.fragment;
        f.kvBegin = std::max(uint64_t(w.kvBegin), uint64_t(f.s2Begin) * c.s2BaseSize);
        f.kvEnd = std::min(uint64_t(w.kvEnd), uint64_t(f.s2End) * c.s2BaseSize);
        f.cost =
            int64_t(f.s2End - f.s2Begin - 1) * w.normalS2Cost + (f.s2End == w.s2End ? w.lastS2Cost : w.normalS2Cost);
    }
    output = result;
    return MlaScheduleStatus::Success;
}

MlaScheduleStatus MlaAutoSplitTndPlan::Merge(uint32_t head, uint32_t part, MlaAutoSplitKvMerge& output) const
{
    const auto& fd = native_.fd;
    if (head >= fd.numOfFdHead || part >= fd.gS1SplitNumOfFdHead[head])
        return MlaScheduleStatus::InvalidSplit;
    MlaAutoSplitKvMerge r{};
    r.head = head;
    r.part = part;
    r.batch = fd.bN2IdxOfFdHead[head];
    r.m = fd.gS1IdxOfFdHead[head];
    r.group = prefix_[r.batch] + r.m;
    r.firstSlot = partialPrefix_[head];
    r.splitCount = fd.s2SplitNumOfFdHead[head];
    r.owner = UINT32_MAX;
    const auto key = std::make_pair(head, part);
    for (uint32_t vector = 0; vector < fd.usedVecNumOfFd; ++vector) {
        if (key <= std::make_pair(fd.gS1IdxEndOfFdHead[vector], fd.gS1IdxEndOfFdHeadSplit[vector])) {
            r.owner = vector;
            break;
        }
    }
    if (r.owner == UINT32_MAX)
        return MlaScheduleStatus::InvalidSplit;
    const uint64_t begin = uint64_t(r.m) * config_.mBaseSize;
    const uint64_t end =
        std::min(begin + config_.mBaseSize, uint64_t(sequences_[r.batch].queryLength) * config_.shape.queryHeads);
    r.rowBegin = begin + uint64_t(part) * 8;
    r.rowEnd = std::min(end, uint64_t(r.rowBegin) + 8);
    if (r.rowBegin >= r.rowEnd)
        return MlaScheduleStatus::InvalidSplit;
    Offsets(layout_, r.firstSlot * config_.mBaseSize + uint64_t(part) * 8, r.outOffsetBytes, r.sumOffsetBytes,
            r.maxOffsetBytes);
    output = r;
    return MlaScheduleStatus::Success;
}

MlaScheduleStatus MlaAutoSplitTndPlan::Row(uint64_t group, uint32_t split, uint32_t row,
                                           MlaAutoSplitKvRow& output) const
{
    MlaAutoSplitTndTask task{};
    auto status = Task(group, split, task);
    const auto& t = task.native;
    if (status != MlaScheduleStatus::Success)
        return status;
    if (row >= t.fragment.whole.rowEnd - t.fragment.whole.rowBegin)
        return MlaScheduleStatus::InvalidSplit;
    const auto& c = config_;
    const uint64_t flat = uint64_t(t.fragment.whole.rowBegin) + row;
    const uint64_t query = task.sequence.queryStart + flat / c.shape.queryHeads, head = flat % c.shape.queryHeads;
    MlaAutoSplitKvRow r{};
    r.out = (query * c.shape.queryHeads + head) * 512 * 2;
    r.lse = (head * c.shape.totalQuery + query) * sizeof(float);
    if (c.aicNum > 1) {
        status = GetSplitAccumulatorOffsets(Accumulator(c), split, 0, head, query, r.accumulatorOut, r.accumulatorLse);
        if (status != MlaScheduleStatus::Success)
            return status;
    }
    r.partialOut = r.partialSum = r.partialMax = UINT64_MAX;
    if (t.kind == MlaAutoSplitKvTaskKind::Partial) {
        Offsets(layout_, t.slot * c.mBaseSize + row, r.partialOut, r.partialSum, r.partialMax);
    }
    output = r;
    return MlaScheduleStatus::Success;
}
} // namespace flash_attn::mla

// Fixed CANN/FA3 provenance and licenses: tiling/mla_auto_split_plan.h.
namespace flash_attn::mla {
namespace {
MlaScheduleStatus CompactConfig(const MlaAutoSplitMixedPlanConfig& c, MlaAutoSplitTndPlanConfig& output)
{
    const auto& s = c.shape;
    // Bound both factors before multiplying; the rectangular KV total may
    // exceed int32/uint32 and is never narrowed to a TND prefix terminal.
    if (s.batch < 0 || s.kvCapacity < 0)
        return MlaScheduleStatus::UnsupportedShape;
    if (uint64_t(s.batch) > UINT32_MAX || s.kvCapacity > kMlaMaxKvLength)
        return MlaScheduleStatus::FieldOverflow;
    output = {{s.batch, s.totalQuery, s.batch * s.kvCapacity, s.queryHeads, s.maxQueryLength, s.kvCapacity},
              c.mBaseSize,
              c.s2BaseSize,
              c.aicNum,
              c.cvRatio,
              c.libApiBytes};
    return MlaScheduleStatus::Success;
}
} // namespace
MlaScheduleStatus BuildAutoSplitMixedPlanLayout(const MlaAutoSplitMixedPlanConfig& c, MlaAutoSplitKvLayout& output)
{
    MlaAutoSplitTndPlanConfig compact{};
    const auto status = CompactConfig(c, compact);
    if (status != MlaScheduleStatus::Success)
        return status;
    return BuildAutoSplitCompactPlanLayout(compact, MlaSequenceLayout::Bsnd, output);
}
MlaScheduleStatus BuildAutoSplitMixedPlan(const MlaAutoSplitMixedPlanConfig& c, const MlaWindowConfig& window,
                                          const MlaUsedInputs& inputs, MlaAutoSplitMixedPlan& output)
{
    MlaAutoSplitTndPlanConfig compact{};
    const auto status = CompactConfig(c, compact);
    if (status != MlaScheduleStatus::Success)
        return status;
    return BuildAutoSplitCompactPlan(compact, MlaSequenceLayout::Bsnd, window, inputs, output);
}
} // namespace flash_attn::mla

// Reuse pinned PA geometry and native variable partition arithmetic, without flattening
// physical pages into a fictitious contiguous KV allocation.

namespace flash_attn::mla {
namespace {

MlaAutoSplitKvConfig AutoBsndConfig(const MlaPagedShape& s, const MlaPlatformInfo& p, uint32_t m, uint32_t n)
{
    return {s.batch, s.query.maxLength, s.logicalKvCapacity, s.queryHeads, m, n, p.aicNum, 2, p.libApiWorkspaceBytes};
}
} // namespace

MlaScheduleStatus BuildAutoSplitPagedBsndBudget(const MlaPagedShape& s, const MlaPlatformInfo& p,
                                                MlaAutoSplitPagedBsndBudget& output)
{
    if (s.query.layout != MlaSequenceLayout::Bsnd || s.pageSize < 16 || s.pageSize > 1024 || s.pageSize % 16 ||
        s.logicalKvCapacity < 0 || s.logicalKvCapacity > kMlaMaxKvLength || s.logicalKvCapacity % s.pageSize ||
        s.batch < 0 || uint64_t(s.batch) > (UINT32_MAX - 756) / 7)
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
    MlaAutoSplitKvLayout bound{};
    status = BuildAutoSplitKvLayout(AutoBsndConfig(s, p, 256, PagedSplitCapacityBase(s)), bound);
    if (status != MlaScheduleStatus::Success)
        return status;
    MlaAutoSplitPagedBsndBudget result{};
    const uint64_t rows = uint64_t(s.query.maxLength) * s.queryHeads;
    result.maxGroups = bound.emptyCapacity ? 0 : uint64_t(s.batch) * ((rows + 127) / 128);
    result.maxPartialSlots = bound.maxPartialSlots;
    result.partialRows = bound.partialRows;
    result.scratchBytes = bound.outOffsetBytes;
    result.userBytes = bound.userBytes;
    result.totalBytes = bound.totalBytes;
    result.accumulator = bound.accumulator;
    output = result;
    return MlaScheduleStatus::Success;
}

MlaScheduleStatus BuildAutoSplitPagedBsndProductionBudget(const MlaPagedShape& s, const MlaPlatformInfo& p,
                                                          MlaAutoSplitPagedBsndBudget& output)
{
    MlaAutoSplitPagedBsndBudget result{};
    auto status = BuildAutoSplitPagedBsndBudget(s, p, result);
    if (status != MlaScheduleStatus::Success)
        return status;
    status = BuildSplitAccumulatorLayout(
        {MlaAccumulatorQueryLayout::Bsnd, s.batch, s.query.maxLength, s.query.totalTokens, s.queryHeads, kMlaMaxAic},
        result.accumulator);
    if (status != MlaScheduleStatus::Success)
        return status;
    const uint64_t direct =
        s.batch && s.query.maxLength && s.logicalKvCapacity ? uint64_t(p.aicNum) * 256 * 528 * sizeof(float) : 0;
    if (direct > uint64_t(INT64_MAX) - result.totalBytes)
        return MlaScheduleStatus::WorkspaceOverflow;
    result.userBytes += direct;
    result.totalBytes += direct;
    output = result;
    return MlaScheduleStatus::Success;
}

MlaScheduleStatus BuildAutoSplitPagedBsndPlan(const MlaPagedShape& shape, const MlaPlatformInfo& platform,
                                              const MlaPagedInputs& inputs, const MlaWindowConfig& window,
                                              MlaAutoSplitPagedBsndPlan& output)
{
    try {
        MlaAutoSplitPagedBsndPlan result;
        auto status = BuildAutoSplitPagedBsndBudget(shape, platform, result.budget);
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
        status = BuildAutoSplitKvPlan(AutoBsndConfig(shape, platform, result.mBase, result.nBase), window, lengths,
                                      result.split);
        if (status != MlaScheduleStatus::Success)
            return status;
        const auto& l = result.split.Layout();
        if (result.split.Groups() > result.budget.maxGroups ||
            result.split.PartialSlots() > result.budget.maxPartialSlots || l.partialRows > result.budget.partialRows ||
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

MlaAutoSplitMixedPlanConfig AutoTndConfig(const MlaPagedShape& s, const MlaPlatformInfo& p, uint32_t m, uint32_t n)
{
    return {{s.batch, s.query.totalTokens, s.queryHeads, s.query.maxLength, s.logicalKvCapacity},
            m,
            n,
            p.aicNum,
            2,
            p.libApiWorkspaceBytes};
}
} // namespace

MlaScheduleStatus BuildAutoSplitPagedTndBudget(const MlaPagedShape& s, const MlaPlatformInfo& p,
                                               MlaAutoSplitPagedTndBudget& output)
{
    if (s.query.layout != MlaSequenceLayout::Tnd || s.pageSize < 16 || s.pageSize > 1024 || s.pageSize % 16 ||
        s.logicalKvCapacity < 0 || s.logicalKvCapacity > kMlaMaxKvLength || s.logicalKvCapacity % s.pageSize ||
        s.batch < 0 || uint64_t(s.batch) > (UINT32_MAX - 756) / 7)
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
    MlaAutoSplitKvLayout bound{};
    status = BuildAutoSplitMixedPlanLayout(AutoTndConfig(s, p, 256, PagedSplitCapacityBase(s)), bound);
    if (status != MlaScheduleStatus::Success)
        return status;
    MlaAutoSplitPagedTndBudget result{};
    const uint64_t rows = uint64_t(s.query.maxLength) * s.queryHeads;
    if (!bound.emptyCapacity) {
        const uint64_t totalRows = uint64_t(s.query.totalTokens) * s.queryHeads;
        const uint64_t active = std::min(uint64_t(s.batch), uint64_t(s.query.totalTokens));
        result.maxGroups = std::min(active * ((rows + 127) / 128), (totalRows + active * 127) / 128);
    }
    result.maxPartialSlots = bound.maxPartialSlots;
    result.partialRows = bound.partialRows;
    result.scratchBytes = bound.outOffsetBytes;
    result.userBytes = bound.userBytes;
    result.totalBytes = bound.totalBytes;
    result.accumulator = bound.accumulator;
    output = result;
    return MlaScheduleStatus::Success;
}

MlaScheduleStatus BuildAutoSplitPagedTndProductionBudget(const MlaPagedShape& s, const MlaPlatformInfo& p,
                                                         MlaAutoSplitPagedTndBudget& output)
{
    MlaAutoSplitPagedTndBudget result{};
    auto status = BuildAutoSplitPagedTndBudget(s, p, result);
    if (status != MlaScheduleStatus::Success)
        return status;
    status = BuildSplitAccumulatorLayout(
        {MlaAccumulatorQueryLayout::Tnd, s.batch, s.query.maxLength, s.query.totalTokens, s.queryHeads, kMlaMaxAic},
        result.accumulator);
    if (status != MlaScheduleStatus::Success)
        return status;
    const uint64_t direct =
        s.batch && s.query.totalTokens && s.logicalKvCapacity ? uint64_t(p.aicNum) * 256 * 528 * sizeof(float) : 0;
    if (direct > uint64_t(INT64_MAX) - result.totalBytes)
        return MlaScheduleStatus::WorkspaceOverflow;
    result.userBytes += direct;
    result.totalBytes += direct;
    output = result;
    return MlaScheduleStatus::Success;
}

MlaScheduleStatus BuildAutoSplitPagedTndPlan(const MlaPagedShape& shape, const MlaPlatformInfo& platform,
                                             const MlaPagedInputs& inputs, const MlaWindowConfig& window,
                                             MlaAutoSplitPagedTndPlan& output)
{
    try {
        MlaAutoSplitPagedTndPlan result;
        auto status = BuildAutoSplitPagedTndBudget(shape, platform, result.budget);
        if (status != MlaScheduleStatus::Success)
            return status;
        status = BuildPagedSchedule(shape, platform, inputs, result.paged);
        if (status != MlaScheduleStatus::Success)
            return status;
        const auto& s = result.paged.schedule;
        result.mBase = s.mBaseSize ? s.mBaseSize : 256;
        result.nBase = s.s2BaseSize ? s.s2BaseSize : PagedSplitCapacityBase(shape);
        // Only logical KV intervals are reused; physical pages remain invocation state.
        status = BuildAutoSplitMixedPlan(AutoTndConfig(shape, platform, result.mBase, result.nBase), window,
                                         {inputs.cuQuery, {}, inputs.usedQuery, inputs.kvLengths}, result.split);
        if (status != MlaScheduleStatus::Success)
            return status;
        const auto& l = result.split.Layout();
        if (result.split.Groups() > result.budget.maxGroups ||
            result.split.PartialSlots() > result.budget.maxPartialSlots || l.partialRows > result.budget.partialRows ||
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
} // namespace flash_attn::mla
