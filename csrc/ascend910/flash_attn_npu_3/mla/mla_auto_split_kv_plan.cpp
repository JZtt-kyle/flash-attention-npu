// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
// Fixed CANN/FA3 provenance is listed in mla_auto_split_kv_plan.h.
// Native partition and FD endpoints are retained; persistent addresses and
// zero-cost output ownership are local adaptations, not uniform resplitting.
#include "mla_auto_split_kv_plan.h"
#include <algorithm>
#include <new>
#include <stdexcept>
#include <tuple>
#include <utility>

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
