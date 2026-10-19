/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// This translation unit executes only on host/AICPU. Bisheng's ASC front end
// also defines __NPU_ARCH__ for host-only files; keep reused scalar consumers
// free of AICore annotations here.
#ifdef __NPU_ARCH__
#undef __NPU_ARCH__
#endif
#include "mla_split.h"
#include "cann_split_core.h"
#include <algorithm>
#include <new>
#include <stdexcept>
#include <utility>
#include <tuple>
#include "mla_split_metadata_consumer.h"

// Adapted from the fixed sources listed in mla_split.h. Native algorithms
// remain in detail/cann_split_core; checked CPU layout/FA3 adaptation is local.
// FA3 portions: Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang,
// Vijay Thakkar, Pradeep Ramani, Tri Dao; BSD 3-Clause (LICENSE.fa3).

namespace flash_attn::mla {
namespace {
MlaCausalStatus Native(const MlaCausalScheduleInput& in, const MlaWindowConfig& config,
                       MlaNativeSplitKvSchedule& result)
{
    MlaWindow window{};
    auto status = NormalizeWindow(config, window);
    if (status != MlaCausalStatus::Success) {
        return status;
    }
    MlaWindowBlock block{};
    status = BuildWindowBlock({0, 0, in.queryHeads, in.mBaseSize, in.s2BaseSize, 0}, config, block);
    if (status != MlaCausalStatus::Success) {
        return status;
    }
    if (in.aicNum < 1 || in.aicNum > kMlaMaxAic || (in.cvRatio != 1 && in.cvRatio != 2)) {
        return MlaCausalStatus::InvalidPlatform;
    }
    if (in.sequences.size() > UINT32_MAX) {
        return MlaCausalStatus::FieldOverflow;
    }
    uint64_t rows = 0, tiles = 0;
    for (const auto& seq : in.sequences) {
        status = BuildWindowBlock({seq.queryLength, seq.kvLength, in.queryHeads, in.mBaseSize, in.s2BaseSize, 0},
                                  config, block);
        if (status != MlaCausalStatus::Success) {
            return status;
        }
        const uint64_t count = uint64_t(seq.queryLength) * in.queryHeads;
        const uint64_t m = (count + in.mBaseSize - 1) / in.mBaseSize;
        rows += count;
        tiles += m * ((seq.kvLength + in.s2BaseSize - 1) / in.s2BaseSize);
        if (rows > kMlaMaxBatchRows || tiles > UINT32_MAX) {
            return MlaCausalStatus::FieldOverflow;
        }
        result.totalMBlocks += m;
    }
    detail::BaseInfo base{};
    base.bSize = in.sequences.size();
    base.n2Size = 1;
    base.gSize = in.queryHeads;
    base.attenMaskFlag = window.kind != MlaWindowKind::None;
    base.sparseMode = window.kind == MlaWindowKind::Local ? 4 : window.kind == MlaWindowKind::Causal ? 3 : 0;
    base.preToken = window.kind == MlaWindowKind::Causal ? INT32_MAX : window.left;
    base.nextToken = window.kind == MlaWindowKind::Causal ? 0 : window.right;
    base.actualLenQDims = base.actualLenKvDims = base.bSize;
    for (const auto& seq : in.sequences) {
        base.actualSeqS1Size.push_back(seq.queryLength);
        base.actualSeqS2Size.push_back(seq.kvLength);
    }
    detail::SplitParam param{};
    param.mBaseSize = in.mBaseSize;
    param.s2BaseSize = in.s2BaseSize;
    param.gS1BaseSizeOfFd = 8;
    detail::SplitContext context(base, param);
    detail::CalcSplitInfo(context);
    detail::CalcCostInfo(context);
    result.totalCost = context.costInfo.totalCost;
    result.totalS2Blocks = context.costInfo.totalBlockNum;
    detail::SplitResult native{uint32_t(in.aicNum), uint32_t(in.cvRatio)};
    detail::SplitCore(in.aicNum, context, native);
    if (native.usedCoreNum < 1 || native.usedCoreNum > in.aicNum || native.numOfFdHead > in.aicNum ||
        native.maxS2SplitNum > in.aicNum + 1 || native.usedVecNumOfFd > in.aicNum * in.cvRatio) {
        return MlaCausalStatus::InvalidSplit;
    }
    result.usedCoreNum = native.usedCoreNum;
    result.maxCost = native.maxCost;
    std::copy(native.bN2End.begin(), native.bN2End.end(), result.bN2End.begin());
    std::copy(native.gS1End.begin(), native.gS1End.end(), result.gS1End.begin());
    std::copy(native.s2End.begin(), native.s2End.end(), result.s2End.begin());
    auto& fd = result.fd;
    fd.numOfFdHead = native.numOfFdHead;
    fd.maxS2SplitNum = native.maxS2SplitNum;
    fd.usedVecNumOfFd = native.usedVecNumOfFd;
    fd.gS1BaseSizeOfFd = param.gS1BaseSizeOfFd;
#define COPY_FD(field) std::copy(native.fdRes.field.begin(), native.fdRes.field.end(), fd.field.begin())
    COPY_FD(bN2IdxOfFdHead);
    COPY_FD(gS1IdxOfFdHead);
    COPY_FD(s2SplitNumOfFdHead);
    COPY_FD(s2SplitStartIdxOfCore);
    COPY_FD(gS1SplitNumOfFdHead);
    COPY_FD(gS1LastPartSizeOfFdHead);
    COPY_FD(gS1IdxEndOfFdHead);
    COPY_FD(gS1IdxEndOfFdHeadSplit);
#undef COPY_FD
    return MlaCausalStatus::Success;
}

bool TensorLayout(MlaAccumulatorTensor& tensor)
{
    uint64_t stride = 1, elements = 1;
    for (uint32_t i = tensor.rank; i-- > 0;) {
        tensor.strides[i] = stride;
        const uint64_t dimension = std::max(uint64_t(1), tensor.shape[i]);
        if (stride > uint64_t(INT64_MAX) / dimension) {
            return false;
        }
        stride *= dimension;
        elements *= tensor.shape[i]; // bounded by stride, including zero extents
    }
    if (elements > uint64_t(INT64_MAX) / sizeof(float)) {
        return false;
    }
    tensor.elements = elements;
    tensor.bytes = elements * sizeof(float);
    return true;
}
} // namespace

MlaCausalStatus BuildNativeSplitKvSchedule(const MlaCausalScheduleInput& input, const MlaWindowConfig& window,
                                           MlaNativeSplitKvSchedule& output)
{
    try {
        MlaNativeSplitKvSchedule result{};
        const auto status = Native(input, window, result);
        if (status == MlaCausalStatus::Success) {
            output = result;
        }
        return status;
    } catch (const std::bad_alloc&) {
        return MlaCausalStatus::AllocationFailure;
    }
}

MlaCausalStatus ResolveSplitKvRequest(int64_t requested, int64_t autoSplits, uint32_t& output)
{
    const int64_t count = requested <= 0 ? autoSplits : requested;
    if (count < 1 || count > 256) {
        return MlaCausalStatus::InvalidSplit;
    }
    output = uint32_t(count);
    return MlaCausalStatus::Success;
}

MlaCausalStatus BuildSplitKvBlock(const MlaCausalBlockInput& input, const MlaWindowConfig& window, int64_t numSplits,
                                  int64_t splitIndex, MlaSplitKvBlock& output)
{
    if (numSplits < 1 || numSplits > 256 || splitIndex < 0 || splitIndex >= numSplits) {
        return MlaCausalStatus::InvalidSplit;
    }
    MlaSplitKvBlock result{};
    const auto status = BuildWindowBlock(input, window, result.whole);
    if (status != MlaCausalStatus::Success) {
        return status;
    }
    const auto& whole = result.whole;
    const uint64_t perSplit = (whole.s2End - whole.s2Begin + numSplits - 1) / numSplits;
    result.s2Begin = std::min(uint64_t(whole.s2End), whole.s2Begin + uint64_t(splitIndex) * perSplit);
    result.s2End = std::min(uint64_t(whole.s2End), result.s2Begin + perSplit);
    result.kvBegin = result.kvEnd = whole.kvEnd;
    if (result.s2Begin < result.s2End) {
        result.kvBegin = std::max(uint64_t(whole.kvBegin), uint64_t(result.s2Begin) * input.s2BaseSize);
        result.kvEnd = std::min(uint64_t(whole.kvEnd), uint64_t(result.s2End) * input.s2BaseSize);
        result.cost = int64_t(result.s2End - result.s2Begin - 1) * whole.normalS2Cost +
                      (result.s2End == whole.s2End ? whole.lastS2Cost : whole.normalS2Cost);
    }
    output = result;
    return MlaCausalStatus::Success;
}

MlaScheduleStatus BuildSplitKvWorkspace(const MlaSplitKvWorkspaceInput& in, MlaSplitKvWorkspace& output)
{
    if (in.mBaseSize < 1 || in.mBaseSize > 256 || in.s2BaseSize < 1 || in.s2BaseSize > 512 || in.maxQueryRows < 0) {
        return MlaScheduleStatus::UnsupportedShape;
    }
    if (uint64_t(in.maxQueryRows) > kMlaMaxDenseRows) {
        return MlaScheduleStatus::FieldOverflow;
    }
    if (in.aicNum < 1 || in.aicNum > kMlaMaxAic || in.blockDim < 1 || in.blockDim > in.aicNum) {
        return MlaScheduleStatus::InvalidPlatform;
    }
    MlaSplitKvWorkspace result{};
    auto& w = result.attention;
    const uint64_t m = (std::min(in.maxQueryRows, in.mBaseSize) + 15) / 16 * 16;
    w.mm1ResSize = m * ((in.s2BaseSize + 31) / 32 * 32);
    w.mm2ResSize = m * 512;
    w.fdAccumOutSize = uint64_t(in.aicNum) * 2 * in.mBaseSize * 512;
    w.fdLogSumExpSize = uint64_t(in.aicNum) * 2 * 2 * in.mBaseSize * 8;
    const uint64_t slots = 2 * in.blockDim;
    w.pOffsetBytes = slots * w.mm1ResSize * 4;
    w.pvOffsetBytes = w.pOffsetBytes + slots * w.mm1ResSize * 2;
    w.vec2OffsetBytes = w.pvOffsetBytes + slots * w.mm2ResSize * 4;
    w.kernelBytes = w.vec2OffsetBytes + slots * w.mm2ResSize * 4;
    w.userBytes = uint64_t(2) * in.aicNum * (6 * w.mm1ResSize + 8 * w.mm2ResSize + 8 * in.mBaseSize);
    if (in.flashDecode) {
        result.fdAccumOffsetBytes = w.kernelBytes;
        result.fdSumOffsetBytes = result.fdAccumOffsetBytes + 4 * w.fdAccumOutSize;
        result.fdMaxOffsetBytes = result.fdSumOffsetBytes + 2 * w.fdLogSumExpSize;
        const uint64_t fdBytes = 4 * (w.fdAccumOutSize + w.fdLogSumExpSize);
        w.kernelBytes += fdBytes;
        w.userBytes += fdBytes;
    }
    w.libApiBytes = in.libApiBytes;
    // Keep a byte budget representable by both uint64 and the runtime's int64 sizes.
    if (w.libApiBytes > uint64_t(INT64_MAX) - w.userBytes) {
        return MlaScheduleStatus::WorkspaceOverflow;
    }
    w.totalBytes = w.libApiBytes + w.userBytes;
    output = result;
    return MlaScheduleStatus::Success;
}

MlaScheduleStatus BuildSplitAccumulatorLayout(const MlaAccumulatorInput& in, MlaAccumulatorLayout& output)
{
    if ((in.layout != MlaAccumulatorQueryLayout::Bsnd && in.layout != MlaAccumulatorQueryLayout::Tnd) || in.batch < 0 ||
        in.queryCapacity < 0 || in.totalQuery < 0 || in.queryHeads < 1 || in.numSplits < 1 || in.numSplits > 256) {
        return MlaScheduleStatus::UnsupportedShape;
    }
    if (uint64_t(in.batch) > UINT32_MAX || in.queryCapacity > kMlaMaxQueryLength || in.queryHeads > kMlaMaxQueryHeads ||
        (in.layout == MlaAccumulatorQueryLayout::Tnd && in.totalQuery > INT32_MAX)) {
        return MlaScheduleStatus::FieldOverflow;
    }
    if (in.layout == MlaAccumulatorQueryLayout::Bsnd &&
        uint64_t(in.totalQuery) != uint64_t(in.batch) * in.queryCapacity) {
        return MlaScheduleStatus::UnsupportedShape;
    }
    MlaAccumulatorLayout result{};
    if (in.numSplits > 1) {
        auto& o = result.out;
        auto& l = result.lse;
        if (in.layout == MlaAccumulatorQueryLayout::Bsnd) {
            o.rank = 5;
            o.shape = {uint64_t(in.numSplits), uint64_t(in.batch), uint64_t(in.queryHeads), uint64_t(in.queryCapacity),
                       512};
        } else {
            o.rank = 4;
            o.shape = {uint64_t(in.numSplits), uint64_t(in.queryHeads), uint64_t(in.totalQuery), 512, 0};
        }
        l.rank = o.rank - 1;
        std::copy_n(o.shape.begin(), l.rank, l.shape.begin());
        if (!TensorLayout(o) || !TensorLayout(l)) {
            return MlaScheduleStatus::WorkspaceOverflow;
        }
    }
    output = result;
    return MlaScheduleStatus::Success;
}

MlaScheduleStatus GetSplitAccumulatorOffsets(const MlaAccumulatorInput& in, int64_t split, int64_t batch, int64_t head,
                                             int64_t query, uint64_t& outBytes, uint64_t& lseBytes)
{
    MlaAccumulatorLayout layout{};
    const auto status = BuildSplitAccumulatorLayout(in, layout);
    if (status != MlaScheduleStatus::Success) {
        return status;
    }
    const bool bsnd = in.layout == MlaAccumulatorQueryLayout::Bsnd;
    if (in.numSplits == 1 || split < 0 || split >= in.numSplits || batch < 0 ||
        (bsnd ? batch >= in.batch : batch != 0) || head < 0 || head >= in.queryHeads || query < 0 ||
        query >= (bsnd ? in.queryCapacity : in.totalQuery)) {
        return MlaScheduleStatus::UnsupportedShape;
    }
    const auto& l = layout.lse;
    const uint64_t index = uint64_t(split) * l.strides[0] + (bsnd ? uint64_t(batch) * l.strides[1] : 0) +
                           uint64_t(head) * l.strides[bsnd ? 2 : 1] + uint64_t(query);
    outBytes = index * 512 * sizeof(float);
    lseBytes = index * sizeof(float);
    return MlaScheduleStatus::Success;
}
} // namespace flash_attn::mla

// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.

// Fixed CANN/FA3 provenance and licenses are listed in mla_split.h.
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

// Fixed CANN/FA3 provenance and licenses: mla_split.h.
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

// Fixed CANN/FA3 provenance and licenses: mla_split.h.
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

// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.

// Fixed CANN/FA3 provenance is listed in mla_split.h.
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

// Fixed CANN/FA3 provenance is listed in mla_split.h.
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

// Fixed CANN/FA3 provenance and licenses: mla_split.h.
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

// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
// Local GQA coordinate adapter; source algorithms/provenance in mla_split.h.

namespace flash_attn::mla {
MlaScheduleStatus ResolveGqaResources(int64_t heads, int64_t kvHeads, const MlaGqaOptions& o, const MlaPlatformInfo& p,
                                      MlaGqaResources& output)
{
    if (heads < 1 || kvHeads < 1 || heads % kvHeads || o.packGqa < -1 || o.packGqa > 1)
        return MlaScheduleStatus::UnsupportedShape;
    if (heads > kMlaMaxQueryHeads)
        return MlaScheduleStatus::FieldOverflow;
    if (!p.aicNum || p.aicNum > kMlaMaxAic || p.aivNum != 2 * p.aicNum || o.smMargin < 0 || o.smMargin >= p.aicNum)
        return MlaScheduleStatus::InvalidPlatform;
    MlaGqaResources r{};
    r.platform = p;
    r.platform.aicNum -= uint32_t(o.smMargin);
    r.platform.aivNum = 2 * r.platform.aicNum;
    for (uint32_t i = 0; i < kMlaMaxAic; ++i) {
        if (i >= r.platform.aicNum)
            r.platform.blockDims[i] = 0;
        else if (!p.blockDims[i])
            return MlaScheduleStatus::InvalidPlatform;
    }
    r.packed = o.packGqa != 0 && heads != kvHeads;
    r.headGroups = r.packed ? kvHeads : heads;
    r.headsPerGroup = r.packed ? heads / kvHeads : 1;
    output = r;
    return MlaScheduleStatus::Success;
}

MlaScheduleStatus BuildGqaSchedule(const MlaGqaConfig& c, const MlaUsedInputs& inputs, const MlaWindowConfig& window,
                                   const MlaGqaOptions& options, const MlaPlatformInfo& physical,
                                   MlaGqaSchedule& output)
{
    MlaGqaSchedule r;
    auto status = ResolveGqaResources(c.shape.queryHeads, c.kvHeads, options, physical, r.resources_);
    if (status != MlaScheduleStatus::Success)
        return status;
    if ((c.valueDim != 128 && c.valueDim != 512) || (c.valueDim == 512 && c.kvHeads != 1) || c.splits < 0 ||
        c.splits > 256 || c.mBase < 32 || c.mBase > 512 || (c.mBase & (c.mBase - 1)) || c.nBase < 1 || c.nBase > 8192 ||
        window.maxQueryLength != c.shape.query.maxLength || window.maxKvLength != c.shape.kv.maxLength ||
        c.pageSize < 0 ||
        (c.pageSize && (c.pageSize < 16 || c.pageSize > 1024 || c.pageSize % 16 ||
                        c.shape.kv.layout != MlaSequenceLayout::Bsnd || c.shape.kv.maxLength % c.pageSize)))
        return MlaScheduleStatus::UnsupportedShape;
    if (c.shape.batch < 0 || uint64_t(c.shape.batch) > UINT32_MAX / r.resources_.headGroups)
        return MlaScheduleStatus::FieldOverflow;
    const auto normalized = NormalizeWindow(window, r.window_);
    if (normalized != MlaCausalStatus::Success)
        return normalized == MlaCausalStatus::FieldOverflow ? MlaScheduleStatus::FieldOverflow
                                                            : MlaScheduleStatus::UnsupportedShape;
    status = ParseUsedLengths(c.shape, inputs, r.lengths_);
    if (status != MlaScheduleStatus::Success)
        return status;
    r.capacity_ = c.splits == 0 ? kMlaMaxAic : c.splits;
    // Reuse the existing public allocation/stride guard, including empty
    // tensors. Its 512-wide envelope also bounds the 128-wide representation.
    MlaAccumulatorLayout accumulator{};
    status = BuildSplitAccumulatorLayout(
        {c.shape.query.layout == MlaSequenceLayout::Tnd ? MlaAccumulatorQueryLayout::Tnd
                                                        : MlaAccumulatorQueryLayout::Bsnd,
         c.shape.batch, c.shape.query.maxLength, c.shape.query.totalTokens, c.shape.queryHeads, r.capacity_},
        accumulator);
    if (status != MlaScheduleStatus::Success)
        return status;
    try {
        r.config_ = c;
        detail::BaseInfo base{};
        base.bSize = c.shape.batch;
        base.n2Size = r.resources_.headGroups;
        base.gSize = r.resources_.headsPerGroup;
        base.isS1G = true;
        base.s1Size = c.shape.query.maxLength;
        base.s2Size = c.shape.kv.maxLength;
        base.actualLenQDims = base.actualLenKvDims = c.shape.batch;
        base.attenMaskFlag = r.window_.kind != MlaWindowKind::None;
        base.sparseMode = r.window_.kind == MlaWindowKind::Local ? 4 : base.attenMaskFlag ? 3 : 0;
        base.preToken = base.sparseMode == 3 ? INT32_MAX : r.window_.left;
        base.nextToken = base.sparseMode == 3 ? 0 : r.window_.right;
        uint64_t tiles = 0;
        // Check source uint32 products and ceil-div BEFORE allocating prefixes.
        for (const auto& s : r.lengths_.sequences) {
            const uint64_t rows = uint64_t(s.queryLength) * base.gSize;
            if (rows > uint64_t(UINT32_MAX) - (c.mBase - 1))
                return MlaScheduleStatus::FieldOverflow;
            const uint64_t groups = (rows + c.mBase - 1) / c.mBase;
            const uint64_t count = groups * ((uint64_t(s.kvLength) + c.nBase - 1) / c.nBase);
            if (count > (UINT32_MAX - tiles) / base.n2Size)
                return MlaScheduleStatus::FieldOverflow;
            tiles += count * base.n2Size;
        }
        r.prefix_.reserve(uint64_t(c.shape.batch) * base.n2Size + 1);
        r.prefix_.push_back(0);
        for (const auto& s : r.lengths_.sequences) {
            base.actualSeqS1Size.push_back(s.queryLength);
            base.actualSeqS2Size.push_back(s.kvLength);
            const uint64_t groups = (uint64_t(s.queryLength) * base.gSize + c.mBase - 1) / c.mBase;
            for (uint32_t h = 0; h < base.n2Size; ++h)
                r.prefix_.push_back(r.prefix_.back() + groups);
        }
        if (c.splits == 0) {
            detail::SplitParam param{};
            param.mBaseSize = c.mBase;
            param.s2BaseSize = c.nBase;
            param.gS1BaseSizeOfFd = 8;
            detail::SplitContext context(base, param);
            detail::CalcSplitInfo(context);
            detail::CalcCostInfo(context);
            detail::SplitResult n{r.resources_.platform.aicNum, 2};
            detail::SplitCore(r.resources_.platform.aicNum, context, n);
            auto& native = r.native_;
            native.usedCoreNum = n.usedCoreNum;
            native.maxCost = n.maxCost;
            native.totalCost = context.costInfo.totalCost;
            native.totalMBlocks = r.Groups();
            native.totalS2Blocks = context.costInfo.totalBlockNum;
            std::copy(n.bN2End.begin(), n.bN2End.end(), native.bN2End.begin());
            std::copy(n.gS1End.begin(), n.gS1End.end(), native.gS1End.begin());
            std::copy(n.s2End.begin(), n.s2End.end(), native.s2End.begin());
            auto& fd = native.fd;
            fd.numOfFdHead = n.numOfFdHead;
            fd.maxS2SplitNum = n.maxS2SplitNum;
            fd.usedVecNumOfFd = n.usedVecNumOfFd;
            fd.gS1BaseSizeOfFd = 8;
#define COPY_GQA_FD(field) std::copy(n.fdRes.field.begin(), n.fdRes.field.end(), fd.field.begin())
            COPY_GQA_FD(bN2IdxOfFdHead);
            COPY_GQA_FD(gS1IdxOfFdHead);
            COPY_GQA_FD(s2SplitNumOfFdHead);
            COPY_GQA_FD(s2SplitStartIdxOfCore);
            COPY_GQA_FD(gS1SplitNumOfFdHead);
            COPY_GQA_FD(gS1LastPartSizeOfFdHead);
            COPY_GQA_FD(gS1IdxEndOfFdHead);
            COPY_GQA_FD(gS1IdxEndOfFdHeadSplit);
#undef COPY_GQA_FD
            if (!n.usedCoreNum || n.usedCoreNum > r.resources_.platform.aicNum ||
                fd.numOfFdHead > r.resources_.platform.aicNum || fd.maxS2SplitNum > r.resources_.platform.aicNum ||
                fd.usedVecNumOfFd > 2 * n.usedCoreNum)
                return MlaScheduleStatus::InvalidSplit;
        }
        output = std::move(r);
        return MlaScheduleStatus::Success;
    } catch (const std::bad_alloc&) {
        return MlaScheduleStatus::AllocationFailure;
    } catch (const std::length_error&) {
        return MlaScheduleStatus::AllocationFailure;
    }
}

MlaScheduleStatus MlaGqaSchedule::Task(uint64_t group, uint32_t split, MlaGqaTask& output) const
{
    if (group >= Groups() || split >= capacity_)
        return MlaScheduleStatus::InvalidSplit;
    const uint32_t slot = std::upper_bound(prefix_.begin(), prefix_.end(), group) - prefix_.begin() - 1;
    const uint32_t b = slot / resources_.headGroups, h = slot % resources_.headGroups;
    const auto& s = lengths_.sequences[b];
    const auto& c = config_;
    MlaGqaTask t{};
    t.batch = b;
    t.headGroup = h;
    t.kvHead = resources_.packed ? h : h / (c.shape.queryHeads / c.kvHeads);
    t.m = group - prefix_[slot];
    t.split = split;
    t.rowBegin = uint64_t(t.m) * c.mBase;
    t.rowEnd = std::min(uint64_t(t.rowBegin) + c.mBase, uint64_t(s.queryLength) * resources_.headsPerGroup);
    const MlaWindowKernelParams w{uint32_t(window_.kind), window_.left, window_.right};
    SplitBsndFragment(s.queryLength, s.kvLength, resources_.headsPerGroup, t.m, c.mBase, c.nBase, w,
                      c.splits ? c.splits : 1, c.splits ? split : 0, t.s2Begin, t.s2End);
    t.splitCount = c.splits;
    t.owner = (group * (c.splits ? c.splits : 1) + split) % resources_.platform.aicNum;
    if (!c.splits) {
        MlaAutoSplitBsndSchedule schedule{};
        schedule.usedCoreNum = native_.usedCoreNum;
        std::copy(native_.bN2End.begin(), native_.bN2End.end(), schedule.bEnd);
        std::copy(native_.gS1End.begin(), native_.gS1End.end(), schedule.mEnd);
        std::copy(native_.s2End.begin(), native_.s2End.end(), schedule.nEnd);
        MlaAutoSplitBsndTask part{};
        part.batch = slot;
        part.m = t.m;
        part.split = split;
        part.s2Begin = t.s2Begin;
        part.s2End = t.s2End;
        t.splitCount = t.s2Begin == t.s2End ? 1 : auto_split_detail::Parts(schedule, part);
        if (!t.splitCount || split >= t.splitCount)
            return MlaScheduleStatus::InvalidSplit;
        if (t.s2Begin != t.s2End) {
            t.owner = part.owner;
            t.s2Begin = part.s2Begin;
            t.s2End = part.s2End;
        }
    }
    output = t;
    return MlaScheduleStatus::Success;
}

MlaScheduleStatus MlaGqaSchedule::Row(uint64_t group, uint32_t split, uint32_t localRow, MlaGqaRow& output) const
{
    MlaGqaTask t{};
    if (Task(group, split, t) != MlaScheduleStatus::Success || localRow >= t.rowEnd - t.rowBegin)
        return MlaScheduleStatus::InvalidSplit;
    const auto& c = config_;
    const auto& s = lengths_.sequences[t.batch];
    const uint32_t flat = t.rowBegin + localRow;
    MlaGqaRow r{};
    r.batch = t.batch;
    r.kvHead = t.kvHead;
    r.head = t.headGroup * resources_.headsPerGroup + flat % resources_.headsPerGroup;
    r.query = flat / resources_.headsPerGroup;
    r.queryToken = s.queryStart + r.query;
    r.kvStart = s.kvStart;
    r.out = (r.queryToken * c.shape.queryHeads + r.head) * c.valueDim * 2;
    const uint64_t lse = c.shape.query.layout == MlaSequenceLayout::Tnd
                             ? uint64_t(r.head) * c.shape.query.totalTokens + r.queryToken
                             : (uint64_t(t.batch) * c.shape.queryHeads + r.head) * c.shape.query.maxLength + r.query;
    r.lse = lse * 4;
    r.accumulatorOut = r.accumulatorLse = UINT64_MAX;
    if (c.splits != 1) {
        const uint64_t row = uint64_t(split) * c.shape.query.totalTokens * c.shape.queryHeads + lse;
        r.accumulatorOut = row * c.valueDim * 4;
        r.accumulatorLse = row * 4;
    }
    output = r;
    return MlaScheduleStatus::Success;
}
} // namespace flash_attn::mla
