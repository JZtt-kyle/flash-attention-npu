/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Adapted from the fixed sources listed in mla_split_kv.h. Native algorithms
// remain in detail/cann_split_core; checked CPU layout/FA3 adaptation is local.
// FA3 portions: Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang,
// Vijay Thakkar, Pradeep Ramani, Tri Dao; BSD 3-Clause (LICENSE.fa3).
#include "mla_split_kv.h"
#include "detail/cann_split_core.h"
#include <algorithm>
#include <new>

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
