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
// remain in cann_split_core.cpp; checked CPU layout/FA3 adaptation is local.
// FA3 portions: Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang,
// Vijay Thakkar, Pradeep Ramani, Tri Dao; BSD 3-Clause (root LICENSE).

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
    w.totalBytes = w.userBytes;
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
template <typename Config>
MlaScheduleStatus ValidateSplitPlanConfig(const Config& c)
{
    if (c.mBaseSize < 1 || c.mBaseSize > 256 || c.s2BaseSize < 1 || c.s2BaseSize > 512)
        return MlaScheduleStatus::UnsupportedShape;
    if (c.numSplits < 1 || c.numSplits > 256)
        return MlaScheduleStatus::InvalidSplit;
    if (c.aicNum < 1 || c.aicNum > kMlaMaxAic || c.blockDim < 1 || c.blockDim > c.aicNum ||
        (c.cvRatio != 1 && c.cvRatio != 2))
        return MlaScheduleStatus::InvalidPlatform;
    return MlaScheduleStatus::Success;
}

// Explicit and native split retain different row budgets, but use the same
// scratch reserve and aligned FP32 partial-output/statistics storage.
template <typename Config, typename Layout>
MlaScheduleStatus BuildPartialStorage(const Config& c, uint64_t rows, uint32_t blockDim, Layout& result)
{
    uint64_t outBytes, statsBytes;
    if (!Multiply(result.partialRows, 512 * sizeof(float), outBytes) ||
        !Multiply(result.partialRows, 8 * sizeof(float), statsBytes))
        return MlaScheduleStatus::WorkspaceOverflow;
    const auto status =
        BuildSplitKvWorkspace({c.mBaseSize, c.s2BaseSize, int64_t(rows), c.aicNum, blockDim}, result.scratch);
    if (status != MlaScheduleStatus::Success)
        return status;
    // Logical M can be odd. Preserve 32-byte alignment of persistent FP32 data.
    result.outOffsetBytes = (result.scratch.attention.userBytes + 31) / 32 * 32;
    if (!Add(result.outOffsetBytes, outBytes, result.sumOffsetBytes) ||
        !Add(result.sumOffsetBytes, statsBytes, result.maxOffsetBytes) ||
        !Add(result.maxOffsetBytes, statsBytes, result.userBytes) ||
        !Add(result.userBytes, c.libApiBytes, result.totalBytes))
        return MlaScheduleStatus::WorkspaceOverflow;
    return MlaScheduleStatus::Success;
}

MlaAccumulatorInput AccumulatorInput(const MlaSplitKvPlanConfig& c)
{
    return {MlaAccumulatorQueryLayout::Bsnd, c.batch,      c.queryCapacity,
            c.batch * c.queryCapacity,       c.queryHeads, c.numSplits};
}
} // namespace

MlaScheduleStatus BuildSplitKvPlanLayout(const MlaSplitKvPlanConfig& c, MlaSplitKvPlanLayout& output)
{
    if (c.batch < 0 || c.queryCapacity < 0 || c.kvCapacity < 0 || c.queryHeads < 1)
        return MlaScheduleStatus::UnsupportedShape;
    auto status = ValidateSplitPlanConfig(c);
    if (status != MlaScheduleStatus::Success)
        return status;
    if (uint64_t(c.batch) > UINT32_MAX || c.queryCapacity > kMlaMaxQueryLength || c.kvCapacity > kMlaMaxKvLength ||
        c.queryHeads > kMlaMaxQueryHeads) {
        return MlaScheduleStatus::FieldOverflow;
    }
    const uint64_t rows = uint64_t(c.queryCapacity) * c.queryHeads;
    if (rows > kMlaMaxDenseRows || (rows && uint64_t(c.batch) > kMlaMaxBatchRows / rows)) {
        return MlaScheduleStatus::FieldOverflow;
    }
    MlaSplitKvPlanLayout result{};
    status = BuildSplitAccumulatorLayout(AccumulatorInput(c), result.accumulator);
    if (status != MlaScheduleStatus::Success) {
        return status;
    }
    if (c.libApiBytes > uint64_t(INT64_MAX)) {
        return MlaScheduleStatus::WorkspaceOverflow;
    }
    result.emptyCapacity = c.batch == 0 || rows == 0 || c.kvCapacity == 0;
    if (!result.emptyCapacity) {
        const uint64_t perSample = (rows + c.mBaseSize - 1) / c.mBaseSize;
        if (!Multiply(c.batch, perSample, result.maxGroups) ||
            !Multiply(result.maxGroups, c.numSplits, result.maxTasks) ||
            !Multiply(result.maxTasks, c.mBaseSize, result.partialRows))
            return MlaScheduleStatus::WorkspaceOverflow;
        status = BuildPartialStorage(c, rows, c.blockDim, result);
        if (status != MlaScheduleStatus::Success)
            return status;
    }
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

MlaScheduleStatus BuildSplitCompactPlanLayout(const MlaSplitTndPlanConfig& c, bool compactKv,
                                              MlaSplitKvPlanLayout& output)
{
    const auto& s = c.shape;
    if (s.batch < 0 || s.totalQuery < 0 || (compactKv && s.totalKv < 0) || s.maxQueryLength < 0 || s.maxKvLength < 0 ||
        s.queryHeads < 1)
        return MlaScheduleStatus::UnsupportedShape;
    auto status = ValidateSplitPlanConfig(c);
    if (status != MlaScheduleStatus::Success)
        return status;
    if (uint64_t(s.batch) > UINT32_MAX || s.totalQuery > INT32_MAX || (compactKv && s.totalKv > INT32_MAX) ||
        s.maxQueryLength > kMlaMaxQueryLength || s.maxKvLength > kMlaMaxKvLength || s.queryHeads > kMlaMaxQueryHeads) {
        return MlaScheduleStatus::FieldOverflow;
    }
    const uint64_t rows = uint64_t(s.maxQueryLength) * s.queryHeads;
    const uint64_t totalRows = uint64_t(s.totalQuery) * s.queryHeads;
    if (rows > kMlaMaxDenseRows || totalRows > kMlaMaxBatchRows)
        return MlaScheduleStatus::FieldOverflow;
    if (uint64_t(s.totalQuery) > uint64_t(s.batch) * s.maxQueryLength ||
        (compactKv && uint64_t(s.totalKv) > uint64_t(s.batch) * s.maxKvLength))
        return MlaScheduleStatus::UnsupportedShape;
    MlaSplitKvPlanLayout result{};
    status = BuildSplitAccumulatorLayout(AccumulatorInput(c), result.accumulator);
    if (status != MlaScheduleStatus::Success)
        return status;
    if (c.libApiBytes > uint64_t(INT64_MAX))
        return MlaScheduleStatus::WorkspaceOverflow;
    result.emptyCapacity = s.totalQuery == 0 || (compactKv ? s.totalKv == 0 : s.batch == 0 || s.maxKvLength == 0);
    if (!result.emptyCapacity) {
        const uint64_t active = std::min(uint64_t(s.batch), uint64_t(s.totalQuery));
        // Each positive segment contributes at most M-1 padding rows. This
        // bound remains valid with arbitrary used lengths and zero segments.
        const uint64_t perSample = (rows + c.mBaseSize - 1) / c.mBaseSize;
        result.maxGroups = std::min(active * perSample, (totalRows + active * (c.mBaseSize - 1)) / c.mBaseSize);
        if (!Multiply(result.maxGroups, c.numSplits, result.maxTasks) ||
            !Multiply(result.maxTasks, c.mBaseSize, result.partialRows))
            return MlaScheduleStatus::WorkspaceOverflow;
        status = BuildPartialStorage(c, rows, c.blockDim, result);
        if (status != MlaScheduleStatus::Success)
            return status;
    }
    output = result;
    return MlaScheduleStatus::Success;
}
} // namespace

MlaScheduleStatus BuildSplitTndPlanLayout(const MlaSplitTndPlanConfig& c, MlaSplitKvPlanLayout& output)
{
    return BuildSplitCompactPlanLayout(c, true, output);
}

} // namespace flash_attn::mla

// Fixed CANN/FA3 provenance and licenses: mla_split.h.
// Compact Q / rectangular KV allocation bounds and addressing are local adapters.
// Reuses the explicit range, workspace and accumulator arithmetic; KV origins
// remain 64-bit batch*capacity even beyond the TND prefix range.

namespace flash_attn::mla {
MlaScheduleStatus BuildSplitMixedPlanLayout(const MlaSplitMixedPlanConfig& c, MlaSplitKvPlanLayout& output)
{
    const auto& s = c.shape;
    // Rectangular KV is bounded by capacity, without forming or narrowing B*C
    // as a compact prefix terminal. The shared helper ignores totalKv here.
    return BuildSplitCompactPlanLayout({{s.batch, s.totalQuery, 0, s.queryHeads, s.maxQueryLength, s.kvCapacity},
                                        c.mBaseSize,
                                        c.s2BaseSize,
                                        c.numSplits,
                                        c.aicNum,
                                        c.blockDim,
                                        c.cvRatio,
                                        c.libApiBytes},
                                       false, output);
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

MlaScheduleStatus BuildSplitPagedBudget(const MlaPagedShape& s, const MlaPlatformInfo& p, int64_t splits,
                                        MlaSplitPagedBudget& output)
{
    const bool tnd = s.query.layout == MlaSequenceLayout::Tnd;
    if ((s.query.layout != MlaSequenceLayout::Tnd && s.query.layout != MlaSequenceLayout::Bsnd) || s.pageSize < 16 ||
        s.pageSize > 1024 || s.pageSize % 16 || s.logicalKvCapacity < 0 || s.logicalKvCapacity > kMlaMaxKvLength ||
        s.logicalKvCapacity % s.pageSize || s.batch < 0 || uint64_t(s.batch) > (UINT32_MAX - 414) / 7)
        return MlaScheduleStatus::UnsupportedShape;
    MlaSchedule validation{};
    auto status = BuildDenseSchedule({tnd ? 0 : s.batch, s.query.maxLength, s.queryHeads, 0}, p, validation);
    if (status != MlaScheduleStatus::Success)
        return status;
    if (!tnd && (s.query.totalTokens < 0 || uint64_t(s.query.totalTokens) != uint64_t(s.batch) * s.query.maxLength))
        return MlaScheduleStatus::UnsupportedShape;
    const uint32_t blocks = s.batch && s.query.totalTokens && s.logicalKvCapacity ? p.aicNum : 1;
    if (p.aivNum != 2 * p.aicNum)
        return MlaScheduleStatus::InvalidPlatform;
    for (uint32_t i = 0; i < p.aicNum; ++i)
        if (p.blockDims[i] != blocks)
            return MlaScheduleStatus::InvalidPlatform;
    MlaSplitKvPlanLayout bound{};
    status = tnd ? BuildSplitMixedPlanLayout(ExplicitTndConfig(s, p, splits, 256, PagedSplitCapacityBase(s)), bound)
                 : BuildSplitKvPlanLayout(ExplicitBsndConfig(s, p, splits, 256, PagedSplitCapacityBase(s)), bound);
    if (status != MlaScheduleStatus::Success)
        return status;
    MlaSplitPagedBudget result{};
    const uint64_t rows = uint64_t(s.query.maxLength) * s.queryHeads;
    if (!bound.emptyCapacity) {
        const uint64_t totalRows = uint64_t(s.query.totalTokens) * s.queryHeads;
        const uint64_t active = std::min(uint64_t(s.batch), uint64_t(s.query.totalTokens));
        result.maxGroups = tnd ? std::min(active * ((rows + 127) / 128), (totalRows + active * 127) / 128)
                               : uint64_t(s.batch) * ((rows + 127) / 128);
        result.maxTasks = result.maxGroups * splits;
        // A source divisor M<256 can pad unequal TND segments differently.
        // Bound every M in [128,256] by at most 255 extra rows per segment;
        // B*maxQ and a rectangular M=256 slot count are not valid substitutes.
        result.partialRows = tnd ? (totalRows + active * 255) * splits : bound.partialRows;
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

MlaScheduleStatus BuildSplitPlan(const MlaUsedShape& s, const MlaPlatformInfo& p, uint32_t m, uint32_t n,
                                 uint32_t blocks, const MlaWindowConfig& window, int64_t splits,
                                 const std::vector<MlaCausalSequence>& lengths, MlaSplitPlan& output)
{
    MlaSplitPlan result;
    MlaScheduleStatus status;
    if (s.query.layout == MlaSequenceLayout::Bsnd) {
        status = BuildSplitKvPlanLayout({s.batch, s.query.maxLength, s.kv.maxLength, s.queryHeads, m, n, splits,
                                         p.aicNum, blocks, p.aicNum ? p.aivNum / p.aicNum : 0, p.libApiWorkspaceBytes},
                                        result.layout);
    } else if (s.kv.layout == MlaSequenceLayout::Tnd) {
        status = BuildSplitTndPlanLayout(
            {{s.batch, s.query.totalTokens, s.kv.totalTokens, s.queryHeads, s.query.maxLength, s.kv.maxLength},
             m,
             n,
             splits,
             p.aicNum,
             blocks,
             p.aicNum ? p.aivNum / p.aicNum : 0,
             p.libApiWorkspaceBytes},
            result.layout);
    } else {
        status =
            BuildSplitMixedPlanLayout({{s.batch, s.query.totalTokens, s.queryHeads, s.query.maxLength, s.kv.maxLength},
                                       m,
                                       n,
                                       splits,
                                       p.aicNum,
                                       blocks,
                                       p.aicNum ? p.aivNum / p.aicNum : 0,
                                       p.libApiWorkspaceBytes},
                                      result.layout);
    }
    if (status != MlaScheduleStatus::Success)
        return status;
    MlaWindow normalized{};
    if (window.maxQueryLength != s.query.maxLength || window.maxKvLength != s.kv.maxLength ||
        NormalizeWindow(window, normalized) != MlaCausalStatus::Success || lengths.size() != uint64_t(s.batch))
        return MlaScheduleStatus::UnsupportedShape;
    try {
        result.prefix.reserve(lengths.size() + 1);
        result.prefix.push_back(0);
        for (const auto& seq : lengths) {
            if (seq.queryLength < 0 || seq.queryLength > s.query.maxLength || seq.kvLength < 0 ||
                seq.kvLength > s.kv.maxLength)
                return MlaScheduleStatus::UnsupportedShape;
            const uint64_t rows = uint64_t(seq.queryLength) * s.queryHeads;
            result.prefix.push_back(result.prefix.back() + (result.layout.emptyCapacity ? 0 : (rows + m - 1) / m));
        }
        if (result.Groups() > result.layout.maxGroups)
            return MlaScheduleStatus::WorkspaceOverflow;
        output = std::move(result);
        return MlaScheduleStatus::Success;
    } catch (const std::bad_alloc&) {
        return MlaScheduleStatus::AllocationFailure;
    } catch (const std::length_error&) {
        return MlaScheduleStatus::AllocationFailure;
    }
}

MlaScheduleStatus BuildSplitPagedPlan(const MlaPagedShape& shape, const MlaPlatformInfo& platform,
                                      const MlaPagedInputs& inputs, const MlaWindowConfig& window, int64_t splits,
                                      MlaSplitPagedPlan& output)
{
    try {
        MlaSplitPagedPlan result;
        auto status = BuildSplitPagedBudget(shape, platform, splits, result.budget);
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
        status =
            BuildSplitPlan({shape.batch,
                            shape.queryHeads,
                            shape.query,
                            {MlaSequenceLayout::Bsnd, shape.batch * shape.logicalKvCapacity, shape.logicalKvCapacity}},
                           platform, result.mBase, result.nBase,
                           shape.batch && shape.query.totalTokens && shape.logicalKvCapacity ? platform.aicNum : 1,
                           window, splits, lengths, result.split);
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

} // namespace flash_attn::mla

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
        status = BuildPartialStorage(c, rows, c.aicNum, result);
        if (status != MlaScheduleStatus::Success)
            return status;
    }
    output = result;
    return MlaScheduleStatus::Success;
}

MlaScheduleStatus BuildAutoSplitPlan(const MlaUsedShape& shape, const MlaPlatformInfo& platform, uint32_t m, uint32_t n,
                                     const MlaWindowConfig& window, const std::vector<MlaCausalSequence>& lengths,
                                     MlaAutoSplitPlan& output)
{
    const MlaAutoSplitKvConfig c{shape.batch,
                                 shape.query.maxLength,
                                 shape.kv.maxLength,
                                 shape.queryHeads,
                                 m,
                                 n,
                                 platform.aicNum,
                                 platform.aicNum ? platform.aivNum / platform.aicNum : 0,
                                 platform.libApiWorkspaceBytes};
    MlaAutoSplitPlan result;
    MlaScheduleStatus status;
    if (shape.query.layout == MlaSequenceLayout::Bsnd) {
        status = BuildAutoSplitKvLayout(c, result.layout_);
    } else {
        status = BuildAutoSplitCompactPlanLayout({{shape.batch, shape.query.totalTokens, shape.kv.totalTokens,
                                                   shape.queryHeads, shape.query.maxLength, shape.kv.maxLength},
                                                  m,
                                                  n,
                                                  c.aicNum,
                                                  c.cvRatio,
                                                  c.libApiBytes},
                                                 shape.kv.layout, result.layout_);
    }
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
        if (result.Groups() > result.layout_.maxGroups)
            return MlaScheduleStatus::WorkspaceOverflow;
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

MlaScheduleStatus MlaAutoSplitPlan::Task(uint64_t group, uint32_t split, MlaAutoSplitKvTask& output) const
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

MlaScheduleStatus MlaAutoSplitPlan::Merge(uint32_t head, uint32_t part, MlaAutoSplitKvMerge& output) const
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
        status = BuildPartialStorage(c, rows, c.aicNum, result);
        if (status != MlaScheduleStatus::Success)
            return status;
    }
    output = result;
    return MlaScheduleStatus::Success;
}

MlaScheduleStatus BuildAutoSplitTndPlanLayout(const MlaAutoSplitTndPlanConfig& c, MlaAutoSplitKvLayout& output)
{
    return BuildAutoSplitCompactPlanLayout(c, MlaSequenceLayout::Tnd, output);
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

MlaAutoSplitMixedPlanConfig AutoTndConfig(const MlaPagedShape& s, const MlaPlatformInfo& p, uint32_t m, uint32_t n)
{
    return {{s.batch, s.query.totalTokens, s.queryHeads, s.query.maxLength, s.logicalKvCapacity},
            m,
            n,
            p.aicNum,
            2,
            p.libApiWorkspaceBytes};
}

MlaScheduleStatus BuildAutoSplitPagedBudget(const MlaPagedShape& s, const MlaPlatformInfo& p,
                                            MlaAutoSplitPagedBudget& output)
{
    const bool tnd = s.query.layout == MlaSequenceLayout::Tnd;
    if ((s.query.layout != MlaSequenceLayout::Tnd && s.query.layout != MlaSequenceLayout::Bsnd) || s.pageSize < 16 ||
        s.pageSize > 1024 || s.pageSize % 16 || s.logicalKvCapacity < 0 || s.logicalKvCapacity > kMlaMaxKvLength ||
        s.logicalKvCapacity % s.pageSize || s.batch < 0 || uint64_t(s.batch) > (UINT32_MAX - 756) / 7)
        return MlaScheduleStatus::UnsupportedShape;
    MlaSchedule validation{};
    auto status = BuildDenseSchedule({tnd ? 0 : s.batch, s.query.maxLength, s.queryHeads, 0}, p, validation);
    if (status != MlaScheduleStatus::Success)
        return status;
    if (!tnd && (s.query.totalTokens < 0 || uint64_t(s.query.totalTokens) != uint64_t(s.batch) * s.query.maxLength))
        return MlaScheduleStatus::UnsupportedShape;
    const uint32_t blocks = s.batch && s.query.totalTokens && s.logicalKvCapacity ? p.aicNum : 1;
    if (p.aivNum != 2 * p.aicNum)
        return MlaScheduleStatus::InvalidPlatform;
    for (uint32_t i = 0; i < p.aicNum; ++i)
        if (p.blockDims[i] != blocks)
            return MlaScheduleStatus::InvalidPlatform;
    MlaAutoSplitKvLayout bound{};
    status = tnd ? BuildAutoSplitMixedPlanLayout(AutoTndConfig(s, p, 256, PagedSplitCapacityBase(s)), bound)
                 : BuildAutoSplitKvLayout(AutoBsndConfig(s, p, 256, PagedSplitCapacityBase(s)), bound);
    if (status != MlaScheduleStatus::Success)
        return status;
    MlaAutoSplitPagedBudget result{};
    const uint64_t rows = uint64_t(s.query.maxLength) * s.queryHeads;
    if (!bound.emptyCapacity) {
        const uint64_t totalRows = uint64_t(s.query.totalTokens) * s.queryHeads;
        const uint64_t active = std::min(uint64_t(s.batch), uint64_t(s.query.totalTokens));
        result.maxGroups = tnd ? std::min(active * ((rows + 127) / 128), (totalRows + active * 127) / 128)
                               : uint64_t(s.batch) * ((rows + 127) / 128);
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

MlaScheduleStatus BuildAutoSplitPagedProductionBudget(const MlaPagedShape& s, const MlaPlatformInfo& p,
                                                      MlaAutoSplitPagedBudget& output)
{
    MlaAutoSplitPagedBudget result{};
    auto status = BuildAutoSplitPagedBudget(s, p, result);
    if (status != MlaScheduleStatus::Success)
        return status;
    status = BuildSplitAccumulatorLayout(
        {(s.query.layout == MlaSequenceLayout::Tnd ? MlaAccumulatorQueryLayout::Tnd : MlaAccumulatorQueryLayout::Bsnd),
         s.batch, s.query.maxLength, s.query.totalTokens, s.queryHeads, kMlaMaxAic},
        result.accumulator);
    if (status != MlaScheduleStatus::Success)
        return status;
    output = result;
    return MlaScheduleStatus::Success;
}

MlaScheduleStatus BuildAutoSplitPagedPlan(const MlaPagedShape& shape, const MlaPlatformInfo& platform,
                                          const MlaPagedInputs& inputs, const MlaWindowConfig& window,
                                          MlaAutoSplitPagedPlan& output)
{
    try {
        MlaAutoSplitPagedPlan result;
        auto status = BuildAutoSplitPagedBudget(shape, platform, result.budget);
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
        status = BuildAutoSplitPlan(
            {shape.batch,
             shape.queryHeads,
             shape.query,
             {MlaSequenceLayout::Bsnd, shape.batch * shape.logicalKvCapacity, shape.logicalKvCapacity}},
            platform, result.mBase, result.nBase, window, lengths, result.split);
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

} // namespace flash_attn::mla
