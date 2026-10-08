/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Derived from CANN ops-transformer ca18aee610729ca5270a44e2d84c751bc687fbf0,
// attention/fused_infer_attention_score/op_host/arch22/fia_tiling_nonquant_mla.cpp:
// DealSameSeqEachBatch, CalcInnerSize, CalcMBaseSize, CreateSplitInput, Split, CalcMmResSize,
// FillTilingWorkspaceParams, CalcNormalWorkspaceSize, CalcNumBlocks, CalcScheduleMode.
// The upstream license is reproduced in LICENSE.cann next to this file.
// Changes: independent BSND actual-KV-length entry; validate int32 lengths before
// numerical conversion to the source int64 vector. Reuse detail::SplitCore and
// constrain FD endpoints to complete M blocks for FA3 num_splits=1, with per-batch
// costs. Static field/resource/workspace checks follow mla_schedule.cpp while
// production BuildDenseSchedule and its metadata consumers remain unchanged.

#include "mla_bsnd_schedule.h"
#include "detail/cann_split_core.h"

#include <algorithm>
#include <limits>
#include <new>
#include <numeric>

namespace flash_attn::mla {
namespace bsnd_schedule_detail {

constexpr uint64_t Align(uint64_t value, uint64_t alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

uint32_t CalcMBaseSize(uint32_t batch, uint32_t rows, uint32_t aicNum, bool sameKvLengths)
{
    uint32_t mBaseSize = 256;
    // DealSameSeqEachBatch: contiguous BSND uses equality of actual KV lengths.
    if (batch % aicNum == 0 || !sameKvLengths) {
        return mBaseSize;
    }
    uint32_t lcm = (batch / std::gcd(batch, aicNum)) * aicNum;
    uint32_t splitOfS1G = lcm / batch;
    if (splitOfS1G == 0 || rows % splitOfS1G != 0) {
        return mBaseSize;
    }
    uint32_t candidate = rows / splitOfS1G;
    if (candidate <= mBaseSize && candidate >= 128) {
        mBaseSize = candidate;
    }
    return mBaseSize;
}

MlaLocalMemory RequiredLocalMemory(uint32_t cvRatio)
{
    // fia_block_cube_nonquant_mla.h::InitBuffers: four Q/P and three K/V L1 slots.
    // fia_block_vec_nonquant_mla.h::InitBuffers: outputBuff2 doubles for C1V1.
    return {7 * 128 * 288 * 2, (cvRatio == 1 ? 174U : 170U) * 1024, 2 * 32768, 2 * 32768, 2 * 65536};
}

bool HasLocalMemory(const MlaLocalMemory& have, const MlaLocalMemory& need)
{
    return have.l1Bytes >= need.l1Bytes && have.ubBytes >= need.ubBytes && have.l0aBytes >= need.l0aBytes &&
           have.l0bBytes >= need.l0bBytes && have.l0cBytes >= need.l0cBytes;
}

template <size_t N>
void CopyField(std::array<uint32_t, N>& dst, const std::vector<uint32_t>& src)
{
    std::copy(src.begin(), src.end(), dst.begin());
}

bool ValidSplit(const detail::SplitResult& split, const detail::BaseInfo& base, uint32_t mBlocks, uint32_t aicNum)
{
    if (split.usedCoreNum == 0 || split.usedCoreNum > aicNum) {
        return false;
    }
    const uint32_t batch = base.bSize;
    uint64_t previous = 0;
    for (uint32_t i = 0; i < split.usedCoreNum; ++i) {
        // num_splits=1: every boundary must be between complete M blocks.
        if (split.bN2End[i] > batch || split.gS1End[i] >= mBlocks || split.s2End[i] != 0) {
            return false;
        }
        uint64_t end = uint64_t(split.bN2End[i]) * mBlocks + split.gS1End[i];
        if (end <= previous || end > uint64_t(batch) * mBlocks) {
            return false;
        }
        previous = end;
    }
    // Source CalcSplitPlan stops when unassignedCost reaches zero, so a zero-KV
    // suffix can be outside its final endpoint. InitOutputSingleCore in the
    // source Kernel initializes empty outputs independently of these partitions.
    // Preserve source endpoints; require every omitted sample to have zero KV.
    if (previous < uint64_t(batch) * mBlocks && previous % mBlocks != 0) {
        return false;
    }
    for (uint32_t b = previous / mBlocks; b < batch; ++b) {
        if (detail::GetS2SeqSize(b, base) != 0) {
            return false;
        }
    }
    return true;
}

void SplitWithoutFlashDecode(uint32_t cores, const detail::BaseInfo& base, const detail::SplitParam& param,
                             uint32_t mBlocks, detail::SplitResult& split)
{
    detail::SplitCore(cores, base, param, split);
    if (split.numOfFdHead == 0) {
        return;
    }
    // FA3 num_splits=1 requires complete M blocks on each core. Round each
    // source endpoint up to the end of its M block and coalesce empty ranges.
    // Rebuild the partition and its costs; never just clear an FD plan's flags.
    // Filtering out all source FD candidates can otherwise discard balanced
    // partitions even when there are enough M blocks to occupy every core.
    detail::SplitContext context(base, param);
    detail::CalcSplitInfo(context);
    // Q uses full capacity, so every batch has the same M block count. KV
    // costs differ by batch; zero-length batches contribute zero. Output
    // initialization is independent of the compute endpoints (see ValidSplit).
    std::vector<int64_t> batchPrefix(uint64_t(base.bSize) + 1, 0);
    std::vector<int64_t> firstCost(base.bSize, 0);
    for (uint32_t b = 0; b < base.bSize; ++b) {
        detail::BatchCache batch{};
        detail::CalcBatchCache(b, context, batch);
        detail::S1GCache first{}, last{};
        detail::CalcS1GCache(0, context, batch, first);
        detail::CalcS1GCache(mBlocks - 1, context, batch, last);
        firstCost[b] = first.s1GCost;
        batchPrefix[b + 1] = batchPrefix[b] + int64_t(mBlocks - 1) * first.s1GCost + last.s1GCost;
    }
    const auto costBefore = [&](uint64_t end) {
        const uint64_t b = end / mBlocks;
        return batchPrefix[b] + (b < base.bSize ? int64_t(end % mBlocks) * firstCost[b] : 0);
    };
    detail::SplitResult whole{cores, split.vecCubeRatio};
    uint64_t previous = 0;
    for (uint32_t i = 0; i < split.usedCoreNum; ++i) {
        const uint64_t end = uint64_t(split.bN2End[i]) * mBlocks + split.gS1End[i] + (split.s2End[i] != 0);
        if (end > previous) {
            const uint32_t core = whole.usedCoreNum++;
            whole.bN2End[core] = end / mBlocks;
            whole.gS1End[core] = end % mBlocks;
            whole.maxCost = std::max(whole.maxCost, costBefore(end) - costBefore(previous));
            previous = end;
        }
    }
    split = std::move(whole);
}

MlaScheduleStatus Calculate(const MlaDenseShape& shape, const MlaPlatformInfo& platform, const int32_t* kvLengths,
                            uint64_t lengthCount, MlaSchedule& output)
{
    constexpr auto u32Max = std::numeric_limits<uint32_t>::max();
    for (int64_t size : {shape.batch, shape.queryLength, shape.queryHeads, shape.kvLength}) {
        if (size < 0) {
            return MlaScheduleStatus::UnsupportedShape;
        }
        if (uint64_t(size) > u32Max) {
            return MlaScheduleStatus::FieldOverflow;
        }
    }
    // Multiplication is wide even for rejected input; never overflow the caller's int64 shape.
    uint64_t rows = uint64_t(shape.queryLength) * uint64_t(shape.queryHeads);
    if (shape.queryHeads == 0 || shape.kvLength > kMlaMaxKvLength) {
        return MlaScheduleStatus::UnsupportedShape;
    }
    if (rows > kMlaMaxDenseRows || shape.queryHeads > kMlaMaxQueryHeads || shape.queryLength > kMlaMaxQueryLength ||
        uint64_t(shape.batch) * rows > kMlaMaxBatchRows) {
        return MlaScheduleStatus::FieldOverflow;
    }
    if (platform.aicNum == 0 || platform.aicNum > kMlaMaxAic || platform.aivNum == 0 ||
        platform.aivNum % platform.aicNum != 0) {
        return MlaScheduleStatus::InvalidPlatform;
    }
    uint32_t ratio = platform.aivNum / platform.aicNum;
    if (ratio != 1 && ratio != 2) {
        return MlaScheduleStatus::InvalidPlatform;
    }
    for (uint32_t c = 0; c < platform.aicNum; ++c) {
        if (platform.blockDims[c] == 0 || platform.blockDims[c] > platform.aicNum) {
            return MlaScheduleStatus::InvalidPlatform;
        }
    }
    auto requiredMemory = RequiredLocalMemory(ratio);
    if (!HasLocalMemory(platform.localMemory, requiredMemory)) {
        return MlaScheduleStatus::InsufficientLocalMemory;
    }
    if ((kvLengths == nullptr && lengthCount != 0) || (kvLengths != nullptr && lengthCount != uint64_t(shape.batch))) {
        return MlaScheduleStatus::UnsupportedShape;
    }
    bool allFull = true;
    bool sameKvLengths = true;
    // Complete validation precedes any source-vector allocation or narrowing.
    for (uint64_t b = 0; b < lengthCount; ++b) {
        if (kvLengths[b] < 0 || kvLengths[b] > shape.kvLength) {
            return MlaScheduleStatus::UnsupportedShape;
        }
        allFull = allFull && kvLengths[b] == shape.kvLength;
        sameKvLengths = sameKvLengths && kvLengths[b] == kvLengths[0];
    }
    if (allFull || shape.batch == 0 || rows == 0 || shape.kvLength == 0) {
        return BuildDenseSchedule(shape, platform, output);
    }
    uint32_t batch = static_cast<uint32_t>(shape.batch);
    MlaSchedule schedule{};
    schedule.bSize = batch;
    schedule.n2Size = 1;
    schedule.gSize = static_cast<uint32_t>(shape.queryHeads);
    schedule.s1Size = static_cast<uint32_t>(shape.queryLength);
    schedule.s2Size = static_cast<uint32_t>(shape.kvLength);
    schedule.headDim = 512;
    schedule.headDimRope = 64;
    schedule.aicNum = platform.aicNum;
    schedule.aivNum = platform.aivNum;
    schedule.cvRatio = ratio;
    schedule.scheduleMode = 1;
    if (sameKvLengths && uint64_t(batch / std::gcd(batch, platform.aicNum)) * platform.aicNum > u32Max) {
        return MlaScheduleStatus::FieldOverflow;
    }
    schedule.mBaseSize = CalcMBaseSize(batch, static_cast<uint32_t>(rows), platform.aicNum, sameKvLengths);
    // First and only CalcInnerSize pass: splitKVFlag=false, pageAttentionFlag=false.
    schedule.s2BaseSize = std::min(schedule.s2Size, 512U);
    schedule.requiredLocalMemory = requiredMemory;
    uint32_t mBlocks = (rows + schedule.mBaseSize - 1) / schedule.mBaseSize;
    uint32_t s2Blocks = (schedule.s2Size + schedule.s2BaseSize - 1) / schedule.s2BaseSize;
    if (uint64_t(batch) * mBlocks * s2Blocks > u32Max) {
        return MlaScheduleStatus::FieldOverflow;
    }

    auto& workspace = schedule.workspace;
    uint64_t mSize = Align(std::min(rows, uint64_t(schedule.mBaseSize)), 16);
    workspace.mm1ResSize = mSize * Align(schedule.s2BaseSize, 32);
    workspace.mm2ResSize = mSize * schedule.headDim;
    workspace.fdAccumOutSize = uint64_t(platform.aicNum) * 2 * schedule.mBaseSize * schedule.headDim;
    workspace.fdLogSumExpSize = uint64_t(platform.aicNum) * 2 * 2 * schedule.mBaseSize * (32 / sizeof(float));
    // Source InitParams sets coreNum_=aicNum_, not usedCoreNum_. Preserve its reserve.
    workspace.userBytes =
        uint64_t(2) * platform.aicNum * (6 * workspace.mm1ResSize + 8 * workspace.mm2ResSize + 8 * schedule.mBaseSize);
    workspace.libApiBytes = platform.libApiWorkspaceBytes;
    if (workspace.libApiBytes > std::numeric_limits<uint64_t>::max() - workspace.userBytes) {
        return MlaScheduleStatus::WorkspaceOverflow;
    }
    workspace.totalBytes = workspace.userBytes + workspace.libApiBytes;

    detail::BaseInfo base{};
    base.bSize = batch;
    base.n2Size = 1;
    base.gSize = schedule.gSize;
    base.s1Size = schedule.s1Size;
    base.s2Size = schedule.s2Size;
    base.actualLenKvDims = batch;
    base.actualSeqS2Size.reserve(batch);
    for (uint32_t b = 0; b < batch; ++b) {
        base.actualSeqS2Size.push_back(static_cast<int64_t>(kvLengths[b]));
    }
    detail::SplitParam param{};
    param.mBaseSize = schedule.mBaseSize;
    param.s2BaseSize = schedule.s2BaseSize;
    detail::SplitResult split{platform.aicNum, ratio};
    SplitWithoutFlashDecode(platform.aicNum, base, param, mBlocks, split);
    // Do not force FD fields to zero to make a non-FD launch appear valid.
    if (split.numOfFdHead != 0 || split.maxS2SplitNum != 0 || split.usedVecNumOfFd != 0) {
        return MlaScheduleStatus::UnexpectedFlashDecode;
    }
    if (!ValidSplit(split, base, mBlocks, platform.aicNum)) {
        return MlaScheduleStatus::InvalidSplit;
    }
    schedule.usedCoreNum = split.usedCoreNum;
    schedule.blockDim = platform.blockDims[split.usedCoreNum - 1];
    // AIC endpoints must all have a launched block. The source kernel uses GetBlockNum().
    if (schedule.blockDim < schedule.usedCoreNum) {
        return MlaScheduleStatus::InvalidPlatform;
    }
    schedule.maxCost = split.maxCost;
    CopyField(schedule.bN2End, split.bN2End);
    CopyField(schedule.gS1End, split.gS1End);
    CopyField(schedule.s2End, split.s2End);
    auto& fd = schedule.fd;
    fd.numOfFdHead = split.numOfFdHead;
    fd.maxS2SplitNum = split.maxS2SplitNum;
    fd.usedVecNumOfFd = split.usedVecNumOfFd;
    fd.gS1BaseSizeOfFd = param.gS1BaseSizeOfFd;
    CopyField(fd.bN2IdxOfFdHead, split.fdRes.bN2IdxOfFdHead);
    CopyField(fd.gS1IdxOfFdHead, split.fdRes.gS1IdxOfFdHead);
    CopyField(fd.s2SplitNumOfFdHead, split.fdRes.s2SplitNumOfFdHead);
    CopyField(fd.s2SplitStartIdxOfCore, split.fdRes.s2SplitStartIdxOfCore);
    CopyField(fd.gS1SplitNumOfFdHead, split.fdRes.gS1SplitNumOfFdHead);
    CopyField(fd.gS1LastPartSizeOfFdHead, split.fdRes.gS1LastPartSizeOfFdHead);
    // Source SetSplitOutput uses usedCoreNum*2 even for C1V1. Read only the actual
    // aicNum*cvRatio vector entries; the rest of our fixed-capacity arrays stay zero.
    CopyField(fd.gS1IdxEndOfFdHead, split.fdRes.gS1IdxEndOfFdHead);
    CopyField(fd.gS1IdxEndOfFdHeadSplit, split.fdRes.gS1IdxEndOfFdHeadSplit);

    // fia_kernel_nonquant_mla.h::Init: two GM slots, offsets in bytes from user workspace.
    uint64_t slots = uint64_t(2) * schedule.blockDim;
    workspace.pOffsetBytes = slots * workspace.mm1ResSize * 4;
    workspace.pvOffsetBytes = workspace.pOffsetBytes + slots * workspace.mm1ResSize * 2;
    workspace.vec2OffsetBytes = workspace.pvOffsetBytes + slots * workspace.mm2ResSize * 4;
    workspace.kernelBytes = workspace.vec2OffsetBytes + slots * workspace.mm2ResSize * 4;
    if (workspace.kernelBytes > workspace.userBytes) {
        return MlaScheduleStatus::WorkspaceOverflow;
    }
    output = schedule;
    return MlaScheduleStatus::Success;
}

} // namespace bsnd_schedule_detail

MlaScheduleStatus BuildBsndKvLengthSchedule(const MlaDenseShape& shape, const MlaPlatformInfo& platform,
                                            const int32_t* kvLengths, uint64_t lengthCount, MlaSchedule& output)
{
    try {
        return bsnd_schedule_detail::Calculate(shape, platform, kvLengths, lengthCount, output);
    } catch (const std::bad_alloc&) {
        return MlaScheduleStatus::AllocationFailure;
    }
}

} // namespace flash_attn::mla
