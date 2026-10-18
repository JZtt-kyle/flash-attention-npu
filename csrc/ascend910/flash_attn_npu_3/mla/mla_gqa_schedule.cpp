// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
// Local GQA coordinate adapter; source algorithms/provenance in mla_gqa_schedule.h.
// This translation unit executes only on host/AICPU. Bisheng's ASC front end
// also defines __NPU_ARCH__ for host-only files; keep reused scalar consumers
// free of AICore annotations here.
#ifdef __NPU_ARCH__
#undef __NPU_ARCH__
#endif
#include "mla_gqa_schedule.h"
#include "detail/cann_split_core.h"
#include "mla_auto_split_bsnd_consumer.h"
#include "mla_split_bsnd_consumer.h"
#include <algorithm>
#include <new>
#include <stdexcept>
#include <utility>

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
