// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
// Reuse pinned PA geometry and native variable partition arithmetic, without flattening
// physical pages into a fictitious contiguous KV allocation.
#include "mla_auto_split_paged_tnd_plan.h"
#include <algorithm>
#include <new>
#include <stdexcept>
#include <utility>

namespace flash_attn::mla {
namespace {
uint32_t CapacityBase(const MlaPagedShape& s)
{
    const uint32_t base = s.pageSize < 512 && 512 % s.pageSize ? 512 / s.pageSize * s.pageSize : 512;
    return std::max(uint32_t(1), std::min(uint32_t(s.logicalKvCapacity), base));
}
MlaAutoSplitMixedPlanConfig Config(const MlaPagedShape& s, const MlaPlatformInfo& p, uint32_t m, uint32_t n)
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
    status = BuildAutoSplitMixedPlanLayout(Config(s, p, 256, CapacityBase(s)), bound);
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
        result.nBase = s.s2BaseSize ? s.s2BaseSize : CapacityBase(shape);
        // Only logical KV intervals are reused; physical pages remain invocation state.
        status = BuildAutoSplitMixedPlan(Config(shape, platform, result.mBase, result.nBase), window,
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
