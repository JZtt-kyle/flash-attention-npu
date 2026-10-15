// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
// Reuse pinned PA geometry and explicit split arithmetic, without flattening
// physical pages into a fictitious contiguous KV allocation.
#include "mla_split_paged_tnd_plan.h"
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
MlaSplitMixedPlanConfig Config(const MlaPagedShape& s, const MlaPlatformInfo& p, int64_t splits, uint32_t m, uint32_t n)
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
    status = BuildSplitMixedPlanLayout(Config(s, p, splits, 256, CapacityBase(s)), bound);
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
        result.nBase = s.s2BaseSize ? s.s2BaseSize : CapacityBase(shape);
        // Only reuse the mixed plan's logical intervals and compact Q layout.
        // Its virtual KV origins never escape this PA adapter.
        status = BuildSplitMixedPlan(Config(shape, platform, splits, result.mBase, result.nBase), window,
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
