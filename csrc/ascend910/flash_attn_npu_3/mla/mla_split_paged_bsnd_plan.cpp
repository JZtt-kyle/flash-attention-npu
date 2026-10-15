// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
// Reuse pinned PA geometry and explicit split arithmetic, without flattening
// physical pages into a fictitious contiguous KV allocation.
#include "mla_split_paged_bsnd_plan.h"
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
MlaSplitKvPlanConfig Config(const MlaPagedShape& s, const MlaPlatformInfo& p, int64_t splits, uint32_t m, uint32_t n)
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
    status = BuildSplitKvPlanLayout(Config(s, p, splits, 256, CapacityBase(s)), bound);
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
        result.nBase = s.s2BaseSize ? s.s2BaseSize : CapacityBase(shape);
        std::vector<MlaCausalSequence> lengths;
        lengths.reserve(result.paged.lengths.sequences.size());
        for (const auto& v : result.paged.lengths.sequences)
            lengths.push_back({v.queryLength, v.kvLength});
        status = BuildSplitKvPlan(Config(shape, platform, splits, result.mBase, result.nBase), window, lengths,
                                  result.split);
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
