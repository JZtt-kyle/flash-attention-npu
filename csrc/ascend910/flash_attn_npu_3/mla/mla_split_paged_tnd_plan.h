// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
// Pinned PA geometry: mla_paged_schedule.h / mla_tnd_schedule.cpp.
// FA3 split/accumulator provenance: mla_split_mixed_plan.h (LICENSE.fa3).
#pragma once
#include "mla_paged_schedule.h"
#include "mla_split_mixed_plan.h"

namespace flash_attn::mla {
struct MlaSplitPagedTndBudget {
    uint64_t maxGroups, maxTasks, partialRows, scratchBytes, userBytes, totalBytes;
    MlaAccumulatorLayout accumulator;
};
// Static compact Q envelope; no prefix, used-length or page reads. C1V2.
// Failure leaves output unchanged, including overflow at empty KV capacity.
MlaScheduleStatus BuildSplitPagedTndBudget(const MlaPagedShape&, const MlaPlatformInfo&, int64_t splits,
                                           MlaSplitPagedTndBudget&);

struct MlaSplitPagedTndPlannedTask {
    MlaSplitKvPlannedTask partial;
    uint64_t queryStart;
    uint32_t queryCapacity, queryLength, kvLength;
};

class MlaSplitPagedTndPlan {
  public:
    MlaSplitPagedTndBudget budget{};
    MlaPagedSchedule paged;
    uint32_t mBase = 0, nBase = 0;
    const MlaSplitKvPlanLayout& Layout() const
    {
        return logical_.Layout();
    }
    const std::vector<uint64_t>& GroupPrefix() const
    {
        return logical_.GroupPrefix();
    }
    uint64_t Groups() const
    {
        return logical_.Groups();
    }
    uint64_t Tasks() const
    {
        return logical_.Tasks();
    }
    MlaScheduleStatus Task(uint64_t, MlaSplitPagedTndPlannedTask&) const;
    MlaScheduleStatus Merge(uint64_t index, MlaSplitKvPlannedMerge& out) const
    {
        return logical_.Merge(index, out);
    }
    MlaScheduleStatus Row(uint64_t task, uint32_t row, MlaSplitKvPlannedRow& out) const
    {
        return logical_.Row(task, row, out);
    }

  private:
    friend MlaScheduleStatus BuildSplitPagedTndPlan(const MlaPagedShape&, const MlaPlatformInfo&, const MlaPagedInputs&,
                                                    const MlaWindowConfig&, int64_t, MlaSplitPagedTndPlan&);
    MlaSplitMixedPlan logical_;
};
// Owns all spans. Queries preserve allocation Q tokens and return only logical
// KV intervals; physical addresses require the invocation's checked page view.
// All-zero KV-used retains initialization tasks; globally empty capacity has
// none. This component does not launch kernels or expose public split support.
MlaScheduleStatus BuildSplitPagedTndPlan(const MlaPagedShape&, const MlaPlatformInfo&, const MlaPagedInputs&,
                                         const MlaWindowConfig&, int64_t splits, MlaSplitPagedTndPlan&);
} // namespace flash_attn::mla
