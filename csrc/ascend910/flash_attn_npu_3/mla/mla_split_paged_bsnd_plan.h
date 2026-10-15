// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
// Pinned source geometry: mla_paged_schedule.h / mla_tnd_schedule.cpp.
// Explicit split and accumulator provenance: mla_split_kv_plan.h (LICENSE.fa3).
#pragma once
#include "mla_paged_schedule.h"
#include "mla_split_kv_plan.h"

namespace flash_attn::mla {
// Independent bounds: M can be any source divisor in [128,256], or 256
// with a tail. maxGroups*256 is NOT the persistent row bound.
struct MlaSplitPagedBsndBudget {
    uint64_t maxGroups, maxTasks, partialRows, scratchBytes, userBytes, totalBytes;
    MlaAccumulatorLayout accumulator;
};
// Static BSND allocation/P/C only; no lengths, page table or cache reads.
// C1V2, fixed launch size aicNum (one core for empty static capacity).
MlaScheduleStatus BuildSplitPagedBsndBudget(const MlaPagedShape&, const MlaPlatformInfo&, int64_t splits,
                                            MlaSplitPagedBsndBudget&);

struct MlaSplitPagedBsndPlan {
    MlaSplitPagedBsndBudget budget{};
    MlaPagedSchedule paged;
    MlaSplitKvPlan split;
    uint32_t mBase = 0, nBase = 0;
};
// Owned snapshot. Task/Merge/Row are split's checked queries; KV ranges are
// logical tokens, not physical pool addresses. Paged Copy resolves every page
// at the eventual call. All-zero effective KV uses M=256 and the capacity's
// PA N base to retain initialization tasks; static empty capacity has none.
// Builders leave output unchanged on failure, including invalid empty inputs.
MlaScheduleStatus BuildSplitPagedBsndPlan(const MlaPagedShape&, const MlaPlatformInfo&, const MlaPagedInputs&,
                                          const MlaWindowConfig&, int64_t splits, MlaSplitPagedBsndPlan&);
} // namespace flash_attn::mla
