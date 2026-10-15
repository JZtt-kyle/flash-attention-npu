// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
// Pinned source geometry: mla_paged_schedule.h / mla_tnd_schedule.cpp.
// Native split and accumulator provenance: mla_auto_split_kv_plan.h (LICENSE.fa3).
#pragma once
#include "mla_paged_schedule.h"
#include "mla_auto_split_mixed_plan.h"

namespace flash_attn::mla {
// Compact Q allocation bounds retain native variable partitions: at most 2*AIC FD slots,
// each at most 256 rows. No uniform explicit split or per-M allocation.
struct MlaAutoSplitPagedTndBudget {
    uint64_t maxGroups, maxPartialSlots, partialRows, scratchBytes, userBytes, totalBytes;
    MlaAccumulatorLayout accumulator;
};
// Static compact TND allocation/P/C only; no lengths, page table or cache reads.
// C1V2, fixed launch size aicNum (one core for empty static capacity).
MlaScheduleStatus BuildAutoSplitPagedTndBudget(const MlaPagedShape&, const MlaPlatformInfo&,
                                               MlaAutoSplitPagedTndBudget&);

// Production adds one Direct slot per AIC and a static 26-slice public accumulator.
MlaScheduleStatus BuildAutoSplitPagedTndProductionBudget(const MlaPagedShape&, const MlaPlatformInfo&,
                                                         MlaAutoSplitPagedTndBudget&);

struct MlaAutoSplitPagedTndPlan {
    MlaAutoSplitPagedTndBudget budget{};
    MlaPagedSchedule paged;
    MlaAutoSplitMixedPlan split;
    uint32_t mBase = 0, nBase = 0;
};
// Owned snapshot. Task(group,split)/Merge(head,part)/Row are checked queries; KV ranges are
// logical tokens, not physical pool addresses. Paged Copy resolves every page
// at the eventual call. All-zero effective KV uses M=256 and the capacity's
// PA N base to retain initialization tasks; static empty capacity has none.
// Builders leave output unchanged on failure, including invalid empty inputs.
MlaScheduleStatus BuildAutoSplitPagedTndPlan(const MlaPagedShape&, const MlaPlatformInfo&, const MlaPagedInputs&,
                                             const MlaWindowConfig&, MlaAutoSplitPagedTndPlan&);
} // namespace flash_attn::mla
