// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
// Source partition/FD arithmetic: CANN ca18aee610729ca5270a44e2d84c751bc687fbf0
// attention/common/op_host/split_core.cpp and fia_tiling_nonquant_mla.cpp.
// FA3 accumulator layout: d15f1531a460ba456f41b01a774f33ab2db8febf
// hopper/flash_api.cpp, BSD 3-Clause; see LICENSE.fa3.
#pragma once

#include "mla_auto_split_kv_plan.h"
#include "mla_tnd_schedule.h"
#include "mla_used_schedule.h"

namespace flash_attn::mla {

struct MlaAutoSplitTndPlanConfig {
    MlaTndShape shape;
    int64_t mBaseSize, s2BaseSize, aicNum, cvRatio;
    uint64_t libApiBytes;
};
struct MlaAutoSplitTndTask {
    MlaAutoSplitKvTask native;
    // Original allocation spans. Each input applies its own validated stride.
    MlaUsedSequence sequence;
};
// Static compact TND envelope, independent of device prefix/used values.
// maxGroups uses totalQ, never B*maxQ; FD storage remains at most 2*AIC slots.
// Accumulator capacity is AIC with shape [AIC,H,totalQ,512] / [AIC,H,totalQ].
MlaScheduleStatus BuildAutoSplitTndPlanLayout(const MlaAutoSplitTndPlanConfig&, MlaAutoSplitKvLayout&);

// Shared compact-Q implementation used by the TND and mixed adapters. The
// KV layout selects validation/parsing only; native partitions use effective
// lengths and all task addresses retain the owned original allocation spans.
class MlaAutoSplitTndPlan;
MlaScheduleStatus BuildAutoSplitCompactPlanLayout(const MlaAutoSplitTndPlanConfig&, MlaSequenceLayout,
                                                  MlaAutoSplitKvLayout&);
MlaScheduleStatus BuildAutoSplitCompactPlan(const MlaAutoSplitTndPlanConfig&, MlaSequenceLayout, const MlaWindowConfig&,
                                            const MlaUsedInputs&, MlaAutoSplitTndPlan&);

// Owned CPU component only. Native endpoints are exclusive in (B,M,S2), FD
// Vector endpoints inclusive in (head,8-row part). Neither is cyclic ownership.
// Plans store O(B+AIC) metadata, not a descriptor per M block or physical tile.
class MlaAutoSplitTndPlan {
  public:
    const MlaAutoSplitKvLayout& Layout() const
    {
        return layout_;
    }
    const MlaNativeSplitKvSchedule& Native() const
    {
        return native_;
    }
    const std::vector<uint64_t>& GroupPrefix() const
    {
        return prefix_;
    }
    const std::array<uint64_t, kMlaMaxAic + 1>& PartialPrefix() const
    {
        return partialPrefix_;
    }
    uint64_t Groups() const
    {
        return prefix_.empty() ? 0 : prefix_.back();
    }
    uint64_t PartialSlots() const
    {
        return partialPrefix_[native_.fd.numOfFdHead];
    }
    MlaScheduleStatus Task(uint64_t group, uint32_t split, MlaAutoSplitTndTask& output) const;
    MlaScheduleStatus Merge(uint32_t head, uint32_t part, MlaAutoSplitKvMerge& output) const;
    MlaScheduleStatus Row(uint64_t group, uint32_t split, uint32_t row, MlaAutoSplitKvRow& output) const;

  private:
    friend MlaScheduleStatus BuildAutoSplitCompactPlan(const MlaAutoSplitTndPlanConfig&, MlaSequenceLayout,
                                                       const MlaWindowConfig&, const MlaUsedInputs&,
                                                       MlaAutoSplitTndPlan&);
    MlaAutoSplitTndPlanConfig config_{};
    MlaAutoSplitKvLayout layout_{};
    MlaNativeSplitKvSchedule native_{};
    MlaWindowConfig window_{};
    std::vector<MlaUsedSequence> sequences_;
    std::vector<uint64_t> prefix_;
    std::array<uint64_t, kMlaMaxAic + 1> partialPrefix_{};
};

// Effective lengths define the original mask geometry; the original allocation
// prefix defines final addresses, never cumulative used lengths or B*maxQ.
// Zero-cost groups have one explicit Empty task, including source-omitted
// suffixes, and no FD merge.
// Only native FD heads reserve persistent slots, at most 2*AIC, ordered by
// (head,split). Complete/direct tasks never overwrite these slots. All partial
// writes must finish before merge (e.g. a later kernel on the same stream).
// Global empty total Q/KV allocation has no tasks/workspace but validates every input.
// Owns parsed spans; input buffers may be destroyed after build.
// CPU component only; no device metadata or public auto TND path.
// Logical M/N bounds do not validate device resources. Failures are atomic.
MlaScheduleStatus BuildAutoSplitTndPlan(const MlaAutoSplitTndPlanConfig& config, const MlaWindowConfig& window,
                                        const MlaUsedInputs& inputs, MlaAutoSplitTndPlan& output);

} // namespace flash_attn::mla
