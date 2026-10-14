// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
// Source arithmetic: CANN ca18aee610729ca5270a44e2d84c751bc687fbf0
// split_core.cpp / fia_tiling_nonquant_mla.cpp and FA3
// d15f1531a460ba456f41b01a774f33ab2db8febf hopper/block.h / flash_api.cpp.
// FA3 portions are BSD 3-Clause; see LICENSE.fa3.
#pragma once

#include "mla_split_kv.h"

namespace flash_attn::mla {

struct MlaSplitKvPlanConfig {
    int64_t batch, queryCapacity, kvCapacity, queryHeads;
    int64_t mBaseSize, s2BaseSize, numSplits;
    int64_t aicNum, blockDim, cvRatio;
    uint64_t libApiBytes;
};

struct MlaSplitKvPlanLayout {
    uint64_t maxGroups, maxTasks, partialRows;
    // Scratch retains native per-core double buffers and host reserve, without
    // native FD slots. Persistent partials start AFTER the entire user budget.
    MlaSplitKvWorkspace scratch;
    uint64_t outOffsetBytes, sumOffsetBytes, maxOffsetBytes, userBytes, totalBytes;
    MlaAccumulatorLayout accumulator;
    bool emptyCapacity;
};

// Static allocation envelope, independent of device-resident actual lengths.
// No large tensor/task array is allocated. Logical tiles are not a promise
// that the corresponding Kernel resource configuration is supported.
MlaScheduleStatus BuildSplitKvPlanLayout(const MlaSplitKvPlanConfig& config, MlaSplitKvPlanLayout& output);

struct MlaSplitKvPlannedMerge {
    uint64_t group, iteration;
    uint32_t batch, m, owner, rowBegin, rowEnd;
    uint64_t outOffsetBytes, sumOffsetBytes, maxOffsetBytes;
};

struct MlaSplitKvPlannedTask {
    uint64_t task, group, iteration;
    uint32_t batch, m, split, owner;
    MlaSplitKvBlock fragment;
    uint64_t outOffsetBytes, sumOffsetBytes, maxOffsetBytes;
};

struct MlaSplitKvPlannedRow {
    // Byte offsets into separate final O (input dtype), LSE (FP32), and FA3
    // FP32 accumulators. Physical Q capacity, never cumulative used Q.
    uint64_t out, lse, accumulatorOut, accumulatorLse;
    // Byte offsets from the user workspace for this partial row.
    uint64_t partialOut, partialSum, partialMax;
};

// Owned, immutable CPU plan, NOT a device metadata record or launcher. A later
// device consumer must validate its serialized protocol independently.
class MlaSplitKvPlan {
  public:
    const MlaSplitKvPlanLayout& Layout() const
    {
        return layout_;
    }
    const std::vector<uint64_t>& GroupPrefix() const
    {
        return prefix_;
    }
    uint64_t Groups() const
    {
        return prefix_.empty() ? 0 : prefix_.back();
    }
    uint64_t Tasks() const
    {
        return Groups() * config_.numSplits;
    }

    // Cyclic ownership: task = core + iteration*blockDim; merge group =
    // vector + iteration*(blockDim*cvRatio). Zero-cost/empty-KV fragments keep
    // their owners so their valid Q rows can be initialized. Zero Q has none.
    MlaScheduleStatus Task(uint64_t index, MlaSplitKvPlannedTask& output) const;
    MlaScheduleStatus Merge(uint64_t index, MlaSplitKvPlannedMerge& output) const;
    MlaScheduleStatus Row(uint64_t task, uint32_t row, MlaSplitKvPlannedRow& output) const;

  private:
    friend MlaScheduleStatus BuildSplitKvPlan(const MlaSplitKvPlanConfig&, const MlaWindowConfig&,
                                              const std::vector<MlaCausalSequence>&, MlaSplitKvPlan&);
    MlaSplitKvPlanConfig config_{};
    MlaSplitKvPlanLayout layout_{};
    MlaWindowConfig window_{};
    std::vector<MlaCausalSequence> lengths_;
    std::vector<uint64_t> prefix_;
};

// Lengths must already come from validated allocation/used parsing; recheck
// every sample against BSND capacities here. The compact M prefix describes
// effective Q rows, while buffers are bounded by Layout's static envelope.
// Raw partial layout is [compact M group, split, M row, 512 or 8]. Slots are
// never reused across waves; merge may start only AFTER ALL partial writes
// complete (e.g. a separate Kernel ordered on the same stream).
// Global empty capacity has no tasks/workspace, but retains accumulator shapes;
// its public initialization/validation is a future caller's responsibility.
// Explicit S=2..256 only. All builders/queries leave outputs unchanged on error.
MlaScheduleStatus BuildSplitKvPlan(const MlaSplitKvPlanConfig& config, const MlaWindowConfig& window,
                                   const std::vector<MlaCausalSequence>& lengths, MlaSplitKvPlan& output);

} // namespace flash_attn::mla
