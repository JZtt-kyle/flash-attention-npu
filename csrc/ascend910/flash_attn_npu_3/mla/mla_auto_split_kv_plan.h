// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
// Source partition/FD arithmetic: CANN ca18aee610729ca5270a44e2d84c751bc687fbf0
// attention/common/op_host/split_core.cpp and fia_tiling_nonquant_mla.cpp.
// FA3 accumulator layout: d15f1531a460ba456f41b01a774f33ab2db8febf
// hopper/flash_api.cpp, BSD 3-Clause; see LICENSE.fa3.
#pragma once

#include "mla_split_kv.h"

namespace flash_attn::mla {

struct MlaAutoSplitKvConfig {
    int64_t batch, queryCapacity, kvCapacity, queryHeads;
    int64_t mBaseSize, s2BaseSize, aicNum, cvRatio;
    uint64_t libApiBytes;
};
struct MlaAutoSplitKvLayout {
    uint64_t maxGroups, maxPartialSlots, partialRows;
    MlaSplitKvWorkspace scratch;
    uint64_t outOffsetBytes, sumOffsetBytes, maxOffsetBytes, userBytes, totalBytes;
    // Static backend capacity AIC, not a reinterpretation of native partitions.
    // With capacity>1, every used row's unused slots must be initialized 0/-inf;
    // a direct (non-FD) result belongs to slot 0. No public ABI is opened here.
    MlaAccumulatorLayout accumulator;
    bool emptyCapacity;
};
MlaScheduleStatus BuildAutoSplitKvLayout(const MlaAutoSplitKvConfig& config, MlaAutoSplitKvLayout& output);

enum class MlaAutoSplitKvTaskKind {
    Direct,
    Partial,
    Empty
};
struct MlaAutoSplitKvTask {
    uint64_t group, slot; // slot=UINT64_MAX for direct/empty; persistent for FD
    uint32_t batch, m, split, splitCount, owner, fdHead;
    MlaAutoSplitKvTaskKind kind;
    MlaSplitKvBlock fragment;
    uint64_t outOffsetBytes, sumOffsetBytes, maxOffsetBytes; // only for Partial
};
struct MlaAutoSplitKvMerge {
    uint64_t group, firstSlot;
    uint32_t batch, m, head, part, owner, splitCount, rowBegin, rowEnd;
    uint64_t outOffsetBytes, sumOffsetBytes, maxOffsetBytes;
};
struct MlaAutoSplitKvRow {
    uint64_t out, lse, accumulatorOut, accumulatorLse;
    uint64_t partialOut, partialSum, partialMax; // UINT64_MAX unless Partial
};

// Owned CPU component only. Native endpoints are exclusive in (B,M,S2), FD
// Vector endpoints inclusive in (head,8-row part). Neither is cyclic ownership.
// Plans store O(B+AIC) metadata, not a descriptor per M block or physical tile.
class MlaAutoSplitKvPlan {
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
    MlaScheduleStatus Task(uint64_t group, uint32_t split, MlaAutoSplitKvTask& output) const;
    MlaScheduleStatus Merge(uint32_t head, uint32_t part, MlaAutoSplitKvMerge& output) const;
    MlaScheduleStatus Row(uint64_t group, uint32_t split, uint32_t row, MlaAutoSplitKvRow& output) const;

  private:
    friend MlaScheduleStatus BuildAutoSplitKvPlan(const MlaAutoSplitKvConfig&, const MlaWindowConfig&,
                                                  const std::vector<MlaCausalSequence>&, MlaAutoSplitKvPlan&);
    MlaAutoSplitKvConfig config_{};
    MlaAutoSplitKvLayout layout_{};
    MlaNativeSplitKvSchedule native_{};
    MlaWindowConfig window_{};
    std::vector<MlaCausalSequence> lengths_;
    std::vector<uint64_t> prefix_;
    std::array<uint64_t, kMlaMaxAic + 1> partialPrefix_{};
};

// Effective lengths define the original mask geometry; BSND capacity defines
// final addresses, never cumulative used lengths. Zero-cost groups have one
// explicit Empty task, including source-omitted suffixes, and no FD merge.
// Only native FD heads reserve persistent slots, at most 2*AIC, ordered by
// (head,split). Complete/direct tasks never overwrite these slots. All partial
// writes must finish before merge (e.g. a later kernel on the same stream).
// Global empty B/Q/KV capacity has no tasks/workspace but validates every input.
// Logical M/N bounds do not validate device resources. Failures are atomic.
MlaScheduleStatus BuildAutoSplitKvPlan(const MlaAutoSplitKvConfig& config, const MlaWindowConfig& window,
                                       const std::vector<MlaCausalSequence>& lengths, MlaAutoSplitKvPlan& output);

} // namespace flash_attn::mla
