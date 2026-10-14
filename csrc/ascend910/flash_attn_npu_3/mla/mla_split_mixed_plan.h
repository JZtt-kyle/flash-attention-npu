// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
// CANN ca18aee610729ca5270a44e2d84c751bc687fbf0 range/workspace arithmetic;
// FA3 d15f1531a460ba456f41b01a774f33ab2db8febf split/accumulator convention.
// FA3 portions are BSD 3-Clause; see LICENSE.fa3.
#pragma once

#include "mla_split_kv_plan.h"
#include "mla_tnd_bsnd_schedule.h"
#include "mla_used_schedule.h"

namespace flash_attn::mla {

struct MlaSplitMixedPlanConfig {
    MlaTndBsndShape shape;
    int64_t mBaseSize, s2BaseSize, numSplits;
    int64_t aicNum, blockDim, cvRatio;
    uint64_t libApiBytes;
};

struct MlaSplitMixedPlannedTask {
    MlaSplitKvPlannedTask partial;
    // Original allocation-token origins, independent of Q/KV-used. Q, qv,
    // K and V each apply their own validated token/head strides at dispatch.
    MlaUsedSequence sequence;
};

// Static mixed allocation envelope (TND Q, BSND KV), without reading prefixes/used or allocating
// tasks/tensors. Layout is shared with BSND; its accumulator is [S,H,T,512].
// At most A=min(B,T) samples have Q tokens. The persistent M-group bound is
// min(A*ceil(maxQ*H/M), floor((T*H+A*(M-1))/M)), not B*ceil(maxQ*H/M).
MlaScheduleStatus BuildSplitMixedPlanLayout(const MlaSplitMixedPlanConfig&, MlaSplitKvPlanLayout&);

class MlaSplitMixedPlan {
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
    MlaScheduleStatus Task(uint64_t, MlaSplitMixedPlannedTask&) const;
    MlaScheduleStatus Merge(uint64_t, MlaSplitKvPlannedMerge&) const;
    MlaScheduleStatus Row(uint64_t task, uint32_t row, MlaSplitKvPlannedRow&) const;

  private:
    friend MlaScheduleStatus BuildSplitMixedPlan(const MlaSplitMixedPlanConfig&, const MlaWindowConfig&,
                                                 const MlaUsedInputs&, MlaSplitMixedPlan&);
    MlaSplitMixedPlanConfig config_{};
    MlaSplitKvPlanLayout layout_{};
    MlaWindowConfig window_{};
    std::vector<MlaUsedSequence> sequences_;
    std::vector<uint64_t> prefix_;
};

// Parse the Q allocation prefix and optional Q/KV used with ParseUsedLengths;
// KV has no prefix; its original physical origin is uint64(batch)*kvCapacity.
// Retain owned spans only. Window normalization uses call-level max lengths,
// while mask alignment and M tasks use each sample's effective lengths.
// Ownership/persistent slot order matches the BSND explicit plan. Final O/LSE
// and accumulator addresses use allocation tokens, never packed used tokens.
// Global empty total Q or BSND KV capacity has no work/workspace; zero used KV in a nonempty
// allocation retains initialization tasks. All failures leave output unchanged.
// CPU component only: no metadata, device synchronization or public split path.
MlaScheduleStatus BuildSplitMixedPlan(const MlaSplitMixedPlanConfig&, const MlaWindowConfig&, const MlaUsedInputs&,
                                      MlaSplitMixedPlan&);

} // namespace flash_attn::mla
