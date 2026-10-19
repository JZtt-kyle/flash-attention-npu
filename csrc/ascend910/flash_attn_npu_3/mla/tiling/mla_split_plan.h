// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
#pragma once
#include "mla_split_kv.h"
#include "mla_sequence.h"

// Source arithmetic: CANN ca18aee610729ca5270a44e2d84c751bc687fbf0
// split_core.cpp / fia_tiling_nonquant_mla.cpp and FA3
// d15f1531a460ba456f41b01a774f33ab2db8febf hopper/block.h / flash_api.cpp.
// FA3 portions are BSD 3-Clause; see LICENSE.fa3.
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
// public callers own its initialization and validation.
// Explicit S=2..256 only. All builders/queries leave outputs unchanged on error.
MlaScheduleStatus BuildSplitKvPlan(const MlaSplitKvPlanConfig& config, const MlaWindowConfig& window,
                                   const std::vector<MlaCausalSequence>& lengths, MlaSplitKvPlan& output);

} // namespace flash_attn::mla

// CANN ca18aee610729ca5270a44e2d84c751bc687fbf0 range/workspace arithmetic;
// FA3 d15f1531a460ba456f41b01a774f33ab2db8febf split/accumulator convention.
// FA3 portions are BSD 3-Clause; see LICENSE.fa3.
namespace flash_attn::mla {

struct MlaSplitTndPlanConfig {
    MlaTndShape shape;
    int64_t mBaseSize, s2BaseSize, numSplits;
    int64_t aicNum, blockDim, cvRatio;
    uint64_t libApiBytes;
};

struct MlaSplitTndPlannedTask {
    MlaSplitKvPlannedTask partial;
    // Original allocation-token origins, independent of Q/KV-used. Q, qv,
    // K and V each apply their own validated token/head strides at dispatch.
    MlaUsedSequence sequence;
};

// Static TND allocation envelope, without reading prefixes/used or allocating
// tasks/tensors. Layout is shared with BSND; its accumulator is [S,H,T,512].
// At most A=min(B,T) samples have Q tokens. The persistent M-group bound is
// min(A*ceil(maxQ*H/M), floor((T*H+A*(M-1))/M)), not B*ceil(maxQ*H/M).
MlaScheduleStatus BuildSplitTndPlanLayout(const MlaSplitTndPlanConfig&, MlaSplitKvPlanLayout&);

class MlaSplitTndPlan {
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
    MlaScheduleStatus Task(uint64_t, MlaSplitTndPlannedTask&) const;
    MlaScheduleStatus Merge(uint64_t, MlaSplitKvPlannedMerge&) const;
    MlaScheduleStatus Row(uint64_t task, uint32_t row, MlaSplitKvPlannedRow&) const;

  private:
    friend MlaScheduleStatus BuildSplitTndPlan(const MlaSplitTndPlanConfig&, const MlaWindowConfig&,
                                               const MlaUsedInputs&, MlaSplitTndPlan&);
    MlaSplitTndPlanConfig config_{};
    MlaSplitKvPlanLayout layout_{};
    MlaWindowConfig window_{};
    std::vector<MlaUsedSequence> sequences_;
    std::vector<uint64_t> prefix_;
};

// Parse both allocation prefixes and optional used arrays with ParseUsedLengths;
// retain owned spans only. Window normalization uses call-level max lengths,
// while mask alignment and M tasks use each sample's effective lengths.
// Ownership/persistent slot order matches the BSND explicit plan. Final O/LSE
// and accumulator addresses use allocation tokens, never packed used tokens.
// Global empty total Q/KV has no work/workspace; zero used KV in a nonempty
// allocation retains initialization tasks. All failures leave output unchanged.
// CPU component only: no metadata, device synchronization or public split path.
MlaScheduleStatus BuildSplitTndPlan(const MlaSplitTndPlanConfig&, const MlaWindowConfig&, const MlaUsedInputs&,
                                    MlaSplitTndPlan&);

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

// Pinned source geometry: tiling/mla_sequence.h / tiling/mla_sequence.cpp.
// Explicit split and accumulator provenance: tiling/mla_split_plan.h (LICENSE.fa3).
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

// Pinned PA geometry: tiling/mla_sequence.h / tiling/mla_sequence.cpp.
// FA3 split/accumulator provenance: tiling/mla_split_plan.h (LICENSE.fa3).
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
