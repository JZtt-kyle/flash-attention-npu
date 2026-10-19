// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
#pragma once
#include "mla_split_kv.h"
#include "mla_sequence.h"

// Source partition/FD arithmetic: CANN ca18aee610729ca5270a44e2d84c751bc687fbf0
// attention/common/op_host/split_core.cpp and fia_tiling_nonquant_mla.cpp.
// FA3 accumulator layout: d15f1531a460ba456f41b01a774f33ab2db8febf
// hopper/flash_api.cpp, BSD 3-Clause; see LICENSE.fa3.
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

// CANN ca18aee610729ca5270a44e2d84c751bc687fbf0 native AIC/FD partition;
// FA3 d15f1531a460ba456f41b01a774f33ab2db8febf accumulator convention.
// FA3 portions are BSD 3-Clause; see LICENSE.fa3.
namespace flash_attn::mla {
struct MlaAutoSplitMixedPlanConfig {
    MlaTndBsndShape shape;
    int64_t mBaseSize, s2BaseSize, aicNum, cvRatio;
    uint64_t libApiBytes;
};
// The owned compact-Q plan and queries are layout independent after parsing.
// Mixed builders parse BSND KV origins as uint64(batch)*capacity, never as a
// TND int32 prefix. Only Q has an allocation prefix; optional used values do
// not move either input's physical origin. Each input applies its own stride.
using MlaAutoSplitMixedPlan = MlaAutoSplitTndPlan;
using MlaAutoSplitMixedTask = MlaAutoSplitTndTask;
// Static budget uses compact totalQ, at most 2*AIC FD slots and
// [AIC,H,totalQ,512] / [AIC,H,totalQ] FP32 accumulator. Empty capacity validates
// all descriptors and has no tasks/workspace. All failures are atomic.
MlaScheduleStatus BuildAutoSplitMixedPlanLayout(const MlaAutoSplitMixedPlanConfig&, MlaAutoSplitKvLayout&);
MlaScheduleStatus BuildAutoSplitMixedPlan(const MlaAutoSplitMixedPlanConfig&, const MlaWindowConfig&,
                                          const MlaUsedInputs&, MlaAutoSplitMixedPlan&);
} // namespace flash_attn::mla

// Pinned source geometry: tiling/mla_sequence.h / tiling/mla_sequence.cpp.
// Native split and accumulator provenance: tiling/mla_auto_split_plan.h (LICENSE.fa3).
namespace flash_attn::mla {
// Static bounds retain native variable partitions: at most 2*AIC FD slots,
// each at most 256 rows. No uniform explicit split or per-M allocation.
struct MlaAutoSplitPagedBsndBudget {
    uint64_t maxGroups, maxPartialSlots, partialRows, scratchBytes, userBytes, totalBytes;
    MlaAccumulatorLayout accumulator;
};
// Static BSND allocation/P/C only; no lengths, page table or cache reads.
// C1V2, fixed launch size aicNum (one core for empty static capacity).
MlaScheduleStatus BuildAutoSplitPagedBsndBudget(const MlaPagedShape&, const MlaPlatformInfo&,
                                                MlaAutoSplitPagedBsndBudget&);

// Production adds one Direct slot per AIC and a static 26-slice public accumulator.
MlaScheduleStatus BuildAutoSplitPagedBsndProductionBudget(const MlaPagedShape&, const MlaPlatformInfo&,
                                                          MlaAutoSplitPagedBsndBudget&);

struct MlaAutoSplitPagedBsndPlan {
    MlaAutoSplitPagedBsndBudget budget{};
    MlaPagedSchedule paged;
    MlaAutoSplitKvPlan split;
    uint32_t mBase = 0, nBase = 0;
};
// Owned snapshot. Task(group,split)/Merge(head,part)/Row are checked queries; KV ranges are
// logical tokens, not physical pool addresses. Paged Copy resolves every page
// at the eventual call. All-zero effective KV uses M=256 and the capacity's
// PA N base to retain initialization tasks; static empty capacity has none.
// Builders leave output unchanged on failure, including invalid empty inputs.
MlaScheduleStatus BuildAutoSplitPagedBsndPlan(const MlaPagedShape&, const MlaPlatformInfo&, const MlaPagedInputs&,
                                              const MlaWindowConfig&, MlaAutoSplitPagedBsndPlan&);
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
