/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "mla_tiling.h"

// Sources: CANN ca18aee610729ca5270a44e2d84c751bc687fbf0 SplitCore and
// fia_tiling_nonquant_mla.cpp; FA3 d15f1531a460ba456f41b01a774f33ab2db8febf
// hopper/flash_api.cpp and block.h. See LICENSE.cann and root LICENSE.

namespace flash_attn::mla {

// Independent CPU protocol, not a production metadata record or a device launch.
// Lengths are already parsed allocation/used lengths; no pointers are retained.
struct MlaNativeSplitKvSchedule {
    uint32_t usedCoreNum;
    std::array<uint32_t, kMlaMaxAic> bN2End, gS1End, s2End;
    int64_t maxCost, totalCost;
    uint64_t totalMBlocks, totalS2Blocks;
    MlaFlashDecodeSchedule fd;
};

// Preserve native variable-count FD partitions and ALL reduction bookkeeping.
// Zero-cost rows can be outside native endpoints and require output initialization.
// Logical tiles do not imply validated device resources. FD reduction M tile = 8.
MlaCausalStatus BuildNativeSplitKvSchedule(const MlaCausalScheduleInput& input, const MlaWindowConfig& window,
                                           MlaNativeSplitKvSchedule& output);

// FA3 <=0 requests backend auto, 1 disables splitting, 2..256 is explicit.
// autoSplits is a backend-selected upper bound (1 if native fd.maxS2SplitNum=0),
// not FA3's CUDA occupancy heuristic. Ignored for positive requests.
MlaCausalStatus ResolveSplitKvRequest(int64_t requested, int64_t autoSplits, uint32_t& output);

struct MlaSplitKvBlock {
    MlaWindowBlock whole;
    uint32_t s2Begin, s2End, kvBegin, kvEnd;
    int64_t cost;
};

// Uniform explicit splits of the validated mask-dependent S2 interval, using
// FA3's ceil-div rule. numSplits is resolved, 1..256. Empty splits are [end,end),
// with zero cost; boundary tiles still need the original row mask. This is
// separate from native auto's cost-balanced endpoints, not a reinterpretation.
MlaCausalStatus BuildSplitKvBlock(const MlaCausalBlockInput& input, const MlaWindowConfig& window, int64_t numSplits,
                                  int64_t splitIndex, MlaSplitKvBlock& output);

struct MlaSplitKvWorkspaceInput {
    int64_t mBaseSize, s2BaseSize, maxQueryRows;
    int64_t aicNum, blockDim;
    bool flashDecode;
    uint64_t libApiBytes;
};
struct MlaSplitKvWorkspace {
    MlaWorkspace attention;
    // Offsets from the user workspace, before libApiBytes is added to the budget.
    // Source FD stores aMLA-normalized FP32 O and separate sum/max (8 floats/row).
    uint64_t fdAccumOffsetBytes, fdSumOffsetBytes, fdMaxOffsetBytes;
};

// Source host budget uses aicNum; kernel segments use blockDim. FD capacity is
// two tasks per physical AIC. Preserves the extra host aMLA accounting. This is
// not the public FP32 O/LSE accumulator, and does not initialize any buffer.
MlaScheduleStatus BuildSplitKvWorkspace(const MlaSplitKvWorkspaceInput& input, MlaSplitKvWorkspace& output);

enum class MlaAccumulatorQueryLayout {
    Bsnd,
    Tnd
};
struct MlaAccumulatorInput {
    MlaAccumulatorQueryLayout layout;
    int64_t batch, queryCapacity, totalQuery, queryHeads, numSplits;
};
struct MlaAccumulatorTensor {
    uint32_t rank;
    std::array<uint64_t, 5> shape, strides; // contiguous ELEMENT strides, unused entries zero
    uint64_t elements, bytes;
};
struct MlaAccumulatorLayout {
    MlaAccumulatorTensor out, lse;
};

// FA3 allocation capacity, not used rows: BSND [S,B,H,Q,512]/[S,B,H,Q],
// TND [S,H,T,512]/[S,H,T]. FP32 for S>1; S=1 has no accumulator storage.
// BSND requires totalQuery=B*queryCapacity; TND totalQuery<=INT32_MAX, with
// prefix/max/used relationships validated by the existing length parser.
MlaScheduleStatus BuildSplitAccumulatorLayout(const MlaAccumulatorInput& input, MlaAccumulatorLayout& output);

// Checked FP32 row offsets in bytes; TND uses absolute allocation-token query
// and batch=0. S=1 or out-of-range coordinates reject. Failure is atomic for
// this and every builder in this header; callers never receive partial output.
MlaScheduleStatus GetSplitAccumulatorOffsets(const MlaAccumulatorInput& input, int64_t split, int64_t batch,
                                             int64_t head, int64_t query, uint64_t& outBytes, uint64_t& lseBytes);

} // namespace flash_attn::mla

// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.

// Source arithmetic: CANN ca18aee610729ca5270a44e2d84c751bc687fbf0
// split_core.cpp / fia_tiling_nonquant_mla.cpp and FA3
// d15f1531a460ba456f41b01a774f33ab2db8febf hopper/block.h / flash_api.cpp.
// FA3 portions are BSD 3-Clause; see root LICENSE.
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

// Owned, immutable CPU plan. Device consumers validate the serialized record independently.
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
// FA3 portions are BSD 3-Clause; see root LICENSE.
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
MlaScheduleStatus BuildSplitMixedPlan(const MlaSplitMixedPlanConfig&, const MlaWindowConfig&, const MlaUsedInputs&,
                                      MlaSplitMixedPlan&);

} // namespace flash_attn::mla

// Pinned source geometry: mla_tiling.h / mla_tiling.cpp.
// Explicit split and accumulator provenance: mla_split.h (root LICENSE).
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

// Pinned PA geometry: mla_tiling.h / mla_tiling.cpp.
// FA3 split/accumulator provenance: mla_split.h (root LICENSE).
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

// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.

// Source partition/FD arithmetic: CANN ca18aee610729ca5270a44e2d84c751bc687fbf0
// attention/common/op_host/split_core.cpp and fia_tiling_nonquant_mla.cpp.
// FA3 accumulator layout: d15f1531a460ba456f41b01a774f33ab2db8febf
// hopper/flash_api.cpp, BSD 3-Clause; see root LICENSE.
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
// Logical M/N bounds do not validate device resources. Failures are atomic.
MlaScheduleStatus BuildAutoSplitTndPlan(const MlaAutoSplitTndPlanConfig& config, const MlaWindowConfig& window,
                                        const MlaUsedInputs& inputs, MlaAutoSplitTndPlan& output);

} // namespace flash_attn::mla

// CANN ca18aee610729ca5270a44e2d84c751bc687fbf0 native AIC/FD partition;
// FA3 d15f1531a460ba456f41b01a774f33ab2db8febf accumulator convention.
// FA3 portions are BSD 3-Clause; see root LICENSE.
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

// Pinned source geometry: mla_tiling.h / mla_tiling.cpp.
// Native split and accumulator provenance: mla_split.h (root LICENSE).
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

// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
// Source scheduling: CANN ca18aee610729ca5270a44e2d84c751bc687fbf0 SplitCore.
// FA3 pack/resource semantics: d15f1531a460ba456f41b01a774f33ab2db8febf
// hopper/flash_api.cpp, BSD 3-Clause; see root LICENSE.

namespace flash_attn::mla {
struct MlaGqaOptions {
    int64_t packGqa = -1; // -1 = backend default, 0 = separate query heads, 1 = packed GQA.
    int64_t smMargin = 0; // Reserve this many AICs and twice as many AIVs.
};
struct MlaGqaResources {
    MlaPlatformInfo platform;
    uint32_t headGroups, headsPerGroup;
    bool packed;
};
// The NPU default preserves the existing CANN S1G path. G=1 normalizes to
// unpacked. Validate signed options before narrowing/subtraction. Keep the
// physical platform's blockDim mapping for the remaining C1V2 resources.
MlaScheduleStatus ResolveGqaResources(int64_t heads, int64_t kvHeads, const MlaGqaOptions& options,
                                      const MlaPlatformInfo& physical, MlaGqaResources& output);

struct MlaGqaConfig {
    MlaUsedShape shape;
    int64_t kvHeads, valueDim, mBase, nBase, splits;
    int64_t pageSize = 0; // PA uses logical rectangular KV spans; page lookup remains separate.
};
struct MlaGqaTask {
    uint32_t batch, headGroup, kvHead, m, split, splitCount, owner;
    uint32_t rowBegin, rowEnd, s2Begin, s2End;
};
struct MlaGqaRow {
    uint32_t batch, head, kvHead, query;
    uint64_t queryToken, kvStart;
    // Byte offsets into separate FP16/BF16 O, FP32 LSE and public accumulators.
    // UINT64_MAX denotes absent accumulators (splits=1). No workspace address
    // is inferred from these public offsets, or from a virtual batch index.
    uint64_t out, lse, accumulatorOut, accumulatorLse;
};

// Owned CPU scheduling/coordinate protocol, not serialized device metadata.
// S1G within a KV head (packed) or S1 within a query head (unpacked) is purely
// logical: tensors keep their original allocation and independent strides.
// Production v47 maps these coordinates to existing partial/merge numerics.
class MlaGqaSchedule {
  public:
    const MlaGqaResources& Resources() const
    {
        return resources_;
    }
    const MlaUsedLengths& Lengths() const
    {
        return lengths_;
    }
    const std::vector<uint64_t>& GroupPrefix() const
    {
        return prefix_;
    }
    const MlaNativeSplitKvSchedule& Native() const
    {
        return native_;
    }
    uint64_t Groups() const
    {
        return prefix_.empty() ? 0 : prefix_.back();
    }
    uint32_t AccumulatorCapacity() const
    {
        return capacity_;
    }
    // Auto retains native endpoint intersections/owners. Positive splits use
    // the existing mask-dependent FA3 fragment rule and cyclic AIC ownership.
    // Empty KV/masked groups retain explicit initialization tasks; zero Q has none.
    MlaScheduleStatus Task(uint64_t group, uint32_t split, MlaGqaTask& output) const;
    MlaScheduleStatus Row(uint64_t group, uint32_t split, uint32_t localRow, MlaGqaRow& output) const;

  private:
    friend MlaScheduleStatus BuildGqaSchedule(const MlaGqaConfig&, const MlaUsedInputs&, const MlaWindowConfig&,
                                              const MlaGqaOptions&, const MlaPlatformInfo&, MlaGqaSchedule&);
    MlaGqaConfig config_{};
    MlaGqaResources resources_{};
    MlaUsedLengths lengths_;
    MlaWindow window_{};
    MlaNativeSplitKvSchedule native_{}; // Populated only for splits=0.
    std::vector<uint64_t> prefix_;      // One prefix per (batch, headGroup), not per task.
    uint32_t capacity_ = 0;
};
// Reuses allocation/used parsing and source SplitCore; all failures are atomic.
// M/N are caller-selected logical tiles (M=32/64/128/256/512, N=1..8192).
// This does not certify local memory, physical pages or a numerical kernel.
// Two representations: valueDim=512 requires Hkv=1; valueDim=128 permits GQA.
MlaScheduleStatus BuildGqaSchedule(const MlaGqaConfig& config, const MlaUsedInputs& inputs,
                                   const MlaWindowConfig& window, const MlaGqaOptions& options,
                                   const MlaPlatformInfo& physical, MlaGqaSchedule& output);
} // namespace flash_attn::mla
