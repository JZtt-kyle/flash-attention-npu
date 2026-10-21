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

struct MlaSplitKvBlock {
    MlaWindowBlock whole;
    uint32_t s2Begin, s2End, kvBegin, kvEnd;
    int64_t cost;
};

struct MlaSplitKvWorkspaceInput {
    int64_t mBaseSize, s2BaseSize, maxQueryRows;
    int64_t aicNum, blockDim;
};
struct MlaSplitKvWorkspace {
    MlaWorkspace attention;
};

// Source host budget uses aicNum; kernel segments use blockDim. Preserves the
// extra host aMLA accounting. Does not initialize any buffer.
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

// Static TND allocation envelope, without reading prefixes/used or allocating
// tasks/tensors. Layout is shared with BSND; its accumulator is [S,H,T,512].
// At most A=min(B,T) samples have Q tokens. The persistent M-group bound is
// min(A*ceil(maxQ*H/M), floor((T*H+A*(M-1))/M)), not B*ceil(maxQ*H/M).
MlaScheduleStatus BuildSplitTndPlanLayout(const MlaSplitTndPlanConfig&, MlaSplitKvPlanLayout&);

struct MlaSplitMixedPlanConfig {
    MlaTndBsndShape shape;
    int64_t mBaseSize, s2BaseSize, numSplits;
    int64_t aicNum, blockDim, cvRatio;
    uint64_t libApiBytes;
};

// Static mixed allocation envelope (TND Q, BSND KV), without reading prefixes/used or allocating
// tasks/tensors. Layout is shared with BSND; its accumulator is [S,H,T,512].
// At most A=min(B,T) samples have Q tokens. The persistent M-group bound is
// min(A*ceil(maxQ*H/M), floor((T*H+A*(M-1))/M)), not B*ceil(maxQ*H/M).
MlaScheduleStatus BuildSplitMixedPlanLayout(const MlaSplitMixedPlanConfig&, MlaSplitKvPlanLayout&);

} // namespace flash_attn::mla

// Pinned source geometry: mla_tiling.h / mla_tiling.cpp.
// Explicit split and accumulator provenance: mla_split.h (root LICENSE).
namespace flash_attn::mla {
// Independent bounds: M can be any source divisor in [128,256], or 256
// with a tail. maxGroups*256 is NOT the persistent row bound.
struct MlaSplitPagedBudget {
    uint64_t maxGroups, maxTasks, partialRows, scratchBytes, userBytes, totalBytes;
    MlaAccumulatorLayout accumulator;
};
// Static allocation/P/C only; no lengths, page table or cache reads.
// C1V2, fixed launch size aicNum (one core for empty static capacity).
MlaScheduleStatus BuildSplitPagedBudget(const MlaPagedShape&, const MlaPlatformInfo&, int64_t splits,
                                        MlaSplitPagedBudget&);

struct MlaSplitPlan {
    MlaSplitKvPlanLayout layout;
    std::vector<uint64_t> prefix;
    const MlaSplitKvPlanLayout& Layout() const
    {
        return layout;
    }
    const std::vector<uint64_t>& GroupPrefix() const
    {
        return prefix;
    }
    uint64_t Groups() const
    {
        return prefix.empty() ? 0 : prefix.back();
    }
};
// Effective lengths have already passed allocation/used parsing. The plan
// retains only the checked static budget and M prefix needed by metadata.
MlaScheduleStatus BuildSplitPlan(const MlaUsedShape&, const MlaPlatformInfo&, uint32_t m, uint32_t n, uint32_t blockDim,
                                 const MlaWindowConfig&, int64_t splits, const std::vector<MlaCausalSequence>&,
                                 MlaSplitPlan&);

struct MlaSplitPagedPlan {
    MlaSplitPagedBudget budget{};
    MlaPagedSchedule paged;
    MlaSplitPlan split;
    uint32_t mBase = 0, nBase = 0;
};
// The snapshot retains logical lengths and checked workspace bounds. Paged Copy resolves every page
// at the eventual call. All-zero effective KV uses M=256 and the capacity's
// PA N base to retain initialization tasks; static empty capacity has none.
// Builders leave output unchanged on failure, including invalid empty inputs.
MlaScheduleStatus BuildSplitPagedPlan(const MlaPagedShape&, const MlaPlatformInfo&, const MlaPagedInputs&,
                                      const MlaWindowConfig&, int64_t splits, MlaSplitPagedPlan&);
} // namespace flash_attn::mla

// Pinned PA geometry: mla_tiling.h / mla_tiling.cpp.
// FA3 split/accumulator provenance: mla_split.h (root LICENSE).

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

// Owned CPU component only. Native endpoints are exclusive in (B,M,S2), FD
// Vector endpoints inclusive in (head,8-row part). Neither is cyclic ownership.
// Plans store O(B+AIC) metadata, not a descriptor per M block or physical tile.
class MlaAutoSplitPlan {
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

  private:
    friend MlaScheduleStatus BuildAutoSplitPlan(const MlaUsedShape&, const MlaPlatformInfo&, uint32_t, uint32_t,
                                                const MlaWindowConfig&, const std::vector<MlaCausalSequence>&,
                                                MlaAutoSplitPlan&);
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

MlaScheduleStatus BuildAutoSplitPlan(const MlaUsedShape&, const MlaPlatformInfo&, uint32_t m, uint32_t n,
                                     const MlaWindowConfig&, const std::vector<MlaCausalSequence>&, MlaAutoSplitPlan&);

struct MlaAutoSplitTndPlanConfig {
    MlaTndShape shape;
    int64_t mBaseSize, s2BaseSize, aicNum, cvRatio;
    uint64_t libApiBytes;
};
// Static compact TND envelope, independent of device prefix/used values.
// maxGroups uses totalQ, never B*maxQ; FD storage remains at most 2*AIC slots.
// Accumulator capacity is AIC with shape [AIC,H,totalQ,512] / [AIC,H,totalQ].
MlaScheduleStatus BuildAutoSplitTndPlanLayout(const MlaAutoSplitTndPlanConfig&, MlaAutoSplitKvLayout&);

// Shared compact-Q implementation used by the TND and mixed adapters. The
// KV layout selects allocation bounds; native partitions use effective lengths.
MlaScheduleStatus BuildAutoSplitCompactPlanLayout(const MlaAutoSplitTndPlanConfig&, MlaSequenceLayout,
                                                  MlaAutoSplitKvLayout&);

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
// Static budget uses compact totalQ, at most 2*AIC FD slots and
// [AIC,H,totalQ,512] / [AIC,H,totalQ] FP32 accumulator. Empty capacity validates
// all descriptors and has no tasks/workspace. All failures are atomic.
MlaScheduleStatus BuildAutoSplitMixedPlanLayout(const MlaAutoSplitMixedPlanConfig&, MlaAutoSplitKvLayout&);
} // namespace flash_attn::mla

// Pinned source geometry: mla_tiling.h / mla_tiling.cpp.
// Native split and accumulator provenance: mla_split.h (root LICENSE).
namespace flash_attn::mla {
// Static bounds retain native variable partitions: at most 2*AIC FD slots,
// each at most 256 rows. No uniform explicit split or per-M allocation.
struct MlaAutoSplitPagedBudget {
    uint64_t maxGroups, maxPartialSlots, partialRows, scratchBytes, userBytes, totalBytes;
    MlaAccumulatorLayout accumulator;
};
// Static BSND allocation/P/C only; no lengths, page table or cache reads.
// C1V2, fixed launch size aicNum (one core for empty static capacity).
MlaScheduleStatus BuildAutoSplitPagedBudget(const MlaPagedShape&, const MlaPlatformInfo&, MlaAutoSplitPagedBudget&);

// Production adds one Direct slot per AIC and a static 26-slice public accumulator.
MlaScheduleStatus BuildAutoSplitPagedProductionBudget(const MlaPagedShape&, const MlaPlatformInfo&,
                                                      MlaAutoSplitPagedBudget&);

struct MlaAutoSplitPagedPlan {
    MlaAutoSplitPagedBudget budget{};
    MlaPagedSchedule paged;
    MlaAutoSplitPlan split;
    uint32_t mBase = 0, nBase = 0;
};
// The snapshot retains logical lengths and native FD partitions. Paged Copy resolves every page
// at the eventual call. All-zero effective KV uses M=256 and the capacity's
// PA N base to retain initialization tasks; static empty capacity has none.
// Builders leave output unchanged on failure, including invalid empty inputs.
MlaScheduleStatus BuildAutoSplitPagedPlan(const MlaPagedShape&, const MlaPlatformInfo&, const MlaPagedInputs&,
                                          const MlaWindowConfig&, MlaAutoSplitPagedPlan&);
} // namespace flash_attn::mla

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

// Owned CPU scheduling/coordinate protocol, not serialized device metadata.
// S1G within a KV head (packed) or S1 within a query head (unpacked) is purely
// logical: tensors keep their original allocation and independent strides.
// Production v47 maps these coordinates to existing partial/merge numerics.
class MlaGqaSchedule {
  public:
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
    // Auto retains native endpoint intersections/owners. Positive splits use
    // the existing mask-dependent FA3 fragment rule and cyclic AIC ownership.
    // Empty KV/masked groups retain explicit initialization tasks; zero Q has none.

  private:
    friend MlaScheduleStatus BuildGqaSchedule(const MlaGqaConfig&, const MlaUsedInputs&, const MlaWindowConfig&,
                                              const MlaGqaOptions&, const MlaPlatformInfo&, MlaGqaSchedule&);
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
