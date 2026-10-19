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

#include <array>
#include <cstdint>
#include <algorithm>
#include <vector>

// The upstream license is reproduced in mla/LICENSE.cann.

namespace flash_attn::mla {

// Source: CANN ops-transformer ca18aee610729ca5270a44e2d84c751bc687fbf0,
// attention/fused_infer_attention_score/op_host/fused_infer_attention_score_tiling.h.
// This is an endpoint capacity, not a hard-coded device core count.
constexpr uint32_t kMlaMaxAic = 26;

// Preserve the source's uint32 S1G ceil-div, dense query stride (Hq*512)
// and LSE DMA stride ((Sq-1)*sizeof(float)) without wrapping any field.
constexpr uint32_t kMlaMaxDenseRows = UINT32_MAX - 255;
constexpr uint32_t kMlaMaxQueryHeads = UINT32_MAX / 512;
constexpr uint32_t kMlaMaxQueryLength = UINT32_MAX / sizeof(float) + 1;
constexpr uint64_t kMlaMaxBatchRows = INT64_MAX / (512 * 2);
// Validated dense KV capacity; larger sequences remain explicitly unsupported.
constexpr uint32_t kMlaMaxKvLength = 16384;

struct MlaDenseShape {
    int64_t batch;
    int64_t queryLength;
    int64_t queryHeads;
    int64_t kvLength;
};

struct MlaLocalMemory {
    uint64_t l1Bytes;
    uint64_t ubBytes;
    uint64_t l0aBytes;
    uint64_t l0bBytes;
    uint64_t l0cBytes;
};

struct MlaPlatformInfo {
    uint32_t aicNum;
    uint32_t aivNum;
    MlaLocalMemory localMemory;
    uint64_t libApiWorkspaceBytes;
    // Populate entry c-1 with PlatformAscendC::CalcTschBlockDim(c*ratio, c, c*ratio).
    // Passing platform results keeps the same computation usable by host/AICPU
    // without a gert dependency or an assumed relationship between cores and blockDim.
    std::array<uint32_t, kMlaMaxAic> blockDims;
};

struct MlaFlashDecodeSchedule {
    uint32_t numOfFdHead;
    uint32_t maxS2SplitNum;
    uint32_t usedVecNumOfFd;
    uint32_t gS1BaseSizeOfFd;
    std::array<uint32_t, kMlaMaxAic> bN2IdxOfFdHead;
    std::array<uint32_t, kMlaMaxAic> gS1IdxOfFdHead;
    std::array<uint32_t, kMlaMaxAic> s2SplitNumOfFdHead;
    std::array<uint32_t, kMlaMaxAic> s2SplitStartIdxOfCore;
    std::array<uint32_t, kMlaMaxAic> gS1SplitNumOfFdHead;
    std::array<uint32_t, kMlaMaxAic> gS1LastPartSizeOfFdHead;
    std::array<uint32_t, kMlaMaxAic * 2> gS1IdxEndOfFdHead;
    std::array<uint32_t, kMlaMaxAic * 2> gS1IdxEndOfFdHeadSplit;
};

struct MlaWorkspace {
    // Source tiling fields: elements per core per slot, NOT bytes.
    uint64_t mm1ResSize;
    uint64_t mm2ResSize;
    // Source fills these even when FD is disabled; no FD workspace is allocated.
    uint64_t fdAccumOutSize;
    uint64_t fdLogSumExpSize;
    // Actual kernel segments use launch blockDim, while the host budget uses aicNum.
    uint64_t qkOffsetBytes;
    uint64_t pOffsetBytes;
    uint64_t pvOffsetBytes;
    uint64_t vec2OffsetBytes;
    uint64_t kernelBytes;
    uint64_t userBytes;
    uint64_t libApiBytes;
    uint64_t totalBytes;
};

struct MlaSchedule {
    uint32_t bSize;
    uint32_t n2Size;
    uint32_t gSize;
    uint32_t s1Size;
    uint32_t s2Size;
    uint32_t headDim;
    uint32_t headDimRope;
    uint32_t mBaseSize;
    uint32_t s2BaseSize;
    uint32_t aicNum;
    uint32_t aivNum;
    uint32_t cvRatio;
    uint32_t usedCoreNum;
    uint32_t blockDim;
    uint32_t scheduleMode; // CANN ScheduleMode::BATCH_MODE = 1.
    // Exclusive ends in (batch*Hkv, S1G block, S2 block) coordinates.
    // Core 0 starts at (0,0,0); subsequent cores start at the previous endpoint.
    std::array<uint32_t, kMlaMaxAic> bN2End;
    std::array<uint32_t, kMlaMaxAic> gS1End;
    std::array<uint32_t, kMlaMaxAic> s2End;
    int64_t maxCost;
    MlaFlashDecodeSchedule fd;
    MlaWorkspace workspace;
    MlaLocalMemory requiredLocalMemory;
};

enum class MlaScheduleStatus {
    Success,
    UnsupportedShape,
    FieldOverflow,
    InvalidPlatform,
    InsufficientLocalMemory,
    WorkspaceOverflow,
    UnexpectedFlashDecode,
    InvalidSplit,
    AllocationFailure,
};

// Pure CPU scheduling for contiguous BSND, shared 512+64 KV, Hkv=1,
// Sq*Hq>=0 within the address/field bounds above, 0<=Sk<=kMlaMaxKvLength, B>=0,
// no mask/length arrays, num_splits=1.
// A zero batch/query/KV capacity produces a validation-only schedule: one
// launch core, zero tiles/endpoints/workspace. No attention Kernel consumes it.
// FP16 and BF16 have identical scheduling. This is not a public forward or
// metadata entry point; callers must separately validate tensor/layout/options.
// No tensor addresses or scale are stored. On failure, output is unchanged.
MlaScheduleStatus BuildDenseSchedule(const MlaDenseShape& shape, const MlaPlatformInfo& platform, MlaSchedule& output);

} // namespace flash_attn::mla

// The upstream license is reproduced in mla/LICENSE.cann.

namespace flash_attn::mla {

// Independent CPU entry for contiguous BSND with full query capacity and
// per-sample KV lengths. Same shape/platform bounds and num_splits=1 as dense.
// kvLengths must point to lengthCount readable int32 values on the executing
// processor and remain stable during this call. nullptr/0 means full capacity;
// otherwise lengthCount must equal B, and every value must be in [0, Sk].
// Invalid counts/values return UnsupportedShape before conversion to source
// int64 lengths. Values are copied numerically, never pointer-reinterpreted.
// No addresses/length pointers survive the call. On failure output is unchanged.
// Nonzero KV capacity with all lengths zero preserves the source batch endpoint,
// base sizes and workspace; global empty capacity follows BuildDenseSchedule.
// Compute endpoints can omit a zero-KV suffix, as in the pinned source. The
// device consumer initializes zero-KV outputs independently of these ranges.
// Variable lengths use the v2 record rather than the fixed-length v1 protocol.
MlaScheduleStatus BuildBsndKvLengthSchedule(const MlaDenseShape& shape, const MlaPlatformInfo& platform,
                                            const int32_t* kvLengths, uint64_t lengthCount, MlaSchedule& output);

struct MlaTndShape {
    int64_t batch;
    int64_t totalQuery;
    int64_t totalKv;
    int64_t queryHeads;
    int64_t maxQueryLength;
    int64_t maxKvLength;
};

struct MlaTndSequence {
    // Token offsets, not element/byte offsets. No used-length compaction.
    uint32_t queryStart, kvStart;
    uint32_t queryLength, kvLength;
};

struct MlaTndLengths {
    uint32_t maxQueryLength = 0, maxKvLength = 0; // Observed maxima, not the caller's upper bounds.
    std::vector<MlaTndSequence> sequences;
};

struct MlaTndSchedule {
    MlaTndLengths lengths;
    MlaSchedule schedule{};
};

// CPU/AICPU component for compact TND Q/KV, shared 512+64 KV, Hkv=1.
// Both arrays are required, including B=0: int32[B+1], starting at zero,
// nondecreasing, ending at the corresponding total capacity (<=INT32_MAX).
// Per-sample differences must fit maxQueryLength/maxKvLength. These are upper
// bounds and need not equal the observed maxima. Dense field/stride bounds
// apply to those maxima bounds; the address bound uses totalQuery*Hq, not B*maxQ*Hq.
// Pointers refer to readable, stable CPU/AICPU memory for the stated counts.
// All values are validated before allocating/copying the normalized result.
// Both APIs leave output unchanged on failure and retain no input pointers.
MlaScheduleStatus ParseTndLengths(const MlaTndShape& shape, const int32_t* cuQuery, uint64_t queryCount,
                                  const int32_t* cuKv, uint64_t kvCount, MlaTndLengths& output);

// No mask, num_splits=1. Reuses the pinned SplitCore with numerical int64
// cumulative tails (without FA3's initial zero). Source isSameActualseq compares
// KV cumulative tails, so even uniform TND can choose a different M base from BSND.
// schedule.s1Size/s2Size are observed per-sample maxima. Endpoints use each
// sample's own M-block count; any omitted suffix must have zero Q or zero KV.
// Global empty total Q/KV produces a validation-only schedule, as in dense.
// Shared by the TND metadata producer and attention launch validation.
MlaScheduleStatus BuildTndSchedule(const MlaTndShape& shape, const MlaPlatformInfo& platform, const int32_t* cuQuery,
                                   uint64_t queryCount, const int32_t* cuKv, uint64_t kvCount, MlaTndSchedule& output);

} // namespace flash_attn::mla

// Source and upstream license: mla_tiling.cpp and LICENSE.cann.
namespace flash_attn::mla {
struct MlaTndBsndShape {
    int64_t batch, totalQuery, queryHeads, maxQueryLength, kvCapacity;
};

struct MlaTndBsndSequence {
    uint32_t queryStart;
    uint64_t kvStart; // Physical token offset b*kvCapacity, never a sum of used lengths.
    uint32_t queryLength, kvLength;
};

struct MlaTndBsndLengths {
    uint32_t maxQueryLength = 0, maxKvLength = 0; // Observed maxima, independent of capacity.
    std::vector<MlaTndBsndSequence> sequences;
};

struct MlaTndBsndSchedule {
    MlaTndBsndLengths lengths;
    MlaSchedule schedule{};
};

// Independent CPU component: compact TND Q, continuous BSND shared 512+64 KV,
// Hkv=1, no mask, num_splits=1. Shared with the v4 AICPU producer, not v3.
// Q requires int32[B+1], including [0] for B=0: initial zero, nondecreasing,
// tail==totalQuery<=INT32_MAX, differences<=maxQueryLength.
// KV requires exactly B int32 lengths in [0,kvCapacity], with 0<=kvCapacity<=16384.
// A null KV pointer is allowed only for B=0/count=0; it never means full capacity.
// B*kvCapacity can exceed INT32_MAX/UINT32_MAX: BSND offsets are 64-bit.
// Inputs are readable and stable for the stated counts. Output owns its spans;
// all values are validated before allocation and failure leaves output unchanged.
MlaScheduleStatus ParseTndBsndLengths(const MlaTndBsndShape& shape, const int32_t* cuQuery, uint64_t queryCount,
                                      const int32_t* kvLengths, uint64_t kvCount, MlaTndBsndLengths& output);

// s1Size is the observed Q maximum, s2Size is the physical KV capacity. Source
// isSameActualseq compares KV lengths (not cumulative tails); SplitCore consumes
// numerical int64 Q cumulative tails and noncumulative KV lengths. Variable-M
// whole-block mapping is shared with compact TND. Nonempty capacity with all KV
// lengths zero retains a zero-cost schedule/workspace, distinct from global empty.
MlaScheduleStatus BuildTndBsndSchedule(const MlaTndBsndShape& shape, const MlaPlatformInfo& platform,
                                       const int32_t* cuQuery, uint64_t queryCount, const int32_t* kvLengths,
                                       uint64_t kvCount, MlaTndBsndSchedule& output);

enum class MlaSequenceLayout {
    Bsnd,
    Tnd
};

struct MlaSequenceAllocation {
    MlaSequenceLayout layout;
    int64_t totalTokens; // Physical allocation, including unused tokens.
    int64_t maxLength;   // BSND capacity; TND upper bound on allocation prefix differences.
};

struct MlaUsedShape {
    int64_t batch, queryHeads;
    MlaSequenceAllocation query, kv;
};

struct MlaLengthArray {
    const int32_t* data = nullptr;
    uint64_t count = 0;
};

struct MlaUsedInputs {
    MlaLengthArray cuQuery, cuKv;
    MlaLengthArray usedQuery, usedKv;
};

struct MlaUsedSequence {
    uint64_t queryStart, kvStart; // Physical token offsets; never cumulative used lengths.
    uint32_t queryCapacity, kvCapacity;
    uint32_t queryLength, kvLength;
};

struct MlaUsedLengths {
    uint32_t maxQueryCapacity = 0, maxKvCapacity = 0;
    uint32_t maxQueryLength = 0, maxKvLength = 0;
    std::vector<MlaUsedSequence> sequences;
};

struct MlaUsedSchedule {
    MlaUsedLengths lengths;
    MlaSchedule schedule{};
};

// Independent CPU component: BSND Q/KV, TND Q/KV or TND Q + BSND KV only.
// TND requires int32[B+1] allocation prefixes, including [0] for B=0; BSND
// requires absent prefixes and totalTokens == B*capacity. Optional used arrays
// are absent iff {nullptr, 0}; otherwise exactly B readable int32 values, each
// in [0, its own allocation length]. Absence means the entire allocated segment.
// All static bounds/counts/values are checked before copying any spans. Inputs
// must remain readable/stable during the call. No pointers are retained, and
// failure leaves the output unchanged.
MlaScheduleStatus ParseUsedLengths(const MlaUsedShape& shape, const MlaUsedInputs& inputs, MlaUsedLengths& output);

// Preserve allocation geometry for base tiles/workspace (observed allocation
// maxima for TND). Explicit used arrays become noncumulative int64 source
// lengths; absent TND used retains the original cumulative allocation tails.
// The shared variable-M SplitCore adapter computes tasks/costs from used lengths.
// Empty allocation has the existing validation-only schedule; all-zero used in
// nonempty allocation retains zero-cost scheduling and workspace. This API is
// not a metadata protocol, AICPU launcher, device consumer or public entry.
MlaScheduleStatus BuildUsedSchedule(const MlaUsedShape& shape, const MlaPlatformInfo& platform,
                                    const MlaUsedInputs& inputs, MlaUsedSchedule& output);

// Validate/snapshot lengths and prepare the same tiles/workspace, without
// running the unmasked partitioner. The caller must supply its own partition
// and final workspace offsets before serializing or launching the schedule.
MlaScheduleStatus PrepareUsedSchedule(const MlaUsedShape& shape, const MlaPlatformInfo& platform,
                                      const MlaUsedInputs& inputs, MlaUsedSchedule& output);

struct MlaPagedShape {
    int64_t batch, queryHeads;
    MlaSequenceAllocation query;
    int64_t logicalKvCapacity; // Page-table width * pageSize, NOT physical pool size.
    int64_t pageSize;
};

struct MlaPagedInputs {
    MlaLengthArray cuQuery, usedQuery, kvLengths;
};

struct MlaPagedSequence {
    uint64_t queryStart; // Physical allocation start; unaffected by Q-used.
    uint32_t queryCapacity, queryLength, kvLength;
};

struct MlaPagedLengths {
    uint32_t maxQueryCapacity = 0, maxQueryLength = 0, maxKvLength = 0;
    std::vector<MlaPagedSequence> sequences;
};

struct MlaPagedSchedule {
    uint32_t pageSize = 0, logicalKvCapacity = 0;
    MlaPagedLengths lengths;
    MlaSchedule schedule{};
};

// Independent CPU/AICPU component: BSND/TND Q, optional Q-used, shared 512+64
// KV, no mask, num_splits=1. Shared by standalone and production v6 generation;
// forward kernels consume the serialized record through device validation.
// P is 16..1024 in multiples of 16; C is 0..16384 and divisible by P.
// Q allocation/prefix/used follow ParseUsedLengths. KV lengths are mandatory
// int32[B], always per sample, in [0,C]; nullptr/count=0 is allowed only at B=0.
// Validate all samples even when Q allocation/used or KV is zero. Inputs are
// readable, stable buffers on the executing CPU/AICPU; no device-to-host copy.
// Output owns its spans, retains no input pointers and is unchanged on failure.
MlaScheduleStatus ParsePagedLengths(const MlaPagedShape& shape, const MlaPagedInputs& inputs, MlaPagedLengths& output);

// Source normal PA tiling: s2Size=ceil(maxKvLength/P)*P, independent of C.
// Apply PA CalcInnerSize before truncating to s2Size, then SplitCore once and
// the existing whole-M mapping. Q geometry remains allocated, not used.
// B=0, zero Q allocation or all-zero KV lengths yields a validation-only
// schedule. This CPU schedule makes no assertion about PA empty O/LSE values.
// No physical page IDs, addresses, pool sizes or strides enter scheduling.
MlaScheduleStatus BuildPagedSchedule(const MlaPagedShape& shape, const MlaPlatformInfo& platform,
                                     const MlaPagedInputs& inputs, MlaPagedSchedule& output);

// Validated PA geometry/workspace without a no-mask partition; used by causal.
MlaScheduleStatus PreparePagedSchedule(const MlaPagedShape& shape, const MlaPlatformInfo& platform,
                                       const MlaPagedInputs& inputs, MlaPagedSchedule& output);

// Source PA tile bound, shared by explicit and native split plans.
inline uint32_t PagedSplitCapacityBase(const MlaPagedShape& s)
{
    const uint32_t base = s.pageSize < 512 && 512 % s.pageSize ? 512 / s.pageSize * s.pageSize : 512;
    return std::max(uint32_t(1), std::min(uint32_t(s.logicalKvCapacity), base));
}
} // namespace flash_attn::mla

// Source: CANN ops-transformer ca18aee610729ca5270a44e2d84c751bc687fbf0,
// attention/common/op_host/split_core.cpp::CalcS2Range / CalcNextTokenLeftUp.
// The upstream license is reproduced in mla/LICENSE.cann.

namespace flash_attn::mla {

struct MlaCausalBlockInput {
    // Effective lengths of ONE sample, not allocation capacities or batch maxima.
    // Layout/address/used validation belongs to the caller. Rows use S1G order.
    int64_t queryLength;
    int64_t kvLength;
    int64_t queryHeads;
    int64_t mBaseSize;
    int64_t s2BaseSize;
    int64_t mBlock;
};

struct MlaCausalBlock {
    int64_t nextTokenLeftUp; // Sk - Sq; key j is visible iff j <= query i + this value.
    uint32_t rowBegin;
    uint32_t rowEnd;
    uint32_t queryBegin;
    uint32_t queryEnd;
    uint32_t kvBegin;
    uint32_t kvEnd;
    uint32_t s2Begin;
    uint32_t s2End;
    uint32_t leadingEmptyRows;
    int64_t cost;
    int64_t normalS2Cost;
    int64_t lastS2Cost;
};

enum class MlaCausalStatus {
    Success,
    UnsupportedShape,
    FieldOverflow,
    InvalidTile,
    InvalidBlock,
    InvalidPlatform,
    InvalidSplit,
    AllocationFailure,
};

// CPU-only geometry for shared 512+64 KV, Hkv=1, right-down causal, num_splits=1.
// Reuses the pinned source's S2 block range. All intervals are half-open and
// relative to one sample. kvBegin/kvEnd are the union of visible logical keys;
// boundary S2 blocks still need a per-row causal mask in the attention kernel.
// leadingEmptyRows counts flattened S1G rows with NO visible key, including
// all rows when Sk=0. It is not a count of query tokens or unused allocation.
// Sq=0 accepts only mBlock=0 and produces empty intervals. Otherwise mBlock
// must name a real M block. Logical tiles are 1..256 by 1..512; accepting a
// size here does not validate device resources or enable a production kernel.
// Uses the existing per-sample MLA shape bounds. Output is unchanged on failure.
// No tensor, metadata format, core partition, workspace or public entry is created.
MlaCausalStatus BuildCausalBlock(const MlaCausalBlockInput& input, MlaCausalBlock& output);

struct MlaCausalSequence {
    int64_t queryLength, kvLength;
};

struct MlaCausalScheduleInput {
    // Owned effective lengths, obtained after the existing layout/used parsers.
    // No allocation prefix, page ID, address or batch maximum is a mask length.
    std::vector<MlaCausalSequence> sequences;
    int64_t queryHeads, mBaseSize, s2BaseSize;
    int64_t aicNum, cvRatio;
};

struct MlaCausalSchedule {
    uint32_t usedCoreNum;
    std::array<uint32_t, kMlaMaxAic> bN2End, gS1End, s2End;
    std::array<int64_t, kMlaMaxAic> coreCost;
    int64_t maxCost, totalCost;
    uint64_t totalMBlocks, totalS2Blocks;
};

// Independent CPU partition at caller-selected logical tile sizes. Retain the
// pinned SplitCore candidate, round partial S2 endpoints up to whole M blocks,
// coalesce duplicates and recompute each core's mask-dependent cost. The last
// endpoint is always (B,0,0), including trailing zero-cost output blocks.
// Every effective Q row has one owner, even if it sees no key. Empty work uses
// one zero-cost core. Unused array entries are zero; output is unchanged on
// failure. No FD, workspace, platform resource validation or production launch.
MlaCausalStatus BuildCausalSchedule(const MlaCausalScheduleInput& input, MlaCausalSchedule& output);

} // namespace flash_attn::mla

// Source: CANN ca18aee610729ca5270a44e2d84c751bc687fbf0 split_core.cpp
// and official FA3 d15f1531a460ba456f41b01a774f33ab2db8febf hopper/flash_api.cpp.
// Licenses: mla/LICENSE.cann and mla/LICENSE.fa3.

namespace flash_attn::mla {

struct MlaWindowConfig {
    // Call-level bounds used by FA3 host normalization, not per-sample used lengths.
    int64_t maxQueryLength, maxKvLength;
    int64_t windowLeft, windowRight;
    bool causal;
};

enum class MlaWindowKind {
    None,
    Causal,
    Local
};
struct MlaWindow {
    MlaWindowKind kind;
    // FA3 parameter values AFTER replacing unlimited sides by bound - 1.
    // Inactive values may be -1 for zero bounds; kind determines the mask.
    int64_t left, right;
};

// Fixed FA3 forward/metadata normalization for 512+64, attention_chunk=0.
// Negative window values denote unlimited sides. Large positive int64 windows
// are normalized before narrowing/arithmetic; host-normalized values below
// INT32_MIN reject (FA3 forward narrows to int, unlike metadata).
// The single-query causal optimization
// follows FA3's exact -1 checks. No public or device capability is enabled.
// Bounds use the existing MLA limits; failure leaves output unchanged.
MlaCausalStatus NormalizeWindow(const MlaWindowConfig& input, MlaWindow& output);

// Extends the causal half-open row/query/key/S2 fields with a lower mask bound.
// Local row i sees keys i-preTokenLeftUp <= j <= i+nextTokenLeftUp.
// None ignores both bounds; Causal ignores the lower bound. Empty key unions
// are [0,0). leadingEmptyRows includes all rows when Sk=0. There cannot be an
// empty suffix for normalized nonnegative windows with right-down alignment.
struct MlaWindowBlock : MlaCausalBlock {
    int64_t preTokenLeftUp;
};

// Effective lengths must fit the call-level bounds. Allocation/used parsers are
// unchanged and must run before this component. Retains causal geometry checks,
// logical tile limits and failure atomicity; boundary tiles need per-row masks.
MlaCausalStatus BuildWindowBlock(const MlaCausalBlockInput& input, const MlaWindowConfig& window,
                                 MlaWindowBlock& output);

// Same complete-M ownership/output format as causal, including zero-cost rows.
// Reuses causal scheduling for the causal special case; otherwise adapts the
// pinned band/no-mask SplitCore candidates. No FD, metadata or production launch.
MlaCausalStatus BuildWindowSchedule(const MlaCausalScheduleInput& input, const MlaWindowConfig& window,
                                    MlaCausalSchedule& output);

} // namespace flash_attn::mla
