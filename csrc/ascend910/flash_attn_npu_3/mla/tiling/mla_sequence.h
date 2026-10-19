#pragma once
#include <algorithm>
#include "mla_schedule.h"
#include <vector>

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

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

// Source and upstream license: tiling/mla_sequence.cpp and LICENSE.cann.
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
