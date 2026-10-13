/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Source and upstream license: mla_tnd_schedule.cpp and LICENSE.cann.
#pragma once
#include "mla_used_schedule.h"

namespace flash_attn::mla {

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

} // namespace flash_attn::mla
