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
#include "mla_schedule.h"
#include <vector>

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
} // namespace flash_attn::mla
