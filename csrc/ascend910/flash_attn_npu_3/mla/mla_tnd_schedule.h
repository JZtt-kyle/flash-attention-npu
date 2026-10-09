/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// The upstream license is reproduced in LICENSE.cann next to this file.
#pragma once

#include "mla_schedule.h"
#include <vector>

namespace flash_attn::mla {

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
// Used by the independent v3 producer; not connected to attention or public TND.
MlaScheduleStatus BuildTndSchedule(const MlaTndShape& shape, const MlaPlatformInfo& platform, const int32_t* cuQuery,
                                   uint64_t queryCount, const int32_t* cuKv, uint64_t kvCount, MlaTndSchedule& output);

} // namespace flash_attn::mla
