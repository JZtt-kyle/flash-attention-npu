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

} // namespace flash_attn::mla
