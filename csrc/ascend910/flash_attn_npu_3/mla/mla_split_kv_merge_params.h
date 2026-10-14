/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Independent BSND FD launch data; not a production metadata record.
#pragma once
#include <cstdint>

namespace flash_attn::mla {
struct MlaSplitKvMergeParams {
    uint32_t batch, sq, heads, batchIndex, rowStart, rows, mBase, splits;
    uint32_t kvLength, maskKind;
    int64_t left, right;
    // Logical KV intervals, clipped to KV <=16384. uint16 keeps the by-value
    // launch record below the 1536-byte compiler/runtime parameter limit.
    uint16_t begin[256], end[256];
    // 1: adapt raw partials and write FA3 accumulators. 0: component timing on
    // already canonical statistics, without accumulator stores.
    uint32_t adapt;
    // Production Q-used controls mask geometry; sq remains allocation stride.
    // Independent components leave this zero and use sq (zero-Q has no task).
    uint32_t queryLength = 0;
};
} // namespace flash_attn::mla
