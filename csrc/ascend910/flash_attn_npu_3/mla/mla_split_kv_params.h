/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Independent validated split-KV launch descriptors; no metadata/public ABI.
#pragma once
#include "mla_window_params.h"

namespace flash_attn::mla {
// Independent partial launch: one validated M/S2 fragment per physical AIC.
// Slots use the M10a native FD workspace (M rows, 512 FP32 O, 8 sum/max each).
// This is launch data, not production metadata or a public accumulator.
struct MlaSplitKvTask {
    uint32_t batch, m, split, s2Begin, s2End;
};
struct MlaSplitKvKernelParams {
    MlaWindowKernelParams window;
    uint64_t outOffsetBytes, sumOffsetBytes, maxOffsetBytes;
    uint64_t totalTasks;
    MlaSplitKvTask tasks[26];
};
// A validated GQA fragment uses the original allocation coordinates while
// scheduling a subset of query heads. Other callers keep their span reader.
struct MlaTaskLengths {
    uint32_t queryStart, queryLength, kvLength;
    uint64_t kvStart;
};
} // namespace flash_attn::mla
