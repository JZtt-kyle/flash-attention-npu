/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Source: CANN ca18aee610729ca5270a44e2d84c751bc687fbf0 split_core.cpp
// and official FA3 d15f1531a460ba456f41b01a774f33ab2db8febf hopper/flash_api.cpp.
// Licenses: LICENSE.cann and LICENSE.fa3 next to this file.
#pragma once

#include "mla_causal.h"

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
