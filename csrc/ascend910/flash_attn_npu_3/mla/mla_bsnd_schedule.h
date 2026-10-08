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
// Compute endpoints can omit a zero-KV suffix, as in the pinned source. A future
// device consumer must initialize zero-KV outputs independently of these ranges.
// Not wired to production forward or the 401-word v1 metadata protocol.
MlaScheduleStatus BuildBsndKvLengthSchedule(const MlaDenseShape& shape, const MlaPlatformInfo& platform,
                                            const int32_t* kvLengths, uint64_t lengthCount, MlaSchedule& output);

} // namespace flash_attn::mla
