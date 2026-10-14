/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Independent Vector FD merge; source arithmetic lives in Block/Tile.
#pragma once
#include "../block/mla_split_kv_merge.hpp"

namespace flash_attn::mla::Kernel {
template <typename OutputT>
class MlaSplitKvMerge {
  public:
    __aicore__ inline void operator()(const MlaSplitKvMergeParams& params, GM_ADDR out, GM_ADDR sum, GM_ADDR maximum,
                                      GM_ADDR result, GM_ADDR lse, GM_ADDR accum, GM_ADDR accumLse, TPipe* pipe)
    {
        if ASCEND_IS_AIV {
            if (GetBlockIdx() != 0) {
                return;
            }
            Block::MlaSplitKvMerge<OutputT> block;
            block.Init(params, out, sum, maximum, result, lse, accum, accumLse, pipe);
            const uint32_t tile = block.TileRows();
            for (uint32_t start = 0; start < params.rows; start += tile) {
                block.Run(start, Min(tile, params.rows - start));
            }
        }
    }
};
} // namespace flash_attn::mla::Kernel
