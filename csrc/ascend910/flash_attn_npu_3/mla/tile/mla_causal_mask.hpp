/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Upstream license: ../LICENSE.cann.
// Source: CANN ops-transformer ca18aee610729ca5270a44e2d84c751bc687fbf0,
// arch22/fia_block_vec_nonquant_mla.h::ElewiseCompute / DealInvalidRows.
// Specialize boolean right-down causal/band masks to S1G rows. Generate the mask
// from validated effective lengths instead of loading a public mask tensor.
// Preserve the source finite sentinel and softmax/aMLA arithmetic; adapt only
// fully masked O/LSE to the FA3 contract in the Block epilogue.
#pragma once
#include "kernel_operator.h"

namespace flash_attn::mla::Tile {
using namespace AscendC;

__aicore__ inline uint32_t CausalEmptyRows(uint32_t row, uint32_t count, uint32_t heads, int64_t nextToken)
{
    if (nextToken >= 0) {
        return 0;
    }
    const uint64_t end = uint64_t(-nextToken) * heads;
    return end <= row ? 0 : Min(uint64_t(count), end - row);
}

__aicore__ inline void CausalMask(LocalTensor<float> scores, uint32_t row, uint32_t rows, uint32_t heads,
                                  int64_t nextToken, uint32_t keyStart, uint32_t columns, uint32_t actualColumns)
{
    // Operate at aligned vector addresses. One bit mask covers 64 FP32 lanes;
    // repeat along heads belonging to the same query, including head tails.
    for (uint32_t start = 0; start < rows;) {
        const uint32_t query = (row + start) / heads;
        const uint32_t count = Min(rows - start, heads - (row + start) % heads);
        const int64_t visible = int64_t(query) + nextToken + 1 - keyStart;
        if (visible < actualColumns) {
            const uint32_t first = visible > 0 ? uint32_t(visible) : 0;
            for (uint32_t col = first / 64 * 64; col < columns; col += 64) {
                const uint32_t begin = col < first ? first - col : 0;
                const uint32_t end = Min(64U, columns - col);
                uint64_t bits[2] = {((~uint64_t(0)) << begin) & ((~uint64_t(0)) >> (64 - end)), 0};
                Duplicate(scores[start * columns + col], -1000000000000.0f, bits, count, 1, columns / 8);
            }
        }
        start += count;
    }
}

__aicore__ inline void WindowMask(LocalTensor<float> scores, uint32_t row, uint32_t rows, uint32_t heads,
                                  int64_t preToken, int64_t nextToken, uint32_t keyStart, uint32_t columns,
                                  uint32_t actualColumns)
{
    CausalMask(scores, row, rows, heads, nextToken, keyStart, columns, actualColumns);
    // The source band combines the upper triangle with the inverted lower
    // triangle. Preserve the same finite sentinel and aligned FP32 writes.
    for (uint32_t start = 0; start < rows;) {
        const uint32_t query = (row + start) / heads;
        const uint32_t count = Min(rows - start, heads - (row + start) % heads);
        const int64_t hidden = int64_t(query) - preToken - keyStart;
        const uint32_t limit = hidden > 0 ? Min(uint64_t(hidden), uint64_t(columns)) : 0;
        for (uint32_t col = 0; col < limit; col += 64) {
            const uint32_t end = Min(64U, limit - col);
            uint64_t bits[2] = {(~uint64_t(0)) >> (64 - end), 0};
            Duplicate(scores[start * columns + col], -1000000000000.0f, bits, count, 1, columns / 8);
        }
        start += count;
    }
}
} // namespace flash_attn::mla::Tile
