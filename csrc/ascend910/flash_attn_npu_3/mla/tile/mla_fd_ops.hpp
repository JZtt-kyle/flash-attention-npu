/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Source: CANN ca18aee610729ca5270a44e2d84c751bc687fbf0,
// attention/common/op_kernel/vector_common.h. See ../LICENSE.cann.
// Split counts above the uint8 repeat bound are segmented by the Block caller.
#pragma once
#include "mla_vector_ops.hpp"
namespace flash_attn::mla::Tile {
__aicore__ inline void MatDivsVec(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                                  uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t dtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t dLoop = actualColumnCount / dtypeMask;
    uint32_t dRemain = actualColumnCount % dtypeMask;

    BinaryRepeatParams repeatParamsDiv;
    repeatParamsDiv.src0BlkStride = 1;
    repeatParamsDiv.src1BlkStride = 1;
    repeatParamsDiv.dstBlkStride = 1;
    repeatParamsDiv.src0RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    repeatParamsDiv.src1RepStride = 0;
    repeatParamsDiv.dstRepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    uint32_t columnRepeatCount = dLoop;
    uint32_t offset = 0;
    for (uint32_t i = 0; i < dLoop; i++) {
        Div(dstUb[offset], src0Ub[offset], src1Ub[offset], dtypeMask, dealRowCount, repeatParamsDiv);
        offset += dtypeMask;
    }

    if (dRemain > 0) {
        Div(dstUb[dLoop * dtypeMask], src0Ub[dLoop * dtypeMask], src1Ub[dLoop * dtypeMask], dRemain, dealRowCount,
            repeatParamsDiv);
    }
}

__aicore__ inline void RowSub(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                              uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t dtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t dLoop = actualColumnCount / dtypeMask;
    uint32_t dRemain = actualColumnCount % dtypeMask;

    BinaryRepeatParams repeatParamsSub;
    repeatParamsSub.src0BlkStride = 1;
    repeatParamsSub.src1BlkStride = 1;
    repeatParamsSub.dstBlkStride = 1;
    repeatParamsSub.src0RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    repeatParamsSub.src1RepStride = 0;
    repeatParamsSub.dstRepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    uint32_t columnRepeatCount = dLoop;
    uint32_t offset = 0;
    for (uint32_t i = 0; i < dLoop; i++) {
        Sub(dstUb[offset], src0Ub[offset], src1Ub[offset], dtypeMask, dealRowCount, repeatParamsSub);
        offset += dtypeMask;
    }

    if (dRemain > 0) {
        Sub(dstUb[dLoop * dtypeMask], src0Ub[dLoop * dtypeMask], src1Ub[dLoop * dtypeMask], dRemain, dealRowCount,
            repeatParamsSub);
    }
}

__aicore__ inline void ColMax(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                              uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t dtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t dLoop = actualColumnCount / dtypeMask;
    uint32_t dRemain = actualColumnCount % dtypeMask;

    BinaryRepeatParams repeatParamsMax;
    repeatParamsMax.src0BlkStride = 1;
    repeatParamsMax.src1BlkStride = 1;
    repeatParamsMax.dstBlkStride = 1;
    repeatParamsMax.src0RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    repeatParamsMax.src1RepStride = 0;
    repeatParamsMax.dstRepStride = 0;
    uint32_t columnRepeatCount = dLoop;
    uint32_t offset = 0;
    for (uint32_t i = 0; i < dLoop; i++) {
        Max(dstUb[offset], src0Ub[offset], src1Ub[offset], dtypeMask, dealRowCount, repeatParamsMax);
        offset += dtypeMask;
    }

    if (dRemain > 0) {
        Max(dstUb[dLoop * dtypeMask], src0Ub[dLoop * dtypeMask], src1Ub[dLoop * dtypeMask], dRemain, dealRowCount,
            repeatParamsMax);
    }
}

__aicore__ inline void ColAdd(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                              uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t dtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t dLoop = actualColumnCount / dtypeMask;
    uint32_t dRemain = actualColumnCount % dtypeMask;

    BinaryRepeatParams repeatParamsAdd;
    repeatParamsAdd.src0BlkStride = 1;
    repeatParamsAdd.src1BlkStride = 1;
    repeatParamsAdd.dstBlkStride = 1;
    repeatParamsAdd.src0RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    repeatParamsAdd.src1RepStride = 0;
    repeatParamsAdd.dstRepStride = 0;
    uint32_t columnRepeatCount = dLoop;
    uint32_t offset = 0;
    for (uint32_t i = 0; i < dLoop; i++) {
        Add(dstUb[offset], src0Ub[offset], src1Ub[offset], dtypeMask, dealRowCount, repeatParamsAdd);
        offset += dtypeMask;
    }

    if (dRemain > 0) {
        Add(dstUb[dLoop * dtypeMask], src0Ub[dLoop * dtypeMask], src1Ub[dLoop * dtypeMask], dRemain, dealRowCount,
            repeatParamsAdd);
    }
}

} // namespace flash_attn::mla::Tile
