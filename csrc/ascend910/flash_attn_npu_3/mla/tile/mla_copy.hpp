/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Upstream license: ../LICENSE.cann (block/ and tile/).
// Source: CANN ops-transformer ca18aee610729ca5270a44e2d84c751bc687fbf0.
// attention/common/op_kernel/memcopy/attn_copy_gm_to_l1.h:
// CopySingle/MultiMatrixNDToNZ, CopyQueryGmToL1::ProcessS1G,
// CopyKvGmToL1::ProcessContinuousOrTensorlist, specialized to dense FP16 BSND.
#pragma once
#include "catlass/gemm/tile/atlasa2/copy_l0c_to_gm.hpp"

namespace flash_attn::mla::Tile {
using namespace AscendC;
__aicore__ inline uint32_t Align(uint32_t n, uint32_t a)
{
    return (n + a - 1) / a * a;
}
struct L1Tensor {
    LocalTensor<half> tensor;
    uint32_t rowCount;
};
struct QueryCoord {
    uint32_t bIdx, n2Idx, gS1Idx, dIdx, gS1DealSize, dDealSize;
};
struct KvCoord {
    uint32_t bIdx, n2Idx, s2Idx, dIdx, s2DealSize, dDealSize;
};
struct DenseTensor {
    GlobalTensor<half> gmTensor;
    uint32_t sequence, heads, dim;
    __aicore__ inline void Init(GM_ADDR ptr, uint32_t s, uint32_t h, uint32_t d)
    {
        gmTensor.SetGlobalBuffer((__gm__ half*)ptr);
        sequence = s;
        heads = h;
        dim = d;
    }
};

__aicore__ inline void CopySingleMatrixNDToNZ(LocalTensor<half> dst, GlobalTensor<half> src, uint32_t rows,
                                              uint32_t cols, uint32_t stride, uint32_t dstRows)
{
    // Source's large-stride fallback is unreachable: dense strides are 64/512.
    Nd2NzParams p;
    p.ndNum = 1;
    p.nValue = rows;
    p.dValue = cols;
    p.srcDValue = stride;
    p.dstNzC0Stride = dstRows;
    p.dstNzNStride = 1;
    p.srcNdMatrixStride = 0;
    p.dstNzMatrixStride = 0;
    DataCopy(dst, src, p);
}

__aicore__ inline void CopyMultiMatrixNDToNZ(LocalTensor<half> dst, GlobalTensor<half> src, uint32_t count,
                                             uint32_t srcMatrixStride, uint32_t dstMatrixStride, uint32_t rows,
                                             uint32_t cols, uint32_t stride, uint32_t dstRows)
{
    // Hq*512 can exceed 65536 even in the first 256-row range.
    if (srcMatrixStride > 65536) {
        uint64_t srcOffset = 0, dstOffset = 0;
        for (uint32_t i = 0; i < count; ++i) {
            CopySingleMatrixNDToNZ(dst[dstOffset], src[srcOffset], rows, cols, stride, dstRows);
            srcOffset += srcMatrixStride;
            dstOffset += dstMatrixStride;
        }
    } else {
        Nd2NzParams p;
        p.ndNum = count;
        p.nValue = rows;
        p.dValue = cols;
        p.srcDValue = stride;
        p.dstNzC0Stride = dstRows;
        p.dstNzNStride = 1;
        p.srcNdMatrixStride = srcMatrixStride;
        p.dstNzMatrixStride = dstMatrixStride;
        DataCopy(dst, src, p);
    }
}

class CopyQueryGmToL1 {
  public:
    __aicore__ inline void operator()(L1Tensor& dst, DenseTensor& src, QueryCoord& c)
    {
        uint32_t sStart = c.gS1Idx / src.heads, gStart = c.gS1Idx % src.heads;
        uint32_t sEnd = (c.gS1Idx + c.gS1DealSize) / src.heads;
        uint32_t gEnd = (c.gS1Idx + c.gS1DealSize) % src.heads;
        uint64_t base = ((uint64_t(c.bIdx) * src.sequence + sStart) * src.heads) * src.dim + c.dIdx;
        uint32_t strideS = src.heads * src.dim;
        if (src.heads == 1) {
            CopySingleMatrixNDToNZ(dst.tensor, src.gmTensor[base], sEnd - sStart, c.dDealSize, strideS, dst.rowCount);
            return;
        }
        uint32_t head = sStart == sEnd ? gEnd - gStart : src.heads - gStart;
        CopySingleMatrixNDToNZ(dst.tensor, src.gmTensor[base + gStart * src.dim], head, c.dDealSize, src.dim,
                               dst.rowCount);
        if (sEnd - sStart >= 1) {
            uint64_t gmOffset = base + strideS;
            uint64_t l1Offset = head * 16;
            if (sEnd - sStart > 1) {
                CopyMultiMatrixNDToNZ(dst.tensor[l1Offset], src.gmTensor[gmOffset], sEnd - sStart - 1, strideS,
                                      src.heads * 16, src.heads, c.dDealSize, src.dim, dst.rowCount);
                gmOffset += (sEnd - sStart - 1) * strideS;
                l1Offset += (sEnd - sStart - 1) * src.heads * 16;
            }
            if (gEnd > 0) {
                CopySingleMatrixNDToNZ(dst.tensor[l1Offset], src.gmTensor[gmOffset], gEnd, c.dDealSize, src.dim,
                                       dst.rowCount);
            }
        }
    }
};
class CopyKvGmToL1 {
  public:
    __aicore__ inline void operator()(L1Tensor& dst, DenseTensor& src, KvCoord& c)
    {
        uint64_t offset = (uint64_t(c.bIdx) * src.sequence + c.s2Idx) * src.dim + c.dIdx;
        CopySingleMatrixNDToNZ(dst.tensor, src.gmTensor[offset], c.s2DealSize, c.dDealSize, src.dim, dst.rowCount);
    }
};

__aicore__ inline void StoreFp32(GlobalTensor<float> dst, LocalTensor<float> src, uint32_t m, uint32_t n,
                                 uint32_t stride)
{
    using GmType = Catlass::Gemm::GemmType<float, Catlass::layout::RowMajor>;
    Catlass::Gemm::Tile::CopyL0CToGm<Catlass::Arch::AtlasA2, float, GmType> copy;
    copy(dst, src, Catlass::layout::RowMajor(m, n, stride), Catlass::layout::zN::template MakeLayout<float>(m, n), 3);
}
} // namespace flash_attn::mla::Tile
