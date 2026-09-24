/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the CATLASS software repository for the full text of the License.
 */
#pragma once

#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "mla_launch.h"

namespace flash_attn::mla {

// Reuse the reduction/broadcast tiles from CATLASS 769cd40 examples/19_mla.
// The local block keeps scale in FP32, normalizes a single KV tile, and writes
// FA3's natural-log LSE in BHS order. The source component's half scale and
// split-oriented rescale output do not implement that contract.
template <typename Element>
class ShortEpilogue {
    using Arch = Catlass::Arch::AtlasA2;
    using FloatType = Catlass::Gemm::GemmType<float, Catlass::layout::RowMajor>;
    using HalfType = Catlass::Gemm::GemmType<Element, Catlass::layout::RowMajor>;
    using Reduction = Catlass::Epilogue::Block::BlockEpilogue<Catlass::Epilogue::EpilogueAtlasA2MLASoftmax, HalfType,
                                                              FloatType, HalfType>;

  public:
    CATLASS_DEVICE
    explicit ShortEpilogue(Catlass::Arch::Resource<Arch>& resource) : reduction(resource, half(1.0f), 1)
    {
        scores = resource.ubBuf.template GetBufferByByte<float>(0);
        probabilities = resource.ubBuf.template GetBufferByByte<Element>(8192);
        maximum = resource.ubBuf.template GetBufferByByte<float>(12288);
        sum = resource.ubBuf.template GetBufferByByte<float>(12544);
        logsum = resource.ubBuf.template GetBufferByByte<float>(12800);
        temporary = resource.ubBuf.template GetBufferByByte<float>(16384);
        output = resource.ubBuf.template GetBufferByByte<float>(32768);
        outputHalf = resource.ubBuf.template GetBufferByByte<Element>(65536);
    }

    CATLASS_DEVICE
    void Softmax(AscendC::GlobalTensor<float> source, AscendC::GlobalTensor<Element> dest,
                 AscendC::GlobalTensor<float> lse, uint32_t rows, uint32_t columns, uint32_t lseStride, float scale)
    {
        if (rows == 0) {
            return;
        }
        const uint32_t stride = ::RoundUp<16>(columns);
        const uint32_t count = rows * stride;
        AscendC::DataCopy(scores, source, count);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::Muls(scores, scores, scale, count);
        AscendC::PipeBarrier<PIPE_V>();
        reduction.ReduceMaxRepeatM(maximum, scores, temporary, rows, columns, stride);
        reduction.TensorSubValueRepeatM(scores, scores, maximum, temporary, rows, ::RoundUp<16>(rows), columns, stride);
        AscendC::Exp(scores, scores, count);
        AscendC::PipeBarrier<PIPE_V>();
        reduction.RowsumTAILTILE(scores, sum, temporary, rows, columns, stride);
        AscendC::Ln(logsum, sum, rows);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(logsum, logsum, maximum, rows);
        AscendC::PipeBarrier<PIPE_V>();

        // Broadcast the denominator to one 32-byte block per row.
        AscendC::Brcb(temporary.template ReinterpretCast<uint32_t>(), sum.template ReinterpretCast<uint32_t>(),
                      ::RoundUp<8>(rows) / 8, AscendC::BrcbRepeatParams(1, 8));
        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t column = 0; column < stride; column += 64) {
            const uint32_t width = ::Min(64U, stride - column);
            AscendC::Div<float>(scores[column], scores[column], temporary, width, rows,
                                AscendC::BinaryRepeatParams(1, 1, 0, stride / 8, stride / 8, 1));
        }
        AscendC::PipeBarrier<PIPE_V>();
        Cast(probabilities, scores, count);
        AscendC::PipeBarrier<PIPE_V>();
        // Cube tiles round K to 16. Keep the probability padding exactly zero.
        if (stride != columns) {
            // AscendC's mask array is {low64, high64}, unlike SetVectorMask.
            uint64_t tailMask[2] = {(0xffffULL << (columns % 16)) & 0xffffULL, 0};
            for (uint32_t row = 0; row < rows; ++row) {
                AscendC::Duplicate(probabilities[row * stride + columns / 16 * 16], Element(0), tailMask, 1, 1, 8);
            }
            AscendC::SetVectorMask<int8_t>(uint64_t(-1), uint64_t(-1));
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Brcb(temporary.template ReinterpretCast<uint32_t>(), logsum.template ReinterpretCast<uint32_t>(),
                      ::RoundUp<8>(rows) / 8, AscendC::BrcbRepeatParams(1, 8));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::DataCopy(dest, probabilities, count);
        AscendC::DataCopyPad(lse, temporary, AscendC::DataCopyExtParams(rows, 4, 0, (lseStride - 1) * 4, 0));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    }

    CATLASS_DEVICE
    void Store(AscendC::GlobalTensor<float> source, AscendC::GlobalTensor<Element> dest, uint32_t rows)
    {
        if (rows == 0) {
            return;
        }
        AscendC::DataCopy(output, source, rows * 512);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        Cast(outputHalf, output, rows * 512);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::DataCopy(dest, outputHalf, rows * 512);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    }

  private:
    CATLASS_DEVICE
    void Cast(AscendC::LocalTensor<Element> dest, AscendC::LocalTensor<float> source, uint32_t count)
    {
        constexpr auto mode =
            std::is_same_v<Element, bfloat16_t> ? AscendC::RoundMode::CAST_RINT : AscendC::RoundMode::CAST_NONE;
        AscendC::Cast(dest, source, mode, count);
    }

    Reduction reduction;
    AscendC::LocalTensor<float> scores, maximum, sum, logsum, temporary, output;
    AscendC::LocalTensor<Element> probabilities, outputHalf;
};

} // namespace flash_attn::mla
