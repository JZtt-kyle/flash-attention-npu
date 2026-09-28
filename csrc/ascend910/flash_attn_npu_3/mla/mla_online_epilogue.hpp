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

#include "mla_short_epilogue.hpp"

namespace flash_attn::mla {

// Online recurrence from pinned CANN ca18aee arch22/fia_block_vec_nonquant.h
// and CATLASS 769cd40 block_epilogue_mla_{softmax,rescale_o}.hpp. Reuse the
// short block's reduction tiles, with FP32 scale, maximum, sum and numerator.
// Two alpha slots allow softmax(i+1) to overlap PV(i) without losing its scale.
template <typename Element>
class OnlineEpilogue : public ShortEpilogue<Element> {
    using Base = ShortEpilogue<Element>;
    using Base::Cast;
    using Base::logsum;
    using Base::maximum;
    using Base::output;
    using Base::outputHalf;
    using Base::probabilities;
    using Base::reduction;
    using Base::scores;
    using Base::sum;
    using Base::temporary;

  public:
    CATLASS_DEVICE
    explicit OnlineEpilogue(Catlass::Arch::Resource<Catlass::Arch::AtlasA2>& resource) : Base(resource)
    {
        runningMaximum = resource.ubBuf.template GetBufferByByte<float>(13056);
        runningSum = resource.ubBuf.template GetBufferByByte<float>(13312);
        alpha[0] = resource.ubBuf.template GetBufferByByte<float>(13568);
        alpha[1] = resource.ubBuf.template GetBufferByByte<float>(13824);
        numerator = resource.ubBuf.template GetBufferByByte<float>(81920);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
    }

    CATLASS_DEVICE
    ~OnlineEpilogue()
    {
        // Each buffer keeps one release token, including lanes with no rows.
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
    }

    CATLASS_DEVICE
    void Softmax(AscendC::GlobalTensor<float> source, AscendC::GlobalTensor<Element> dest, uint32_t rows,
                 uint32_t columns, float scale, uint32_t tile)
    {
        if (rows == 0) {
            return;
        }
        const uint32_t stride = ::RoundUp<16>(columns);
        const uint32_t count = rows * stride;
        auto rescale = alpha[tile % 2];
        // Only the previous softmax reads this buffer. Its release permits
        // MTE2 to load scores while Vector accumulates an independent partial.
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::DataCopy(scores, source, count);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::Muls(scores, scores, scale, count);
        AscendC::PipeBarrier<PIPE_V>();
        reduction.ReduceMaxRepeatM(maximum, scores, temporary, rows, columns, stride);
        if (tile != 0) {
            AscendC::Max(maximum, maximum, runningMaximum, rows);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Sub(rescale, runningMaximum, maximum, rows);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Exp(rescale, rescale, rows);
            AscendC::PipeBarrier<PIPE_V>();
        }
        AscendC::DataCopy(runningMaximum, maximum, ::RoundUp<8>(rows));
        AscendC::PipeBarrier<PIPE_V>();
        reduction.TensorSubValueRepeatM(scores, scores, maximum, temporary, rows, ::RoundUp<16>(rows), columns, stride);
        AscendC::Exp(scores, scores, count);
        AscendC::PipeBarrier<PIPE_V>();
        // Preserve P before using CATLASS's in-place row reduction. The two
        // 64-element halves are added once, avoiding repeated block reductions
        // on every full KV tile. The sum still uses the FP32 exponentials.
        Cast(probabilities, scores, count);
        AscendC::PipeBarrier<PIPE_V>();
        reduction.ReduceSumRepeatM(sum, scores, rows, columns, stride);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        if (tile == 0) {
            AscendC::DataCopy(runningSum, sum, ::RoundUp<8>(rows));
        } else {
            AscendC::Mul(runningSum, runningSum, rescale, rows);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(runningSum, runningSum, sum, rows);
        }
        AscendC::PipeBarrier<PIPE_V>();
        // The last tile may end inside the Cube's K alignment. Padding must
        // contribute neither to the denominator nor to the PV numerator.
        if (stride != columns) {
            uint64_t tailMask[2] = {(0xffffULL << (columns % 16)) & 0xffffULL, 0};
            for (uint32_t row = 0; row < rows; ++row) {
                AscendC::Duplicate(probabilities[row * stride + columns / 16 * 16], Element(0), tailMask, 1, 1, 8);
            }
            AscendC::SetVectorMask<int8_t>(uint64_t(-1), uint64_t(-1));
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::DataCopy(dest, probabilities, count);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    }

    CATLASS_DEVICE
    void Accumulate(AscendC::GlobalTensor<float> source, AscendC::GlobalTensor<Element> dest,
                    AscendC::GlobalTensor<float> lse, uint32_t rows, uint32_t lseStride, uint32_t tile, bool last)
    {
        if (rows == 0) {
            return;
        }
        // The partial buffer has a separate lifetime from scores, so its next
        // load need not wait for the intervening softmax's vector instructions.
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::DataCopy(output, source, rows * 512);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        if (tile == 0) {
            AscendC::DataCopy(numerator, output, rows * 512);
        } else {
            ScaleNumerator<false>(alpha[tile % 2], rows);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Add(numerator, numerator, output, rows * 512);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        if (!last) {
            return;
        }
        ScaleNumerator<true>(runningSum, rows);
        AscendC::PipeBarrier<PIPE_V>();
        Cast(outputHalf, numerator, rows * 512);
        AscendC::Ln(logsum, runningSum, rows);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Add(logsum, logsum, runningMaximum, rows);
        AscendC::PipeBarrier<PIPE_V>();
        Broadcast(logsum, rows);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::DataCopy(dest, outputHalf, rows * 512);
        AscendC::DataCopyPad(lse, temporary, AscendC::DataCopyExtParams(rows, 4, 0, (lseStride - 1) * 4, 0));
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    }

  private:
    template <bool Divide>
    CATLASS_DEVICE void ScaleNumerator(AscendC::LocalTensor<float> factors, uint32_t rows)
    {
        Broadcast(factors, rows);
        // Each factor occupies one 32-byte block. For fewer than eight rows,
        // repeat along the 512 columns of a row and reuse its factor in every
        // repeat. This issues rows instructions instead of eight. Wider tasks
        // keep the column-wise traversal, including C1V1's sixteen-row case.
        const bool byRow = rows < 8;
        const uint32_t instructions = byRow ? rows : 8;
        const uint32_t repeats = byRow ? 8 : rows;
        const AscendC::BinaryRepeatParams strides =
            byRow ? AscendC::BinaryRepeatParams(1, 1, 0, 8, 8, 0) : AscendC::BinaryRepeatParams(1, 1, 0, 64, 64, 1);
        for (uint32_t index = 0; index < instructions; ++index) {
            auto values = numerator[index * (byRow ? 512 : 64)];
            auto factor = temporary[byRow ? index * 8 : 0];
            if constexpr (Divide) {
                AscendC::Div<float>(values, values, factor, 64, repeats, strides);
            } else {
                AscendC::Mul<float>(values, values, factor, 64, repeats, strides);
            }
        }
    }

    CATLASS_DEVICE
    void Broadcast(AscendC::LocalTensor<float> source, uint32_t rows)
    {
        AscendC::Brcb(temporary.template ReinterpretCast<uint32_t>(), source.template ReinterpretCast<uint32_t>(),
                      ::RoundUp<8>(rows) / 8, AscendC::BrcbRepeatParams(1, 8));
        AscendC::PipeBarrier<PIPE_V>();
    }

    AscendC::LocalTensor<float> runningMaximum, runningSum, alpha[2], numerator;
};

} // namespace flash_attn::mla
