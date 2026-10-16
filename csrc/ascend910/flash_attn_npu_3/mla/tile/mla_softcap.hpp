// FA3 softcap applies to the complete scaled score, before masking.
// Reuse AscendC's FP32 tanh and the existing softmax scratch allocation.
#pragma once
#include "lib/math/tanh.h"

namespace flash_attn::mla::Tile {
__aicore__ inline void Softcap(AscendC::LocalTensor<float> scores, AscendC::LocalTensor<uint8_t> scratch,
                               uint32_t count, float cap)
{
    if (cap == 0.0f)
        return;
    // Four 8 KiB regions fit in the existing softmax scratch. Tanh's input,
    // output and temporary tensors are disjoint. All scratch contents are
    // dead before SoftmaxFlashV2; no extra UB or event is needed.
    constexpr uint32_t chunkSize = 2048;
    auto normalized = scratch.ReinterpretCast<float>();
    auto temporary = scratch[chunkSize * sizeof(float)];
    temporary.SetSize(chunkSize * sizeof(float));
    auto polynomial = scratch[2 * chunkSize * sizeof(float)].ReinterpretCast<float>();
    auto square = scratch[3 * chunkSize * sizeof(float)].ReinterpretCast<float>();
    AscendC::PipeBarrier<PIPE_V>();
    for (uint32_t start = 0; start < count; start += chunkSize) {
        uint32_t size = count - start < chunkSize ? count - start : chunkSize;
        AscendC::Duplicate(normalized, cap, size);
        AscendC::PipeBarrier<PIPE_V>();
        // Division also handles very small positive caps without overflowing
        // a precomputed reciprocal (zero scores must remain zero).
        AscendC::Div(normalized, scores[start], normalized, size);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Tanh(scores[start], normalized, temporary, size);
        AscendC::PipeBarrier<PIPE_V>();
        // AscendC tanh uses exp(2*x)-1. For |x| <= 1/8, evaluate its
        // odd Taylor polynomial through x^7 instead: the relative remainder
        // is < 1.4e-9, below FP32 precision, without cancellation near zero.
        // Reuse tanh's now-dead temporary as the Select bit mask.
        AscendC::Mul(square, normalized, normalized, size);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::CompareScalar(temporary, square, 0.015625f, AscendC::CMPMODE::LE, size);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mins(square, square, 0.015625f, size);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(polynomial, square, -17.0f / 315.0f, size);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(polynomial, polynomial, 2.0f / 15.0f, size);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(polynomial, polynomial, square, size);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(polynomial, polynomial, -1.0f / 3.0f, size);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(polynomial, polynomial, square, size);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(polynomial, polynomial, 1.0f, size);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(normalized, normalized, polynomial, size);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Select(scores[start], temporary, normalized, scores[start], AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE,
                        size);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(scores[start], scores[start], cap, size);
        AscendC::PipeBarrier<PIPE_V>();
    }
}
} // namespace flash_attn::mla::Tile
