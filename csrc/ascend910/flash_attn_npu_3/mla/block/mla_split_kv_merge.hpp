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
// arch22/fia_block_vec_flashdecode_arch22.h: ComputeScaleValue,
// ReduceFinalRes, CopyFinalResOut. FP16/BF16; 512+64 and 192/128 numerical specializations.
#pragma once
#include "catlass/catlass.hpp"
#include "../tile/mla_fd_ops.hpp"
#include "../mla_params.h"
namespace flash_attn::mla::Block {
using namespace AscendC;

template <typename OutputT, bool WriteSingleAccumulator = false, bool IsTnd = false, uint32_t HeadDim = 512,
          bool GroupedOutput = HeadDim == 128>
class MlaSplitKvMerge {
    static_assert(HeadDim == 512 || HeadDim == 128);

  public:
    __aicore__ inline void Init(const MlaSplitKvMergeParams& p, GM_ADDR out, GM_ADDR sum, GM_ADDR maximum,
                                GM_ADDR result, GM_ADDR lse, GM_ADDR accum, GM_ADDR accumLse, TPipe* pipe,
                                float softcap = 0.0f)
    {
        params = p;
        roundToEven = softcap != 0.0f;
        partial.SetGlobalBuffer((__gm__ float*)out);
        sums.SetGlobalBuffer((__gm__ float*)sum);
        maxima.SetGlobalBuffer((__gm__ float*)maximum);
        output.SetGlobalBuffer((__gm__ OutputT*)result);
        outputLse.SetGlobalBuffer((__gm__ float*)lse);
        accumulator.SetGlobalBuffer((__gm__ float*)accum);
        accumulatorLse.SetGlobalBuffer((__gm__ float*)accumLse);
        // <=192 split-row pairs use the source 6 KiB bound. 256 explicit
        // splits use one row and 8 KiB. No uint8 vector repeat may be 256.
        pipe->InitBuffer(sumBuf, 8192);
        pipe->InitBuffer(maxBuf, 8192);
        pipe->InitBuffer(weightBuf, 8192);
        pipe->InitBuffer(rowMaxBuf, 256);
        pipe->InitBuffer(rowSumBuf, 256);
        pipe->InitBuffer(lseBuf, 256);
        pipe->InitBuffer(partialBuf[0], 16384);
        pipe->InitBuffer(partialBuf[1], 16384);
        pipe->InitBuffer(reduceBuf, 16384);
        pipe->InitBuffer(castBuf, 8192);
    }

    __aicore__ inline uint32_t TileRows() const
    {
        return Max(1U, Min(8U, 192U / params.splits));
    }

    __aicore__ inline bool Valid(uint32_t split, uint32_t row) const
    {
        if (!params.adapt) {
            return true;
        }
        int64_t lo = params.begin[split], hi = params.end[split];
        const int64_t query = (params.rowStart + row) / params.heads;
        const int64_t center = query + int64_t(params.kvLength) - (params.queryLength ? params.queryLength : params.sq);
        if (params.maskKind == 1) {
            hi = Min(hi, center + 1);
        } else if (params.maskKind == 2) {
            lo = Max(lo, center - params.left);
            hi = Min(hi, center + params.right + 1);
        }
        return lo < hi;
    }

    __aicore__ inline uint64_t RowAddress(uint32_t row) const
    {
        const uint64_t flat = uint64_t(params.rowStart) + row;
        if constexpr (IsTnd)
            return (uint64_t(GroupedOutput ? params.kvHead : 0) * params.heads + flat % params.heads) * params.sq +
                   params.batchIndex + flat / params.heads;
        if constexpr (GroupedOutput)
            return ((uint64_t(params.batchIndex) * params.kvHeads + params.kvHead) * params.heads +
                    flat % params.heads) *
                       params.sq +
                   flat / params.heads;
        return (uint64_t(params.batchIndex) * params.heads + flat % params.heads) * params.sq + flat / params.heads;
    }

    __aicore__ inline void Run(uint32_t start, uint32_t rows)
    {
        auto sum = sumBuf.Get<float>(), maximum = maxBuf.Get<float>(), weights = weightBuf.Get<float>();
        auto rowMax = rowMaxBuf.Get<float>(), rowSum = rowSumBuf.Get<float>(), lse = lseBuf.Get<float>();
        const uint32_t width = rows * 8;
        for (uint32_t split = 0; split < params.splits; ++split) {
            const uint64_t offset = (uint64_t(split) * params.mBase + start) * 8;
            DataCopy(sum[split * width], sums[offset], width);
            DataCopy(maximum[split * width], maxima[offset], width);
        }
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        // Preserve native FD's two preloaded O slots. The statistics and
        // weight calculation overlap these independent GM-to-UB transfers.
        SetFlag<HardEvent::V_MTE2>(2);
        SetFlag<HardEvent::V_MTE2>(3);
        const uint32_t preload = Min(2U, params.splits);
        for (uint32_t split = 0; split < preload; ++split) {
            LoadPartial(split, start, rows);
        }
        bool any[8]{};
        for (uint32_t split = 0; split < params.splits; ++split) {
            for (uint32_t row = 0; row < rows; ++row) {
                const bool valid = Valid(split, start + row);
                any[row] = any[row] || valid;
                if (!valid) {
                    // Never infer row validity from the source's finite mask
                    // sentinel: a real logit may legitimately be that small.
                    Duplicate(sum[split * width + row * 8], 0.0f, 8);
                    Duplicate(maximum[split * width + row * 8], -3.402823466e38f, 8);
                }
            }
        }
        PipeBarrier<PIPE_V>();
        Duplicate(rowMax, -3.402823466e38f, width);
        Duplicate(rowSum, 0.0f, width);
        PipeBarrier<PIPE_V>();
        for (uint32_t s = 0; s < params.splits; s += 255) {
            Tile::ColMax(rowMax, maximum[s * width], rowMax, Min(255U, params.splits - s), width, width);
            PipeBarrier<PIPE_V>();
        }
        for (uint32_t s = 0; s < params.splits; s += 255) {
            Tile::RowSub(weights[s * width], maximum[s * width], rowMax, Min(255U, params.splits - s), width, width);
        }
        PipeBarrier<PIPE_V>();
        Exp(weights, weights, params.splits * width);
        PipeBarrier<PIPE_V>();
        Mul(weights, sum, weights, params.splits * width);
        PipeBarrier<PIPE_V>();
        for (uint32_t s = 0; s < params.splits; s += 255) {
            Tile::ColAdd(rowSum, weights[s * width], rowSum, Min(255U, params.splits - s), width, width);
            PipeBarrier<PIPE_V>();
        }
        Tile::ComputeSoftMaxLse(lse, rowSum, rowMax, rows);
        for (uint32_t row = 0; row < rows; ++row) {
            if (!any[row]) {
                Duplicate(rowSum[row * 8], 1.0f, 8);
                Duplicate(lse[row * 8], -float(1.0 / 0.0), 8);
            }
        }
        PipeBarrier<PIPE_V>();
        for (uint32_t s = 0; s < params.splits; s += 255) {
            Tile::MatDivsVec(weights[s * width], weights[s * width], rowSum, Min(255U, params.splits - s), width,
                             width);
        }
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        for (uint32_t row = 0; row < rows;) {
            const uint64_t gap = (uint64_t(params.sq) - 1) * 4;
            const uint32_t count =
                gap > UINT32_MAX ? 1 : Min(rows - row, params.heads - (params.rowStart + start + row) % params.heads);
            DataCopyPad(
                outputLse[RowAddress(start + row)], lse[row * 8],
                DataCopyExtParams{uint16_t(count), 4, 0, count == 1 ? 0 : uint32_t((uint64_t(params.sq) - 1) * 4), 0});
            row += count;
        }
        SetFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_V>(0);

        auto reduced = reduceBuf.Get<float>();
        for (uint32_t split = 0; split < params.splits; ++split) {
            const uint32_t slot = split % 2;
            if (split >= preload) {
                LoadPartial(split, start, rows);
            }
            WaitFlag<HardEvent::MTE2_V>(2 + slot);
            auto current = partialBuf[slot].Get<float>();
            for (uint32_t row = 0; row < rows; ++row) {
                if (!Valid(split, start + row)) {
                    Duplicate(current[row * HeadDim], 0.0f, HeadDim);
                }
            }
            PipeBarrier<PIPE_V>();
            if (params.adapt && (params.splits > 1 || WriteSingleAccumulator)) {
                auto s = sum[split * width], m = maximum[split * width];
                Tile::ComputeSoftMaxLse(lse, s, m, rows);
                for (uint32_t row = 0; row < rows; ++row) {
                    if (!Valid(split, start + row)) {
                        Duplicate(lse[row * 8], -float(1.0 / 0.0), 8);
                    }
                }
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE3>(0);
                WaitFlag<HardEvent::V_MTE3>(0);
                for (uint32_t row = 0; row < rows;) {
                    const uint64_t gap = (uint64_t(params.sq) - 1) * HeadDim * 4;
                    const uint32_t count =
                        gap > UINT32_MAX
                            ? 1
                            : Min(rows - row, params.heads - (params.rowStart + start + row) % params.heads);
                    const uint64_t offset = uint64_t(split) * params.batch * params.heads * params.sq *
                                                (GroupedOutput ? params.kvHeads : 1) +
                                            RowAddress(start + row);
                    DataCopyPad(accumulator[offset * HeadDim], current[row * HeadDim],
                                DataCopyExtParams{uint16_t(count), HeadDim * 4, 0, count == 1 ? 0 : uint32_t(gap), 0});
                    DataCopyPad(accumulatorLse[offset], lse[row * 8],
                                DataCopyExtParams{uint16_t(count), 4, 0,
                                                  count == 1 ? 0 : uint32_t((uint64_t(params.sq) - 1) * 4), 0});
                    row += count;
                }
                SetFlag<HardEvent::MTE3_V>(0);
                WaitFlag<HardEvent::MTE3_V>(0);
            }
            auto target = split == 0 ? reduced : current;
            Tile::RowMuls(target, current, weights[split * width], rows, HeadDim, HeadDim);
            if (split != 0) {
                PipeBarrier<PIPE_V>();
                Add(reduced, reduced, current, rows * HeadDim);
            }
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE2>(2 + slot);
        }
        WaitFlag<HardEvent::V_MTE2>(2);
        WaitFlag<HardEvent::V_MTE2>(3);
        auto cast = castBuf.Get<OutputT>();
        if constexpr (std::is_same<OutputT, bfloat16_t>::value) {
            Cast(cast, reduced, RoundMode::CAST_RINT, rows * HeadDim);
        } else if (roundToEven) {
            Cast(cast, reduced, RoundMode::CAST_RINT, rows * HeadDim);
        } else {
            Cast(cast, reduced, RoundMode::CAST_ROUND, rows * HeadDim);
        }
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        const uint64_t outRow =
            uint64_t(params.batchIndex) * (IsTnd ? 1 : params.sq) * params.heads + params.rowStart + start;
        if constexpr (GroupedOutput) {
            for (uint32_t row = 0; row < rows; ++row) {
                const uint64_t flat = uint64_t(params.rowStart) + start + row;
                const uint64_t address =
                    ((uint64_t(params.batchIndex) * (IsTnd ? 1 : params.sq) + flat / params.heads) * params.kvHeads +
                     params.kvHead) *
                        params.heads +
                    flat % params.heads;
                DataCopy(output[address * HeadDim], cast[row * HeadDim], HeadDim);
            }
        } else
            DataCopy(output[outRow * HeadDim], cast, rows * HeadDim);
        SetFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_V>(0);
    }

  private:
    __aicore__ inline void LoadPartial(uint32_t split, uint32_t start, uint32_t rows)
    {
        const uint32_t slot = split % 2;
        WaitFlag<HardEvent::V_MTE2>(2 + slot);
        DataCopy(partialBuf[slot].Get<float>(), partial[(uint64_t(split) * params.mBase + start) * HeadDim],
                 rows * HeadDim);
        SetFlag<HardEvent::MTE2_V>(2 + slot);
    }

    MlaSplitKvMergeParams params;
    bool roundToEven = false;
    GlobalTensor<float> partial, sums, maxima, outputLse, accumulator, accumulatorLse;
    GlobalTensor<OutputT> output;
    TBuf<> sumBuf, maxBuf, weightBuf, rowMaxBuf, rowSumBuf, lseBuf, partialBuf[2], reduceBuf, castBuf;
};
} // namespace flash_attn::mla::Block
