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

#include "catlass/gemm/block/block_mmad.hpp"
#include "mla_launch.h"

namespace flash_attn::mla {

// Adapted from CATLASS 769cd40 block_mmad_mla_{qk,pv}.hpp. Keep its
// L1 KV sharing, Tile copies/MMADs and event protocol. Dense online tasks
// have at most sixteen rows, so the complete 512+64 query fits in half
// of L0A. Load it once per task; PV exclusively uses the other half.
template <typename Element>
struct OnlineMmadTypes {
    using Row = Catlass::layout::RowMajor;
    using Column = Catlass::layout::ColumnMajor;
    using Tile = Catlass::GemmShape<128, 128, 576>;
    using HalfRow = Catlass::Gemm::GemmType<Element, Row>;
    using HalfColumn = Catlass::Gemm::GemmType<Element, Column>;
    using FloatRow = Catlass::Gemm::GemmType<float, Row>;
    using QK =
        Catlass::Gemm::Block::BlockMmad<Catlass::Gemm::MmadAtlasA2MLAQK, Tile, Tile, HalfRow, HalfColumn, FloatRow>;
    using PV = Catlass::Gemm::Block::BlockMmad<Catlass::Gemm::MmadAtlasA2MLAPV, Tile, Tile, HalfRow, HalfRow, FloatRow>;
};

template <typename Element>
class CachedQueryQK : public OnlineMmadTypes<Element>::QK {
    using Base = typename OnlineMmadTypes<Element>::QK;
    using ElementA = typename Base::ElementA;
    using ElementB = typename Base::ElementB;
    using ElementC = typename Base::ElementC;
    using LayoutA = typename Base::LayoutA;
    using LayoutB = typename Base::LayoutB;
    using LayoutC = typename Base::LayoutC;
    using LayoutAInL1 = typename Base::LayoutAInL1;
    using LayoutAInL0 = typename Base::LayoutAInL0;
    using LayoutBInL1 = typename Base::LayoutBInL1;
    using LayoutBInL0 = typename Base::LayoutBInL0;
    using LayoutCInL0 = typename Base::LayoutCInL0;
    using L1AAlignHelper = typename Base::L1AAlignHelper;
    using L1BAlignHelper = typename Base::L1BAlignHelper;
    using Base::BLOCK_SIZE;
    using Base::copyGmToL1A;
    using Base::copyGmToL1B;
    using Base::copyL0CToGm;
    using Base::copyL1ToL0A;
    using Base::copyL1ToL0B;
    using Base::EMBED_SPLIT_LOOP;
    using Base::EMBED_SPLIT_SIZE;
    using Base::l0ATensor;
    using Base::l0BTensor;
    using Base::l0CTensor;
    using Base::l1ATensor;
    using Base::l1BTensor;
    using Base::tileMmad;

  public:
    using Base::Base;
    static_assert(kHeadsPerTile <= 16 && 16 * 576 * sizeof(Element) <= Base::L0A_PINGPONG_BUF_SIZE,
                  "Cached Q needs one sixteen-row fractal within its reserved L0A half");

    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<ElementA> gA, AscendC::GlobalTensor<ElementA> gARope,
                    AscendC::GlobalTensor<ElementB> gB, AscendC::GlobalTensor<ElementB> gBRope,
                    AscendC::GlobalTensor<ElementC> gC, LayoutA layoutA, LayoutA layoutARope, LayoutB layoutB,
                    LayoutB layoutBRope, LayoutC layoutC, Catlass::GemmCoord actualShape,
                    Catlass::MatrixCoord qShapeSingleNd, uint32_t& qHeads, uint32_t& nIdx, uint32_t& pingpongIdx)
    {
        uint32_t rowNum = actualShape.m();
        uint32_t kSeqTile = actualShape.n();
        uint32_t embed = layoutB.shape(0);
        uint32_t embedRope = layoutBRope.shape(0);
        uint32_t embedCat = actualShape.k();
        uint32_t embedSplitSize = EMBED_SPLIT_SIZE;
        uint32_t embedSplitLoopK = EMBED_SPLIT_LOOP;
        uint32_t curHeadNum = qShapeSingleNd.row();
        uint32_t tokenNumPerHead = rowNum / curHeadNum;
        uint32_t kSeqTileRound = ::RoundUp<L1BAlignHelper::N_ALIGNED>(kSeqTile);
        uint32_t rowNumRound = ::RoundUp<L1AAlignHelper::M_ALIGNED>(rowNum);
        uint32_t l1KvPingPongFlag = pingpongIdx % 2;

        if (nIdx == 0) {
            // copy Q to L1
            auto layoutASingleNd = layoutA.GetTileLayout(Catlass::MakeCoord(curHeadNum, embed));
            LayoutAInL1 layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(rowNum, embed);
            copyGmToL1A(l1ATensor, gA, layoutAInL1, layoutASingleNd, tokenNumPerHead, qHeads * embed, tokenNumPerHead,
                        BLOCK_SIZE, rowNumRound);

            // copy QRope to L1
            auto layoutARopeSingleNd = layoutARope.GetTileLayout(Catlass::MakeCoord(curHeadNum, embedRope));
            LayoutAInL1 layoutARopeInL1 = LayoutAInL1::template MakeLayout<ElementA>(rowNum, embedRope);
            copyGmToL1A(l1ATensor[rowNumRound * embed], gARope, layoutARopeInL1, layoutARopeSingleNd, tokenNumPerHead,
                        qHeads * embedRope, tokenNumPerHead, BLOCK_SIZE, rowNumRound);

            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
            // With one sixteen-row fractal, the task's content and RoPE
            // remain adjacent when copying L1 zN to L0A zZ. The task drain
            // orders this reload after every previous QK/PV read.
            LayoutAInL1 queryInL1 = LayoutAInL1::template MakeLayout<ElementA>(rowNum, embedCat);
            LayoutAInL0 queryInL0 = LayoutAInL0::template MakeLayout<ElementA>(rowNum, embedCat);
            copyL1ToL0A(l0ATensor[0], l1ATensor, queryInL0, queryInL1);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1KvPingPongFlag);
        // copy K to L1
        LayoutBInL1 layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(embed, kSeqTile);
        copyGmToL1B(l1BTensor[l1KvPingPongFlag], gB, layoutBInL1, layoutB);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1KvPingPongFlag);

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(l1KvPingPongFlag + 2);
        // copy KRope to L1
        LayoutBInL1 layoutBRopeInL1 = LayoutBInL1::template MakeLayout<ElementB>(embedRope, kSeqTile);
        copyGmToL1B(l1BTensor[l1KvPingPongFlag][kSeqTileRound * embed], gBRope, layoutBRopeInL1, layoutBRope);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(l1KvPingPongFlag + 2);

        for (uint32_t embedSplitIdx = 0; embedSplitIdx < embedSplitLoopK; embedSplitIdx++) {
            uint32_t l0ABPingPongFlag = embedSplitIdx % 2;
            if (embedSplitIdx == embedSplitLoopK - 1) {
                embedSplitSize = embedRope;
            }
            if (embedSplitIdx == 0) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1KvPingPongFlag);
            } else if (embedSplitIdx == embedSplitLoopK - 1) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(l1KvPingPongFlag + 2);
            }
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag + 2);
            // copy K from L1 to l0b
            LayoutBInL1 layoutBCatInL1 = LayoutBInL1::template MakeLayout<ElementB>(embedCat, kSeqTile);
            LayoutBInL0 layoutBCatInL0 = LayoutBInL0::template MakeLayout<ElementB>(embedSplitSize, kSeqTile);
            copyL1ToL0B(l0BTensor[l0ABPingPongFlag],
                        l1BTensor[l1KvPingPongFlag][embedSplitIdx * kSeqTileRound * EMBED_SPLIT_SIZE], layoutBCatInL0,
                        layoutBCatInL1);
            if (embedSplitIdx == embedSplitLoopK - 1) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(l1KvPingPongFlag + 2);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0ABPingPongFlag + 2);
            // mmad
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0ABPingPongFlag + 2);
            if (embedSplitIdx == 0) {
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l1KvPingPongFlag);
            }
            // mmad
            tileMmad(l0CTensor[l1KvPingPongFlag], l0ATensor[0][embedSplitIdx * rowNumRound * EMBED_SPLIT_SIZE],
                     l0BTensor[l0ABPingPongFlag], rowNumRound, kSeqTile, embedSplitSize, embedSplitIdx == 0);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag + 2);
        }
        // copy S to gm
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l1KvPingPongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l1KvPingPongFlag);
        auto blockShape = Catlass::MakeCoord(rowNum, kSeqTile);
        auto layoutInL0C = LayoutCInL0::MakeLayoutInL0C(blockShape);
        // copy L0C to gm
        copyL0CToGm(gC, l0CTensor[l1KvPingPongFlag], layoutC, layoutInL0C);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l1KvPingPongFlag);
    }
};

template <typename Element>
class CachedQueryPV : public OnlineMmadTypes<Element>::PV {
    using Base = typename OnlineMmadTypes<Element>::PV;
    using ElementA = typename Base::ElementA;
    using ElementB = typename Base::ElementB;
    using ElementC = typename Base::ElementC;
    using LayoutA = typename Base::LayoutA;
    using LayoutB = typename Base::LayoutB;
    using LayoutC = typename Base::LayoutC;
    using LayoutAInL1 = typename Base::LayoutAInL1;
    using LayoutAInL0 = typename Base::LayoutAInL0;
    using LayoutBInL1 = typename Base::LayoutBInL1;
    using LayoutBInL0 = typename Base::LayoutBInL0;
    using LayoutCInL0 = typename Base::LayoutCInL0;
    using L1AAlignHelper = typename Base::L1AAlignHelper;
    using L1BAlignHelper = typename Base::L1BAlignHelper;
    using Base::copyGmToL1A;
    using Base::copyL0CToGm;
    using Base::copyL1ToL0A;
    using Base::copyL1ToL0B;
    using Base::EMBED_SPLIT_LOOP;
    using Base::EMBED_SPLIT_SIZE;
    using Base::l0ATensor;
    using Base::l0BTensor;
    using Base::l0CTensor;
    using Base::l1ATensor;
    using Base::l1BTensor;
    using Base::tileMmad;

  public:
    using Base::Base;
    static_assert(kHeadsPerTile <= 16 && 16 * 576 * sizeof(Element) <= Base::L0A_PINGPONG_BUF_SIZE,
                  "Cached Q needs one sixteen-row fractal within its reserved L0A half");
    static_assert(16 * 512 * sizeof(ElementC) <= Base::L0C_PINGPONG_BUF_SIZE,
                  "The complete PV output must fit within one L0C half");

    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<ElementA> gA, AscendC::GlobalTensor<ElementC> gC, LayoutA layoutA,
                    LayoutB layoutB, LayoutC layoutC, Catlass::GemmCoord actualShape, uint32_t& nIdx,
                    uint32_t& pingpongIdx, Catlass::Arch::CrossCoreFlag softmaxReady, uint32_t& locPingPongIdx)
    {
        uint32_t rowNum = actualShape.m();
        uint32_t vSeqTile = actualShape.k();
        uint32_t embed = actualShape.n();
        uint32_t embedSplitSize = EMBED_SPLIT_SIZE;
        uint32_t embedSplitLoopV = EMBED_SPLIT_LOOP;
        uint32_t rowNumRound = ::RoundUp<L1AAlignHelper::M_ALIGNED>(rowNum);
        uint32_t embedSplitSizeRound = ::RoundUp<L1BAlignHelper::N_ALIGNED>(embedSplitSize);
        uint32_t vSeqTileRound = ::RoundUp<L1BAlignHelper::K_ALIGNED>(vSeqTile);
        uint32_t L1BPingPongFlag = (pingpongIdx - 1) % 2;
        // QK retains its query in L0A[0] until the task drains.
        constexpr uint32_t L0APingPongFlag = 1;
        const uint32_t L0CPingPongFlag = locPingPongIdx % 2;

        for (uint32_t embedSplitIdx = 0; embedSplitIdx < embedSplitLoopV; embedSplitIdx++) {
            uint32_t L0BPingPongFlag = (embedSplitIdx + 1) % 2;
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0BPingPongFlag + 2);
            LayoutBInL1 layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(vSeqTile, embed);
            LayoutBInL0 layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(vSeqTile, embedSplitSize);
            // copy V from L1 to L0B
            copyL1ToL0B(l0BTensor[L0BPingPongFlag],
                        l1BTensor[L1BPingPongFlag][embedSplitIdx * vSeqTileRound * EMBED_SPLIT_SIZE], layoutBInL0,
                        layoutBInL1);
            if (embedSplitIdx == embedSplitLoopV - 1) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(L1BPingPongFlag);
            }

            if (embedSplitIdx == 0) {
                LayoutAInL1 layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(rowNum, vSeqTile);
                Catlass::Arch::CrossCoreWaitFlag(softmaxReady);
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
                // copy P to L1
                copyGmToL1A(l1ATensor, gA, layoutAInL1, layoutA);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID7);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID7);
                // move p from l1 to l0a
                AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0APingPongFlag);
                LayoutAInL0 layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(rowNum, vSeqTile);
                copyL1ToL0A(l0ATensor[L0APingPongFlag], l1ATensor, layoutAInL0, layoutAInL1);
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
            }

            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(L0BPingPongFlag);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(L0BPingPongFlag);
            if (embedSplitIdx == 0) {
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(L0CPingPongFlag);
            }
            // mmad
            // Adjacent column slices are adjacent in zN at this row stride.
            // Keep all four independent MMAD results in one L0C slot so FIX
            // can write the complete 512-column partial in a single copy.
            tileMmad(l0CTensor[L0CPingPongFlag][embedSplitIdx * rowNumRound * embedSplitSizeRound],
                     l0ATensor[L0APingPongFlag], l0BTensor[L0BPingPongFlag], rowNumRound, embedSplitSize, vSeqTile);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0BPingPongFlag + 2);
            if (embedSplitIdx == embedSplitLoopV - 1) {
                AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(L0APingPongFlag);
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(L0CPingPongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(L0CPingPongFlag);
        auto layoutInL0C = LayoutCInL0::MakeLayoutInL0C(Catlass::MakeCoord(rowNum, embed));
        copyL0CToGm(gC, l0CTensor[L0CPingPongFlag], layoutC, layoutInL0C);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(L0CPingPongFlag);
        locPingPongIdx++;
    }
};

} // namespace flash_attn::mla
