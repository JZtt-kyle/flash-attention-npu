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

constexpr uint32_t kOnlineKvStages = 3;

// Adapted from CATLASS 769cd40 block_mmad_mla_{qk,pv}.hpp. Keep its
// L1 KV sharing and Tile copies/MMADs. Dense online tasks
// have at most sixteen rows, so the complete 512+64 query fits in half
// of L0A. Load it once per task; PV exclusively uses the other half.
// For more than three KV tiles, a sixteen-row L1 query partition leaves
// room for three shared KV slots and lets the next load overlap PV.
// Shorter sequences retain the original two-slot L1 partition and schedule.
template <typename Element, bool PrefetchKv>
struct OnlineMmadTypes {
    using Row = Catlass::layout::RowMajor;
    using Column = Catlass::layout::ColumnMajor;
    using Tile = Catlass::GemmShape<PrefetchKv ? kHeadsPerTile : 128, kShortKvLimit, 576>;
    using HalfRow = Catlass::Gemm::GemmType<Element, Row>;
    using HalfColumn = Catlass::Gemm::GemmType<Element, Column>;
    using FloatRow = Catlass::Gemm::GemmType<float, Row>;
    using QK =
        Catlass::Gemm::Block::BlockMmad<Catlass::Gemm::MmadAtlasA2MLAQK, Tile, Tile, HalfRow, HalfColumn, FloatRow>;
    using PV = Catlass::Gemm::Block::BlockMmad<Catlass::Gemm::MmadAtlasA2MLAPV, Tile, Tile, HalfRow, HalfRow, FloatRow>;
    static constexpr uint32_t kKvStages = PrefetchKv ? kOnlineKvStages : 2;
    static constexpr uint32_t kQueryBytes = Tile::M * 576 * sizeof(Element);
    static constexpr uint32_t kKvBytes = kShortKvLimit * 576 * sizeof(Element);
    static constexpr uint32_t kProbabilityOffset = kQueryBytes + kKvStages * kKvBytes;
    static_assert(kProbabilityOffset + kHeadsPerTile * kShortKvLimit * sizeof(Element) <=
                      Catlass::Arch::AtlasA2::L1_SIZE,
                  "Online query, KV tiles and probabilities must fit in L1");
};

template <typename Element, bool PrefetchKv>
class CachedQueryQK : public OnlineMmadTypes<Element, PrefetchKv>::QK {
    using Base = typename OnlineMmadTypes<Element, PrefetchKv>::QK;
    static constexpr uint32_t kKvStages = OnlineMmadTypes<Element, PrefetchKv>::kKvStages;
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
    using Base::tileMmad;

  public:
    CATLASS_DEVICE
    explicit CachedQueryQK(Catlass::Arch::Resource<Catlass::Arch::AtlasA2>& resource) : Base(resource)
    {
        for (uint32_t slot = 0; slot < kKvStages; ++slot) {
            kvInL1[slot] =
                resource.l1Buf.template GetBufferByByte<Element>(OnlineMmadTypes<Element, PrefetchKv>::kQueryBytes +
                                                                 slot * OnlineMmadTypes<Element, PrefetchKv>::kKvBytes);
        }
    }

    static_assert(kHeadsPerTile <= 16 && 16 * 576 * sizeof(Element) <= Base::L0A_PINGPONG_BUF_SIZE,
                  "Cached Q needs one sixteen-row fractal within its reserved L0A half");

    CATLASS_DEVICE
    void Prefetch(AscendC::GlobalTensor<ElementB> gB, AscendC::GlobalTensor<ElementB> gBRope, LayoutB layoutB,
                  LayoutB layoutBRope, uint32_t tile)
    {
        const uint32_t slot = tile % kKvStages;
        const uint32_t columns = layoutB.shape(1);
        const uint32_t content = layoutB.shape(0);
        const uint32_t stride = ::RoundUp<L1BAlignHelper::N_ALIGNED>(columns);
        // Content stays live through PV; RoPE is released by QK. A third
        // slot lets both loads issue before waiting for the previous PV;
        // the two-slot schedule calls this immediately before its QK.
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(slot);
        auto contentInL1 = LayoutBInL1::template MakeLayout<ElementB>(content, columns);
        copyGmToL1B(kvInL1[slot], gB, contentInL1, layoutB);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(slot);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(slot + kKvStages);
        auto ropeInL1 = LayoutBInL1::template MakeLayout<ElementB>(layoutBRope.shape(0), columns);
        copyGmToL1B(kvInL1[slot][stride * content], gBRope, ropeInL1, layoutBRope);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(slot + kKvStages);
    }

    CATLASS_DEVICE
    void LoadQuery(AscendC::GlobalTensor<ElementA> gA, AscendC::GlobalTensor<ElementA> gARope, LayoutA layoutA,
                   LayoutA layoutARope, uint32_t qHeads)
    {
        const uint32_t rowNum = layoutA.shape(0);
        const uint32_t curHeadNum = rowNum;
        const uint32_t tokenNumPerHead = 1;
        const uint32_t embed = layoutA.shape(1);
        const uint32_t embedRope = layoutARope.shape(1);
        const uint32_t embedCat = embed + embedRope;
        const uint32_t rowNumRound = ::RoundUp<L1AAlignHelper::M_ALIGNED>(rowNum);
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

        // Load Q before the first KV tile so its L1-to-L0A transfer can
        // overlap that prefetch. The three-slot path reserves ids 0..5 for KV.
        constexpr uint32_t queryReady = PrefetchKv ? EVENT_ID6 : EVENT_ID0;
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(queryReady);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(queryReady);
        // With one sixteen-row fractal, the task's content and RoPE
        // remain adjacent when copying L1 zN to L0A zZ. The task drain
        // orders this reload after every previous QK/PV read.
        LayoutAInL1 queryInL1 = LayoutAInL1::template MakeLayout<ElementA>(rowNum, embedCat);
        LayoutAInL0 queryInL0 = LayoutAInL0::template MakeLayout<ElementA>(rowNum, embedCat);
        copyL1ToL0A(l0ATensor[0], l1ATensor, queryInL0, queryInL1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
    }

    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<ElementC> gC, LayoutB layoutB, LayoutB layoutBRope, LayoutC layoutC,
                    Catlass::GemmCoord actualShape, uint32_t nIdx)
    {
        uint32_t rowNum = actualShape.m();
        uint32_t kSeqTile = actualShape.n();
        uint32_t embedRope = layoutBRope.shape(0);
        uint32_t embedCat = actualShape.k();
        uint32_t embedSplitSize = EMBED_SPLIT_SIZE;
        uint32_t embedSplitLoopK = EMBED_SPLIT_LOOP;
        uint32_t kSeqTileRound = ::RoundUp<L1BAlignHelper::N_ALIGNED>(kSeqTile);
        uint32_t rowNumRound = ::RoundUp<L1AAlignHelper::M_ALIGNED>(rowNum);
        uint32_t kvSlot = nIdx % kKvStages;
        uint32_t l0CPingPongFlag = nIdx % 2;

        for (uint32_t embedSplitIdx = 0; embedSplitIdx < embedSplitLoopK; embedSplitIdx++) {
            uint32_t l0ABPingPongFlag = embedSplitIdx % 2;
            if (embedSplitIdx == embedSplitLoopK - 1) {
                embedSplitSize = embedRope;
            }
            if (embedSplitIdx == 0) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(kvSlot);
            } else if (embedSplitIdx == embedSplitLoopK - 1) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(kvSlot + kKvStages);
            }
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag + 2);
            // copy K from L1 to l0b
            LayoutBInL1 layoutBCatInL1 = LayoutBInL1::template MakeLayout<ElementB>(embedCat, kSeqTile);
            LayoutBInL0 layoutBCatInL0 = LayoutBInL0::template MakeLayout<ElementB>(embedSplitSize, kSeqTile);
            copyL1ToL0B(l0BTensor[l0ABPingPongFlag], kvInL1[kvSlot][embedSplitIdx * kSeqTileRound * EMBED_SPLIT_SIZE],
                        layoutBCatInL0, layoutBCatInL1);
            if (embedSplitIdx == embedSplitLoopK - 1) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(kvSlot + kKvStages);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(l0ABPingPongFlag + 2);
            // mmad
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(l0ABPingPongFlag + 2);
            if (embedSplitIdx == 0) {
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(l0CPingPongFlag);
            }
            // mmad
            tileMmad(l0CTensor[l0CPingPongFlag], l0ATensor[0][embedSplitIdx * rowNumRound * EMBED_SPLIT_SIZE],
                     l0BTensor[l0ABPingPongFlag], rowNumRound, kSeqTile, embedSplitSize, embedSplitIdx == 0);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPongFlag + 2);
        }
        // copy S to gm
        AscendC::SetFlag<AscendC::HardEvent::M_FIX>(l0CPingPongFlag);
        AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(l0CPingPongFlag);
        auto blockShape = Catlass::MakeCoord(rowNum, kSeqTile);
        auto layoutInL0C = LayoutCInL0::MakeLayoutInL0C(blockShape);
        // copy L0C to gm
        copyL0CToGm(gC, l0CTensor[l0CPingPongFlag], layoutC, layoutInL0C);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(l0CPingPongFlag);
    }

  private:
    AscendC::LocalTensor<Element> kvInL1[kKvStages];
};

template <typename Element, bool PrefetchKv>
class CachedQueryPV : public OnlineMmadTypes<Element, PrefetchKv>::PV {
    using Base = typename OnlineMmadTypes<Element, PrefetchKv>::PV;
    static constexpr uint32_t kKvStages = OnlineMmadTypes<Element, PrefetchKv>::kKvStages;
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
    using Base::tileMmad;

  public:
    CATLASS_DEVICE
    explicit CachedQueryPV(Catlass::Arch::Resource<Catlass::Arch::AtlasA2>& resource) : Base(resource)
    {
        l1ATensor =
            resource.l1Buf.template GetBufferByByte<Element>(OnlineMmadTypes<Element, PrefetchKv>::kProbabilityOffset);
        for (uint32_t slot = 0; slot < kKvStages; ++slot) {
            kvInL1[slot] =
                resource.l1Buf.template GetBufferByByte<Element>(OnlineMmadTypes<Element, PrefetchKv>::kQueryBytes +
                                                                 slot * OnlineMmadTypes<Element, PrefetchKv>::kKvBytes);
        }
    }

    static_assert(kHeadsPerTile <= 16 && 16 * 576 * sizeof(Element) <= Base::L0A_PINGPONG_BUF_SIZE,
                  "Cached Q needs one sixteen-row fractal within its reserved L0A half");
    static_assert(16 * 512 * sizeof(ElementC) <= Base::L0C_PINGPONG_BUF_SIZE,
                  "The complete PV output must fit within one L0C half");

    CATLASS_DEVICE
    void operator()(AscendC::GlobalTensor<ElementA> gA, AscendC::GlobalTensor<ElementC> gC, LayoutA layoutA,
                    LayoutB layoutB, LayoutC layoutC, Catlass::GemmCoord actualShape, uint32_t& nIdx,
                    Catlass::Arch::CrossCoreFlag softmaxReady, uint32_t& locPingPongIdx)
    {
        uint32_t rowNum = actualShape.m();
        uint32_t vSeqTile = actualShape.k();
        uint32_t embed = actualShape.n();
        uint32_t embedSplitSize = EMBED_SPLIT_SIZE;
        uint32_t embedSplitLoopV = EMBED_SPLIT_LOOP;
        uint32_t rowNumRound = ::RoundUp<L1AAlignHelper::M_ALIGNED>(rowNum);
        uint32_t embedSplitSizeRound = ::RoundUp<L1BAlignHelper::N_ALIGNED>(embedSplitSize);
        uint32_t vSeqTileRound = ::RoundUp<L1BAlignHelper::K_ALIGNED>(vSeqTile);
        uint32_t kvSlot = (nIdx - 1) % kKvStages;
        // QK retains its query in L0A[0] until the task drains.
        constexpr uint32_t L0APingPongFlag = 1;
        const uint32_t L0CPingPongFlag = locPingPongIdx % 2;

        for (uint32_t embedSplitIdx = 0; embedSplitIdx < embedSplitLoopV; embedSplitIdx++) {
            uint32_t L0BPingPongFlag = (embedSplitIdx + 1) % 2;
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(L0BPingPongFlag + 2);
            LayoutBInL1 layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(vSeqTile, embed);
            LayoutBInL0 layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(vSeqTile, embedSplitSize);
            // copy V from L1 to L0B
            copyL1ToL0B(l0BTensor[L0BPingPongFlag], kvInL1[kvSlot][embedSplitIdx * vSeqTileRound * EMBED_SPLIT_SIZE],
                        layoutBInL0, layoutBInL1);
            if (embedSplitIdx == embedSplitLoopV - 1) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(kvSlot);
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

  private:
    AscendC::LocalTensor<Element> kvInL1[kKvStages];
};

} // namespace flash_attn::mla
