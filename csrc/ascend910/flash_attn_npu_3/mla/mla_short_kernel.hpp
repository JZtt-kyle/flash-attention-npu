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

#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "mla_short_epilogue.hpp"

namespace flash_attn::mla {

// Adapted from CATLASS 769cd40 examples/19_mla/mla_kernel.cpp. The two
// dot products accumulate in one L0C tile, and PV reuses the content KV in L1.
// This is the shared-KV subset of pinned CANN ca18aee's arch22 MLA algorithm
// (fia_block_cube_nonquant_mla.h); no FIA launch or tiling-key machinery is used.
template <typename Element>
class ShortKernel {
    using Row = Catlass::layout::RowMajor;
    using Column = Catlass::layout::ColumnMajor;
    using HalfRow = Catlass::Gemm::GemmType<Element, Row>;
    using HalfColumn = Catlass::Gemm::GemmType<Element, Column>;
    using FloatRow = Catlass::Gemm::GemmType<float, Row>;
    using Tile = Catlass::GemmShape<128, 128, 576>;
    using QK =
        Catlass::Gemm::Block::BlockMmad<Catlass::Gemm::MmadAtlasA2MLAQK, Tile, Tile, HalfRow, HalfColumn, FloatRow>;
    using PV = Catlass::Gemm::Block::BlockMmad<Catlass::Gemm::MmadAtlasA2MLAPV, Tile, Tile, HalfRow, HalfRow, FloatRow>;

  public:
    CATLASS_DEVICE
    void operator()(const ShortParams& p)
    {
        Catlass::Arch::Resource<Catlass::Arch::AtlasA2> resource;
        AscendC::GlobalTensor<Element> q, k, v, qv, out, probabilities;
        AscendC::GlobalTensor<float> scores, partial, lse;
        q.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(p.q));
        k.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(p.k));
        v.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(p.v));
        qv.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(p.qv));
        out.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(p.out));
        lse.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(p.lse));
        uint32_t core = AscendC::GetBlockIdx();
#ifdef __DAV_VEC__
        core /= AscendC::GetSubBlockNum();
        ShortEpilogue<Element> epilogue(resource);
#endif
        const uint64_t workspaceOffset = uint64_t(core) * kWorkspaceBytesPerCore;
        scores.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(p.workspace + workspaceOffset));
        probabilities.SetGlobalBuffer(reinterpret_cast<__gm__ Element*>(p.workspace + workspaceOffset + kScoreBytes));
        partial.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(p.workspace + workspaceOffset + kScoreBytes + kProbabilityBytes));
        Catlass::Arch::CrossCoreFlag qkReady(1), softmaxReady(2), pvReady(3), outputReady(4);
#ifdef __DAV_CUBE__
        QK qk(resource);
        PV pv(resource);
        for (uint32_t event = 0; event < 4; ++event) {
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(event);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(event);
        }
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
#endif
        const uint32_t headTiles = ::CeilDiv(p.heads, kHeadsPerTile);
        for (uint64_t task = core; task < p.tasks; task += AscendC::GetBlockNum()) {
            const uint64_t token = task / headTiles;
            const uint32_t batch = token / p.queryLength;
            const uint32_t query = token % p.queryLength;
            const uint32_t firstHead = (task % headTiles) * kHeadsPerTile;
            const uint32_t rows = ::Min(kHeadsPerTile, p.heads - firstHead);
            const uint32_t columns = ::RoundUp<16>(p.kvLength);
            const uint64_t queryRow = token * p.heads + firstHead;
#ifdef __DAV_CUBE__
            uint32_t n = 0, pingpong = 0, localPingpong = 0, heads = p.heads;
            qk(qv[queryRow * 512], q[queryRow * 64], v[uint64_t(batch) * p.kvLength * 512],
               k[uint64_t(batch) * p.kvLength * 64], scores, Row(rows, 512), Row(rows, 64), Column(512, p.kvLength),
               Column(64, p.kvLength), Row(::RoundUp<16>(rows), columns), Catlass::GemmCoord(rows, p.kvLength, 576),
               Catlass::MatrixCoord(rows, 512), heads, n, pingpong);
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady);
            n = 1;
            pingpong = 1;
            pv(probabilities, partial, Row(rows, p.kvLength, columns), Row(512, p.kvLength),
               Row(::RoundUp<16>(rows), 512), Catlass::GemmCoord(rows, 512, p.kvLength), n, pingpong, softmaxReady,
               localPingpong);
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(pvReady);
            // Every task owns its per-core workspace until both vector lanes
            // finish. This also bounds the cross-core flag counter.
            Catlass::Arch::CrossCoreWaitFlag(outputReady);
#endif
#ifdef __DAV_VEC__
            const uint32_t lanes = AscendC::GetSubBlockNum();
            const uint32_t lane = AscendC::GetSubBlockIdx();
            const uint32_t firstRow = lane * (rows / lanes);
            const uint32_t laneRows = lane + 1 == lanes ? rows - firstRow : rows / lanes;
            Catlass::Arch::CrossCoreWaitFlag(qkReady);
            const uint64_t lseOffset = (uint64_t(batch) * p.heads + firstHead + firstRow) * p.queryLength + query;
            epilogue.Softmax(scores[firstRow * columns], probabilities[firstRow * columns], lse[lseOffset], laneRows,
                             p.kvLength, p.queryLength, p.scale);
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(softmaxReady);
            Catlass::Arch::CrossCoreWaitFlag(pvReady);
            epilogue.Store(partial[firstRow * 512], out[(queryRow + firstRow) * 512], laneRows);
            Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(outputReady);
#endif
        }
#ifdef __DAV_CUBE__
        for (uint32_t event = 0; event < 4; ++event) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(event);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(event);
        }
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
#endif
    }
};

template <typename Element>
CATLASS_GLOBAL void MlaShort(uint64_t syncAddress, ShortParams params)
{
    AscendC::SetSyncBaseAddr(syncAddress);
    ShortKernel<Element> kernel;
    kernel(params);
}

template <typename Element>
void LaunchShort(const ShortParams& params, uint32_t blocks, uint64_t syncAddress, aclrtStream stream)
{
    MlaShort<Element><<<blocks, nullptr, stream>>>(syncAddress, params);
}

} // namespace flash_attn::mla
