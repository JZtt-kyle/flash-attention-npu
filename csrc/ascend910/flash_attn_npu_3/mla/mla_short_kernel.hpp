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
#include "mla_online_epilogue.hpp"
#include "mla_online_mmad.hpp"

namespace flash_attn::mla {

// Adapted from CATLASS 769cd40 examples/19_mla/mla_kernel.cpp. The two
// dot products accumulate in one L0C tile, and PV reuses the content KV in L1.
// This is the shared-KV subset of pinned CANN ca18aee's arch22 MLA algorithm
// (fia_block_cube_nonquant_mla.h); no FIA launch or tiling-key machinery is used.
template <typename Element, bool Online = false, bool PrefetchKv = false>
class DenseKernel {
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
    void operator()(const DenseParams& p)
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
        std::conditional_t<Online, OnlineEpilogue<Element>, ShortEpilogue<Element>> epilogue(resource);
#endif
        constexpr uint32_t stages = Online ? 2 : 1;
        constexpr uint32_t workspaceBytes = Online ? kOnlineWorkspaceBytesPerCore : kWorkspaceBytesPerCore;
        const uint64_t workspaceOffset = uint64_t(core) * workspaceBytes;
        scores.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(p.workspace + workspaceOffset));
        probabilities.SetGlobalBuffer(
            reinterpret_cast<__gm__ Element*>(p.workspace + workspaceOffset + stages * kScoreBytes));
        partial.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(p.workspace + workspaceOffset +
                                                                stages * (kScoreBytes + kProbabilityBytes)));
        Catlass::Arch::CrossCoreFlag qkReady(1), softmaxReady(2), pvReady(3), outputReady(4);
#ifdef __DAV_CUBE__
        std::conditional_t<Online, CachedQueryQK<Element, PrefetchKv>, QK> qk(resource);
        std::conditional_t<Online, CachedQueryPV<Element, PrefetchKv>, PV> pv(resource);
        for (uint32_t event = 0; event < 4; ++event) {
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(event);
            if constexpr (!Online || !PrefetchKv) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(event);
            }
        }
        if constexpr (Online && PrefetchKv) {
            for (uint32_t event = 0; event < 2 * kOnlineKvStages; ++event) {
                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(event);
            }
        }
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
#endif
        const uint32_t headsPerTask = Online ? p.headsPerTask : kHeadsPerTile;
        const uint32_t headTiles = ::CeilDiv(p.heads, headsPerTask);
        for (uint64_t task = core; task < p.tasks; task += AscendC::GetBlockNum()) {
            const uint64_t token = task / headTiles;
            const uint32_t batch = token / p.queryLength;
            const uint32_t query = token % p.queryLength;
            const uint32_t firstHead = (task % headTiles) * headsPerTask;
            const uint32_t rows = ::Min(headsPerTask, p.heads - firstHead);
            const uint32_t columns = ::RoundUp<16>(p.kvLength);
            const uint64_t queryRow = token * p.heads + firstHead;
#ifdef __DAV_VEC__
            const uint32_t lanes = AscendC::GetSubBlockNum();
            const uint32_t lane = AscendC::GetSubBlockIdx();
            const uint32_t firstRow = lane * (rows / lanes);
            const uint32_t laneRows = lane + 1 == lanes ? rows - firstRow : rows / lanes;
            const uint64_t lseOffset = (uint64_t(batch) * p.heads + firstHead + firstRow) * p.queryLength + query;
#endif
            if constexpr (!Online) {
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
                Catlass::Arch::CrossCoreWaitFlag(qkReady);
                epilogue.Softmax(scores[firstRow * columns], probabilities[firstRow * columns], lse[lseOffset],
                                 laneRows, p.kvLength, p.queryLength, p.scale);
                Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(softmaxReady);
                Catlass::Arch::CrossCoreWaitFlag(pvReady);
                epilogue.Store(partial[firstRow * 512], out[(queryRow + firstRow) * 512], laneRows);
                Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(outputReady);
#endif
            } else {
                const uint32_t tiles = ::CeilDiv(p.kvLength, kShortKvLimit);
#ifdef __DAV_CUBE__
                uint32_t localPingpong = 0;
                qk.LoadQuery(qv[queryRow * 512], q[queryRow * 64], Row(rows, 512), Row(rows, 64), p.heads);
                if constexpr (PrefetchKv) {
                    qk.Prefetch(v[uint64_t(batch) * p.kvLength * 512], k[uint64_t(batch) * p.kvLength * 64],
                                Column(512, kShortKvLimit), Column(64, kShortKvLimit), 0);
                }
                // Prime KV(0), then overlap QK(i+1) with softmax(i), and
                // softmax(i+1) with PV(i). Three L1 slots also let KV(i+2)
                // load before PV(i) waits for its probabilities. Q stays in L0A.
                for (uint32_t step = 0; step <= tiles; ++step) {
                    if (step < tiles) {
                        const uint32_t width = ::Min(kShortKvLimit, p.kvLength - step * kShortKvLimit);
                        if constexpr (!PrefetchKv) {
                            const uint64_t kvRow = uint64_t(batch) * p.kvLength + step * kShortKvLimit;
                            qk.Prefetch(v[kvRow * 512], k[kvRow * 64], Column(512, width), Column(64, width), step);
                        }
                        qk(scores[(step % 2) * kHeadsPerTile * kShortKvLimit], Column(512, width), Column(64, width),
                           Row(::RoundUp<16>(rows), ::RoundUp<16>(width)), Catlass::GemmCoord(rows, width, 576), step);
                        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady);
                        if (PrefetchKv && step + 1 < tiles) {
                            const uint32_t nextWidth = ::Min(kShortKvLimit, p.kvLength - (step + 1) * kShortKvLimit);
                            const uint64_t nextRow = uint64_t(batch) * p.kvLength + (step + 1) * kShortKvLimit;
                            qk.Prefetch(v[nextRow * 512], k[nextRow * 64], Column(512, nextWidth),
                                        Column(64, nextWidth), step + 1);
                        }
                    }
                    if (step > 0) {
                        const uint32_t width = ::Min(kShortKvLimit, p.kvLength - (step - 1) * kShortKvLimit);
                        // Two partial slots allow PV(i+1) and QK(i+2) to run
                        // while Vector accumulates PV(i). Consume the oldest
                        // completion before reusing that slot, then drain the
                        // remaining completions before the next task.
                        if (step > 2) {
                            Catlass::Arch::CrossCoreWaitFlag(outputReady);
                        }
                        pv(probabilities[((step - 1) % 2) * kHeadsPerTile * kShortKvLimit],
                           partial[((step - 1) % 2) * kHeadsPerTile * 512], Row(rows, width, ::RoundUp<16>(width)),
                           Row(512, width), Row(::RoundUp<16>(rows), 512), Catlass::GemmCoord(rows, 512, width), step,
                           softmaxReady, localPingpong);
                        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(pvReady);
                    }
                }
                for (uint32_t pending = 0; pending < ::Min(tiles, 2U); ++pending) {
                    Catlass::Arch::CrossCoreWaitFlag(outputReady);
                }
#endif
#ifdef __DAV_VEC__
                for (uint32_t step = 0; step <= tiles; ++step) {
                    if (step < tiles) {
                        const uint32_t width = ::Min(kShortKvLimit, p.kvLength - step * kShortKvLimit);
                        const uint32_t offset =
                            (step % 2) * kHeadsPerTile * kShortKvLimit + firstRow * ::RoundUp<16>(width);
                        Catlass::Arch::CrossCoreWaitFlag(qkReady);
                        epilogue.Softmax(scores[offset], probabilities[offset], laneRows, width, p.scale, step);
                        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(softmaxReady);
                    }
                    if (step > 0) {
                        epilogue.RescaleNumerator(laneRows, step - 1);
                        Catlass::Arch::CrossCoreWaitFlag(pvReady);
                        epilogue.Accumulate(partial[(((step - 1) % 2) * kHeadsPerTile + firstRow) * 512],
                                            out[(queryRow + firstRow) * 512], lse[lseOffset], laneRows, p.queryLength,
                                            step - 1, step == tiles);
                        // Intermediate tiles do not write GM. Release partial
                        // only after MTE2 and vector accumulation have finished.
                        Catlass::Arch::CrossCoreSetFlag<0x2, PIPE_V>(outputReady);
                    }
                }
#endif
            }
        }
#ifdef __DAV_CUBE__
        for (uint32_t event = 0; event < 4; ++event) {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(event);
            if constexpr (!Online || !PrefetchKv) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(event);
            }
        }
        if constexpr (Online && PrefetchKv) {
            for (uint32_t event = 0; event < 2 * kOnlineKvStages; ++event) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(event);
            }
        }
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
#endif
    }
};

template <typename Element>
CATLASS_GLOBAL void MlaShort(uint64_t syncAddress, DenseParams params)
{
    AscendC::SetSyncBaseAddr(syncAddress);
    DenseKernel<Element> kernel;
    kernel(params);
}

template <typename Element>
void LaunchShort(const DenseParams& params, uint32_t blocks, uint64_t syncAddress, aclrtStream stream)
{
    MlaShort<Element><<<blocks, nullptr, stream>>>(syncAddress, params);
}

template <typename Element, bool PrefetchKv>
CATLASS_GLOBAL void MlaOnline(uint64_t syncAddress, DenseParams params)
{
    AscendC::SetSyncBaseAddr(syncAddress);
    DenseKernel<Element, true, PrefetchKv> kernel;
    kernel(params);
}

template <typename Element>
void LaunchOnline(const DenseParams& params, uint32_t blocks, uint64_t syncAddress, aclrtStream stream)
{
    // Up to three tiles, extra prefetch slots add startup work without
    // enough steady-state overlap to recover it (notably a one-column tail).
    if (params.kvLength > 3 * kShortKvLimit) {
        MlaOnline<Element, true><<<blocks, nullptr, stream>>>(syncAddress, params);
    } else {
        MlaOnline<Element, false><<<blocks, nullptr, stream>>>(syncAddress, params);
    }
}

} // namespace flash_attn::mla
