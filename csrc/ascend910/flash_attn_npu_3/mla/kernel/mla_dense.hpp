/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Upstream license: ../LICENSE.cann.
// Source: CANN ops-transformer ca18aee610729ca5270a44e2d84c751bc687fbf0,
// attention/fused_infer_attention_score/op_kernel/arch22/fia_kernel_nonquant_mla.h:
// Init, InitOutputSingleCore, InitCalcParamsEach, CalcParams, UpdateAxisInfo,
// FlashAttention, CreateTask, ShouldExecuteTask and ExecuteTask.
// Specialization: nonempty BSND capacity, Hkv=1, no mask/FD,
// Sq*Hq>0, 1<=Sk<=16384. The caller validates source field/address bounds and
// supplies the source M-block partition, including cross-core and tail blocks.
// MlaTaskRing shares the task ring with typed nonpaged and BSND/TND PA wrappers.
// IsTnd uses compact Q token starts and per-sample Q/KV lengths. MixedKv
// selects v4 five-word spans with a 64-bit physical KV start instead of v3.
// HasKvLengths additionally consumes a validated v2 int32 length snapshot.
// HasUsed selects v5 eight-word spans: physical starts, capacities and used.
// Default BSND addressing is uint64(b)*capacity; TND uses validated starts.
// HasStrides selects independent BSND/TND input layouts, without changing the
// task ring, output layout, metadata or numerical/synchronization order.
// IsPaged selects validated v6 five-word spans and the PA Copy Tile. Query
// allocation/used stay separate; the source S2 is not a physical cache stride.
// Source GetTaskDealMode skips zero-KV tasks; initialization is independent of
// compute endpoints. FA3 requires -inf (source +inf) LSE for these empty rows.
#pragma once
#include "../mla_dense_params.h"
#include "../block/mla_vector.hpp"

namespace flash_attn::mla::Kernel {
using namespace AscendC;

template <typename InputT, bool HasKvLengths, bool IsTnd, bool MixedKv = false, bool HasUsed = false,
          bool HasStrides = false, bool IsPaged = false>
class MlaTaskRing {
    static_assert(IsSameType<InputT, half>::value || IsSameType<InputT, bfloat16_t>::value,
                  "MLA dense only supports FP16 and BF16");
    static_assert(!MixedKv || (IsTnd && HasKvLengths));
    static_assert(!HasUsed || HasKvLengths);
    static_assert(!IsPaged || (HasKvLengths && HasStrides && !MixedKv && !HasUsed));
    static constexpr uint32_t kSpanWords = IsPaged ? 5 : HasUsed ? 8 : MixedKv ? 5 : 4;
    static constexpr bool kQueryLengths = IsTnd || HasUsed || IsPaged;

  protected:
    __aicore__ inline void Run(const MlaDenseKernelSchedule& schedule, float scale, GM_ADDR qc, GM_ADDR qr, GM_ADDR kv,
                               GM_ADDR kr, GM_ADDR output, GM_ADDR lse, GM_ADDR workspace, TPipe* pipe,
                               GM_ADDR kvLengths, uint32_t totalQuery = 0, const MlaBsndInputLayouts* layouts = nullptr,
                               bool kvIsTnd = IsTnd && !MixedKv, GM_ADDR pageTable = nullptr,
                               const MlaPagedCacheLayout* pagedLayout = nullptr)
    {
        params = schedule;
        queryTokens = IsTnd ? totalQuery : params.sq;
        if constexpr (HasKvLengths) {
            lengths.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(kvLengths));
        }
        uint32_t index = GetBlockIdx();
        uint32_t ratio = 1;
        if ASCEND_IS_AIV {
            ratio = GetSubBlockNum();
            index /= ratio;
        }
        if (index >= params.usedCoreNum) {
            if ASCEND_IS_AIV {
                SyncAll();
            }
            return;
        }
        core = index;
        bStart = core == 0 ? 0 : params.bN2End[core - 1];
        mStart = core == 0 ? 0 : params.gS1End[core - 1];
        sStart = core == 0 ? 0 : params.s2End[core - 1];
        bEnd = params.bN2End[core];
        mEnd = params.gS1End[core];
        sEnd = params.s2End[core];

        if ASCEND_IS_AIV {
            InitOutputSingleCore(output, lse, ratio);
        }
        const uint64_t r1 = params.mm1ResSize, r2 = params.mm2ResSize;
        uint64_t offset = 0;
        GlobalTensor<float> qk, pv, vec2;
        GlobalTensor<InputT> p;
        GlobalTensor<int32_t> pvInt;
        qk.SetGlobalBuffer((__gm__ float*)(workspace + core * 2 * r1 * sizeof(float)));
        offset += GetBlockNum() * 2 * r1 * sizeof(float);
        p.SetGlobalBuffer((__gm__ InputT*)(workspace + offset + core * 2 * r1 * sizeof(InputT)));
        offset += GetBlockNum() * 2 * r1 * sizeof(InputT);
        pv.SetGlobalBuffer((__gm__ float*)(workspace + offset + core * 2 * r2 * sizeof(float)));
        pvInt.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(pv.GetPhyAddr(0)));
        offset += GetBlockNum() * 2 * r2 * sizeof(float);
        vec2.SetGlobalBuffer((__gm__ float*)(workspace + offset + core * 2 * r2 * sizeof(float)));

        Block::CubeParams cubeParams{queryTokens, params.heads, params.sk, params.mm1ResSize, params.mm2ResSize};
        if ASCEND_IS_AIC {
            cube.Init(pipe, cubeParams, qc, qr, kv, kr, layouts, kvIsTnd, pageTable, pagedLayout);
            cube.InitMm1GlobalTensor(qk);
            cube.InitMm2GlobalTensor(p, pv);
            cube.AllocEventID();
        }
        if ASCEND_IS_AIV {
            Block::VectorParams vectorParams{cubeParams, ratio, scale};
            vector.Init(vectorParams, output, lse);
            vector.InitVec1GlobalTensor(qk, p, pvInt);
            vector.InitVec2GlobalTensor(vec2, pv);
            vector.InitBuffers(pipe);
            vector.AllocEventID();
        }
        FlashAttention();
        if ASCEND_IS_AIC {
            cube.FreeEventID();
        }
        if ASCEND_IS_AIV {
            vector.FreeEventID();
        }
    }

  private:
    using RunInfo = Block::RunInfo;
    static constexpr uint32_t kTaskCount = 3;
    Block::MlaCube<InputT, IsTnd, HasStrides, IsPaged> cube;
    Block::MlaVector<InputT, IsTnd> vector;
    MlaDenseKernelSchedule params;
    GlobalTensor<uint32_t> lengths;
    uint32_t actualKvLength = 0, actualQueryLength = 0;
    uint32_t queryTokens = 0, queryStart = 0;
    uint64_t kvStart = 0;
    uint32_t core, bStart, mStart, sStart, bEnd, mEnd, sEnd;
    uint32_t bn2IdxInCurCore = 0;

    __aicore__ inline void InitOutputSingleCore(GM_ADDR output, GM_ADDR lse, uint32_t ratio)
    {
        GlobalTensor<float> lseGm;
        lseGm.SetGlobalBuffer((__gm__ float*)lse);
        uint32_t event = 0;
        SetFlag<HardEvent::MTE3_V>(event);
        uint64_t total =
            IsTnd ? uint64_t(queryTokens) * params.heads : uint64_t(params.batch) * params.sq * params.heads;
        if constexpr (HasKvLengths) {
            bool needInit = false;
            for (uint32_t b = 0; b < params.batch; ++b) {
                needInit |= lengths.GetValue(kQueryLengths ? kSpanWords * b + kSpanWords - 1 : b) == 0;
            }
            // Like source needInit, cover every output, including zero-KV
            // samples beyond the last compute endpoint. Never read their Q/KV.
            if (needInit) {
                const uint64_t outputSize = total * 512;
                const uint64_t single = (outputSize + params.usedCoreNum * ratio - 1) / (params.usedCoreNum * ratio);
                const uint64_t start = GetBlockIdx() * single;
                WaitFlag<HardEvent::MTE3_V>(event);
                if (start < outputSize) {
                    GlobalTensor<InputT> outGm;
                    outGm.SetGlobalBuffer(reinterpret_cast<__gm__ InputT*>(output));
                    const uint64_t remaining = outputSize - start;
                    matmul::InitOutput<InputT>(outGm[start], remaining < single ? remaining : single, 0);
                }
                SetFlag<HardEvent::MTE3_V>(event);
            }
        }
        uint64_t single = (total + params.usedCoreNum * ratio - 1) / (params.usedCoreNum * ratio);
        uint64_t start = GetBlockIdx() * single;
        uint64_t tail = total - start;
        uint64_t count = tail < single ? tail : single;
        WaitFlag<HardEvent::MTE3_V>(event);
        if (start < total && count > 0) {
            matmul::InitOutput<float>(lseGm[start], count, HasKvLengths ? -float(1.0 / 0.0) : float(1.0 / 0.0));
        }
        SetFlag<HardEvent::MTE3_V>(event);
        WaitFlag<HardEvent::MTE3_V>(event);
        SyncAll();
    }

    __aicore__ inline void CalcParams(uint64_t loop, uint32_t b, uint32_t m, uint32_t s, RunInfo& info)
    {
        info.loop = loop;
        info.isValid = true;
        info.bIdx = b;
        info.n2Idx = 0;
        info.gS1Idx = m * params.mBaseSize;
        uint32_t rows = (kQueryLengths ? actualQueryLength : params.sq) * params.heads;
        if constexpr (IsTnd) {
            info.queryStart = queryStart;
            info.kvStart = kvStart;
        }
        info.actMBaseSize = params.mBaseSize;
        if ((m + 1) * params.mBaseSize > rows) {
            info.actMBaseSize = rows - m * params.mBaseSize;
        }
        info.actualSingleProcessSInnerSize = params.s2BaseSize;
        const uint32_t kvLength = HasKvLengths ? actualKvLength : params.sk;
        if ((s + 1) * params.s2BaseSize > kvLength) {
            info.actualSingleProcessSInnerSize = kvLength - s * params.s2BaseSize;
        }
        info.actualSingleProcessSInnerSizeAlign = Tile::Align(info.actualSingleProcessSInnerSize, 32U);
        // Every non-FD M block starts at S2=0 and consumes all of its S2 blocks.
        info.isFirstSInnerLoop = (loop == 0 || s == 0);
        if (info.isFirstSInnerLoop) {
            ++bn2IdxInCurCore;
        }
        info.bn2IdxInCurCore = bn2IdxInCurCore - 1;
        info.isLastS2Loop = (s + 1 == (kvLength + params.s2BaseSize - 1) / params.s2BaseSize);
        info.s2BatchOffset = s * params.s2BaseSize;
    }

    __aicore__ inline void UpdateAxisInfo(uint32_t& b, uint32_t& m, uint32_t& s)
    {
        uint32_t mLoops =
            ((kQueryLengths ? actualQueryLength : params.sq) * params.heads + params.mBaseSize - 1) / params.mBaseSize;
        uint32_t kvLength = HasKvLengths ? actualKvLength : params.sk;
        uint32_t sLoops = (kvLength + params.s2BaseSize - 1) / params.s2BaseSize;
        if (s + 1 < sLoops) {
            ++s;
            return;
        }
        s = 0;
        if (m + 1 < mLoops) {
            ++m;
            return;
        }
        m = 0;
        ++b;
    }

    __aicore__ inline bool ShouldExecuteTask(RunInfo tasks[kTaskCount])
    {
        for (uint32_t i = 0; i < kTaskCount; ++i) {
            if (tasks[i].isValid) {
                return true;
            }
        }
        return false;
    }

    __aicore__ inline void ExecuteTask(uint64_t loop, RunInfo tasks[kTaskCount])
    {
        RunInfo& current = tasks[loop % kTaskCount];
        RunInfo& previous = tasks[(loop + 2) % kTaskCount];
        RunInfo& oldest = tasks[(loop + 1) % kTaskCount];
        if (current.isValid) {
            if ASCEND_IS_AIC {
                cube.ComputeMm1(current);
            }
        }
        if (previous.isValid) {
            if ASCEND_IS_AIV {
                vector.ProcessVec1L(previous);
            }
            if ASCEND_IS_AIC {
                cube.ComputeMm2(previous);
            }
        }
        if (oldest.isValid) {
            if ASCEND_IS_AIV {
                vector.ProcessVec2L(oldest);
            }
            oldest.isValid = false;
        }
    }

    __aicore__ inline void FlashAttention()
    {
        RunInfo tasks[kTaskCount];
        uint32_t b = bStart, m = mStart, s = sStart;
        uint64_t created = 0, executed = 0;
        uint32_t previousBatch = UINT32_MAX;
        bool dispatch = true, execute = false;
        while (dispatch || execute) {
            dispatch = (b != bEnd || m != mEnd || s != sEnd);
            if (dispatch) {
                if constexpr (HasKvLengths) {
                    if (b != previousBatch) {
                        if constexpr (IsPaged) {
                            const uint32_t start = kSpanWords * b;
                            if constexpr (IsTnd) {
                                // v6 validates TND allocation starts within int32.
                                // Q-used limits rows; it never moves later samples.
                                queryStart = lengths.GetValue(start);
                            }
                            actualQueryLength = lengths.GetValue(start + 3);
                            actualKvLength = lengths.GetValue(start + 4);
                        } else if constexpr (HasUsed) {
                            const uint32_t start = kSpanWords * b;
                            // TND query totals fit int32. BSND uses the existing
                            // 64-bit b*Sq address in Block, equal to this span.
                            queryStart = lengths.GetValue(start);
                            kvStart =
                                uint64_t(lengths.GetValue(start + 2)) | (uint64_t(lengths.GetValue(start + 3)) << 32);
                            actualQueryLength = lengths.GetValue(start + 6);
                            actualKvLength = lengths.GetValue(start + 7);
                        } else if constexpr (IsTnd) {
                            const uint32_t start = kSpanWords * b;
                            queryStart = lengths.GetValue(start);
                            kvStart = lengths.GetValue(start + 1);
                            if constexpr (MixedKv) {
                                kvStart |= uint64_t(lengths.GetValue(start + 2)) << 32;
                            }
                            actualQueryLength = lengths.GetValue(start + kSpanWords - 2);
                            actualKvLength = lengths.GetValue(start + kSpanWords - 1);
                        } else {
                            actualKvLength = lengths.GetValue(b);
                        }
                        previousBatch = b;
                    }
                    if (actualKvLength == 0 || (kQueryLengths && actualQueryLength == 0)) {
                        // Preserve the source task-ring counters while skipping
                        // DEAL_ZERO/SKIP_ZERO. Pending tasks drain after dispatch.
                        UpdateAxisInfo(b, m, s);
                        continue;
                    }
                }
                CalcParams(created, b, m, s, tasks[created % kTaskCount]);
                ++created;
                UpdateAxisInfo(b, m, s);
            }
            execute = ShouldExecuteTask(tasks);
            if (execute) {
                ExecuteTask(executed++, tasks);
            }
        }
    }
};
template <typename InputT, bool HasKvLengths, bool IsTnd, bool MixedKv = false, bool HasUsed = false,
          bool HasStrides = false>
using MlaNonPaged = MlaTaskRing<InputT, HasKvLengths, IsTnd, MixedKv, HasUsed, HasStrides>;
// Typed BSND entry retains the original interface; the common engine is only
// accessible through layout-specific wrappers after metadata validation.
template <typename InputT, bool HasKvLengths = false, bool HasStrides = false>
class MlaDense : private MlaNonPaged<InputT, HasKvLengths, false, false, false, HasStrides> {
  public:
    __aicore__ inline void operator()(const MlaDenseKernelSchedule& schedule, float scale, GM_ADDR qc, GM_ADDR qr,
                                      GM_ADDR kv, GM_ADDR kr, GM_ADDR output, GM_ADDR lse, GM_ADDR workspace,
                                      TPipe* pipe, GM_ADDR kvLengths = nullptr,
                                      const MlaBsndInputLayouts* layouts = nullptr)
    {
        this->Run(schedule, scale, qc, qr, kv, kr, output, lse, workspace, pipe, kvLengths, 0, layouts);
    }
};
} // namespace flash_attn::mla::Kernel
