/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "mla_params.h"
#include "mla_vector_block.hpp"
#include "kernel_operator.h"
#include "mla_metadata_consumer.h"

// Upstream license: LICENSE.cann.
// Source: CANN ops-transformer ca18aee610729ca5270a44e2d84c751bc687fbf0,
// attention/fused_infer_attention_score/op_kernel/arch22/fia_kernel_nonquant_mla.h:
// Init, InitOutputSingleCore, InitCalcParamsEach, CalcParams, UpdateAxisInfo,
// FlashAttention, CreateTask, ShouldExecuteTask and ExecuteTask.
// Default specialization: nonempty BSND capacity, Hkv=1, no mask/FD,
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
// IsCausal specializes S2 ranges/masking at compile time; all existing wrappers default to false.
// IsWindow adds normalized band bounds and nonzero S2 starts for the independent BSND component.
// IsPaged selects validated v6 five-word spans and the PA Copy Tile. Query
// allocation/used stay separate; the source S2 is not a physical cache stride.
// Source GetTaskDealMode skips zero-KV tasks; initialization is independent of
// compute endpoints. FA3 requires -inf (source +inf) LSE for these empty rows.

namespace flash_attn::mla::Kernel {
using namespace AscendC;

template <typename InputT, bool HasKvLengths, bool IsTnd, bool MixedKv = false, bool HasUsed = false,
          bool HasStrides = false, bool IsPaged = false, bool IsCausal = false, bool IsWindow = false,
          bool IsPartial = false, bool AbsolutePartialOffsets = false>
class MlaTaskRing {
    static_assert(IsSameType<InputT, half>::value || IsSameType<InputT, bfloat16_t>::value,
                  "MLA dense only supports FP16 and BF16");
    static_assert(!MixedKv || (IsTnd && HasKvLengths));
    static_assert(!HasUsed || HasKvLengths);
    static_assert(!IsPaged || (HasKvLengths && HasStrides && !MixedKv && !HasUsed));
    static_assert(!IsWindow || (HasKvLengths && !IsCausal));
    static_assert(!IsPartial || IsWindow);
    static constexpr uint32_t kSpanWords = IsPaged ? 5 : HasUsed ? 8 : MixedKv ? 5 : 4;
    static constexpr bool kQueryLengths = IsTnd || HasUsed || IsPaged;

  protected:
    __aicore__ inline void Run(const MlaDenseKernelSchedule& schedule, float scale, GM_ADDR qc, GM_ADDR qr, GM_ADDR kv,
                               GM_ADDR kr, GM_ADDR output, GM_ADDR lse, GM_ADDR workspace, TPipe* pipe,
                               GM_ADDR kvLengths, uint32_t totalQuery = 0, const MlaBsndInputLayouts* layouts = nullptr,
                               bool kvIsTnd = IsTnd && !MixedKv, GM_ADDR pageTable = nullptr,
                               const MlaPagedCacheLayout* pagedLayout = nullptr,
                               MlaWindowKernelParams windowParams = {}, const MlaSplitKvKernelParams* partial = nullptr,
                               float softcap = 0.0f, const MlaTaskLengths* fragmentLengths = nullptr)
    {
        params = schedule;
        taskLengths = fragmentLengths;
        if constexpr (IsWindow) {
            window = windowParams;
        }
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

        if constexpr (IsPartial) {
            const auto& task = partial->tasks[core];
            bStart = bEnd = task.batch;
            mStart = mEnd = task.m;
            sStart = task.s2Begin;
            sEnd = task.s2End;
            if (sStart == sEnd) {
                if ASCEND_IS_AIV {
                    InitEmptyPartial(workspace, *partial, ratio);
                }
                return;
            }
        } else {
            if ASCEND_IS_AIV {
                InitOutputSingleCore(output, lse, ratio);
            }
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
            Block::VectorParams vectorParams{cubeParams, ratio, scale, softcap};
            vector.Init(vectorParams, output, lse);
            if constexpr (IsPartial) {
                vector.InitPartial(workspace + partial->outOffsetBytes, workspace + partial->sumOffsetBytes,
                                   workspace + partial->maxOffsetBytes,
                                   AbsolutePartialOffsets ? 0 : uint64_t(core) * params.mBaseSize);
            }
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
    Block::MlaVector<InputT, IsTnd, IsCausal, IsWindow, IsPartial> vector;
    MlaDenseKernelSchedule params;
    MlaWindowKernelParams window;
    GlobalTensor<uint32_t> lengths;
    uint32_t actualKvLength = 0, actualQueryLength = 0;
    uint32_t queryTokens = 0, queryStart = 0;
    uint64_t kvStart = 0;
    const MlaTaskLengths* taskLengths = nullptr;
    uint32_t core, bStart, mStart, sStart, bEnd, mEnd, sEnd;
    uint32_t bn2IdxInCurCore = 0;

    __aicore__ inline void InitEmptyPartial(GM_ADDR workspace, const MlaSplitKvKernelParams& partial, uint32_t ratio)
    {
        // Empty fragments own their slot even when there are no Cube tasks.
        // Each vector initializes disjoint complete rows; order MTE3 writes
        // before recycling the initialization scratch.
        const uint32_t chunk = (params.mBaseSize + ratio - 1) / ratio;
        const uint32_t begin = (GetBlockIdx() % ratio) * chunk;
        if (begin >= params.mBaseSize) {
            return;
        }
        const uint32_t rows = Min(chunk, params.mBaseSize - begin);
        const uint64_t row = (AbsolutePartialOffsets ? 0 : uint64_t(core) * params.mBaseSize) + begin;
        GlobalTensor<float> gm;
        gm.SetGlobalBuffer((__gm__ float*)(workspace + partial.outOffsetBytes));
        matmul::InitOutput<float>(gm[row * 512], rows * 512, 0);
        SetFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_V>(0);
        gm.SetGlobalBuffer((__gm__ float*)(workspace + partial.sumOffsetBytes));
        matmul::InitOutput<float>(gm[row * 8], rows * 8, 0);
        SetFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_V>(0);
        gm.SetGlobalBuffer((__gm__ float*)(workspace + partial.maxOffsetBytes));
        matmul::InitOutput<float>(gm[row * 8], rows * 8, -float(1.0 / 0.0));
        SetFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_V>(0);
    }

    __aicore__ inline void InitOutputSingleCore(GM_ADDR output, GM_ADDR lse, uint32_t ratio)
    {
        GlobalTensor<float> lseGm;
        lseGm.SetGlobalBuffer((__gm__ float*)lse);
        uint32_t event = 0;
        SetFlag<HardEvent::MTE3_V>(event);
        uint64_t total =
            IsTnd ? uint64_t(queryTokens) * params.heads : uint64_t(params.batch) * params.sq * params.heads;
        if constexpr (HasKvLengths) {
            bool needInit = IsCausal || IsWindow;
            if constexpr (!IsCausal && !IsWindow) {
                for (uint32_t b = 0; b < params.batch; ++b) {
                    needInit |= lengths.GetValue(kQueryLengths ? kSpanWords * b + kSpanWords - 1 : b) == 0;
                }
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
            matmul::InitOutput<float>(lseGm[start], count,
                                      (HasKvLengths && !((IsCausal || IsWindow) && params.sk == 0)) ? -float(1.0 / 0.0)
                                                                                                    : float(1.0 / 0.0));
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
        // Whole-M partitions consume their visible range, which can start
        // above zero for band masks. Softmax/PV and ping-pong ownership must
        // restart on that first emitted tile, never on a skipped S2=0 tile.
        info.isFirstSInnerLoop = (loop == 0 || s == FirstS2(m));
        if (info.isFirstSInnerLoop) {
            ++bn2IdxInCurCore;
        }
        info.bn2IdxInCurCore = bn2IdxInCurCore - 1;
        info.isLastS2Loop = (s + 1 == (IsPartial ? sEnd : S2Loops(m)));
        if constexpr (IsCausal) {
            info.nextToken = int64_t(kvLength) - (kQueryLengths ? actualQueryLength : params.sq);
        }
        if constexpr (IsWindow) {
            const int64_t delta = int64_t(kvLength) - (kQueryLengths ? actualQueryLength : params.sq);
            info.nextToken = window.kind == 0 ? INT32_MAX : delta + (window.kind == 2 ? window.right : 0);
            info.preToken = window.kind == 2 ? window.left - delta : INT32_MAX;
        }
        info.s2BatchOffset = s * params.s2BaseSize;
    }

    // Pinned CalcCurS2StartEndWithSparse, specialized to right-down causal/band.
    // The last query touched by this S1G block determines its union of keys.
    // Keep full S2 tiles; Vector still masks each row within the boundary tile.
    __aicore__ inline uint32_t S2Loops(uint32_t m)
    {
        const uint32_t kvLength = HasKvLengths ? actualKvLength : params.sk;
        if constexpr (IsWindow) {
            if (window.kind == 0) {
                return (kvLength + params.s2BaseSize - 1) / params.s2BaseSize;
            }
        }
        if constexpr (IsCausal || IsWindow) {
            const uint32_t sq = kQueryLengths ? actualQueryLength : params.sq;
            const uint64_t end = Min(uint64_t(m + 1) * params.mBaseSize, uint64_t(sq) * params.heads);
            if (end == 0 || kvLength == 0) {
                return 0;
            }
            const int64_t right = IsWindow && window.kind == 2 ? window.right : 0;
            const int64_t keys = int64_t((end - 1) / params.heads) + int64_t(kvLength) - sq + right + 1;
            return keys <= 0 ? 0
                             : (Min(uint64_t(keys), uint64_t(kvLength)) + params.s2BaseSize - 1) / params.s2BaseSize;
        }
        return (kvLength + params.s2BaseSize - 1) / params.s2BaseSize;
    }

    __aicore__ inline uint32_t FirstS2(uint32_t m)
    {
        if constexpr (IsWindow) {
            if (window.kind == 2) {
                const uint32_t sq = kQueryLengths ? actualQueryLength : params.sq;
                const int64_t first =
                    int64_t(uint64_t(m) * params.mBaseSize / params.heads) + actualKvLength - int64_t(sq) - window.left;
                return first <= 0 ? 0 : Min(uint64_t(first), uint64_t(actualKvLength)) / params.s2BaseSize;
            }
        }
        return 0;
    }

    __aicore__ inline void UpdateAxisInfo(uint32_t& b, uint32_t& m, uint32_t& s)
    {
        if constexpr (IsPartial) {
            ++s;
            return;
        }
        uint32_t mLoops =
            ((kQueryLengths ? actualQueryLength : params.sq) * params.heads + params.mBaseSize - 1) / params.mBaseSize;
        uint32_t sLoops = S2Loops(m);
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
                        if (taskLengths) {
                            queryStart = taskLengths->queryStart;
                            kvStart = taskLengths->kvStart;
                            actualQueryLength = taskLengths->queryLength;
                            actualKvLength = taskLengths->kvLength;
                        } else if constexpr (IsPaged) {
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
                if constexpr (IsCausal || IsWindow) {
                    if (S2Loops(m) == 0) {
                        // Initialization owns entirely masked blocks. Do not
                        // consume ring slots or change ping-pong counters.
                        UpdateAxisInfo(b, m, s);
                        continue;
                    }
                }
                if constexpr (IsWindow) {
                    // Axis endpoints stay at whole-M boundaries. Jump over
                    // the hidden prefix without creating tasks or advancing
                    // the ring counters; pending tasks retain their order.
                    s = Max(s, FirstS2(m));
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

// Device adapter for the private dense schedule protocol. Validate before the
// attention Kernel allocates events, initializes LSE or touches its workspace.

namespace flash_attn::mla::Kernel {
struct MetadataReader {
    AscendC::GlobalTensor<uint32_t> data;
    __aicore__ inline uint32_t operator()(uint32_t word) const
    {
        return data.GetValue(word);
    }
};

// Reused metadata is a snapshot, not proof that the current lengths still
// match. Every Cube/Vector checks all B lengths before any attention barrier
// or write. The host has validated the optional tensor's size/device/stride.
__aicore__ inline bool CheckDenseCacheLengths(GM_ADDR address, uint32_t batch, uint32_t capacity)
{
    if (address == nullptr) {
        return true;
    }
    AscendC::GlobalTensor<int32_t> lengths;
    lengths.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(address));
    for (uint32_t b = 0; b < batch; ++b) {
        if (lengths.GetValue(b) != static_cast<int32_t>(capacity)) {
            AscendC::Trap();
            return false;
        }
    }
    return true;
}

template <typename InputT>
__aicore__ inline MlaMetadataReadStatus ReadDenseMetadata(GM_ADDR address, uint32_t words,
                                                          const MlaDenseMetadataContext& context,
                                                          MlaDenseKernelSchedule& output)
{
    static_assert(AscendC::IsSameType<InputT, half>::value || AscendC::IsSameType<InputT, bfloat16_t>::value);
    if (address == nullptr || reinterpret_cast<uint64_t>(address) % sizeof(uint32_t) != 0) {
        return MlaMetadataReadStatus::InvalidBuffer;
    }
    const uint32_t dtype =
        AscendC::IsSameType<InputT, half>::value ? uint32_t(MlaMetadataDtype::Fp16) : uint32_t(MlaMetadataDtype::Bf16);
    if (context.dtype != dtype) {
        return MlaMetadataReadStatus::DtypeMismatch;
    }
    if (context.blockDim != AscendC::GetBlockNum()) {
        return MlaMetadataReadStatus::PlatformMismatch;
    }
    MetadataReader read;
    read.data.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(address));
    return flash_attn::mla::ReadDenseMetadata(read, words, context, output);
}

// Runtime-facing adapter: an invalid record must fail the stream, not leave
// apparently successful, uninitialized O/LSE. Trap is unconditional on build
// flags (unlike debug assertions) and needs no status buffer or host readback.
// All Cube/Vector cores validate the same immutable record before the caller
// constructs TPipe, allocates events or enters any attention barrier.
template <typename InputT>
__aicore__ inline bool LoadDenseMetadata(GM_ADDR address, uint32_t words, const MlaDenseMetadataContext& context,
                                         MlaDenseKernelSchedule& output)
{
    if (ReadDenseMetadata<InputT>(address, words, context, output) != MlaMetadataReadStatus::Success) {
        AscendC::Trap();
        return false;
    }
    return true;
}
} // namespace flash_attn::mla::Kernel

// Device-only adapter for v2 BSND records. Every Cube/Vector validates before
// attention event allocation, barriers or output/workspace writes .

namespace flash_attn::mla::Kernel {
template <typename InputT>
__aicore__ inline MlaMetadataReadStatus
ReadBsndKvLengthMetadata(GM_ADDR address, uint32_t words, const MlaDenseMetadataContext& context, GM_ADDR lengths,
                         uint64_t lengthCount, MlaDenseKernelSchedule& output)
{
    using Status = MlaMetadataReadStatus;
    static_assert(AscendC::IsSameType<InputT, half>::value || AscendC::IsSameType<InputT, bfloat16_t>::value);
    const uint64_t addr = reinterpret_cast<uint64_t>(address), lenAddr = reinterpret_cast<uint64_t>(lengths);
    if (address == nullptr || addr % sizeof(uint32_t) != 0 || words > (UINT64_MAX - addr) / sizeof(uint32_t)) {
        return Status::InvalidBuffer;
    }
    if (lenAddr % sizeof(int32_t) != 0 || lengthCount > (UINT64_MAX - lenAddr) / sizeof(int32_t)) {
        return Status::InvalidCacheLengths;
    }
    const uint32_t dtype =
        AscendC::IsSameType<InputT, half>::value ? uint32_t(MlaMetadataDtype::Fp16) : uint32_t(MlaMetadataDtype::Bf16);
    if (context.dtype != dtype) {
        return Status::DtypeMismatch;
    }
    if (context.blockDim != AscendC::GetBlockNum()) {
        return Status::PlatformMismatch;
    }
    MetadataReader read, current;
    read.data.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(address));
    if (lengths != nullptr) {
        current.data.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(lengths));
    }
    return flash_attn::mla::ReadBsndKvLengthMetadata(read, words, context, current, lengths != nullptr, lengthCount,
                                                     output);
}

template <typename InputT>
__aicore__ inline bool LoadBsndKvLengthMetadata(GM_ADDR address, uint32_t words, const MlaDenseMetadataContext& context,
                                                GM_ADDR lengths, uint64_t lengthCount, MlaDenseKernelSchedule& output)
{
    if (ReadBsndKvLengthMetadata<InputT>(address, words, context, lengths, lengthCount, output) !=
        MlaMetadataReadStatus::Success) {
        AscendC::Trap();
        return false;
    }
    return true;
}
} // namespace flash_attn::mla::Kernel

// Independent v3 device consumer. All Cube/Vector cores validate before the
// caller allocates TPipe/events, writes outputs/workspace or synchronizes.

namespace flash_attn::mla::Kernel {
template <typename InputT>
__aicore__ inline MlaMetadataReadStatus
ReadTndMetadata(GM_ADDR address, uint32_t words, const MlaTndMetadataContext& context, GM_ADDR cuQuery,
                uint64_t queryCount, GM_ADDR cuKv, uint64_t kvCount, MlaTndKernelSchedule& output)
{
    using Status = MlaMetadataReadStatus;
    static_assert(AscendC::IsSameType<InputT, half>::value || AscendC::IsSameType<InputT, bfloat16_t>::value);
    const uint64_t addr = reinterpret_cast<uint64_t>(address);
    if (address == nullptr || addr % 4 != 0 || words > (UINT64_MAX - addr) / 4) {
        return Status::InvalidBuffer;
    }
    const uint64_t qAddr = reinterpret_cast<uint64_t>(cuQuery), kAddr = reinterpret_cast<uint64_t>(cuKv);
    if (cuQuery == nullptr || cuKv == nullptr || qAddr % 4 != 0 || kAddr % 4 != 0 ||
        queryCount > (UINT64_MAX - qAddr) / 4 || kvCount > (UINT64_MAX - kAddr) / 4) {
        return Status::InvalidPrefixes;
    }
    const uint32_t dtype =
        AscendC::IsSameType<InputT, half>::value ? uint32_t(MlaMetadataDtype::Fp16) : uint32_t(MlaMetadataDtype::Bf16);
    if (context.dtype != dtype) {
        return Status::DtypeMismatch;
    }
    if (context.blockDim != AscendC::GetBlockNum()) {
        return Status::PlatformMismatch;
    }
    MetadataReader read, query, kv;
    read.data.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(address));
    query.data.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(cuQuery));
    kv.data.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(cuKv));
    return flash_attn::mla::ReadTndMetadata(read, words, context, query, queryCount, kv, kvCount, output);
}

template <typename InputT>
__aicore__ inline bool LoadTndMetadata(GM_ADDR address, uint32_t words, const MlaTndMetadataContext& context,
                                       GM_ADDR cuQuery, uint64_t queryCount, GM_ADDR cuKv, uint64_t kvCount,
                                       MlaTndKernelSchedule& output)
{
    if (ReadTndMetadata<InputT>(address, words, context, cuQuery, queryCount, cuKv, kvCount, output) !=
        MlaMetadataReadStatus::Success) {
        AscendC::Trap();
        return false;
    }
    return true;
}
} // namespace flash_attn::mla::Kernel

// Independent v4 mixed-layout device consumer. All Cube/Vector cores validate before the
// caller allocates TPipe/events, writes outputs/workspace or synchronizes.

namespace flash_attn::mla::Kernel {
template <typename InputT>
__aicore__ inline MlaMetadataReadStatus
ReadTndBsndMetadata(GM_ADDR address, uint32_t words, const MlaTndBsndMetadataContext& context, GM_ADDR cuQuery,
                    uint64_t queryCount, GM_ADDR kvLengths, uint64_t kvCount, MlaTndBsndKernelSchedule& output)
{
    using Status = MlaMetadataReadStatus;
    static_assert(AscendC::IsSameType<InputT, half>::value || AscendC::IsSameType<InputT, bfloat16_t>::value);
    const uint64_t addr = reinterpret_cast<uint64_t>(address);
    if (address == nullptr || addr % 4 != 0 || words > (UINT64_MAX - addr) / 4) {
        return Status::InvalidBuffer;
    }
    const uint64_t qAddr = reinterpret_cast<uint64_t>(cuQuery), kAddr = reinterpret_cast<uint64_t>(kvLengths);
    if (cuQuery == nullptr || qAddr % 4 != 0 || queryCount > (UINT64_MAX - qAddr) / 4) {
        return Status::InvalidPrefixes;
    }
    if ((kvLengths == nullptr && (context.batch != 0 || kvCount != 0)) || kAddr % 4 != 0 ||
        kvCount > (UINT64_MAX - kAddr) / 4) {
        return Status::InvalidCacheLengths;
    }
    const uint32_t dtype =
        AscendC::IsSameType<InputT, half>::value ? uint32_t(MlaMetadataDtype::Fp16) : uint32_t(MlaMetadataDtype::Bf16);
    if (context.dtype != dtype) {
        return Status::DtypeMismatch;
    }
    if (context.blockDim != AscendC::GetBlockNum()) {
        return Status::PlatformMismatch;
    }
    MetadataReader read, query, kv;
    read.data.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(address));
    query.data.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(cuQuery));
    if (kvLengths != nullptr) {
        kv.data.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(kvLengths));
    }
    return flash_attn::mla::ReadTndBsndMetadata(read, words, context, query, queryCount, kv, kvCount, output);
}

template <typename InputT>
__aicore__ inline bool LoadTndBsndMetadata(GM_ADDR address, uint32_t words, const MlaTndBsndMetadataContext& context,
                                           GM_ADDR cuQuery, uint64_t queryCount, GM_ADDR kvLengths, uint64_t kvCount,
                                           MlaTndBsndKernelSchedule& output)
{
    if (ReadTndBsndMetadata<InputT>(address, words, context, cuQuery, queryCount, kvLengths, kvCount, output) !=
        MlaMetadataReadStatus::Success) {
        AscendC::Trap();
        return false;
    }
    return true;
}
} // namespace flash_attn::mla::Kernel

// Independent v5 device consumer. Every Cube/Vector validates before its caller
// constructs TPipe/events, writes outputs/workspace or synchronizes.

namespace flash_attn::mla::Kernel {
struct UsedMetadataReaders {
    MetadataReader data[4];
    __aicore__ inline uint32_t operator()(uint32_t axis, uint32_t index) const
    {
        return data[axis](index);
    }
};

template <typename InputT, bool IsCausal = false, bool IsTndCausal = false, bool IsMixedCausal = false,
          bool IsWindow = false, bool IsTndWindow = false, bool IsMixedWindow = false>
__aicore__ inline MlaMetadataReadStatus
ReadUsedMetadata(GM_ADDR address, uint32_t words, const MlaUsedMetadataContext& context,
                 const MlaUsedMetadataInputs& inputs, MlaUsedKernelSchedule& output, MlaWindowKernelParams window = {})
{
    using Status = MlaMetadataReadStatus;
    static_assert(AscendC::IsSameType<InputT, half>::value || AscendC::IsSameType<InputT, bfloat16_t>::value);
    if (address == nullptr || !ValidUsedMetadataAddress(reinterpret_cast<uint64_t>(address), words)) {
        return Status::InvalidBuffer;
    }
    const uint32_t dtype =
        AscendC::IsSameType<InputT, half>::value ? uint32_t(MlaMetadataDtype::Fp16) : uint32_t(MlaMetadataDtype::Bf16);
    if (context.dtype != dtype) {
        return Status::DtypeMismatch;
    }
    if (context.blockDim != AscendC::GetBlockNum()) {
        return Status::PlatformMismatch;
    }
    MetadataReader read;
    read.data.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(address));
    UsedMetadataReaders arrays;
    const uint64_t addresses[4] = {inputs.cuQuery.address, inputs.cuKv.address, inputs.usedQuery.address,
                                   inputs.usedKv.address};
    for (uint32_t i = 0; i < 4; ++i) {
        if (addresses[i] != 0) {
            arrays.data[i].data.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(addresses[i]));
        }
    }
    return flash_attn::mla::ReadUsedMetadata<IsCausal, IsTndCausal, IsMixedCausal, IsWindow, IsTndWindow,
                                             IsMixedWindow>(read, words, context, inputs, arrays, output, window);
}

template <typename InputT, bool IsCausal = false, bool IsTndCausal = false, bool IsMixedCausal = false,
          bool IsWindow = false, bool IsTndWindow = false, bool IsMixedWindow = false>
__aicore__ inline bool LoadUsedMetadata(GM_ADDR address, uint32_t words, const MlaUsedMetadataContext& context,
                                        const MlaUsedMetadataInputs& inputs, MlaUsedKernelSchedule& output,
                                        MlaWindowKernelParams window = {})
{
    if (ReadUsedMetadata<InputT, IsCausal, IsTndCausal, IsMixedCausal, IsWindow, IsTndWindow, IsMixedWindow>(
            address, words, context, inputs, output, window) != MlaMetadataReadStatus::Success) {
        AscendC::Trap();
        return false;
    }
    return true;
}
} // namespace flash_attn::mla::Kernel

// Independent v6 device validation. Every Cube/Vector must pass before its
// caller constructs TPipe/events, writes outputs/workspace or synchronizes.

namespace flash_attn::mla::Kernel {
struct PagedMetadataReaders {
    MetadataReader data[3];
    __aicore__ inline uint32_t operator()(uint32_t axis, uint32_t index) const
    {
        return data[axis](index);
    }
};

struct PageTableReader {
    AscendC::GlobalTensor<uint32_t> data;
    __aicore__ inline uint32_t operator()(uint64_t index) const
    {
        return data.GetValue(index);
    }
};

template <typename InputT, bool IsCausal = false, bool IsWindow = false>
__aicore__ inline MlaMetadataReadStatus
ReadPagedMetadata(GM_ADDR address, uint32_t words, const MlaPagedMetadataContext& context,
                  const MlaPagedMetadataInputs& inputs, const MlaPagedCacheInputs& cache,
                  MlaPagedKernelSchedule& output, MlaWindowKernelParams window = {})
{
    using Status = MlaMetadataReadStatus;
    static_assert(AscendC::IsSameType<InputT, half>::value || AscendC::IsSameType<InputT, bfloat16_t>::value);
    if (address == nullptr || !ValidPagedMetadataAddress(reinterpret_cast<uint64_t>(address), words)) {
        return Status::InvalidBuffer;
    }
    const uint32_t dtype =
        AscendC::IsSameType<InputT, half>::value ? uint32_t(MlaMetadataDtype::Fp16) : uint32_t(MlaMetadataDtype::Bf16);
    if (context.dtype != dtype) {
        return Status::DtypeMismatch;
    }
    if (context.blockDim != AscendC::GetBlockNum()) {
        return Status::PlatformMismatch;
    }
    MetadataReader read;
    read.data.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(address));
    PagedMetadataReaders arrays;
    const uint64_t addresses[3] = {inputs.cuQuery.address, inputs.usedQuery.address, inputs.kvLengths.address};
    for (uint32_t i = 0; i < 3; ++i) {
        if (addresses[i] != 0) {
            arrays.data[i].data.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(addresses[i]));
        }
    }
    PageTableReader pages;
    if (cache.table.address != 0) {
        pages.data.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(cache.table.address));
    }
    return flash_attn::mla::ReadPagedMetadata<IsCausal, IsWindow>(read, words, context, inputs, arrays, cache, pages,
                                                                  output, window);
}

template <typename InputT, bool IsCausal = false, bool IsWindow = false>
__aicore__ inline bool LoadPagedMetadata(GM_ADDR address, uint32_t words, const MlaPagedMetadataContext& context,
                                         const MlaPagedMetadataInputs& inputs, const MlaPagedCacheInputs& cache,
                                         MlaPagedKernelSchedule& output, MlaWindowKernelParams window = {})
{
    if (ReadPagedMetadata<InputT, IsCausal, IsWindow>(address, words, context, inputs, cache, output, window) !=
        MlaMetadataReadStatus::Success) {
        AscendC::Trap();
        return false;
    }
    return true;
}
} // namespace flash_attn::mla::Kernel

// Compact TND specialization of the migrated CANN nonquant MLA task ring.
// Source and license: mla_dense.hpp and LICENSE.cann. Token starts follow
// CANN ca18aee610729ca5270a44e2d84c751bc687fbf0 ActualSeqLensParser;
// the validated v3 snapshot replaces the source int64 cumulative arrays.

namespace flash_attn::mla::Kernel {
template <typename InputT, bool HasStrides = false>
class MlaTnd : private MlaNonPaged<InputT, true, true, false, false, HasStrides> {
  public:
    // Caller must validate v3 and current prefixes on every core before TPipe,
    // synchronization or writes. Empty capacity still passes that validation.
    __aicore__ inline void operator()(const MlaTndKernelSchedule& schedule, float scale, GM_ADDR qc, GM_ADDR qr,
                                      GM_ADDR kv, GM_ADDR kr, GM_ADDR output, GM_ADDR lse, GM_ADDR workspace,
                                      TPipe* pipe, GM_ADDR spans, const MlaBsndInputLayouts* layouts = nullptr)
    {
        if (schedule.totalQuery == 0) {
            return;
        }
        if (schedule.totalKv == 0) {
            if ASCEND_IS_AIV {
                // FA3 global-empty capacity uses +inf; empty samples within a
                // nonempty capacity use -inf in the common initialization.
                const uint64_t rows = uint64_t(schedule.totalQuery) * schedule.tiling.heads;
                const uint64_t single = (rows + GetSubBlockNum() - 1) / GetSubBlockNum();
                const uint64_t start = GetBlockIdx() * single;
                if (start < rows) {
                    const uint64_t count = rows - start < single ? rows - start : single;
                    GlobalTensor<InputT> outGm;
                    GlobalTensor<float> lseGm;
                    outGm.SetGlobalBuffer(reinterpret_cast<__gm__ InputT*>(output));
                    lseGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(lse));
                    matmul::InitOutput<InputT>(outGm[start * 512], count * 512, 0);
                    SetFlag<HardEvent::MTE3_V>(0);
                    WaitFlag<HardEvent::MTE3_V>(0);
                    matmul::InitOutput<float>(lseGm[start], count, float(1.0 / 0.0));
                    SetFlag<HardEvent::MTE3_V>(0);
                    WaitFlag<HardEvent::MTE3_V>(0);
                }
            }
            return;
        }
        this->Run(schedule.tiling, scale, qc, qr, kv, kr, output, lse, workspace, pipe, spans, schedule.totalQuery,
                  layouts);
    }
};
} // namespace flash_attn::mla::Kernel

// TND-Q/BSND-KV specialization of the migrated CANN nonquant MLA task ring.
// Source and license: mla_dense.hpp and LICENSE.cann. Token starts follow
// CANN ca18aee610729ca5270a44e2d84c751bc687fbf0 ActualSeqLensParser;
// the validated v4 snapshot supplies compact Q starts and 64-bit b*Sk KV starts.

namespace flash_attn::mla::Kernel {
template <typename InputT, bool HasStrides = false>
class MlaTndBsnd : private MlaNonPaged<InputT, true, true, true, false, HasStrides> {
  public:
    // Caller must validate v4 and current Q prefixes/KV lengths on every core before TPipe,
    // synchronization or writes. Empty capacity still passes that validation.
    __aicore__ inline void operator()(const MlaTndBsndKernelSchedule& schedule, float scale, GM_ADDR qc, GM_ADDR qr,
                                      GM_ADDR kv, GM_ADDR kr, GM_ADDR output, GM_ADDR lse, GM_ADDR workspace,
                                      TPipe* pipe, GM_ADDR spans, const MlaBsndInputLayouts* layouts = nullptr)
    {
        if (schedule.totalQuery == 0) {
            return;
        }
        if (schedule.totalKvCapacity == 0) {
            if ASCEND_IS_AIV {
                // FA3 global-empty capacity uses +inf; empty samples within a
                // nonempty capacity use -inf in the common initialization.
                const uint64_t rows = uint64_t(schedule.totalQuery) * schedule.tiling.heads;
                const uint64_t single = (rows + GetSubBlockNum() - 1) / GetSubBlockNum();
                const uint64_t start = GetBlockIdx() * single;
                if (start < rows) {
                    const uint64_t count = rows - start < single ? rows - start : single;
                    GlobalTensor<InputT> outGm;
                    GlobalTensor<float> lseGm;
                    outGm.SetGlobalBuffer(reinterpret_cast<__gm__ InputT*>(output));
                    lseGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(lse));
                    matmul::InitOutput<InputT>(outGm[start * 512], count * 512, 0);
                    SetFlag<HardEvent::MTE3_V>(0);
                    WaitFlag<HardEvent::MTE3_V>(0);
                    matmul::InitOutput<float>(lseGm[start], count, float(1.0 / 0.0));
                    SetFlag<HardEvent::MTE3_V>(0);
                    WaitFlag<HardEvent::MTE3_V>(0);
                }
            }
            return;
        }
        this->Run(schedule.tiling, scale, qc, qr, kv, kr, output, lse, workspace, pipe, spans, schedule.totalQuery,
                  layouts);
    }
};
} // namespace flash_attn::mla::Kernel

// Allocation/used specialization of the CANN nonquant MLA task ring.
// Source and license: mla_dense.hpp and LICENSE.cann. v5 separates the
// FA3 physical allocation from the source task lengths; used never repacks Q/KV.

namespace flash_attn::mla::Kernel {
template <typename InputT, bool IsTnd, bool HasStrides = false, bool IsCausal = false, bool IsWindow = false>
class MlaUsed : private MlaTaskRing<InputT, /*HasKvLengths=*/true, IsTnd, /*MixedKv=*/false, /*HasUsed=*/true,
                                    HasStrides, /*IsPaged=*/false, IsCausal, IsWindow> {
  public:
    // Validate v5 and all four current arrays on every core before TPipe,
    // synchronization or writes. IsTnd must match the validated Q layout.
    __aicore__ inline void operator()(const MlaUsedKernelSchedule& schedule, float scale, GM_ADDR qc, GM_ADDR qr,
                                      GM_ADDR kv, GM_ADDR kr, GM_ADDR output, GM_ADDR lse, GM_ADDR workspace,
                                      TPipe* pipe, GM_ADDR spans, const MlaBsndInputLayouts* layouts = nullptr,
                                      MlaWindowKernelParams window = {})
    {
        const uint64_t totalQuery = UsedMetadataTotal(schedule.query);
        if (totalQuery == 0) {
            return;
        }
        if (UsedMetadataTotal(schedule.kv) == 0) {
            if ASCEND_IS_AIV {
                // The FA3 global-empty branch initializes the allocation.
                // Nonempty capacity with zero used is handled by Run (-inf).
                const uint64_t rows = totalQuery * schedule.tiling.heads;
                const uint64_t single = (rows + GetSubBlockNum() - 1) / GetSubBlockNum();
                const uint64_t start = GetBlockIdx() * single;
                if (start < rows) {
                    const uint64_t count = rows - start < single ? rows - start : single;
                    GlobalTensor<InputT> outGm;
                    GlobalTensor<float> lseGm;
                    outGm.SetGlobalBuffer(reinterpret_cast<__gm__ InputT*>(output));
                    lseGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(lse));
                    matmul::InitOutput<InputT>(outGm[start * 512], count * 512, 0);
                    SetFlag<HardEvent::MTE3_V>(0);
                    WaitFlag<HardEvent::MTE3_V>(0);
                    matmul::InitOutput<float>(lseGm[start], count, float(1.0 / 0.0));
                    SetFlag<HardEvent::MTE3_V>(0);
                    WaitFlag<HardEvent::MTE3_V>(0);
                }
            }
            return;
        }
        this->Run(schedule.tiling, scale, qc, qr, kv, kr, output, lse, workspace, pipe, spans,
                  IsTnd ? uint32_t(totalQuery) : 0, layouts, schedule.kv.layout == uint32_t(MlaSequenceLayout::Tnd),
                  nullptr, nullptr, window);
    }
};
} // namespace flash_attn::mla::Kernel

// BSND/TND PA specializations of the CANN nonquant MLA task ring.
// Source and license: mla_dense.hpp and LICENSE.cann. v6/v10/v11 validate current
// lengths, all effective pages and complete cache spans before TPipe/events.
// Production launchers explicitly select the query layout and mask specialization.

namespace flash_attn::mla::Kernel {
template <typename InputT, bool IsTnd, bool IsCausal = false, bool IsWindow = false>
class MlaPaged : private MlaTaskRing<InputT, /*HasKvLengths=*/true, IsTnd, /*MixedKv=*/false, /*HasUsed=*/false,
                                     /*HasStrides=*/true, /*IsPaged=*/true, IsCausal, IsWindow> {
  public:
    // The caller has passed LoadPagedMetadata on every core, checked Q/qv
    // storage and matching query layout, and bound cache addresses to this call.
    __aicore__ inline void operator()(const MlaPagedKernelSchedule& schedule, float scale, GM_ADDR qc, GM_ADDR qr,
                                      const MlaPagedCacheInputs& cache, GM_ADDR output, GM_ADDR lse, GM_ADDR workspace,
                                      TPipe* pipe, GM_ADDR spans, const MlaBsndInputLayouts& layouts,
                                      MlaWindowKernelParams window = {})
    {
        // TND total and allocation starts have been validated to fit int32.
        const uint32_t totalQuery = IsTnd ? schedule.query.totalTokens.low : 0;
        const uint64_t tokens = IsTnd ? totalQuery : uint64_t(schedule.tiling.batch) * schedule.tiling.sq;
        const uint64_t rows = tokens * schedule.tiling.heads;
        if (rows == 0) {
            return;
        }
        if (schedule.tiling.sk == 0) {
            // PA has positive P even with an empty table/pool. Every effective
            // row has zero KV, so use FA3 device-empty -inf, not dense +inf.
            // One C1V2 initializes the rows. A statically nonempty allocation
            // may launch extra cores; their start is past the output extent.
            if ASCEND_IS_AIV {
                const uint64_t single = (rows + GetSubBlockNum() - 1) / GetSubBlockNum();
                const uint64_t start = GetBlockIdx() * single;
                if (start < rows) {
                    const uint64_t count = rows - start < single ? rows - start : single;
                    GlobalTensor<InputT> outGm;
                    GlobalTensor<float> lseGm;
                    outGm.SetGlobalBuffer(reinterpret_cast<__gm__ InputT*>(output));
                    lseGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(lse));
                    matmul::InitOutput<InputT>(outGm[start * 512], count * 512, 0);
                    SetFlag<HardEvent::MTE3_V>(0);
                    WaitFlag<HardEvent::MTE3_V>(0);
                    matmul::InitOutput<float>(lseGm[start], count, -float(1.0 / 0.0));
                    SetFlag<HardEvent::MTE3_V>(0);
                    WaitFlag<HardEvent::MTE3_V>(0);
                }
            }
            return;
        }
        this->Run(schedule.tiling, scale, qc, qr, reinterpret_cast<GM_ADDR>(cache.v.address),
                  reinterpret_cast<GM_ADDR>(cache.k.address), output, lse, workspace, pipe, spans, totalQuery, &layouts,
                  false, reinterpret_cast<GM_ADDR>(cache.table.address), &cache.layout, window);
    }
};
} // namespace flash_attn::mla::Kernel
