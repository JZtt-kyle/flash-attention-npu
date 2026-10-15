// Independent PA FD partial probe; same physical page table as production.
// Adapted from mla_split_kv_attention_probe.cpp.
// Independent FD partial probe. Algorithm headers are unmodified pinned
// CANN blobs (ca18aee610729ca5270a44e2d84c751bc687fbf0). Source receives a real
// compressed triangular bool mask, sparse mode 4/3/0, independently normalized
// pre/next bounds and int64 KV lengths. Only the public source initialization,
// task creation/execution and original FD Block writeback run; no FD merge.
// Fragment descriptors are checked against independent Python key enumeration.
#include <cmath>
#include "kernel_operator.h"
#include "acl/acl.h"
#include "runtime/rt.h"
#include "mla/mla_dense_params.h"
#include "mla/mla_split_kv_params.h"
using namespace AscendC;
using flash_attn::mla::MlaDenseKernelSchedule;
using flash_attn::mla::MlaSplitKvKernelParams;
#ifdef MLA_DENSE_BF16
using InputT = bfloat16_t;
#else
using InputT = half;
#endif
#include "mla_probe_batch_launch.h"
#ifdef MLA_DENSE_ORACLE
#ifdef MLA_DENSE_BF16
#define PARTIAL_ENTRY cann_split_paged_bsnd_attention_bf16
#else
#define PARTIAL_ENTRY cann_split_paged_bsnd_attention
#endif
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
#include "oracle_tiling.h"
#include "fia_public_define.h"
using fa_base_vector::FLOAT_ZERO;
constexpr float FLOAT_INF = AttentionCommon::ConstInfo::FLOAT_INF;
constexpr uint32_t BUFFER_SIZE_BYTE_1K = AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_1K;
#include "fia_kernel_nonquant_mla.h"
using Types = AttentionCommon::FIAType<InputT, InputT, InputT, InputT, true, true>;
using OracleCube = typename std::conditional<g_coreType == AscendC::AIC, FiaBlockCubeNonQuantMla<Types>,
                                             FiaBlockCubeNonQuantMlaDummy<Types>>::type;
using OracleVector = typename std::conditional<g_coreType == AscendC::AIC, FiaBlockVecNonQuantMlaDummy<Types>,
                                               FiaBlockVecNonQuantMla<Types>>::type;
using OracleFd = typename std::conditional<g_coreType == AscendC::AIC, FiaBlockVecFlashDecodeDummy<Types>,
                                           FiaBlockVecFlashDecode<Types>>::type;
#endif
#else
#ifdef MLA_DENSE_BF16
#define PARTIAL_ENTRY mla_split_kv_attention_bf16
#else
#define PARTIAL_ENTRY mla_split_kv_attention
#endif
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
#include "mla/kernel/mla_split_kv.hpp"
#endif
#endif

__global__ __aicore__ __mix__(1, 2) void PARTIAL_ENTRY(uint64_t ffts, GM_ADDR qc, GM_ADDR qr, GM_ADDR kv, GM_ADDR kr,
                                                       GM_ADDR kvList, GM_ADDR output, GM_ADDR lse, GM_ADDR workspace,
                                                       MlaDenseKernelSchedule params, float scale, GM_ADDR lengths,
                                                       GM_ADDR mask, MlaSplitKvKernelParams partial, uint32_t pageSize,
                                                       uint32_t maxPages, GM_ADDR table)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
    SetSyncBaseAddr(ffts);
    TPipe pipe;
    const auto& window = partial.window;
#ifdef MLA_DENSE_ORACLE
    FusedInferAttentionScoreTilingData tiling{};
    tiling.baseParams.bSize = params.batch;
    tiling.baseParams.n2Size = 1;
    tiling.baseParams.gSize = params.heads;
    tiling.baseParams.s1Size = params.sq;
    tiling.baseParams.s2Size = params.sk;
    tiling.baseParams.headDim = 512;
    tiling.baseParams.headDimRope = 64;
    tiling.baseParams.scaleValue = scale;
    tiling.baseParams.usedCoreNum = params.usedCoreNum;
    tiling.baseParams.batchContinuous = 1;
    tiling.baseParams.softmaxLseFlag = 0;
    tiling.baseParams.actualSeqS2Dims = params.batch;
    tiling.baseParams.needInit = 0;
    tiling.maskParams.attenMaskFlag = window.kind != 0;
    tiling.maskParams.attenMaskStride = 2048;
    tiling.maskParams.attenMaskBatchStride = 0;
    tiling.maskParams.sparseMode = window.kind == 2 ? 4 : window.kind == 1 ? 3 : 0;
    tiling.maskParams.preToken = window.kind == 1 ? INT32_MAX : window.left;
    tiling.maskParams.nextToken = window.kind == 1 ? 0 : window.right;
    tiling.pageAttenParams.blockSize = pageSize;
    tiling.pageAttenParams.maxBlockNumPerBatch = maxPages;
    tiling.kvStrideParams.keyBnStride = uint64_t(pageSize) * 512;
    tiling.kvStrideParams.valueBnStride = uint64_t(pageSize) * 512;
    tiling.kvStrideParams.keyRopeBnStride = uint64_t(pageSize) * 64;
    tiling.innerSplitParams.mBaseSize = params.mBaseSize;
    tiling.innerSplitParams.s2BaseSize = params.s2BaseSize;
    tiling.workspaceParams.mm1ResSize = params.mm1ResSize;
    tiling.workspaceParams.mm2ResSize = params.mm2ResSize;
    for (uint32_t i = 0; i < 26; ++i) {
        tiling.outerSplitParams.bN2End[i] = params.bN2End[i];
        tiling.outerSplitParams.gS1End[i] = params.gS1End[i];
        tiling.outerSplitParams.s2End[i] = params.s2End[i];
    }
    uint32_t core = GetBlockIdx();
    if ASCEND_IS_AIV {
        core /= GetSubBlockNum();
    }
    const auto& task = partial.tasks[core];
    tiling.workspaceParams.fdAccumOutSize = (partial.sumOffsetBytes - partial.outOffsetBytes) / sizeof(float);
    tiling.workspaceParams.fdLogSumExpSize = (partial.maxOffsetBytes - partial.sumOffsetBytes) / 2;
    tiling.fdParams.bN2IdxOfFdHead[0] = task.batch;
    tiling.fdParams.gS1IdxOfFdHead[0] = task.m;
    tiling.fdParams.s2SplitNumOfFdHead[0] = params.usedCoreNum;
    FiaKernelNonQuantMla<Types, OracleCube, OracleVector, OracleFd> op;
    op.Init(qc, kvList, kvList, nullptr, mask, nullptr, lengths, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            nullptr, table, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, qr, kr,
            nullptr, nullptr, output, lse, workspace, &tiling, nullptr, &pipe);
    if (task.s2Begin == task.s2End) {
        // Fixed source emits no buffers for an empty fragment. The test
        // harness supplies its independently specified canonical empty slot.
        if ASCEND_IS_AIV {
            const uint32_t halfRows = (params.mBaseSize + 1) / 2;
            const uint32_t start = (GetBlockIdx() % 2) * halfRows;
            if (start < params.mBaseSize) {
                const uint32_t count = Min(halfRows, params.mBaseSize - start);
                const uint64_t row = uint64_t(core) * params.mBaseSize + start;
                matmul::InitOutput<float>(op.accumOutGm[row * 512], count * 512, 0);
                SetFlag<HardEvent::MTE3_V>(0);
                WaitFlag<HardEvent::MTE3_V>(0);
                matmul::InitOutput<float>(op.lseSumFdGm[row * 8], count * 8, 0);
                SetFlag<HardEvent::MTE3_V>(0);
                WaitFlag<HardEvent::MTE3_V>(0);
                matmul::InitOutput<float>(op.lseMaxFdGm[row * 8], count * 8, -float(1.0 / 0.0));
                SetFlag<HardEvent::MTE3_V>(0);
                WaitFlag<HardEvent::MTE3_V>(0);
            }
        }
        return;
    }
    // All methods and arithmetic below belong to the unmodified pinned
    // source. A single-fragment harness supplies endpoints; source still
    // independently calculates sparse bounds, masks, row sizes and statistics.
    op.constInfo.bN2Start = op.constInfo.bN2End = task.batch;
    op.constInfo.gS1Start = op.constInfo.gS1End = task.m;
    op.constInfo.s2Start = task.s2Begin;
    op.constInfo.s2End = task.s2End;
    op.constInfo.coreStartKVSplitPos = core;
    op.GetTaskDealMode(task.batch, task.m, task.s2Begin);
    if (op.curS2Start != task.s2Begin || op.curS2End != task.s2End) {
        Trap();
    }
    if ASCEND_IS_AIV {
        op.vectorService.InitBuffers(&pipe);
        op.vectorService.AllocEventID();
    }
    if ASCEND_IS_AIC {
        op.matmulService.InitBuffers(&pipe);
        op.matmulService.AllocEventID();
    }
    AttentionCommon::RunInfo tasks[3];
    const uint64_t count = task.s2End - task.s2Begin;
    for (uint64_t loop = 0; loop < count + 2; ++loop) {
        if (loop < count) {
            op.CreateTask(loop, task.batch, task.m, task.s2Begin + loop, tasks);
        }
        op.ExecuteTask(loop, tasks);
    }
    if ASCEND_IS_AIV {
        op.vectorService.FreeEventID();
    }
    if ASCEND_IS_AIC {
        op.matmulService.FreeEventID();
    }
#else
    flash_attn::mla::Kernel::MlaSplitKvBsnd<InputT> op;
    op(params, partial, scale, qc, qr, kv, kr, workspace, &pipe, lengths);
#endif
#endif
}

extern "C" int launch_paged_partial(void* stream, void* qc, void* qr, void* kv, void* kr, void* kvList, void* output,
                                    void* lse, void* workspace, const MlaDenseKernelSchedule* params, float scale,
                                    void* lengths, void* mask, const MlaSplitKvKernelParams* partial, uint32_t pageSize,
                                    uint32_t maxPages, void* table)
{
    if (!table || pageSize < 16 || pageSize > 1024 || pageSize % 16 || !params || !partial ||
        partial->window.kind > 2 ||
        (partial->window.kind == 2 &&
         ((partial->window.left < 0 && params->sk != 0) || (partial->window.right < 0 && params->sq != 0))) ||
        !std::isfinite(scale) || !lengths || !mask || params->cvRatio != 2 || params->usedCoreNum == 0 ||
        params->usedCoreNum > 26 || params->blockDim != params->usedCoreNum) {
        return -1;
    }
    if (params->batch == 0 || params->sq == 0) {
        return 0;
    }
    if (!qc || !qr || !kv || !kr || !kvList || !output || !lse || !workspace) {
        return -1;
    }
    uint64_t ffts = 0;
    uint32_t len = 0;
    auto error = rtGetC2cCtrlAddr(&ffts, &len);
    if (error != RT_ERROR_NONE) {
        return error;
    }
    error = aclrtSetOpExecuteTimeOut(15);
    if (error != ACL_SUCCESS) {
        return error;
    }
#if !defined(__NPU_ARCH__)
    auto before = batchLaunchCount;
#endif
    PARTIAL_ENTRY<<<params->blockDim, nullptr, stream>>>(ffts, (uint8_t*)qc, (uint8_t*)qr, (uint8_t*)kv, (uint8_t*)kr,
                                                         (uint8_t*)kvList, (uint8_t*)output, (uint8_t*)lse,
                                                         (uint8_t*)workspace, *params, scale, (uint8_t*)lengths,
                                                         (uint8_t*)mask, *partial, pageSize, maxPages, (uint8_t*)table);
#if !defined(__NPU_ARCH__)
    return batchLaunchCount == before + 1 ? batchLaunchError : -2;
#else
    return 0;
#endif
}
