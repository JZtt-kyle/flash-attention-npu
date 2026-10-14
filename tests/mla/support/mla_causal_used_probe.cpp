// Test-only BSND Q-used source wrapper. Independent non-FD causal probe. The source algorithm headers are unmodified
// pinned CANN blobs (ca18aee610729ca5270a44e2d84c751bc687fbf0).
// Source receives an actual 2048x2048 triangular bool mask, sparse mode 3,
// independent int64 KV lengths, and the same whole-M schedule as migration.
#include <cmath>
#include "kernel_operator.h"
#include "acl/acl.h"
#include "runtime/rt.h"
#include "mla/mla_dense_params.h"
using namespace AscendC;
using flash_attn::mla::MlaDenseKernelSchedule;
#ifdef MLA_DENSE_BF16
using InputT = bfloat16_t;
#else
using InputT = half;
#endif
#include "mla_probe_batch_launch.h"
#ifdef MLA_DENSE_BF16
#define CAUSAL_ENTRY cann_causal_used_bf16
#else
#define CAUSAL_ENTRY cann_causal_used
#endif
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
#include "oracle_tiling.h"
#include "fia_public_define.h"
using fa_base_vector::FLOAT_ZERO;
constexpr float FLOAT_INF = AttentionCommon::ConstInfo::FLOAT_INF;
constexpr uint32_t BUFFER_SIZE_BYTE_1K = AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_1K;
#include "fia_kernel_nonquant_mla.h"
using Types = AttentionCommon::FIAType<InputT, InputT, InputT, InputT>;
using OracleCube = typename std::conditional<g_coreType == AscendC::AIC, FiaBlockCubeNonQuantMla<Types>,
                                             FiaBlockCubeNonQuantMlaDummy<Types>>::type;
using OracleVector = typename std::conditional<g_coreType == AscendC::AIC, FiaBlockVecNonQuantMlaDummy<Types>,
                                               FiaBlockVecNonQuantMla<Types>>::type;
using OracleFd = typename std::conditional<g_coreType == AscendC::AIC, FiaBlockVecFlashDecodeDummy<Types>,
                                           FiaBlockVecFlashDecode<Types>>::type;
#endif

__global__ __aicore__ __mix__(1, 2) void CAUSAL_ENTRY(uint64_t ffts, GM_ADDR qc, GM_ADDR qr, GM_ADDR kv, GM_ADDR kr,
                                                      GM_ADDR kvList, GM_ADDR output, GM_ADDR lse, GM_ADDR workspace,
                                                      MlaDenseKernelSchedule params, float scale, GM_ADDR lengths,
                                                      GM_ADDR mask, GM_ADDR queryLengths)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
    SetSyncBaseAddr(ffts);
    TPipe pipe;
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
    tiling.baseParams.softmaxLseFlag = 1;
    tiling.baseParams.actualSeqS2Dims = params.batch;
    tiling.baseParams.actualSeqS1Dims = params.batch;
    tiling.baseParams.needInit = 1;
    tiling.maskParams.attenMaskFlag = 1;
    tiling.maskParams.attenMaskStride = 2048;
    tiling.maskParams.attenMaskBatchStride = 0;
    tiling.maskParams.sparseMode = 3;
    tiling.maskParams.preToken = INT32_MAX;
    tiling.maskParams.nextToken = 0;
    tiling.innerSplitParams.mBaseSize = params.mBaseSize;
    tiling.innerSplitParams.s2BaseSize = params.s2BaseSize;
    tiling.workspaceParams.mm1ResSize = params.mm1ResSize;
    tiling.workspaceParams.mm2ResSize = params.mm2ResSize;
    for (uint32_t i = 0; i < 26; ++i) {
        tiling.outerSplitParams.bN2End[i] = params.bN2End[i];
        tiling.outerSplitParams.gS1End[i] = params.gS1End[i];
        tiling.outerSplitParams.s2End[i] = params.s2End[i];
    }
    FiaKernelNonQuantMla<Types, OracleCube, OracleVector, OracleFd> op;
    op.Init(qc, kvList, kvList, nullptr, mask, queryLengths, lengths, nullptr, nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            qr, kr, nullptr, nullptr, output, lse, workspace, &tiling, nullptr, &pipe);
    op.Process();
#endif
}

extern "C" int launch_causal_used(void* stream, void* qc, void* qr, void* kv, void* kr, void* kvList, void* output,
                                  void* lse, void* workspace, const MlaDenseKernelSchedule* params, float scale,
                                  void* lengths, void* mask, void* queryLengths)
{
    if (!params || !std::isfinite(scale) || !lengths || !mask || params->cvRatio != 2 || params->usedCoreNum == 0 ||
        params->usedCoreNum > 26 || params->blockDim < params->usedCoreNum || params->blockDim > 26 || !queryLengths) {
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
    CAUSAL_ENTRY<<<params->blockDim, nullptr, stream>>>(
        ffts, (uint8_t*)qc, (uint8_t*)qr, (uint8_t*)kv, (uint8_t*)kr, (uint8_t*)kvList, (uint8_t*)output, (uint8_t*)lse,
        (uint8_t*)workspace, *params, scale, (uint8_t*)lengths, (uint8_t*)mask, (uint8_t*)queryLengths);
#if !defined(__NPU_ARCH__)
    return batchLaunchCount == before + 1 ? batchLaunchError : -2;
#else
    return 0;
#endif
}
