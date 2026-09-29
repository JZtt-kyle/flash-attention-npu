// Test-only dense launcher, built twice. Never linked into the FA3 extension.
#include <cstdint>
#include <cmath>
#include "acl/acl.h"
#include "runtime/rt_ffts.h"
#include "kernel_operator.h"
#include "mla/mla_dense_params.h"
using namespace AscendC;
using flash_attn::mla::MlaDenseKernelSchedule;
#ifdef MLA_DENSE_BF16
using InputT = bfloat16_t;
#else
using InputT = half;
#endif

#if !defined(__NPU_ARCH__)
// Bisheng's standalone <<<>>> launcher defaults to NORMAL_MODE. Only this
// isolated probe DSO wraps the runtime call to preserve source BATCH_MODE.
// The counter below makes a compiler/runtime path that bypasses it fail.
static uint64_t batchLaunchCount = 0;
static aclError batchLaunchError = ACL_SUCCESS;
extern "C" aclError __real_aclrtLaunchKernelWithHostArgs(aclrtFuncHandle, uint32_t, aclrtStream, aclrtLaunchKernelCfg*,
                                                         void*, size_t, aclrtPlaceHolderInfo*, size_t);
extern "C" aclError __wrap_aclrtLaunchKernelWithHostArgs(aclrtFuncHandle function, uint32_t blocks, aclrtStream stream,
                                                         aclrtLaunchKernelCfg* config, void* args, size_t argsSize,
                                                         aclrtPlaceHolderInfo* placeholders, size_t placeholderCount)
{
    ++batchLaunchCount;
    aclrtLaunchKernelAttr attrs[9]{};
    size_t count = 0;
    if (config) {
        if (config->numAttrs > 8 || (config->numAttrs && !config->attrs)) {
            batchLaunchError = ACL_ERROR_INVALID_PARAM;
            return batchLaunchError;
        }
        for (size_t i = 0; i < config->numAttrs; ++i) {
            if (config->attrs[i].id != ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE) {
                attrs[count++] = config->attrs[i];
            }
        }
    }
    attrs[count].id = ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE;
    attrs[count++].value.schemMode = 1;
    aclrtLaunchKernelCfg batchConfig{attrs, count};
    batchLaunchError = __real_aclrtLaunchKernelWithHostArgs(function, blocks, stream, &batchConfig, args, argsSize,
                                                            placeholders, placeholderCount);
    return batchLaunchError;
}
#endif

#ifdef MLA_DENSE_ORACLE
#ifdef MLA_DENSE_BF16
#define DENSE_ENTRY cann_dense_bf16_probe
#else
#define DENSE_ENTRY cann_dense_probe
#endif
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
#include "oracle_tiling.h"
#include "fia_public_define.h"
// Constants normally introduced by the FIA translation unit's earlier headers.
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
#else
#ifdef MLA_DENSE_BF16
#define DENSE_ENTRY mla_dense_bf16_probe
#else
#define DENSE_ENTRY mla_dense_probe
#endif
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
#include "mla/kernel/mla_dense.hpp"
#endif
#endif

__global__ __aicore__ __mix__(1, 2) void DENSE_ENTRY(uint64_t ffts, GM_ADDR qc, GM_ADDR qr, GM_ADDR kv, GM_ADDR kr,
                                                     GM_ADDR kvList, GM_ADDR output, GM_ADDR lse, GM_ADDR workspace,
                                                     MlaDenseKernelSchedule params, float scale)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
    SetSyncBaseAddr(ffts);
    TPipe pipe;
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
    tiling.baseParams.softmaxLseFlag = 1;
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
    op.Init(qc, kvList, kvList, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            qr, kr, nullptr, nullptr, output, lse, workspace, &tiling, nullptr, &pipe);
    op.Process();
#else
    flash_attn::mla::Kernel::MlaDense<InputT> op;
    op(params, scale, qc, qr, kv, kr, output, lse, workspace, &pipe);
#endif
#endif
}

extern "C" int launch_dense(void* stream, void* qc, void* qr, void* kv, void* kr, void* kvList, void* output, void* lse,
                            void* workspace, const MlaDenseKernelSchedule* params, float scale)
{
    if (!params || !qc || !qr || !kv || !kr || !kvList || !output || !lse || !workspace || !std::isfinite(scale) ||
        params->batch == 0 || params->sq == 0 || params->heads == 0 || uint64_t(params->sq) * params->heads > 256 ||
        params->sk == 0 || params->sk > 512 || params->cvRatio != 2 || params->usedCoreNum == 0 ||
        params->usedCoreNum > 26 || params->blockDim < params->usedCoreNum) {
        return -1;
    }
    // This standalone probe, like the validated production range, is C1V2 only.
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
    DENSE_ENTRY<<<params->blockDim, nullptr, stream>>>(ffts, (uint8_t*)qc, (uint8_t*)qr, (uint8_t*)kv, (uint8_t*)kr,
                                                       (uint8_t*)kvList, (uint8_t*)output, (uint8_t*)lse,
                                                       (uint8_t*)workspace, *params, scale);
#if !defined(__NPU_ARCH__)
    return batchLaunchCount == before + 1 ? batchLaunchError : -2;
#else
    return 0;
#endif
}
