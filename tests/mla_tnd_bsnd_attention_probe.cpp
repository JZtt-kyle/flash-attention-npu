// Isolated TND-Q/BSND-KV attention driver. Fixed CANN headers stay unmodified;
// migrated execution reads validated v4 metadata, never a host schedule.
#include <cmath>
#include "acl/acl.h"
#include "runtime/rt_ffts.h"
#include "kernel_operator.h"
#include "mla/mla_tnd_bsnd_metadata_consumer.h"
#include "mla_probe_batch_launch.h"
using namespace AscendC;
using namespace flash_attn::mla;
#ifdef MLA_DENSE_BF16
using InputT = bfloat16_t;
#else
using InputT = half;
#endif
#ifdef MLA_DENSE_ORACLE
#ifdef MLA_DENSE_BF16
#define MIXED_ENTRY cann_mixed_bf16_probe
#else
#define MIXED_ENTRY cann_mixed_fp16_probe
#endif
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
#include "oracle_tiling.h"
#include "fia_public_define.h"
using fa_base_vector::FLOAT_ZERO;
constexpr float FLOAT_INF = AttentionCommon::ConstInfo::FLOAT_INF;
constexpr uint32_t BUFFER_SIZE_BYTE_1K = AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_1K;
#include "fia_kernel_nonquant_mla.h"
// Source arch22 chooses the KV length parser from Q layout (TND), so the
// isolated oracle takes int64 cumulative KV ends. Its BSH KV format still
// addresses physical b*Sk; no source algorithm/header is modified.
using Types = AttentionCommon::FIAType<InputT, InputT, InputT, InputT, false, false, AttentionCommon::FIA_LAYOUT::TND,
                                       0, false, AttentionCommon::FIA_LAYOUT::BSH>;
using OracleCube = typename std::conditional<g_coreType == AscendC::AIC, FiaBlockCubeNonQuantMla<Types>,
                                             FiaBlockCubeNonQuantMlaDummy<Types>>::type;
using OracleVector = typename std::conditional<g_coreType == AscendC::AIC, FiaBlockVecNonQuantMlaDummy<Types>,
                                               FiaBlockVecNonQuantMla<Types>>::type;
using OracleFd = typename std::conditional<g_coreType == AscendC::AIC, FiaBlockVecFlashDecodeDummy<Types>,
                                           FiaBlockVecFlashDecode<Types>>::type;
#endif
#else
#ifdef MLA_DENSE_BF16
#define MIXED_ENTRY mla_mixed_bf16_probe
#else
#define MIXED_ENTRY mla_mixed_fp16_probe
#endif
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
#include "mla/kernel/mla_tnd_bsnd.hpp"
#include "mla/kernel/mla_tnd_bsnd_metadata.hpp"
#endif
#endif

__global__ __aicore__ __mix__(1, 2) void MIXED_ENTRY(uint64_t ffts, GM_ADDR qc, GM_ADDR qr, GM_ADDR kv, GM_ADDR kr,
                                                     GM_ADDR kvList, GM_ADDR output, GM_ADDR lse, GM_ADDR workspace,
                                                     GM_ADDR metadata, uint32_t words, GM_ADDR cuQ, uint64_t qcount,
                                                     GM_ADDR cuK, uint64_t kcount, MlaTndBsndMetadataContext context,
                                                     MlaTndBsndKernelSchedule sourceSchedule, float scale,
                                                     GM_ADDR sourceQ, GM_ADDR sourceK, bool sourceNeedInit,
                                                     GM_ADDR status, bool checked)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
#ifndef MLA_DENSE_ORACLE
    MlaTndBsndKernelSchedule schedule{};
    if (checked) {
        if (!Kernel::LoadTndBsndMetadata<InputT>(metadata, words, context, cuQ, qcount, cuK, kcount, schedule))
            return;
    } else {
        const auto result =
            Kernel::ReadTndBsndMetadata<InputT>(metadata, words, context, cuQ, qcount, cuK, kcount, schedule);
        uint32_t index = GetBlockIdx();
        if ASCEND_IS_AIV {
            index += GetBlockNum();
        }
        GlobalTensor<uint32_t> state;
        state.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(status) + index * 16);
        state.SetValue(0, uint32_t(result));
        DataCacheCleanAndInvalid<uint32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(state);
        if (result != MlaMetadataReadStatus::Success)
            return;
    }
#endif
    // No attention synchronization, TPipe or writes precede validation.
    SetSyncBaseAddr(ffts);
    TPipe pipe;
#ifdef MLA_DENSE_ORACLE
    const auto& params = sourceSchedule.tiling;
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
    // FillTilingBaseParams carries the host-selected TND output layout.
    tiling.baseParams.outputLayout = uint32_t(AttentionCommon::FIA_LAYOUT::TND);
    tiling.baseParams.actualSeqS1Dims = params.batch;
    tiling.baseParams.actualSeqS2Dims = params.batch;
    tiling.baseParams.needInit = sourceNeedInit;
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
    op.Init(qc, kvList, kvList, nullptr, nullptr, sourceQ, sourceK, nullptr, nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            qr, kr, nullptr, nullptr, output, lse, workspace, &tiling, nullptr, &pipe);
    op.Process();
#else
    Kernel::MlaTndBsnd<InputT> op;
    op(schedule, scale, qc, qr, kv, kr, output, lse, workspace, &pipe,
       metadata + kMlaTndBsndMetadataPrefixWords * sizeof(uint32_t));
#endif
#endif
}

extern "C" int launch_mixed_attention(void* stream, void* qc, void* qr, void* kv, void* kr, void* kvList, void* output,
                                      void* lse, void* workspace, void* metadata, uint32_t words, void* cuQ,
                                      uint64_t qcount, void* cuK, uint64_t kcount,
                                      const MlaTndBsndMetadataContext* context,
                                      const MlaTndBsndKernelSchedule* sourceSchedule, float scale, void* sourceQ,
                                      void* sourceK, bool sourceNeedInit, void* status, bool checked)
{
    if (!context || !sourceSchedule || !std::isfinite(scale) || context->blockDim == 0 || context->blockDim > 26 ||
        (!checked && !status))
        return -1;
    uint64_t ffts = 0;
    uint32_t len = 0;
    auto error = rtGetC2cCtrlAddr(&ffts, &len);
    if (error != RT_ERROR_NONE)
        return error;
    error = aclrtSetOpExecuteTimeOut(15);
    if (error != ACL_SUCCESS)
        return error;
#if !defined(__NPU_ARCH__)
    auto before = batchLaunchCount;
#endif
    MIXED_ENTRY<<<context->blockDim, nullptr, stream>>>(
        ffts, (uint8_t*)qc, (uint8_t*)qr, (uint8_t*)kv, (uint8_t*)kr, (uint8_t*)kvList, (uint8_t*)output, (uint8_t*)lse,
        (uint8_t*)workspace, (uint8_t*)metadata, words, (uint8_t*)cuQ, qcount, (uint8_t*)cuK, kcount, *context,
        *sourceSchedule, scale, (uint8_t*)sourceQ, (uint8_t*)sourceK, sourceNeedInit, (uint8_t*)status, checked);
#if !defined(__NPU_ARCH__)
    return batchLaunchCount == before + 1 ? batchLaunchError : -2;
#else
    return 0;
#endif
}
