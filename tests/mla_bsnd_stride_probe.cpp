// Isolated BSND stride attention driver; never linked into production.
// Reuses the fixed-source adapter from mla_used_attention_probe.cpp. Source
// receives independently materialized logical inputs; its headers are unchanged.
#include <cmath>
#include "acl/acl.h"
#include "runtime/rt_ffts.h"
#include "kernel_operator.h"
#include "mla/mla_used_metadata_consumer.h"
#include "mla/mla_bsnd_layout.h"
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
#define STRIDE_ENTRY cann_bsnd_stride_bf16_probe
#else
#define STRIDE_ENTRY cann_bsnd_stride_fp16_probe
#endif
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
#include "oracle_tiling.h"
#include "fia_public_define.h"
using fa_base_vector::FLOAT_ZERO;
constexpr float FLOAT_INF = AttentionCommon::ConstInfo::FLOAT_INF;
constexpr uint32_t BUFFER_SIZE_BYTE_1K = AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_1K;
#include "fia_kernel_nonquant_mla.h"
#endif
#else
#ifdef MLA_DENSE_BF16
#define STRIDE_ENTRY mla_bsnd_stride_bf16_probe
#else
#define STRIDE_ENTRY mla_bsnd_stride_fp16_probe
#endif
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
#include "mla/kernel/mla_used.hpp"
#include "mla/kernel/mla_used_metadata.hpp"
#include "mla/kernel/mla_bsnd_metadata.hpp"
#endif
#endif

#if defined(MLA_DENSE_ORACLE) && defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
template <AttentionCommon::FIA_LAYOUT QLayout, AttentionCommon::FIA_LAYOUT KvLayout>
__aicore__ inline void RunSource(GM_ADDR qc, GM_ADDR qr, GM_ADDR kvList, GM_ADDR kr, GM_ADDR output, GM_ADDR lse,
                                 GM_ADDR workspace, const MlaUsedKernelSchedule& sourceSchedule, float scale,
                                 GM_ADDR sourceQ, GM_ADDR sourceK, bool sourceNeedInit, TPipe& pipe)
{
    using Types = AttentionCommon::FIAType<InputT, InputT, InputT, InputT, false, false, QLayout, 0, false, KvLayout>;
    using OracleCube = typename std::conditional<g_coreType == AscendC::AIC, FiaBlockCubeNonQuantMla<Types>,
                                                 FiaBlockCubeNonQuantMlaDummy<Types>>::type;
    using OracleVector = typename std::conditional<g_coreType == AscendC::AIC, FiaBlockVecNonQuantMlaDummy<Types>,
                                                   FiaBlockVecNonQuantMla<Types>>::type;
    using OracleFd = typename std::conditional<g_coreType == AscendC::AIC, FiaBlockVecFlashDecodeDummy<Types>,
                                               FiaBlockVecFlashDecode<Types>>::type;
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
    tiling.baseParams.outputLayout = uint32_t(QLayout);
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
}
#endif

__global__ __aicore__ __mix__(1, 2) void STRIDE_ENTRY(
    uint64_t ffts, GM_ADDR qc, GM_ADDR qr, GM_ADDR kv, GM_ADDR kr, GM_ADDR kvList, GM_ADDR output, GM_ADDR lse,
    GM_ADDR workspace, GM_ADDR metadata, uint32_t words, MlaDenseMetadataContext denseContext,
    MlaUsedMetadataContext usedContext, MlaUsedMetadataInputs inputs, MlaUsedKernelSchedule sourceSchedule, float scale,
    GM_ADDR sourceQ, GM_ADDR sourceK, bool sourceNeedInit, uint32_t version, MlaBsndInputLayouts layouts)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
#ifndef MLA_DENSE_ORACLE
    MlaUsedKernelSchedule schedule{};
    if (version == 5) {
        if (!Kernel::LoadUsedMetadata<InputT>(metadata, words, usedContext, inputs, schedule))
            return;
    } else if (version == 2) {
        if (!Kernel::LoadBsndKvLengthMetadata<InputT>(metadata, words, denseContext,
                                                      reinterpret_cast<GM_ADDR>(inputs.usedKv.address),
                                                      inputs.usedKv.count, schedule.tiling))
            return;
    } else if (!Kernel::LoadDenseMetadata<InputT>(metadata, words, denseContext, schedule.tiling)) {
        return;
    }
#endif
    SetSyncBaseAddr(ffts);
    TPipe pipe;
#ifdef MLA_DENSE_ORACLE
    using Layout = AttentionCommon::FIA_LAYOUT;
    RunSource<Layout::BSH, Layout::BSH>(qc, qr, kvList, kr, output, lse, workspace, sourceSchedule, scale, sourceQ,
                                        sourceK, sourceNeedInit, pipe);
#else
    if (version == 5) {
        Kernel::MlaUsed<InputT, false, true> op;
        op(schedule, scale, qc, qr, kv, kr, output, lse, workspace, &pipe,
           metadata + kMlaUsedMetadataPrefixWords * sizeof(uint32_t), &layouts);
    } else if (version == 2) {
        Kernel::MlaDense<InputT, true, true> op;
        op(schedule.tiling, scale, qc, qr, kv, kr, output, lse, workspace, &pipe,
           metadata + kMlaBsndKvLengthMetadataPrefixWords * sizeof(uint32_t), &layouts);
    } else {
        Kernel::MlaDense<InputT, false, true> op;
        op(schedule.tiling, scale, qc, qr, kv, kr, output, lse, workspace, &pipe, nullptr, &layouts);
    }
#endif
#endif
}

extern "C" int launch_bsnd_stride(void* stream, void* qc, void* qr, void* kv, void* kr, void* kvList, void* output,
                                  void* lse, void* workspace, void* metadata, uint32_t words,
                                  const MlaDenseMetadataContext* denseContext,
                                  const MlaUsedMetadataContext* usedContext, const MlaUsedMetadataInputs* inputs,
                                  const MlaUsedKernelSchedule* sourceSchedule, float scale, void* sourceQ,
                                  void* sourceK, bool sourceNeedInit, uint32_t blocks, uint32_t version,
                                  const MlaBsndTensorView* views)
{
    if (!denseContext || !usedContext || !inputs || !sourceSchedule || !views || !std::isfinite(scale) || blocks == 0 ||
        blocks > 26 || (version != 1 && version != 2 && version != 5) || usedContext->query.layout != 0 ||
        usedContext->kv.layout != 0)
        return -1;
    const auto& p = sourceSchedule->tiling;
    if (version != 5 && (p.batch == 0 || p.sq == 0 || p.sk == 0))
        return -1; // Dense global-empty initialization belongs to the host entry.
    MlaBsndInputLayouts layouts{};
#if !defined(__NPU_ARCH__)
    MlaBsndTensorLayout* targets[] = {&layouts.q, &layouts.qv, &layouts.k, &layouts.v};
    for (uint32_t i = 0; i < 4; ++i) {
        if (views[i].sizes[0] != p.batch || views[i].sizes[1] != (i < 2 ? p.sq : p.sk) ||
            views[i].sizes[2] != (i < 2 ? p.heads : 1) || views[i].sizes[3] != (i % 2 == 0 ? 64 : 512) ||
            BuildBsndTensorLayout(views[i], *targets[i]) != MlaLayoutStatus::Success)
            return -3;
    }
#endif
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
    STRIDE_ENTRY<<<blocks, nullptr, stream>>>(
        ffts, (uint8_t*)qc, (uint8_t*)qr, (uint8_t*)kv, (uint8_t*)kr, (uint8_t*)kvList, (uint8_t*)output, (uint8_t*)lse,
        (uint8_t*)workspace, (uint8_t*)metadata, words, *denseContext, *usedContext, *inputs, *sourceSchedule, scale,
        (uint8_t*)sourceQ, (uint8_t*)sourceK, sourceNeedInit, version, layouts);
#if !defined(__NPU_ARCH__)
    return batchLaunchCount == before + 1 ? batchLaunchError : -2;
#else
    return 0;
#endif
}
