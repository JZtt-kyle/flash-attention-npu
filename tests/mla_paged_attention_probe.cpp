// Independent BSND/TND PA attention probe, never linked into the public extension.
// Oracle instantiates pinned unmodified CANN PA/non-FD headers. Q/row/table
// strides outside its ABI are materialized by the test before launch/timing.
#include <cmath>
#include "acl/acl.h"
#include "runtime/rt_ffts.h"
#include "kernel_operator.h"
#include "mla/mla_paged_metadata_consumer.h"
#include "mla/mla_tnd_layout.h"
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
#define PAGED_ENTRY cann_paged_attention_bf16_probe
#else
#define PAGED_ENTRY cann_paged_attention_fp16_probe
#endif
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
#include "oracle_tiling.h"
#include "fia_public_define.h"
using fa_base_vector::FLOAT_ZERO;
constexpr float FLOAT_INF = AttentionCommon::ConstInfo::FLOAT_INF;
constexpr uint32_t BUFFER_SIZE_BYTE_1K = AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_1K;
#include "fia_kernel_nonquant_mla.h"
template <AttentionCommon::FIA_LAYOUT QLayout>
__aicore__ inline void RunSource(GM_ADDR qc, GM_ADDR qr, GM_ADDR kvList, GM_ADDR kr, GM_ADDR output, GM_ADDR lse,
                                 GM_ADDR workspace, const MlaPagedKernelSchedule& sourceSchedule, float scale,
                                 GM_ADDR sourceQ, GM_ADDR sourceK, const MlaPagedCacheInputs& cache,
                                 bool sourceNeedInit, TPipe& pipe)
{
    using Types = AttentionCommon::FIAType<InputT, InputT, InputT, InputT, true, false, QLayout, 0, false,
                                           AttentionCommon::FIA_LAYOUT::BSH>;
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
    tiling.pageAttenParams.blockSize = sourceSchedule.pageSize;
    tiling.pageAttenParams.maxBlockNumPerBatch = cache.layout.maxPages;
    tiling.kvStrideParams.keyBnStride = cache.layout.v.batchStride;
    tiling.kvStrideParams.valueBnStride = cache.layout.v.batchStride;
    tiling.kvStrideParams.keyRopeBnStride = cache.layout.k.batchStride;
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
            nullptr, nullptr, reinterpret_cast<GM_ADDR>(cache.table.address), nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr, nullptr, nullptr, qr, kr, nullptr, nullptr, output, lse, workspace, &tiling,
            nullptr, &pipe);
    op.Process();
}
#endif
#else
#ifdef MLA_DENSE_BF16
#define PAGED_ENTRY mla_paged_attention_bf16_probe
#else
#define PAGED_ENTRY mla_paged_attention_fp16_probe
#endif
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
#include "mla/kernel/mla_paged.hpp"
#endif
#endif

__global__ __aicore__ __mix__(1, 2) void PAGED_ENTRY(uint64_t ffts, GM_ADDR qc, GM_ADDR qr, GM_ADDR kvList,
                                                     GM_ADDR output, GM_ADDR lse, GM_ADDR workspace, GM_ADDR metadata,
                                                     uint32_t words, MlaPagedMetadataContext context,
                                                     MlaPagedMetadataInputs inputs, MlaPagedCacheInputs cache,
                                                     MlaPagedKernelSchedule sourceSchedule, MlaBsndInputLayouts layouts,
                                                     float scale, GM_ADDR sourceQ, GM_ADDR sourceK, bool sourceNeedInit,
                                                     GM_ADDR status, bool checked)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
#ifndef MLA_DENSE_ORACLE
    MlaPagedKernelSchedule schedule{};
    if (checked && !Kernel::LoadPagedMetadata<InputT>(metadata, words, context, inputs, cache, schedule))
        return;
    const auto result = checked ? MlaMetadataReadStatus::Success
                                : Kernel::ReadPagedMetadata<InputT>(metadata, words, context, inputs, cache, schedule);
    if (status != nullptr) {
        uint32_t index = GetBlockIdx();
        if ASCEND_IS_AIV {
            index += GetBlockNum();
        }
        GlobalTensor<uint32_t> state;
        state.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(status) + index * 16);
        state.SetValue(0, uint32_t(result));
        DataCacheCleanAndInvalid<uint32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(state);
    }
    if (result != MlaMetadataReadStatus::Success)
        return;
#endif
    SetSyncBaseAddr(ffts);
    TPipe pipe;
#ifdef MLA_DENSE_ORACLE
    using Layout = AttentionCommon::FIA_LAYOUT;
    if (context.query.layout == 1) {
        // PA keeps KV lengths non-cumulative even with TND query prefixes.
        RunSource<Layout::TND>(qc, qr, kvList, reinterpret_cast<GM_ADDR>(cache.k.address), output, lse, workspace,
                               sourceSchedule, scale, sourceQ, sourceK, cache, sourceNeedInit, pipe);
    } else {
        RunSource<Layout::BSH>(qc, qr, kvList, reinterpret_cast<GM_ADDR>(cache.k.address), output, lse, workspace,
                               sourceSchedule, scale, sourceQ, sourceK, cache, sourceNeedInit, pipe);
    }
#else
    GM_ADDR spans = metadata + kMlaPagedMetadataPrefixWords * sizeof(uint32_t);
    if (context.query.layout == 1) {
        Kernel::MlaPagedTnd<InputT> op;
        op(schedule, scale, qc, qr, cache, output, lse, workspace, &pipe, spans, layouts);
    } else {
        Kernel::MlaPagedBsnd<InputT> op;
        op(schedule, scale, qc, qr, cache, output, lse, workspace, &pipe, spans, layouts);
    }
#endif
#endif
}

extern "C" uint32_t build_attention_cache(const MlaPageTableView* table, const MlaBsndTensorView* k,
                                          const MlaBsndTensorView* v, MlaPagedCacheLayout* output)
{
#if !defined(__NPU_ARCH__)
    return uint32_t(BuildPagedCacheLayout(*table, *k, *v, *output));
#else
    return 0;
#endif
}

extern "C" int launch_paged_attention(void* stream, void* qc, void* qr, void* kvList, void* output, void* lse,
                                      void* workspace, void* metadata, uint32_t words,
                                      const MlaPagedMetadataContext* context, const MlaPagedMetadataInputs* inputs,
                                      const MlaPagedCacheInputs* cache, const MlaPagedKernelSchedule* sourceSchedule,
                                      const MlaBsndTensorView* views, float scale, void* sourceQ, void* sourceK,
                                      bool sourceNeedInit, void* status, bool checked, uint32_t blocks)
{
    if (!context || !inputs || !cache || !sourceSchedule || !views || !std::isfinite(scale) || blocks == 0 ||
        blocks > 26 || context->query.layout > 1)
        return -1;
    MlaBsndInputLayouts layouts{};
#if !defined(__NPU_ARCH__)
    MlaBsndTensorLayout* targets[] = {&layouts.q, &layouts.qv};
    const void* pointers[] = {qr, qc};
    const bool tnd = context->query.layout == 1;
    for (uint32_t i = 0; i < 2; ++i) {
        const auto& v = views[i];
        if (v.sizes[0] != (tnd ? 1 : context->batch) || v.sizes[2] != context->heads ||
            v.sizes[3] != (i == 0 ? 64 : 512))
            return -3;
#ifndef MLA_DENSE_ORACLE
        const uint64_t tokens =
            uint64_t(context->query.totalTokens.low) | (uint64_t(context->query.totalTokens.high) << 32);
        if (v.sizes[1] != (tnd ? tokens : context->query.maxLength))
            return -3;
#endif
        if (tnd) {
            const MlaTndTensorView view{{v.sizes[1], v.sizes[2], v.sizes[3]},
                                        {v.strides[1], v.strides[2], v.strides[3]},
                                        v.storageOffset,
                                        v.storageBytes};
            MlaTndTensorLayout layout{};
            if (BuildTndTensorLayout(view, layout) != MlaLayoutStatus::Success)
                return -3;
            *targets[i] = {0, layout.tokenStride, layout.headStride, layout.spanElements};
        } else if (BuildBsndTensorLayout(v, *targets[i]) != MlaLayoutStatus::Success) {
            return -3;
        }
        const auto span = targets[i]->spanElements;
        if (span && (!pointers[i] || uintptr_t(pointers[i]) > UINTPTR_MAX - span * sizeof(InputT)))
            return -3;
    }
#ifdef MLA_DENSE_ORACLE
    if (sourceSchedule->tiling.batch == 0 || sourceSchedule->tiling.sq == 0 || sourceSchedule->tiling.sk == 0)
        return -4; // Source does not accept a zero-work tiling/empty table.
#endif
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
    PAGED_ENTRY<<<blocks, nullptr, stream>>>(ffts, (uint8_t*)qc, (uint8_t*)qr, (uint8_t*)kvList, (uint8_t*)output,
                                             (uint8_t*)lse, (uint8_t*)workspace, (uint8_t*)metadata, words, *context,
                                             *inputs, *cache, *sourceSchedule, layouts, scale, (uint8_t*)sourceQ,
                                             (uint8_t*)sourceK, sourceNeedInit, (uint8_t*)status, checked);
#if !defined(__NPU_ARCH__)
    return batchLaunchCount == before + 1 ? batchLaunchError : -2;
#else
    return 0;
#endif
}
