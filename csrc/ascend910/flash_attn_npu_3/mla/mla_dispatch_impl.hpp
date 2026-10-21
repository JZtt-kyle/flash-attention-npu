// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
// Layout variants share the dispatch implementation, as in fwd_dispatch_impl.hpp.
// autogen/ selects one dtype/layout per compiler unit; numerical layers are reused.
#pragma once
#include "kernel_operator.h"
#include "mla_dispatch.h"
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
#include "mla_kernel.hpp"
#endif

namespace flash_attn::mla {
template <typename DType, bool HasStrides>
__global__ __aicore__ __mix__(1, 2) void DenseKernel(uint64_t ffts, GM_ADDR qc, GM_ADDR qr, GM_ADDR kv, GM_ADDR kr,
                                                     GM_ADDR output, GM_ADDR lse, GM_ADDR workspace, GM_ADDR metadata,
                                                     GM_ADDR cacheSeqlens, MlaDenseMetadataContext context, float scale,
                                                     bool hasKvLengths, MlaBsndInputLayouts layouts)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
    MlaDenseKernelSchedule schedule{};
    if (hasKvLengths) {
        if (!Kernel::LoadBsndKvLengthMetadata<DType>(metadata, kMlaBsndKvLengthMetadataPrefixWords + context.batch,
                                                     context, cacheSeqlens, cacheSeqlens ? context.batch : 0,
                                                     schedule)) {
            return;
        }
    } else {
        if (!Kernel::LoadDenseMetadata<DType>(metadata, kMlaDenseMetadataWords, context, schedule) ||
            !Kernel::CheckDenseCacheLengths(cacheSeqlens, context.batch, context.sk)) {
            return;
        }
    }
    if (context.batch == 0 || context.sq == 0 || context.sk == 0) {
        return;
    }
    AscendC::SetSyncBaseAddr(ffts);
    AscendC::TPipe pipe;
    if (hasKvLengths) {
        Kernel::MlaDense<DType, true, HasStrides> op;
        op(schedule, scale, qc, qr, kv, kr, output, lse, workspace, &pipe,
           metadata + kMlaBsndKvLengthMetadataPrefixWords * sizeof(int32_t), &layouts);
    } else {
        Kernel::MlaDense<DType, false, HasStrides> op;
        op(schedule, scale, qc, qr, kv, kr, output, lse, workspace, &pipe, nullptr, &layouts);
    }
#endif
}

template <typename DType>
aclError LaunchDense(const MlaLaunchArgs& a)
{
    MlaBatchLaunchScope launch;
    if (a.hasStrides) {
        DenseKernel<DType, true><<<a.context.blockDim, nullptr, a.stream>>>(
            a.ffts, a.qc, a.qr, a.kv, a.kr, a.output, a.lse, a.workspace, a.metadata, a.cacheSeqlens, a.context,
            a.scale, a.hasKvLengths, a.layouts);
    } else {
        DenseKernel<DType, false><<<a.context.blockDim, nullptr, a.stream>>>(
            a.ffts, a.qc, a.qr, a.kv, a.kr, a.output, a.lse, a.workspace, a.metadata, a.cacheSeqlens, a.context,
            a.scale, a.hasKvLengths, a.layouts);
    }
    return launch.Result();
}
} // namespace flash_attn::mla

namespace flash_attn::mla {
template <typename DType, bool HasStrides>
__global__ __aicore__ __mix__(1, 2) void TndKernel(uint64_t ffts, GM_ADDR qc, GM_ADDR qr, GM_ADDR kv, GM_ADDR kr,
                                                   GM_ADDR output, GM_ADDR lse, GM_ADDR workspace, GM_ADDR metadata,
                                                   GM_ADDR cuQuery, GM_ADDR cuKv, MlaTndMetadataContext context,
                                                   float scale, MlaBsndInputLayouts layouts)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
    MlaTndKernelSchedule schedule{};
    const uint32_t words = kMlaTndMetadataPrefixWords + kMlaTndMetadataSequenceWords * context.batch;
    const uint64_t count = uint64_t(context.batch) + 1;
    if (!Kernel::LoadTndMetadata<DType>(metadata, words, context, cuQuery, count, cuKv, count, schedule)) {
        return;
    }
    // Every core validates before TPipe, attention synchronization or writes,
    // including empty total capacity. The TND Kernel implements empty O/LSE.
    AscendC::SetSyncBaseAddr(ffts);
    AscendC::TPipe pipe;
    Kernel::MlaTnd<DType, HasStrides> op;
    op(schedule, scale, qc, qr, kv, kr, output, lse, workspace, &pipe,
       metadata + kMlaTndMetadataPrefixWords * sizeof(int32_t), &layouts);
#endif
}

template <typename DType>
aclError LaunchTnd(const MlaTndLaunchArgs& a)
{
    MlaBatchLaunchScope launch;
    if (a.hasStrides) {
        TndKernel<DType, true><<<a.context.blockDim, nullptr, a.stream>>>(a.ffts, a.qc, a.qr, a.kv, a.kr, a.output,
                                                                          a.lse, a.workspace, a.metadata, a.cuQuery,
                                                                          a.cuKv, a.context, a.scale, a.layouts);
    } else {
        TndKernel<DType, false><<<a.context.blockDim, nullptr, a.stream>>>(a.ffts, a.qc, a.qr, a.kv, a.kr, a.output,
                                                                           a.lse, a.workspace, a.metadata, a.cuQuery,
                                                                           a.cuKv, a.context, a.scale, a.layouts);
    }
    return launch.Result();
}
} // namespace flash_attn::mla

namespace flash_attn::mla {
template <typename DType, bool HasStrides>
__global__ __aicore__ __mix__(1, 2) void MixedKernel(uint64_t ffts, GM_ADDR qc, GM_ADDR qr, GM_ADDR kv, GM_ADDR kr,
                                                     GM_ADDR output, GM_ADDR lse, GM_ADDR workspace, GM_ADDR metadata,
                                                     GM_ADDR cuQuery, GM_ADDR cacheSeqlens,
                                                     MlaTndBsndMetadataContext context, float scale,
                                                     MlaBsndInputLayouts layouts)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
    MlaTndBsndKernelSchedule schedule{};
    const uint32_t words = kMlaTndBsndMetadataPrefixWords + kMlaTndBsndMetadataSequenceWords * context.batch;
    if (!Kernel::LoadTndBsndMetadata<DType>(metadata, words, context, cuQuery, uint64_t(context.batch) + 1,
                                            cacheSeqlens, context.batch, schedule)) {
        return;
    }
    // Every core validates before TPipe, attention synchronization or writes,
    // including empty capacity. The mixed Kernel implements empty O/LSE.
    AscendC::SetSyncBaseAddr(ffts);
    AscendC::TPipe pipe;
    Kernel::MlaTndBsnd<DType, HasStrides> op;
    op(schedule, scale, qc, qr, kv, kr, output, lse, workspace, &pipe,
       metadata + kMlaTndBsndMetadataPrefixWords * sizeof(int32_t), &layouts);
#endif
}

template <typename DType>
aclError LaunchTndBsnd(const MlaTndBsndLaunchArgs& a)
{
    MlaBatchLaunchScope launch;
    if (a.hasStrides) {
        MixedKernel<DType, true><<<a.context.blockDim, nullptr, a.stream>>>(
            a.ffts, a.qc, a.qr, a.kv, a.kr, a.output, a.lse, a.workspace, a.metadata, a.cuQuery, a.cacheSeqlens,
            a.context, a.scale, a.layouts);
    } else {
        MixedKernel<DType, false><<<a.context.blockDim, nullptr, a.stream>>>(
            a.ffts, a.qc, a.qr, a.kv, a.kr, a.output, a.lse, a.workspace, a.metadata, a.cuQuery, a.cacheSeqlens,
            a.context, a.scale, a.layouts);
    }
    return launch.Result();
}
} // namespace flash_attn::mla

namespace flash_attn::mla {
template <typename DType, bool HasStrides, bool IsCausal = false, bool IsWindow = false>
__global__ __aicore__ __mix__(1, 2) void BsndUsedKernel(uint64_t ffts, GM_ADDR qc, GM_ADDR qr, GM_ADDR kv, GM_ADDR kr,
                                                        GM_ADDR output, GM_ADDR lse, GM_ADDR workspace,
                                                        GM_ADDR metadata, MlaUsedMetadataContext context,
                                                        MlaUsedMetadataInputs inputs, float scale,
                                                        MlaBsndInputLayouts layouts, MlaWindowKernelParams window)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
    MlaUsedKernelSchedule schedule{};
    const uint32_t words = kMlaUsedMetadataPrefixWords + kMlaUsedMetadataSequenceWords * context.batch +
                           (IsWindow ? kMlaWindowMetadataTailWords : 0);
    if (!Kernel::LoadUsedMetadata<DType, IsCausal, false, false, IsWindow>(metadata, words, context, inputs, schedule,
                                                                           window)) {
        return;
    }
    // Every core validates before TPipe, synchronization or output writes,
    // even for empty capacity or all-zero used lengths.
    AscendC::SetSyncBaseAddr(ffts);
    AscendC::TPipe pipe;
    Kernel::MlaUsed<DType, false, HasStrides, IsCausal, IsWindow> op;
    op(schedule, scale, qc, qr, kv, kr, output, lse, workspace, &pipe,
       metadata + kMlaUsedMetadataPrefixWords * sizeof(int32_t), &layouts, window);
#endif
}

template <typename DType>
aclError LaunchBsndUsed(const MlaUsedLaunchArgs& a)
{
    MlaBatchLaunchScope launch;
    if (a.isWindow && a.hasStrides) {
        BsndUsedKernel<DType, true, false, true>
            <<<a.context.blockDim, nullptr, a.stream>>>(a.ffts, a.qc, a.qr, a.kv, a.kr, a.output, a.lse, a.workspace,
                                                        a.metadata, a.context, a.inputs, a.scale, a.layouts, a.window);
    } else if (a.isWindow) {
        BsndUsedKernel<DType, false, false, true>
            <<<a.context.blockDim, nullptr, a.stream>>>(a.ffts, a.qc, a.qr, a.kv, a.kr, a.output, a.lse, a.workspace,
                                                        a.metadata, a.context, a.inputs, a.scale, a.layouts, a.window);
    } else if (a.causal && a.hasStrides) {
        BsndUsedKernel<DType, true, true>
            <<<a.context.blockDim, nullptr, a.stream>>>(a.ffts, a.qc, a.qr, a.kv, a.kr, a.output, a.lse, a.workspace,
                                                        a.metadata, a.context, a.inputs, a.scale, a.layouts, a.window);
    } else if (a.causal) {
        BsndUsedKernel<DType, false, true>
            <<<a.context.blockDim, nullptr, a.stream>>>(a.ffts, a.qc, a.qr, a.kv, a.kr, a.output, a.lse, a.workspace,
                                                        a.metadata, a.context, a.inputs, a.scale, a.layouts, a.window);
    } else if (a.hasStrides) {
        BsndUsedKernel<DType, true>
            <<<a.context.blockDim, nullptr, a.stream>>>(a.ffts, a.qc, a.qr, a.kv, a.kr, a.output, a.lse, a.workspace,
                                                        a.metadata, a.context, a.inputs, a.scale, a.layouts, a.window);
    } else {
        BsndUsedKernel<DType, false>
            <<<a.context.blockDim, nullptr, a.stream>>>(a.ffts, a.qc, a.qr, a.kv, a.kr, a.output, a.lse, a.workspace,
                                                        a.metadata, a.context, a.inputs, a.scale, a.layouts, a.window);
    }
    return launch.Result();
}
} // namespace flash_attn::mla

namespace flash_attn::mla {
template <typename DType, bool HasStrides, bool IsCausal = false, bool IsMixedCausal = false, bool IsWindow = false,
          bool IsMixedWindow = false>
__global__ __aicore__ __mix__(1, 2) void TndUsedKernel(uint64_t ffts, GM_ADDR qc, GM_ADDR qr, GM_ADDR kv, GM_ADDR kr,
                                                       GM_ADDR output, GM_ADDR lse, GM_ADDR workspace, GM_ADDR metadata,
                                                       MlaUsedMetadataContext context, MlaUsedMetadataInputs inputs,
                                                       float scale, MlaBsndInputLayouts layouts,
                                                       MlaWindowKernelParams window)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
    MlaUsedKernelSchedule schedule{};
    const uint32_t words = kMlaUsedMetadataPrefixWords + kMlaUsedMetadataSequenceWords * context.batch +
                           (IsWindow ? kMlaWindowMetadataTailWords : 0);
    if (!Kernel::LoadUsedMetadata < DType, IsCausal, IsCausal && !IsMixedCausal, IsMixedCausal, IsWindow,
        IsWindow && !IsMixedWindow, IsMixedWindow > (metadata, words, context, inputs, schedule, window)) {
        return;
    }
    // Every core validates before TPipe, synchronization or output writes,
    // even for empty capacity or all-zero used lengths.
    AscendC::SetSyncBaseAddr(ffts);
    AscendC::TPipe pipe;
    Kernel::MlaUsed<DType, true, HasStrides, IsCausal, IsWindow> op;
    op(schedule, scale, qc, qr, kv, kr, output, lse, workspace, &pipe,
       metadata + kMlaUsedMetadataPrefixWords * sizeof(int32_t), &layouts, window);
#endif
}

template <typename DType>
aclError LaunchTndUsed(const MlaUsedLaunchArgs& a)
{
    MlaBatchLaunchScope launch;
    if (a.isWindow && a.context.kv.layout == uint32_t(MlaSequenceLayout::Bsnd) && a.hasStrides) {
        TndUsedKernel<DType, true, false, false, true, true>
            <<<a.context.blockDim, nullptr, a.stream>>>(a.ffts, a.qc, a.qr, a.kv, a.kr, a.output, a.lse, a.workspace,
                                                        a.metadata, a.context, a.inputs, a.scale, a.layouts, a.window);
    } else if (a.isWindow && a.context.kv.layout == uint32_t(MlaSequenceLayout::Bsnd)) {
        TndUsedKernel<DType, false, false, false, true, true>
            <<<a.context.blockDim, nullptr, a.stream>>>(a.ffts, a.qc, a.qr, a.kv, a.kr, a.output, a.lse, a.workspace,
                                                        a.metadata, a.context, a.inputs, a.scale, a.layouts, a.window);
    } else if (a.isWindow && a.hasStrides) {
        TndUsedKernel<DType, true, false, false, true>
            <<<a.context.blockDim, nullptr, a.stream>>>(a.ffts, a.qc, a.qr, a.kv, a.kr, a.output, a.lse, a.workspace,
                                                        a.metadata, a.context, a.inputs, a.scale, a.layouts, a.window);
    } else if (a.isWindow) {
        TndUsedKernel<DType, false, false, false, true>
            <<<a.context.blockDim, nullptr, a.stream>>>(a.ffts, a.qc, a.qr, a.kv, a.kr, a.output, a.lse, a.workspace,
                                                        a.metadata, a.context, a.inputs, a.scale, a.layouts, a.window);
    } else if (a.causal && a.context.kv.layout == uint32_t(MlaSequenceLayout::Bsnd) && a.hasStrides) {
        TndUsedKernel<DType, true, true, true>
            <<<a.context.blockDim, nullptr, a.stream>>>(a.ffts, a.qc, a.qr, a.kv, a.kr, a.output, a.lse, a.workspace,
                                                        a.metadata, a.context, a.inputs, a.scale, a.layouts, a.window);
    } else if (a.causal && a.context.kv.layout == uint32_t(MlaSequenceLayout::Bsnd)) {
        TndUsedKernel<DType, false, true, true>
            <<<a.context.blockDim, nullptr, a.stream>>>(a.ffts, a.qc, a.qr, a.kv, a.kr, a.output, a.lse, a.workspace,
                                                        a.metadata, a.context, a.inputs, a.scale, a.layouts, a.window);
    } else if (a.causal && a.hasStrides) {
        TndUsedKernel<DType, true, true>
            <<<a.context.blockDim, nullptr, a.stream>>>(a.ffts, a.qc, a.qr, a.kv, a.kr, a.output, a.lse, a.workspace,
                                                        a.metadata, a.context, a.inputs, a.scale, a.layouts, a.window);
    } else if (a.causal) {
        TndUsedKernel<DType, false, true>
            <<<a.context.blockDim, nullptr, a.stream>>>(a.ffts, a.qc, a.qr, a.kv, a.kr, a.output, a.lse, a.workspace,
                                                        a.metadata, a.context, a.inputs, a.scale, a.layouts, a.window);
    } else if (a.hasStrides) {
        TndUsedKernel<DType, true><<<a.context.blockDim, nullptr, a.stream>>>(a.ffts, a.qc, a.qr, a.kv, a.kr, a.output,
                                                                              a.lse, a.workspace, a.metadata, a.context,
                                                                              a.inputs, a.scale, a.layouts, a.window);
    } else {
        TndUsedKernel<DType, false>
            <<<a.context.blockDim, nullptr, a.stream>>>(a.ffts, a.qc, a.qr, a.kv, a.kr, a.output, a.lse, a.workspace,
                                                        a.metadata, a.context, a.inputs, a.scale, a.layouts, a.window);
    }
    return launch.Result();
}
} // namespace flash_attn::mla

namespace flash_attn::mla {
template <typename DType, bool IsTnd, bool IsCausal, bool IsWindow = false>
__global__ __aicore__ __mix__(1, 2) void PagedKernel(uint64_t ffts, GM_ADDR qc, GM_ADDR qr, GM_ADDR output, GM_ADDR lse,
                                                     GM_ADDR workspace, GM_ADDR metadata,
                                                     MlaPagedMetadataContext context, MlaPagedMetadataInputs inputs,
                                                     MlaPagedCacheInputs cache, MlaBsndInputLayouts layouts,
                                                     float scale, MlaWindowKernelParams window)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
    MlaPagedKernelSchedule schedule{};
    const uint32_t words = kMlaPagedMetadataPrefixWords + kMlaPagedMetadataSequenceWords * context.batch +
                           (IsWindow ? kMlaWindowPagedMetadataTailWords : 0);
    // All cores validate the current lengths and every effective page before
    // TPipe, events, synchronization or output/workspace writes, even if empty.
    if (!Kernel::LoadPagedMetadata<DType, IsCausal, IsWindow>(metadata, words, context, inputs, cache, schedule,
                                                              window)) {
        return;
    }
    AscendC::SetSyncBaseAddr(ffts);
    AscendC::TPipe pipe;
    Kernel::MlaPaged<DType, IsTnd, IsCausal, IsWindow> op;
    op(schedule, scale, qc, qr, cache, output, lse, workspace, &pipe,
       metadata + kMlaPagedMetadataPrefixWords * sizeof(int32_t), layouts, window);
#endif
}

template <typename DType, bool IsTnd>
aclError LaunchPaged(const MlaPagedLaunchArgs& a)
{
    MlaBatchLaunchScope launch;
    if (a.isWindow) {
        PagedKernel<DType, IsTnd, false, true>
            <<<a.context.blockDim, nullptr, a.stream>>>(a.ffts, a.qc, a.qr, a.output, a.lse, a.workspace, a.metadata,
                                                        a.context, a.inputs, a.cache, a.layouts, a.scale, a.window);
    } else if (a.causal) {
        PagedKernel<DType, IsTnd, true>
            <<<a.context.blockDim, nullptr, a.stream>>>(a.ffts, a.qc, a.qr, a.output, a.lse, a.workspace, a.metadata,
                                                        a.context, a.inputs, a.cache, a.layouts, a.scale, a.window);
    } else {
        PagedKernel<DType, IsTnd, false>
            <<<a.context.blockDim, nullptr, a.stream>>>(a.ffts, a.qc, a.qr, a.output, a.lse, a.workspace, a.metadata,
                                                        a.context, a.inputs, a.cache, a.layouts, a.scale, a.window);
    }
    return launch.Result();
}
} // namespace flash_attn::mla
