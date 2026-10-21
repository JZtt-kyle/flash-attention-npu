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

#if defined(MLA_DENSE_TYPE)
namespace flash_attn::mla {
template <bool HasStrides>
__global__ __aicore__ __mix__(1, 2) void MLA_DENSE_ENTRY(uint64_t ffts, GM_ADDR qc, GM_ADDR qr, GM_ADDR kv, GM_ADDR kr,
                                                         GM_ADDR output, GM_ADDR lse, GM_ADDR workspace,
                                                         GM_ADDR metadata, GM_ADDR cacheSeqlens,
                                                         MlaDenseMetadataContext context, float scale,
                                                         bool hasKvLengths, MlaBsndInputLayouts layouts)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
    MlaDenseKernelSchedule schedule{};
    if (hasKvLengths) {
        if (!Kernel::LoadBsndKvLengthMetadata<MLA_DENSE_TYPE>(
                metadata, kMlaBsndKvLengthMetadataPrefixWords + context.batch, context, cacheSeqlens,
                cacheSeqlens ? context.batch : 0, schedule)) {
            return;
        }
    } else {
        if (!Kernel::LoadDenseMetadata<MLA_DENSE_TYPE>(metadata, kMlaDenseMetadataWords, context, schedule) ||
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
        Kernel::MlaDense<MLA_DENSE_TYPE, true, HasStrides> op;
        op(schedule, scale, qc, qr, kv, kr, output, lse, workspace, &pipe,
           metadata + kMlaBsndKvLengthMetadataPrefixWords * sizeof(int32_t), &layouts);
    } else {
        Kernel::MlaDense<MLA_DENSE_TYPE, false, HasStrides> op;
        op(schedule, scale, qc, qr, kv, kr, output, lse, workspace, &pipe, nullptr, &layouts);
    }
#endif
}

aclError MLA_DENSE_LAUNCH(const MlaLaunchArgs& a)
{
    MlaBatchLaunchScope launch;
    if (a.hasStrides) {
        MLA_DENSE_ENTRY<true><<<a.context.blockDim, nullptr, a.stream>>>(a.ffts, a.qc, a.qr, a.kv, a.kr, a.output,
                                                                         a.lse, a.workspace, a.metadata, a.cacheSeqlens,
                                                                         a.context, a.scale, a.hasKvLengths, a.layouts);
    } else {
        MLA_DENSE_ENTRY<false><<<a.context.blockDim, nullptr, a.stream>>>(
            a.ffts, a.qc, a.qr, a.kv, a.kr, a.output, a.lse, a.workspace, a.metadata, a.cacheSeqlens, a.context,
            a.scale, a.hasKvLengths, a.layouts);
    }
    return launch.Result();
}
} // namespace flash_attn::mla

#elif defined(MLA_TND_TYPE)
namespace flash_attn::mla {
template <bool HasStrides>
__global__ __aicore__ __mix__(1, 2) void MLA_TND_ENTRY(uint64_t ffts, GM_ADDR qc, GM_ADDR qr, GM_ADDR kv, GM_ADDR kr,
                                                       GM_ADDR output, GM_ADDR lse, GM_ADDR workspace, GM_ADDR metadata,
                                                       GM_ADDR cuQuery, GM_ADDR cuKv, MlaTndMetadataContext context,
                                                       float scale, MlaBsndInputLayouts layouts)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
    MlaTndKernelSchedule schedule{};
    const uint32_t words = kMlaTndMetadataPrefixWords + kMlaTndMetadataSequenceWords * context.batch;
    const uint64_t count = uint64_t(context.batch) + 1;
    if (!Kernel::LoadTndMetadata<MLA_TND_TYPE>(metadata, words, context, cuQuery, count, cuKv, count, schedule)) {
        return;
    }
    // Every core validates before TPipe, attention synchronization or writes,
    // including empty total capacity. The TND Kernel implements empty O/LSE.
    AscendC::SetSyncBaseAddr(ffts);
    AscendC::TPipe pipe;
    Kernel::MlaTnd<MLA_TND_TYPE, HasStrides> op;
    op(schedule, scale, qc, qr, kv, kr, output, lse, workspace, &pipe,
       metadata + kMlaTndMetadataPrefixWords * sizeof(int32_t), &layouts);
#endif
}

aclError MLA_TND_LAUNCH(const MlaTndLaunchArgs& a)
{
    MlaBatchLaunchScope launch;
    if (a.hasStrides) {
        MLA_TND_ENTRY<true><<<a.context.blockDim, nullptr, a.stream>>>(a.ffts, a.qc, a.qr, a.kv, a.kr, a.output, a.lse,
                                                                       a.workspace, a.metadata, a.cuQuery, a.cuKv,
                                                                       a.context, a.scale, a.layouts);
    } else {
        MLA_TND_ENTRY<false><<<a.context.blockDim, nullptr, a.stream>>>(a.ffts, a.qc, a.qr, a.kv, a.kr, a.output, a.lse,
                                                                        a.workspace, a.metadata, a.cuQuery, a.cuKv,
                                                                        a.context, a.scale, a.layouts);
    }
    return launch.Result();
}
} // namespace flash_attn::mla

#elif defined(MLA_MIXED_TYPE)
namespace flash_attn::mla {
template <bool HasStrides>
__global__ __aicore__ __mix__(1, 2) void MLA_MIXED_ENTRY(uint64_t ffts, GM_ADDR qc, GM_ADDR qr, GM_ADDR kv, GM_ADDR kr,
                                                         GM_ADDR output, GM_ADDR lse, GM_ADDR workspace,
                                                         GM_ADDR metadata, GM_ADDR cuQuery, GM_ADDR cacheSeqlens,
                                                         MlaTndBsndMetadataContext context, float scale,
                                                         MlaBsndInputLayouts layouts)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
    MlaTndBsndKernelSchedule schedule{};
    const uint32_t words = kMlaTndBsndMetadataPrefixWords + kMlaTndBsndMetadataSequenceWords * context.batch;
    if (!Kernel::LoadTndBsndMetadata<MLA_MIXED_TYPE>(metadata, words, context, cuQuery, uint64_t(context.batch) + 1,
                                                     cacheSeqlens, context.batch, schedule)) {
        return;
    }
    // Every core validates before TPipe, attention synchronization or writes,
    // including empty capacity. The mixed Kernel implements empty O/LSE.
    AscendC::SetSyncBaseAddr(ffts);
    AscendC::TPipe pipe;
    Kernel::MlaTndBsnd<MLA_MIXED_TYPE, HasStrides> op;
    op(schedule, scale, qc, qr, kv, kr, output, lse, workspace, &pipe,
       metadata + kMlaTndBsndMetadataPrefixWords * sizeof(int32_t), &layouts);
#endif
}

aclError MLA_MIXED_LAUNCH(const MlaTndBsndLaunchArgs& a)
{
    MlaBatchLaunchScope launch;
    if (a.hasStrides) {
        MLA_MIXED_ENTRY<true><<<a.context.blockDim, nullptr, a.stream>>>(a.ffts, a.qc, a.qr, a.kv, a.kr, a.output,
                                                                         a.lse, a.workspace, a.metadata, a.cuQuery,
                                                                         a.cacheSeqlens, a.context, a.scale, a.layouts);
    } else {
        MLA_MIXED_ENTRY<false><<<a.context.blockDim, nullptr, a.stream>>>(
            a.ffts, a.qc, a.qr, a.kv, a.kr, a.output, a.lse, a.workspace, a.metadata, a.cuQuery, a.cacheSeqlens,
            a.context, a.scale, a.layouts);
    }
    return launch.Result();
}
} // namespace flash_attn::mla

#elif defined(MLA_USED_TYPE) && !MLA_USED_TND
namespace flash_attn::mla {
template <bool HasStrides, bool IsCausal = false, bool IsWindow = false>
__global__ __aicore__ __mix__(1, 2) void MLA_USED_ENTRY(uint64_t ffts, GM_ADDR qc, GM_ADDR qr, GM_ADDR kv, GM_ADDR kr,
                                                        GM_ADDR output, GM_ADDR lse, GM_ADDR workspace,
                                                        GM_ADDR metadata, MlaUsedMetadataContext context,
                                                        MlaUsedMetadataInputs inputs, float scale,
                                                        MlaBsndInputLayouts layouts, MlaWindowKernelParams window)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
    MlaUsedKernelSchedule schedule{};
    const uint32_t words = kMlaUsedMetadataPrefixWords + kMlaUsedMetadataSequenceWords * context.batch +
                           (IsWindow ? kMlaWindowMetadataTailWords : 0);
    if (!Kernel::LoadUsedMetadata<MLA_USED_TYPE, IsCausal, false, false, IsWindow>(metadata, words, context, inputs,
                                                                                   schedule, window)) {
        return;
    }
    // Every core validates before TPipe, synchronization or output writes,
    // even for empty capacity or all-zero used lengths.
    AscendC::SetSyncBaseAddr(ffts);
    AscendC::TPipe pipe;
    Kernel::MlaUsed<MLA_USED_TYPE, false, HasStrides, IsCausal, IsWindow> op;
    op(schedule, scale, qc, qr, kv, kr, output, lse, workspace, &pipe,
       metadata + kMlaUsedMetadataPrefixWords * sizeof(int32_t), &layouts, window);
#endif
}

aclError MLA_USED_LAUNCH(const MlaUsedLaunchArgs& a)
{
    MlaBatchLaunchScope launch;
    if (a.isWindow && a.hasStrides) {
        MLA_USED_ENTRY<true, false, true>
            <<<a.context.blockDim, nullptr, a.stream>>>(a.ffts, a.qc, a.qr, a.kv, a.kr, a.output, a.lse, a.workspace,
                                                        a.metadata, a.context, a.inputs, a.scale, a.layouts, a.window);
    } else if (a.isWindow) {
        MLA_USED_ENTRY<false, false, true>
            <<<a.context.blockDim, nullptr, a.stream>>>(a.ffts, a.qc, a.qr, a.kv, a.kr, a.output, a.lse, a.workspace,
                                                        a.metadata, a.context, a.inputs, a.scale, a.layouts, a.window);
    } else if (a.causal && a.hasStrides) {
        MLA_USED_ENTRY<true, true><<<a.context.blockDim, nullptr, a.stream>>>(a.ffts, a.qc, a.qr, a.kv, a.kr, a.output,
                                                                              a.lse, a.workspace, a.metadata, a.context,
                                                                              a.inputs, a.scale, a.layouts, a.window);
    } else if (a.causal) {
        MLA_USED_ENTRY<false, true>
            <<<a.context.blockDim, nullptr, a.stream>>>(a.ffts, a.qc, a.qr, a.kv, a.kr, a.output, a.lse, a.workspace,
                                                        a.metadata, a.context, a.inputs, a.scale, a.layouts, a.window);
    } else if (a.hasStrides) {
        MLA_USED_ENTRY<true><<<a.context.blockDim, nullptr, a.stream>>>(a.ffts, a.qc, a.qr, a.kv, a.kr, a.output, a.lse,
                                                                        a.workspace, a.metadata, a.context, a.inputs,
                                                                        a.scale, a.layouts, a.window);
    } else {
        MLA_USED_ENTRY<false><<<a.context.blockDim, nullptr, a.stream>>>(a.ffts, a.qc, a.qr, a.kv, a.kr, a.output,
                                                                         a.lse, a.workspace, a.metadata, a.context,
                                                                         a.inputs, a.scale, a.layouts, a.window);
    }
    return launch.Result();
}
} // namespace flash_attn::mla

#elif defined(MLA_USED_TYPE)
namespace flash_attn::mla {
template <bool HasStrides, bool IsCausal = false, bool IsMixedCausal = false, bool IsWindow = false,
          bool IsMixedWindow = false>
__global__ __aicore__ __mix__(1, 2) void MLA_USED_ENTRY(uint64_t ffts, GM_ADDR qc, GM_ADDR qr, GM_ADDR kv, GM_ADDR kr,
                                                        GM_ADDR output, GM_ADDR lse, GM_ADDR workspace,
                                                        GM_ADDR metadata, MlaUsedMetadataContext context,
                                                        MlaUsedMetadataInputs inputs, float scale,
                                                        MlaBsndInputLayouts layouts, MlaWindowKernelParams window)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
    MlaUsedKernelSchedule schedule{};
    const uint32_t words = kMlaUsedMetadataPrefixWords + kMlaUsedMetadataSequenceWords * context.batch +
                           (IsWindow ? kMlaWindowMetadataTailWords : 0);
    if (!Kernel::LoadUsedMetadata < MLA_USED_TYPE, IsCausal, IsCausal && !IsMixedCausal, IsMixedCausal, IsWindow,
        IsWindow && !IsMixedWindow, IsMixedWindow > (metadata, words, context, inputs, schedule, window)) {
        return;
    }
    // Every core validates before TPipe, synchronization or output writes,
    // even for empty capacity or all-zero used lengths.
    AscendC::SetSyncBaseAddr(ffts);
    AscendC::TPipe pipe;
    Kernel::MlaUsed<MLA_USED_TYPE, true, HasStrides, IsCausal, IsWindow> op;
    op(schedule, scale, qc, qr, kv, kr, output, lse, workspace, &pipe,
       metadata + kMlaUsedMetadataPrefixWords * sizeof(int32_t), &layouts, window);
#endif
}

aclError MLA_USED_LAUNCH(const MlaUsedLaunchArgs& a)
{
    MlaBatchLaunchScope launch;
    if (a.isWindow && a.context.kv.layout == uint32_t(MlaSequenceLayout::Bsnd) && a.hasStrides) {
        MLA_USED_ENTRY<true, false, false, true, true>
            <<<a.context.blockDim, nullptr, a.stream>>>(a.ffts, a.qc, a.qr, a.kv, a.kr, a.output, a.lse, a.workspace,
                                                        a.metadata, a.context, a.inputs, a.scale, a.layouts, a.window);
    } else if (a.isWindow && a.context.kv.layout == uint32_t(MlaSequenceLayout::Bsnd)) {
        MLA_USED_ENTRY<false, false, false, true, true>
            <<<a.context.blockDim, nullptr, a.stream>>>(a.ffts, a.qc, a.qr, a.kv, a.kr, a.output, a.lse, a.workspace,
                                                        a.metadata, a.context, a.inputs, a.scale, a.layouts, a.window);
    } else if (a.isWindow && a.hasStrides) {
        MLA_USED_ENTRY<true, false, false, true>
            <<<a.context.blockDim, nullptr, a.stream>>>(a.ffts, a.qc, a.qr, a.kv, a.kr, a.output, a.lse, a.workspace,
                                                        a.metadata, a.context, a.inputs, a.scale, a.layouts, a.window);
    } else if (a.isWindow) {
        MLA_USED_ENTRY<false, false, false, true>
            <<<a.context.blockDim, nullptr, a.stream>>>(a.ffts, a.qc, a.qr, a.kv, a.kr, a.output, a.lse, a.workspace,
                                                        a.metadata, a.context, a.inputs, a.scale, a.layouts, a.window);
    } else if (a.causal && a.context.kv.layout == uint32_t(MlaSequenceLayout::Bsnd) && a.hasStrides) {
        MLA_USED_ENTRY<true, true, true>
            <<<a.context.blockDim, nullptr, a.stream>>>(a.ffts, a.qc, a.qr, a.kv, a.kr, a.output, a.lse, a.workspace,
                                                        a.metadata, a.context, a.inputs, a.scale, a.layouts, a.window);
    } else if (a.causal && a.context.kv.layout == uint32_t(MlaSequenceLayout::Bsnd)) {
        MLA_USED_ENTRY<false, true, true>
            <<<a.context.blockDim, nullptr, a.stream>>>(a.ffts, a.qc, a.qr, a.kv, a.kr, a.output, a.lse, a.workspace,
                                                        a.metadata, a.context, a.inputs, a.scale, a.layouts, a.window);
    } else if (a.causal && a.hasStrides) {
        MLA_USED_ENTRY<true, true><<<a.context.blockDim, nullptr, a.stream>>>(a.ffts, a.qc, a.qr, a.kv, a.kr, a.output,
                                                                              a.lse, a.workspace, a.metadata, a.context,
                                                                              a.inputs, a.scale, a.layouts, a.window);
    } else if (a.causal) {
        MLA_USED_ENTRY<false, true>
            <<<a.context.blockDim, nullptr, a.stream>>>(a.ffts, a.qc, a.qr, a.kv, a.kr, a.output, a.lse, a.workspace,
                                                        a.metadata, a.context, a.inputs, a.scale, a.layouts, a.window);
    } else if (a.hasStrides) {
        MLA_USED_ENTRY<true><<<a.context.blockDim, nullptr, a.stream>>>(a.ffts, a.qc, a.qr, a.kv, a.kr, a.output, a.lse,
                                                                        a.workspace, a.metadata, a.context, a.inputs,
                                                                        a.scale, a.layouts, a.window);
    } else {
        MLA_USED_ENTRY<false><<<a.context.blockDim, nullptr, a.stream>>>(a.ffts, a.qc, a.qr, a.kv, a.kr, a.output,
                                                                         a.lse, a.workspace, a.metadata, a.context,
                                                                         a.inputs, a.scale, a.layouts, a.window);
    }
    return launch.Result();
}
} // namespace flash_attn::mla

#elif defined(MLA_PAGED_TYPE)
namespace flash_attn::mla {
template <bool IsCausal, bool IsWindow = false>
__global__ __aicore__ __mix__(1, 2) void MLA_PAGED_ENTRY(uint64_t ffts, GM_ADDR qc, GM_ADDR qr, GM_ADDR output,
                                                         GM_ADDR lse, GM_ADDR workspace, GM_ADDR metadata,
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
    if (!Kernel::LoadPagedMetadata<MLA_PAGED_TYPE, IsCausal, IsWindow>(metadata, words, context, inputs, cache,
                                                                       schedule, window)) {
        return;
    }
    AscendC::SetSyncBaseAddr(ffts);
    AscendC::TPipe pipe;
    Kernel::MlaPaged<MLA_PAGED_TYPE, MLA_PAGED_TND, IsCausal, IsWindow> op;
    op(schedule, scale, qc, qr, cache, output, lse, workspace, &pipe,
       metadata + kMlaPagedMetadataPrefixWords * sizeof(int32_t), layouts, window);
#endif
}

aclError MLA_PAGED_LAUNCH(const MlaPagedLaunchArgs& a)
{
    MlaBatchLaunchScope launch;
    if (a.isWindow) {
        MLA_PAGED_ENTRY<false, true>
            <<<a.context.blockDim, nullptr, a.stream>>>(a.ffts, a.qc, a.qr, a.output, a.lse, a.workspace, a.metadata,
                                                        a.context, a.inputs, a.cache, a.layouts, a.scale, a.window);
    } else if (a.causal) {
        MLA_PAGED_ENTRY<true><<<a.context.blockDim, nullptr, a.stream>>>(a.ffts, a.qc, a.qr, a.output, a.lse,
                                                                         a.workspace, a.metadata, a.context, a.inputs,
                                                                         a.cache, a.layouts, a.scale, a.window);
    } else {
        MLA_PAGED_ENTRY<false><<<a.context.blockDim, nullptr, a.stream>>>(a.ffts, a.qc, a.qr, a.output, a.lse,
                                                                          a.workspace, a.metadata, a.context, a.inputs,
                                                                          a.cache, a.layouts, a.scale, a.window);
    }
    return launch.Result();
}
} // namespace flash_attn::mla

#endif
