// Production TND PA launch. Source and license are retained in
// kernel/mla_dense.hpp and LICENSE.cann; Kernel/Block/Tile math is unchanged.
#pragma once
#include "mla_launch.h"
#include "kernel_operator.h"
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
#include "../kernel/mla_paged.hpp"
#endif

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
    Kernel::MlaPagedTnd<MLA_PAGED_TYPE, IsCausal, IsWindow> op;
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
