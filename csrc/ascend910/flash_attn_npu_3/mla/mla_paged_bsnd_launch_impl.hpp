// Production BSND PA launch. Source and license are retained in
// kernel/mla_dense.hpp and LICENSE.cann; Kernel/Block/Tile math is unchanged.
#pragma once
#include "mla_paged_launch.h"
#include "kernel_operator.h"
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
#include "kernel/mla_paged.hpp"
#endif

namespace flash_attn::mla {
__global__ __aicore__ __mix__(1, 2) void MLA_PAGED_ENTRY(uint64_t ffts, GM_ADDR qc, GM_ADDR qr, GM_ADDR output,
                                                         GM_ADDR lse, GM_ADDR workspace, GM_ADDR metadata,
                                                         MlaPagedMetadataContext context, MlaPagedMetadataInputs inputs,
                                                         MlaPagedCacheInputs cache, MlaBsndInputLayouts layouts,
                                                         float scale)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
    MlaPagedKernelSchedule schedule{};
    const uint32_t words = kMlaPagedMetadataPrefixWords + kMlaPagedMetadataSequenceWords * context.batch;
    // All cores validate the current lengths and every effective page before
    // TPipe, events, synchronization or output/workspace writes, even if empty.
    if (!Kernel::LoadPagedMetadata<MLA_PAGED_TYPE>(metadata, words, context, inputs, cache, schedule)) {
        return;
    }
    AscendC::SetSyncBaseAddr(ffts);
    AscendC::TPipe pipe;
    Kernel::MlaPagedBsnd<MLA_PAGED_TYPE> op;
    op(schedule, scale, qc, qr, cache, output, lse, workspace, &pipe,
       metadata + kMlaPagedMetadataPrefixWords * sizeof(int32_t), layouts);
#endif
}

aclError MLA_PAGED_LAUNCH(const MlaPagedLaunchArgs& a)
{
    MlaBatchLaunchScope launch;
    MLA_PAGED_ENTRY<<<a.context.blockDim, nullptr, a.stream>>>(
        a.ffts, a.qc, a.qr, a.output, a.lse, a.workspace, a.metadata, a.context, a.inputs, a.cache, a.layouts, a.scale);
    return launch.Result();
}
} // namespace flash_attn::mla
