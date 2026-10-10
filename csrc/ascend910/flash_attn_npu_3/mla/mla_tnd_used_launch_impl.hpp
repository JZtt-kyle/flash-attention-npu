// Production TND allocation/used launch. Source and license are retained in
// kernel/mla_used.hpp and kernel/mla_dense.hpp; numerical blocks are unchanged.
#pragma once
#include "mla_launch.h"
#include "kernel_operator.h"
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
#include "kernel/mla_used.hpp"
#include "kernel/mla_used_metadata.hpp"
#endif

namespace flash_attn::mla {
template <bool HasStrides>
__global__ __aicore__ __mix__(1, 2) void MLA_USED_ENTRY(uint64_t ffts, GM_ADDR qc, GM_ADDR qr, GM_ADDR kv, GM_ADDR kr,
                                                        GM_ADDR output, GM_ADDR lse, GM_ADDR workspace,
                                                        GM_ADDR metadata, MlaUsedMetadataContext context,
                                                        MlaUsedMetadataInputs inputs, float scale,
                                                        MlaBsndInputLayouts layouts)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
    MlaUsedKernelSchedule schedule{};
    const uint32_t words = kMlaUsedMetadataPrefixWords + kMlaUsedMetadataSequenceWords * context.batch;
    if (!Kernel::LoadUsedMetadata<MLA_USED_TYPE>(metadata, words, context, inputs, schedule)) {
        return;
    }
    // Every core validates before TPipe, synchronization or output writes,
    // even for empty capacity or all-zero used lengths.
    AscendC::SetSyncBaseAddr(ffts);
    AscendC::TPipe pipe;
    Kernel::MlaUsed<MLA_USED_TYPE, true, HasStrides> op;
    op(schedule, scale, qc, qr, kv, kr, output, lse, workspace, &pipe,
       metadata + kMlaUsedMetadataPrefixWords * sizeof(int32_t), &layouts);
#endif
}

aclError MLA_USED_LAUNCH(const MlaUsedLaunchArgs& a)
{
    MlaBatchLaunchScope launch;
    if (a.hasStrides) {
        MLA_USED_ENTRY<true><<<a.context.blockDim, nullptr, a.stream>>>(a.ffts, a.qc, a.qr, a.kv, a.kr, a.output, a.lse,
                                                                        a.workspace, a.metadata, a.context, a.inputs,
                                                                        a.scale, a.layouts);
    } else {
        MLA_USED_ENTRY<false><<<a.context.blockDim, nullptr, a.stream>>>(a.ffts, a.qc, a.qr, a.kv, a.kr, a.output,
                                                                         a.lse, a.workspace, a.metadata, a.context,
                                                                         a.inputs, a.scale, a.layouts);
    }
    return launch.Result();
}
} // namespace flash_attn::mla
