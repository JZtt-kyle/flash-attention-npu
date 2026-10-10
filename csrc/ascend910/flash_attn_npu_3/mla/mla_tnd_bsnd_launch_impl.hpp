// Production TND-Q/BSND-KV launch using the independently validated v4
// consumer and CANN-derived Kernel/Block/Tile implementation. Source/license
// are retained in kernel/mla_tnd_bsnd.hpp and kernel/mla_dense.hpp.
#pragma once
#include "mla_launch.h"
#include "kernel_operator.h"
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
#include "kernel/mla_tnd_bsnd.hpp"
#include "kernel/mla_tnd_bsnd_metadata.hpp"
#endif

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
