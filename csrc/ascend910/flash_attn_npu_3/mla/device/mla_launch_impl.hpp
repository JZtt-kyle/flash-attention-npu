// Source launch: CANN ops-transformer ca18aee610729ca5270a44e2d84c751bc687fbf0,
// fused_infer_attention_score_v3.cpp BSH_KVBSH_MLA_TILING (FP16/BF16, C1V2).
// The migrated algorithm and its copyright/license live under kernel/block/tile.
#pragma once
#include "mla_launch.h"
#include "kernel_operator.h"
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
#include "../kernel/mla_dense.hpp"
#include "../kernel/mla_metadata.hpp"
#include "../kernel/mla_bsnd_metadata.hpp"
#endif

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
