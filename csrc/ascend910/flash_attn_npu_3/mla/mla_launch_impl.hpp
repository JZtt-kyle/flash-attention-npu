// Source launch: CANN ops-transformer ca18aee610729ca5270a44e2d84c751bc687fbf0,
// fused_infer_attention_score_v3.cpp BSH_KVBSH_MLA_TILING (FP16/BF16, C1V2).
// The migrated algorithm and its copyright/license live under kernel/block/tile.
#pragma once
#include "mla_launch.h"
#include "kernel_operator.h"
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
#include "kernel/mla_dense.hpp"
#include "kernel/mla_metadata.hpp"
#endif

namespace flash_attn::mla {
__global__ __aicore__ __mix__(1, 2) void MLA_DENSE_ENTRY(uint64_t ffts, GM_ADDR qc, GM_ADDR qr, GM_ADDR kv, GM_ADDR kr,
                                                         GM_ADDR output, GM_ADDR lse, GM_ADDR workspace,
                                                         GM_ADDR metadata, MlaDenseMetadataContext context, float scale)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
    MlaDenseKernelSchedule schedule{};
    if (!Kernel::LoadDenseMetadata<MLA_DENSE_TYPE>(metadata, kMlaDenseMetadataWords, context, schedule)) {
        return;
    }
    AscendC::SetSyncBaseAddr(ffts);
    AscendC::TPipe pipe;
    Kernel::MlaDense<MLA_DENSE_TYPE> op;
    op(schedule, scale, qc, qr, kv, kr, output, lse, workspace, &pipe);
#endif
}

aclError MLA_DENSE_LAUNCH(const MlaLaunchArgs& a)
{
    MlaBatchLaunchScope launch;
    MLA_DENSE_ENTRY<<<a.context.blockDim, nullptr, a.stream>>>(a.ffts, a.qc, a.qr, a.kv, a.kr, a.output, a.lse,
                                                               a.workspace, a.metadata, a.context, a.scale);
    return launch.Result();
}
} // namespace flash_attn::mla
