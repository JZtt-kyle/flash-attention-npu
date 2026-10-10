// Production compact TND launch, sharing the independently validated v3
// consumer and CANN-derived Kernel/Block/Tile implementation. Source/license
// are retained in kernel/mla_tnd.hpp and kernel/mla_dense.hpp.
#pragma once
#include "mla_launch.h"
#include "kernel_operator.h"
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
#include "kernel/mla_tnd.hpp"
#include "kernel/mla_tnd_metadata.hpp"
#endif

namespace flash_attn::mla {
__global__ __aicore__ __mix__(1, 2) void MLA_TND_ENTRY(uint64_t ffts, GM_ADDR qc, GM_ADDR qr, GM_ADDR kv, GM_ADDR kr,
                                                       GM_ADDR output, GM_ADDR lse, GM_ADDR workspace, GM_ADDR metadata,
                                                       GM_ADDR cuQuery, GM_ADDR cuKv, MlaTndMetadataContext context,
                                                       float scale)
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
    Kernel::MlaTnd<MLA_TND_TYPE> op;
    op(schedule, scale, qc, qr, kv, kr, output, lse, workspace, &pipe,
       metadata + kMlaTndMetadataPrefixWords * sizeof(int32_t));
#endif
}

aclError MLA_TND_LAUNCH(const MlaTndLaunchArgs& a)
{
    MlaBatchLaunchScope launch;
    MLA_TND_ENTRY<<<a.context.blockDim, nullptr, a.stream>>>(a.ffts, a.qc, a.qr, a.kv, a.kr, a.output, a.lse,
                                                             a.workspace, a.metadata, a.cuQuery, a.cuKv, a.context,
                                                             a.scale);
    return launch.Result();
}
} // namespace flash_attn::mla
