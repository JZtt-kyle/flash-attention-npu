// BSND PA specialization of the CANN nonquant MLA task ring.
// Source and license: mla_dense.hpp and ../LICENSE.cann. v6 validates current
// lengths, all effective pages and complete cache spans before TPipe/events.
// This independent component does not enable a public PA forward entry.
#pragma once
#include "mla_dense.hpp"
#include "mla_paged_metadata.hpp"

namespace flash_attn::mla::Kernel {
template <typename InputT>
class MlaPagedBsnd : private MlaTaskRing<InputT, true, false, false, false, true, true> {
  public:
    // The caller has passed LoadPagedMetadata on every core, checked BSND Q
    // and qv storage, and bound cache addresses to this invocation. No TND.
    __aicore__ inline void operator()(const MlaPagedKernelSchedule& schedule, float scale, GM_ADDR qc, GM_ADDR qr,
                                      const MlaPagedCacheInputs& cache, GM_ADDR output, GM_ADDR lse, GM_ADDR workspace,
                                      TPipe* pipe, GM_ADDR spans, const MlaBsndInputLayouts& layouts)
    {
        const uint64_t rows = uint64_t(schedule.tiling.batch) * schedule.tiling.sq * schedule.tiling.heads;
        if (rows == 0) {
            return;
        }
        if (schedule.tiling.sk == 0) {
            // PA has positive P even with an empty table/pool. Every effective
            // row has zero KV, so use FA3 device-empty -inf, not dense +inf.
            // The validated zero-work schedule launches exactly one C1V2.
            if ASCEND_IS_AIV {
                const uint64_t single = (rows + GetSubBlockNum() - 1) / GetSubBlockNum();
                const uint64_t start = GetBlockIdx() * single;
                if (start < rows) {
                    const uint64_t count = rows - start < single ? rows - start : single;
                    GlobalTensor<InputT> outGm;
                    GlobalTensor<float> lseGm;
                    outGm.SetGlobalBuffer(reinterpret_cast<__gm__ InputT*>(output));
                    lseGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(lse));
                    matmul::InitOutput<InputT>(outGm[start * 512], count * 512, 0);
                    SetFlag<HardEvent::MTE3_V>(0);
                    WaitFlag<HardEvent::MTE3_V>(0);
                    matmul::InitOutput<float>(lseGm[start], count, -float(1.0 / 0.0));
                    SetFlag<HardEvent::MTE3_V>(0);
                    WaitFlag<HardEvent::MTE3_V>(0);
                }
            }
            return;
        }
        this->Run(schedule.tiling, scale, qc, qr, reinterpret_cast<GM_ADDR>(cache.v.address),
                  reinterpret_cast<GM_ADDR>(cache.k.address), output, lse, workspace, pipe, spans, 0, &layouts, false,
                  reinterpret_cast<GM_ADDR>(cache.table.address), &cache.layout);
    }
};
} // namespace flash_attn::mla::Kernel
