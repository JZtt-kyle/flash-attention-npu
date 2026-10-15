// BSND/TND PA specializations of the CANN nonquant MLA task ring.
// Source and license: mla_dense.hpp and ../LICENSE.cann. v6/v10/v11 validate current
// lengths, all effective pages and complete cache spans before TPipe/events.
// Production launchers explicitly select the query layout and mask specialization.
#pragma once
#include "mla_dense.hpp"
#include "mla_paged_metadata.hpp"

namespace flash_attn::mla::Kernel {
template <typename InputT, bool IsTnd, bool IsCausal = false, bool IsWindow = false>
class MlaPaged : private MlaTaskRing<InputT, /*HasKvLengths=*/true, IsTnd, /*MixedKv=*/false, /*HasUsed=*/false,
                                     /*HasStrides=*/true, /*IsPaged=*/true, IsCausal, IsWindow> {
  public:
    // The caller has passed LoadPagedMetadata on every core, checked Q/qv
    // storage and matching query layout, and bound cache addresses to this call.
    __aicore__ inline void operator()(const MlaPagedKernelSchedule& schedule, float scale, GM_ADDR qc, GM_ADDR qr,
                                      const MlaPagedCacheInputs& cache, GM_ADDR output, GM_ADDR lse, GM_ADDR workspace,
                                      TPipe* pipe, GM_ADDR spans, const MlaBsndInputLayouts& layouts,
                                      MlaWindowKernelParams window = {})
    {
        // TND total and allocation starts have been validated to fit int32.
        const uint32_t totalQuery = IsTnd ? schedule.query.totalTokens.low : 0;
        const uint64_t tokens = IsTnd ? totalQuery : uint64_t(schedule.tiling.batch) * schedule.tiling.sq;
        const uint64_t rows = tokens * schedule.tiling.heads;
        if (rows == 0) {
            return;
        }
        if (schedule.tiling.sk == 0) {
            // PA has positive P even with an empty table/pool. Every effective
            // row has zero KV, so use FA3 device-empty -inf, not dense +inf.
            // One C1V2 initializes the rows. A statically nonempty allocation
            // may launch extra cores; their start is past the output extent.
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
                  reinterpret_cast<GM_ADDR>(cache.k.address), output, lse, workspace, pipe, spans, totalQuery, &layouts,
                  false, reinterpret_cast<GM_ADDR>(cache.table.address), &cache.layout, window);
    }
};
template <typename InputT, bool IsCausal = false, bool IsWindow = false>
using MlaPagedBsnd = MlaPaged<InputT, false, IsCausal, IsWindow>;
template <typename InputT, bool IsCausal = false, bool IsWindow = false>
using MlaPagedTnd = MlaPaged<InputT, true, IsCausal, IsWindow>;
} // namespace flash_attn::mla::Kernel
