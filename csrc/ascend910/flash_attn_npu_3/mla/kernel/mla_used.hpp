// Allocation/used specialization of the CANN nonquant MLA task ring.
// Source and license: mla_dense.hpp and ../LICENSE.cann. v5 separates the
// FA3 physical allocation from the source task lengths; used never repacks Q/KV.
#pragma once
#include "mla_dense.hpp"
#include "../mla_used_metadata_consumer.h"

namespace flash_attn::mla::Kernel {
template <typename InputT, bool IsTnd>
class MlaUsed : private MlaNonPaged<InputT, true, IsTnd, false, true> {
  public:
    // Validate v5 and all four current arrays on every core before TPipe,
    // synchronization or writes. IsTnd must match the validated Q layout.
    __aicore__ inline void operator()(const MlaUsedKernelSchedule& schedule, float scale, GM_ADDR qc, GM_ADDR qr,
                                      GM_ADDR kv, GM_ADDR kr, GM_ADDR output, GM_ADDR lse, GM_ADDR workspace,
                                      TPipe* pipe, GM_ADDR spans)
    {
        const uint64_t totalQuery = UsedMetadataTotal(schedule.query);
        if (totalQuery == 0) {
            return;
        }
        if (UsedMetadataTotal(schedule.kv) == 0) {
            if ASCEND_IS_AIV {
                // The FA3 global-empty branch initializes the allocation.
                // Nonempty capacity with zero used is handled by Run (-inf).
                const uint64_t rows = totalQuery * schedule.tiling.heads;
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
                    matmul::InitOutput<float>(lseGm[start], count, float(1.0 / 0.0));
                    SetFlag<HardEvent::MTE3_V>(0);
                    WaitFlag<HardEvent::MTE3_V>(0);
                }
            }
            return;
        }
        this->Run(schedule.tiling, scale, qc, qr, kv, kr, output, lse, workspace, pipe, spans,
                  IsTnd ? uint32_t(totalQuery) : 0);
    }
};
} // namespace flash_attn::mla::Kernel
