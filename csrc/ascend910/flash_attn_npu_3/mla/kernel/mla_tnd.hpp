// Compact TND specialization of the migrated CANN nonquant MLA task ring.
// Source and license: mla_dense.hpp and ../LICENSE.cann. Token starts follow
// CANN ca18aee610729ca5270a44e2d84c751bc687fbf0 ActualSeqLensParser;
// the validated v3 snapshot replaces the source int64 cumulative arrays.
#pragma once
#include "mla_dense.hpp"
#include "../mla_tnd_metadata_consumer.h"

namespace flash_attn::mla::Kernel {
template <typename InputT, bool HasStrides = false>
class MlaTnd : private MlaNonPaged<InputT, true, true, false, false, HasStrides> {
  public:
    // Caller must validate v3 and current prefixes on every core before TPipe,
    // synchronization or writes. Empty capacity still passes that validation.
    __aicore__ inline void operator()(const MlaTndKernelSchedule& schedule, float scale, GM_ADDR qc, GM_ADDR qr,
                                      GM_ADDR kv, GM_ADDR kr, GM_ADDR output, GM_ADDR lse, GM_ADDR workspace,
                                      TPipe* pipe, GM_ADDR spans, const MlaBsndInputLayouts* layouts = nullptr)
    {
        if (schedule.totalQuery == 0) {
            return;
        }
        if (schedule.totalKv == 0) {
            if ASCEND_IS_AIV {
                // FA3 global-empty capacity uses +inf; empty samples within a
                // nonempty capacity use -inf in the common initialization.
                const uint64_t rows = uint64_t(schedule.totalQuery) * schedule.tiling.heads;
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
        this->Run(schedule.tiling, scale, qc, qr, kv, kr, output, lse, workspace, pipe, spans, schedule.totalQuery,
                  layouts);
    }
};
} // namespace flash_attn::mla::Kernel
