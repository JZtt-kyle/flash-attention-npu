// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
// Public ND BSND adapter around the migrated fixed-source numerical kernel.
#include "kernel_operator.h"
#define MLA_INT8_DEVICE_SCHEDULE
#include "mla_int8_launch.h"
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
#include "kernel/mla_int8.hpp"
using namespace AscendC;

namespace flash_attn::mla::Int8 {
struct LengthReader {
    GlobalTensor<int32_t> data;
    bool present;
    uint32_t capacity;
    __aicore__ inline int64_t operator()(uint32_t b) const
    {
        return present ? data.GetValue(b) : capacity;
    }
};
template <bool Fd, bool Balance>
__aicore__ inline void Execute(const LaunchArgs& a, const Int8TilingParams& p)
{
    TPipe pipe;
    MlaInt8Kernel<Int8Type<false, Fd, Balance>> op;
    op.SetInt32Lengths((GM_ADDR)a.lengths);
    op.SetPublicPartials((GM_ADDR)a.accum, (GM_ADDR)a.accumLse);
    op.Init((GM_ADDR)a.qv, (GM_ADDR)a.v, (GM_ADDR)a.q, (GM_ADDR)a.k, (GM_ADDR)a.mask, nullptr, (GM_ADDR)a.lengths,
            (GM_ADDR)a.table, (GM_ADDR)a.out, (GM_ADDR)a.lse, (GM_ADDR)a.workspace, &p, &pipe);
    op.SetCacheLayout(a.cache);
    op.InitQuant((GM_ADDR)a.qs, (GM_ADDR)a.ks, (GM_ADDR)a.vs);
    op.Process();
}
} // namespace flash_attn::mla::Int8
#endif

__global__ __aicore__ __mix__(1, 2) void mla_int8_bsnd(flash_attn::mla::Int8::LaunchArgs a)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
    using namespace flash_attn::mla::Int8;
    SetSyncBaseAddr(a.ffts);
    GlobalTensor<int32_t> lengths, table;
    lengths.SetGlobalBuffer((__gm__ int32_t*)a.lengths);
    table.SetGlobalBuffer((__gm__ int32_t*)a.table);
    LengthReader length{lengths, bool(a.lengths), a.config.capacity};
    Int8TilingParams p{};
    uint32_t mode = 0;
    if (!Schedule(a.config, length, p, mode)) {
        Trap();
        return;
    }
    if ASCEND_IS_AIV {
        if (a.metadata) {
            // Every word is checked once across the full vector grid, including
            // lanes without attention work. No barrier is needed: execution uses
            // only the independently checked canonical p, never record contents.
            // A mismatch on any lane fails the whole asynchronous invocation.
            GlobalTensor<uint32_t> metadata;
            metadata.SetGlobalBuffer((__gm__ uint32_t*)a.metadata);
            const auto* words = reinterpret_cast<const uint32_t*>(&p);
            const uint32_t lanes = GetBlockNum() * GetSubBlockNum();
            for (uint32_t w = GetBlockIdx(); w < kFixedWords + a.config.batch; w += lanes) {
                const uint32_t expected = w < kHeaderWords  ? HeaderWord(a.config, w, mode, true)
                                          : w < kFixedWords ? words[w - kHeaderWords]
                                                            : uint32_t(length(w - kFixedWords));
                if (metadata.GetValue(w) != expected) {
                    Trap();
                    return;
                }
            }
        }
    }
    // Validate every used page even for empty Q; padding entries are ignored.
    for (uint32_t b = 0; b < a.config.batch; ++b) {
        for (uint32_t page = 0; page < Div(uint32_t(length(b)), 128); ++page) {
            int32_t physical = table.GetValue(uint64_t(b) * a.cache.tableRowStride + page);
            if (physical < 0 || uint64_t(physical) >= a.cache.physicalPages) {
                Trap();
                return;
            }
        }
    }
    if (!a.config.batch || !a.config.query)
        return;
    p.baseParams.scaleValue = a.scale;
    p.baseParams.softmaxLseFlag = 1;
    switch (mode) {
    case 0:
        Execute<false, false>(a, p);
        break;
    case 2:
        Execute<true, false>(a, p);
        break;
    case 4:
        Execute<false, true>(a, p);
        break;
    case 6:
        Execute<true, true>(a, p);
        break;
    default:
        Trap();
    }
#endif
}

namespace flash_attn::mla::Int8 {
aclError Launch(const LaunchArgs& a)
{
    mla_int8_bsnd<<<a.config.cores, nullptr, a.stream>>>(a);
    return ACL_SUCCESS;
}
} // namespace flash_attn::mla::Int8
