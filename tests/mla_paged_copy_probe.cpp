// Standalone Copy component only: no attention, metadata generation or public
// dispatch. Both builds share the diagnostic sink and two-slot event protocol.
// The oracle uses unmodified pinned CANN Copy/offset/parser headers. Injecting
// outer strides into its layout is not a claim about the native FIA host ABI.
#include <cstdint>
#include "acl/acl.h"
#include "kernel_operator.h"

struct CopyGeometry {
    uint32_t page, batch, columns, physicalPages;
    uint64_t tableStride, kPage, kRow, vPage, vRow;
};
struct CopyTask {
    uint32_t batch, start, rows, dstRows, mode;
};
constexpr uint32_t SLOT_ELEMENTS = 256 * 288 + 512;
constexpr uint32_t GUARD_ELEMENTS = 256;

using namespace AscendC;
#ifdef MLA_PAGED_COPY_BF16
using InputT = bfloat16_t;
#else
using InputT = half;
#endif

#ifdef MLA_PAGED_COPY_ORACLE
#define COPY_ENTRY cann_paged_copy
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
#include "attn_copy_gm_to_l1.h"
struct SourceTensor {
    GlobalTensor<InputT> gmTensor;
    OffsetCalculatorImpl<GmFormat::PA_BnBsND, FormatCategory::GM_KV_PA_BNBD, int64_t> offsetCalculator;
};
#endif
#else
#define COPY_ENTRY mla_paged_copy
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
#include "mla/tile/mla_paged_copy.hpp"
#endif
#endif

template <bool Dump>
__global__ __aicore__ __mix__(1, 2) void COPY_ENTRY(GM_ADDR kr, GM_ADDR kv, GM_ADDR table, GM_ADDR taskData,
                                                    GM_ADDR initial, GM_ADDR output, CopyGeometry g, uint32_t taskCount,
                                                    uint32_t rounds)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
    if ASCEND_IS_AIV {
        return;
    }
    GlobalTensor<uint32_t> tasks;
    tasks.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(taskData));
    GlobalTensor<InputT> init, out;
    init.SetGlobalBuffer(reinterpret_cast<__gm__ InputT*>(initial));
    out.SetGlobalBuffer(reinterpret_cast<__gm__ InputT*>(output));
#ifdef MLA_PAGED_COPY_ORACLE
    GlobalTensor<int32_t> pages;
    pages.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(table));
    SourceTensor k, v;
    k.gmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ InputT*>(kr));
    v.gmTensor.SetGlobalBuffer(reinterpret_cast<__gm__ InputT*>(kv));
    k.offsetCalculator.Init(1, g.page, 64, pages, g.tableStride);
    v.offsetCalculator.Init(1, g.page, 512, pages, g.tableStride);
    k.offsetCalculator.gmLayout.stride = MakeStride(g.kPage, uint64_t(0), g.kRow, uint64_t(1));
    v.offsetCalculator.gmLayout.stride = MakeStride(g.vPage, uint64_t(0), g.vRow, uint64_t(1));
#else
    using namespace flash_attn::mla;
    MlaPagedCacheLayout layout{};
    layout.pageSize = g.page;
    layout.tableRowStride = g.tableStride;
    layout.k.batchStride = g.kPage;
    layout.k.rowStride = g.kRow;
    layout.v.batchStride = g.vPage;
    layout.v.rowStride = g.vRow;
    Tile::PagedCache<InputT> cache;
    cache.Init(kr, kv, table, layout);
#endif
    TPipe pipe;
    TBuf<TPosition::A1> storage;
    TBuf<TPosition::B2> sink;
    pipe.InitBuffer(storage, 2 * SLOT_ELEMENTS * sizeof(InputT));
    pipe.InitBuffer(sink, 512);
    auto l1 = storage.Get<InputT>();
    auto l0 = sink.Get<InputT>();
    for (uint32_t slot = 0; slot < 2; ++slot) {
        SetFlag<HardEvent::MTE1_MTE2>(slot);
    }
    uint32_t total = taskCount * rounds;
    for (uint32_t iteration = 0; iteration <= total; ++iteration) {
        if (iteration < total) {
            uint32_t slot = iteration % 2, index = iteration % taskCount;
            uint64_t base = uint64_t(index) * 5;
            CopyTask t{tasks.GetValue(base), tasks.GetValue(base + 1), tasks.GetValue(base + 2),
                       tasks.GetValue(base + 3), tasks.GetValue(base + 4)};
            WaitFlag<HardEvent::MTE1_MTE2>(slot);
            auto buffer = l1[slot * SLOT_ELEMENTS];
            if constexpr (Dump) {
                DataCopy(buffer, init, DataCopyParams{1, SLOT_ELEMENTS / 16, 0, 0});
                PipeBarrier<PIPE_MTE2>();
            }
            uint32_t contentOffset = t.mode == 1 ? t.dstRows * 32 : 0;
            uint32_t ropeOffset = t.mode == 0 ? t.dstRows * 256 : 0;
#ifdef MLA_PAGED_COPY_ORACLE
            FaL1Tensor<InputT, L1Format::NZ> content{buffer[GUARD_ELEMENTS + contentOffset], t.dstRows};
            FaL1Tensor<InputT, L1Format::NZ> rope{buffer[GUARD_ELEMENTS + ropeOffset], t.dstRows};
            GmKvCoord cc{
                t.batch, 0, t.start, t.mode < 2 ? t.mode * 256 : (t.mode - 2) * 128, t.rows, t.mode < 2 ? 256U : 128U};
            GmKvCoord rc{t.batch, 0, t.start, t.mode * 32, t.rows, 32};
            if (t.mode < 2) {
                CopyKKropePAGmToL1<InputT, GmFormat::PA_BnBsND>{}(content, rope, v, k, cc, rc);
            } else {
                CopyKvGmToL1<InputT, GmFormat::PA_BnBsND>{}(content, v, cc);
            }
#else
            Tile::L1Tensor<InputT> content{buffer[GUARD_ELEMENTS + contentOffset], t.dstRows};
            Tile::L1Tensor<InputT> rope{buffer[GUARD_ELEMENTS + ropeOffset], t.dstRows};
            Tile::KvCoord cc{
                0,      0, t.start, t.mode < 2 ? t.mode * 256 : (t.mode - 2) * 128, t.rows, t.mode < 2 ? 256U : 128U,
                t.batch};
            Tile::KvCoord rc{0, 0, t.start, t.mode * 32, t.rows, 32, t.batch};
            if (t.mode < 2) {
                Tile::CopyKKropePAGmToL1<InputT>{}(content, rope, cache, cc, rc);
            } else {
                Tile::CopyPagedKvGmToL1<InputT>{}(content, cache, cc);
            }
#endif
            SetFlag<HardEvent::MTE2_MTE1>(slot);
        }
        if (iteration > 0) {
            uint32_t slot = (iteration - 1) % 2, index = (iteration - 1) % taskCount;
            WaitFlag<HardEvent::MTE2_MTE1>(slot);
            auto buffer = l1[slot * SLOT_ELEMENTS];
            // A real MTE1 consumer retains the source readiness/release
            // dependency. The optional diagnostic dump runs on MTE3.
            LoadData2DParams p;
            p.repeatTimes = 1;
            p.srcStride = 1;
            LoadData(l0, buffer[GUARD_ELEMENTS], p);
            if constexpr (Dump) {
                SetFlag<HardEvent::MTE1_MTE3>(slot);
                WaitFlag<HardEvent::MTE1_MTE3>(slot);
                DataCopy(out[uint64_t(index) * SLOT_ELEMENTS], buffer, DataCopyParams{1, SLOT_ELEMENTS / 16, 0, 0});
                SetFlag<HardEvent::MTE3_MTE1>(slot);
                WaitFlag<HardEvent::MTE3_MTE1>(slot);
            }
            SetFlag<HardEvent::MTE1_MTE2>(slot);
        }
    }
    for (uint32_t slot = 0; slot < 2; ++slot) {
        WaitFlag<HardEvent::MTE1_MTE2>(slot);
    }
    PipeBarrier<PIPE_ALL>();
#endif
}

// The test supplies bounded, valid immutable device tasks/pages. Static host
// view and v6 device-consumer rejection are covered by their separate suites.
extern "C" int launch_paged_copy(void* stream, void* k, void* v, void* table, void* tasks, void* initial, void* output,
                                 const CopyGeometry* geometry, uint32_t taskCount, uint32_t rounds, bool dump)
{
    if (geometry == nullptr || taskCount == 0 || taskCount > 100000 || rounds == 0 || rounds > 1000 ||
        geometry->page < 16 || geometry->page > 1024 || geometry->page % 16 != 0) {
        return -1;
    }
    auto status = aclrtSetOpExecuteTimeOut(15);
    if (status != ACL_SUCCESS) {
        return status;
    }
    if (dump) {
        COPY_ENTRY<true><<<1, nullptr, stream>>>((uint8_t*)k, (uint8_t*)v, (uint8_t*)table, (uint8_t*)tasks,
                                                 (uint8_t*)initial, (uint8_t*)output, *geometry, taskCount, rounds);
    } else {
        COPY_ENTRY<false><<<1, nullptr, stream>>>((uint8_t*)k, (uint8_t*)v, (uint8_t*)table, (uint8_t*)tasks,
                                                  (uint8_t*)initial, (uint8_t*)output, *geometry, taskCount, rounds);
    }
    return 0;
}
