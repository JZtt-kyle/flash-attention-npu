// Isolated numerical probe. The oracle compiles untouched fixed CANN blobs;
// only its raw tiling declarations and buffer launch harness are generated.
#include <cstring>
#include <limits>
#include "kernel_operator.h"
#include "acl/acl.h"
#include "runtime/rt.h"
#include "mla/mla_int8_params.h"
#include "mla/mla_paged_layout.h"
#include "mla_probe_batch_launch.h"
using flash_attn::mla::MlaPagedCacheLayout;
using flash_attn::mla::Int8::Int8TilingParams;
#ifdef MLA_INT8_ORACLE
#define INT8_ENTRY cann_int8_component
#else
#define INT8_ENTRY mla_int8_component
#endif
using namespace AscendC;
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
#ifdef MLA_INT8_ORACLE
#include "oracle_tiling.h"
#include "incre_flash_attention_preload_mla.h"
template <bool Tnd, bool Fd, bool Balance>
using Types = IFAType<int8_t, int8_t, bfloat16_t, bfloat16_t, true, Fd, Tnd ? LAYOUT::TND : LAYOUT::BSH, 0, false,
                      LAYOUT::NZ, AMLAMODE::NORMAL, Balance>;
template <bool Tnd, bool Fd, bool Balance>
class Kernel : public IncreFlashAttentionAttenPreloadMla<Types<Tnd, Fd, Balance>> {
  public:
    // The source's InitCalcParamsEach reads lengths before InitActualSeqLen.
    // Seed only these raw-buffer views in the oracle harness; every numerical
    // and traversal method remains an unmodified fixed source definition.
    __aicore__ inline void SeedLengths(GM_ADDR qlen, GM_ADDR klen)
    {
        this->actualSeqLengthsGmQ.SetGlobalBuffer((__gm__ uint64_t*)qlen);
        this->actualSeqLengthsGm.SetGlobalBuffer((__gm__ uint64_t*)klen);
    }
};
#else
#include "mla/kernel/mla_int8.hpp"
template <bool Tnd, bool Fd, bool Balance>
using Kernel = flash_attn::mla::Int8::MlaInt8Kernel<flash_attn::mla::Int8::Int8Type<Tnd, Fd, Balance>>;
#endif
#endif

template <bool Tnd, bool Fd, bool Balance>
__global__ __aicore__ __mix__(1, 2) void INT8_ENTRY(uint64_t ffts, GM_ADDR qv, GM_ADDR qr, GM_ADDR kv, GM_ADDR kr,
                                                    GM_ADDR qs, GM_ADDR ks, GM_ADDR vs, GM_ADDR table, GM_ADDR qlen,
                                                    GM_ADDR klen, GM_ADDR mask, GM_ADDR out, GM_ADDR lse, GM_ADDR ws,
                                                    Int8TilingParams p, MlaPagedCacheLayout layout)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
    SetSyncBaseAddr(ffts);
    TPipe pipe;
    Kernel<Tnd, Fd, Balance> op;
#ifdef MLA_INT8_ORACLE
    static_assert(sizeof(p) == sizeof(IncreFlashAttentionTilingDataMla));
    auto* params = reinterpret_cast<const IncreFlashAttentionTilingDataMla*>(&p);
    op.SeedLengths(qlen, klen);
    op.Init(qv, kv, kv, nullptr, mask, qlen, klen, table, nullptr, qr, kr, out, lse, ws, params, nullptr, &pipe);
    op.InitQuant(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, ks, nullptr, vs, nullptr, nullptr, qs,
                 ws);
#else
    op.Init(qv, kv, qr, kr, mask, qlen, klen, table, out, lse, ws, &p, &pipe);
    op.SetCacheLayout(layout);
    op.InitQuant(qs, ks, vs);
#endif
    op.Process();
#endif
}

extern "C" uint32_t int8_params_size()
{
    return sizeof(Int8TilingParams);
}
extern "C" int launch_int8(void* stream, void* qv, void* qr, void* kv, void* kr, void* qs, void* ks, void* vs,
                           void* table, void* qlen, void* klen, void* mask, void* out, void* lse, void* ws,
                           const Int8TilingParams* p, int mode, const MlaPagedCacheLayout* layout)
{
    if (!p || !qv || !qr || !kv || !kr || !qs || !ks || !vs || !table || !qlen || !klen || !out || !lse || !ws ||
        p->increFlashAttentionSingleCoreParams.usedCoreNum == 0 ||
        p->increFlashAttentionSingleCoreParams.usedCoreNum > 20 || p->baseParams.blockSize != 128 || mode < 0 ||
        mode > 7 || !layout)
        return -1;
    uint64_t ffts = 0;
    uint32_t len = 0;
    auto error = rtGetC2cCtrlAddr(&ffts, &len);
    if (error != RT_ERROR_NONE)
        return error;
    error = aclrtSetOpExecuteTimeOut(15);
    if (error != ACL_SUCCESS)
        return error;
#if !defined(__NPU_ARCH__)
    auto before = batchLaunchCount;
#endif
#define CALL(T, F, B)                                                                                                  \
    INT8_ENTRY<T, F, B><<<p->increFlashAttentionSingleCoreParams.usedCoreNum, nullptr, stream>>>(                      \
        ffts, (uint8_t*)qv, (uint8_t*)qr, (uint8_t*)kv, (uint8_t*)kr, (uint8_t*)qs, (uint8_t*)ks, (uint8_t*)vs,        \
        (uint8_t*)table, (uint8_t*)qlen, (uint8_t*)klen, (uint8_t*)mask, (uint8_t*)out, (uint8_t*)lse, (uint8_t*)ws,   \
        *p, *layout)
    switch (mode) {
    case 0:
        CALL(false, false, false);
        break;
    case 1:
        CALL(true, false, false);
        break;
    case 2:
        CALL(false, true, false);
        break;
    case 3:
        CALL(true, true, false);
        break;
    case 4:
        CALL(false, false, true);
        break;
    case 5:
        CALL(true, false, true);
        break;
    case 6:
        CALL(false, true, true);
        break;
    case 7:
        CALL(true, true, true);
        break;
    }
#undef CALL
#if !defined(__NPU_ARCH__)
    return batchLaunchCount == before + 1 ? batchLaunchError : -2;
#else
    return 0;
#endif
}
