// Test-only numerical launcher; never linked into the FA3 extension.
// Oracle headers are unmodified ca18aee Git blobs. Only the fragment harness
// supplies endpoints and raw-buffer addresses, preserving generic source math.
#include <cmath>
#include "kernel_operator.h"
#include "acl/acl.h"
#include "runtime/rt.h"
#include "mla/mla_prefill_params.h"
using namespace AscendC;
using flash_attn::mla::MlaPrefillParams;
using flash_attn::mla::MlaPrefillTask;
#ifdef MLA_DENSE_BF16
using InputT = bfloat16_t;
#else
using InputT = half;
#endif
#include "mla_probe_batch_launch.h"
struct PrefillTasks {
    MlaPrefillTask tasks[26];
};
#ifdef MLA_DENSE_ORACLE
#define PREFILL_ENTRY cann_prefill_attention
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
#include "oracle_tiling.h"
#include "fia_public_define.h"
using fa_base_vector::FLOAT_ZERO;
using fa_base_vector::RowMuls;
constexpr uint32_t BUFFER_SIZE_BYTE_32B = AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_32B;
constexpr uint32_t BUFFER_SIZE_BYTE_8K = AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_8K;
constexpr uint32_t BUFFER_SIZE_BYTE_16K = AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_16K;
constexpr float FLOAT_INF = AttentionCommon::ConstInfo::FLOAT_INF;
constexpr uint32_t BUFFER_SIZE_BYTE_1K = AttentionCommon::ConstInfo::BUFFER_SIZE_BYTE_1K;
#include "fia_kernel_nonquant.h"
template <bool Brc>
using Types = AttentionCommon::FIAType<InputT, InputT, InputT, InputT, false, true, AttentionCommon::FIA_LAYOUT::BSH, 0,
                                       false, AttentionCommon::FIA_LAYOUT::BSH, Brc>;
template <bool Brc>
using Cube = typename std::conditional<g_coreType == AscendC::AIC, FiaBlockCubeNonQuantGqa<Types<Brc>>,
                                       FiaBlockCubeNonQuantGqaDummy<Types<Brc>>>::type;
template <bool Brc>
using Vec = typename std::conditional<g_coreType == AscendC::AIC, FiaBlockVecNonQuantDummy<Types<Brc>>,
                                      FiaBlockVecNonQuant<Types<Brc>>>::type;
template <bool Brc>
using Fd = typename std::conditional<g_coreType == AscendC::AIC, FiaBlockVecFlashDecodeDummy<Types<Brc>>,
                                     FiaBlockVecFlashDecode<Types<Brc>>>::type;
template <bool Brc>
class Oracle : public FiaKernelNonQuant<Types<Brc>, Cube<Brc>, Vec<Brc>, Fd<Brc>> {
    using Base = FiaKernelNonQuant<Types<Brc>, Cube<Brc>, Vec<Brc>, Fd<Brc>>;

  public:
    __aicore__ inline void Fragment(const MlaPrefillParams& p, const MlaPrefillTask& t, uint32_t core, bool partial,
                                    GM_ADDR po, GM_ADDR su, GM_ADDR ma, TPipe* pipe)
    {
        if (t.begin == t.end)
            return; // Harness initializes canonical empty slots.
        uint32_t bn = t.batch * p.kvHeads + t.kvHead, mb = t.rowStart / p.mBase;
        this->constInfo.bN2Start = this->constInfo.bN2End = bn;
        this->constInfo.gS1Start = this->constInfo.gS1End = mb;
        this->constInfo.s2Start = t.begin;
        this->constInfo.s2End = t.end;
        this->constInfo.coreStartKVSplitPos = 0;
        this->GetTaskDealMode(bn, mb, t.begin);
        if (this->curS2Start != t.begin || this->curS2End != t.end)
            Trap();
        if ASCEND_IS_AIV {
            GlobalTensor<float> o, s, m;
            o.SetGlobalBuffer((__gm__ float*)po);
            s.SetGlobalBuffer((__gm__ float*)su);
            m.SetGlobalBuffer((__gm__ float*)ma);
            this->vectorService.InitFlashDecodeGlobalTensor(o, m, s);
            this->vectorService.InitBuffers(pipe);
            this->vectorService.AllocEventID();
        }
        if ASCEND_IS_AIC {
            this->matmulService.InitBuffers(pipe);
            this->matmulService.AllocEventID();
        }
        AttentionCommon::RunInfo ring[3]{};
        uint32_t count = t.end - t.begin;
        for (uint32_t loop = 0; loop < count + 2; ++loop) {
            if (loop < count) {
                this->CreateTask(loop, bn, mb, t.begin + loop, ring);
                ring[loop % 3].tndIsS2SplitCore = partial;
                ring[loop % 3].tndCoreStartKVSplitPos = 0;
                ring[loop % 3].accumTmpOutNum = 0;
            }
            this->ExecuteTask(loop, ring);
        }
        if ASCEND_IS_AIV {
            this->vectorService.FreeEventID();
        }
        if ASCEND_IS_AIC {
            this->matmulService.FreeEventID();
        }
    }
};
#endif
#else
#define PREFILL_ENTRY mla_prefill_attention
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
#include "mla/kernel/mla_prefill.hpp"
#endif
#endif

template <bool Brc, bool Partial>
__global__ __aicore__ __mix__(1, 2) void PREFILL_ENTRY(uint64_t ffts, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR qr,
                                                       GM_ADDR kr, GM_ADDR out, GM_ADDR lse, GM_ADDR ws, GM_ADDR po,
                                                       GM_ADDR su, GM_ADDR ma, GM_ADDR qlens, GM_ADDR klens,
                                                       GM_ADDR mask, MlaPrefillParams p, PrefillTasks tasks)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
    SetSyncBaseAddr(ffts);
    TPipe pipe;
    uint32_t core = GetBlockIdx();
    if ASCEND_IS_AIV {
        core /= 2;
    }
    auto t = tasks.tasks[core];
    po += uint64_t(core) * p.mBase * 128 * 4;
    su += uint64_t(core) * p.mBase * 8 * 4;
    ma += uint64_t(core) * p.mBase * 8 * 4;
#ifdef MLA_DENSE_ORACLE
    FusedInferAttentionScoreTilingData tiling{};
    tiling.baseParams.bSize = p.batch;
    tiling.baseParams.n2Size = p.kvHeads;
    tiling.baseParams.gSize = p.heads / p.kvHeads;
    tiling.baseParams.s1Size = p.sq;
    tiling.baseParams.s2Size = p.sk;
    tiling.baseParams.headDim = 128;
    tiling.baseParams.headDimRope = 64;
    tiling.baseParams.scaleValue = p.scale;
    tiling.baseParams.usedCoreNum = p.cores;
    tiling.baseParams.batchContinuous = 1;
    tiling.baseParams.softmaxLseFlag = !Partial;
    tiling.baseParams.actualSeqS1Dims = p.batch;
    tiling.baseParams.actualSeqS2Dims = p.batch;
    tiling.baseParams.outputLayout = 0;
    tiling.innerSplitParams.mBaseSize = p.mBase;
    tiling.innerSplitParams.s2BaseSize = p.nBase;
    tiling.workspaceParams.mm1ResSize = p.mBase * ((p.nBase + 31) / 32 * 32);
    tiling.workspaceParams.mm2ResSize = p.mBase * 128;
    tiling.maskParams.attenMaskFlag = p.window.kind != 0;
    tiling.maskParams.attenMaskStride = 2048;
    tiling.maskParams.sparseMode = p.window.kind == 2 ? 4 : p.window.kind == 1 ? 3 : 0;
    tiling.maskParams.preToken = p.window.kind == 1 ? INT32_MAX : p.window.left;
    tiling.maskParams.nextToken = p.window.kind == 1 ? 0 : p.window.right;
    tiling.maskParams.isRowInvalid = 1;
    tiling.maskParams.isExistRowInvalid = 1;
    tiling.fdParams.bN2IdxOfFdHead[0] = t.batch * p.kvHeads + t.kvHead;
    tiling.fdParams.gS1IdxOfFdHead[0] = t.rowStart / p.mBase;
    tiling.fdParams.s2SplitNumOfFdHead[0] = 1;
    Oracle<Brc> op;
    op.Init(q, k, v, nullptr, mask, qlens, klens, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, qr, kr, nullptr,
            nullptr, out, lse, ws, &tiling, nullptr, &pipe);
    op.Fragment(p, t, core, Partial, po, su, ma, &pipe);
#else
    flash_attn::mla::Kernel::MlaPrefill<InputT, Brc, Partial> op;
    op.Run(p, t, core, q, k, v, out, lse, ws, po, su, ma, &pipe);
#endif
#endif
}
extern "C" int launch_prefill(void* stream, void* q, void* k, void* v, void* qr, void* kr, void* out, void* lse,
                              void* ws, void* po, void* su, void* ma, void* qlens, void* klens, void* mask,
                              const MlaPrefillParams* p, const PrefillTasks* tasks, int partial)
{
    if (!p || !tasks || !q || !k || !v || !out || !lse || !ws || !po || !su || !ma || !qlens || !klens || !mask ||
        p->cores < 1 || p->cores > 26 || !p->kvHeads || !p->heads || p->heads % p->kvHeads ||
        !std::isfinite(p->scale) || !p->mBase || p->mBase > 512 || !p->nBase || p->nBase > 8192 || p->window.kind > 2 ||
        partial < 0 || partial > 1)
        return -1;
    if (p->mBase > 128 && p->nBase > 1024)
        return -1;
    for (uint32_t i = 0; i < p->cores; ++i) {
        auto t = tasks->tasks[i];
        if (t.batch >= p->batch || t.kvHead >= p->kvHeads || t.rowStart % p->mBase || !t.rows || t.rows > p->mBase ||
            t.queryLength > p->sq || t.kvLength > p->sk ||
            uint64_t(t.rowStart) + t.rows > uint64_t(t.queryLength) * (p->heads / p->kvHeads) || t.begin > t.end ||
            t.end > (uint64_t(t.kvLength) + p->nBase - 1) / p->nBase)
            return -1;
    }
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
#define CALL(B, P)                                                                                                     \
    PREFILL_ENTRY<B, P><<<p->cores, nullptr, stream>>>(ffts, (uint8_t*)q, (uint8_t*)k, (uint8_t*)v, (uint8_t*)qr,      \
                                                       (uint8_t*)kr, (uint8_t*)out, (uint8_t*)lse, (uint8_t*)ws,       \
                                                       (uint8_t*)po, (uint8_t*)su, (uint8_t*)ma, (uint8_t*)qlens,      \
                                                       (uint8_t*)klens, (uint8_t*)mask, *p, *tasks)
    if (p->mBase <= 128) {
        if (partial) {
            CALL(true, true);
        } else {
            CALL(true, false);
        }
    } else {
        if (partial) {
            CALL(false, true);
        } else {
            CALL(false, false);
        }
    }
#undef CALL
#if !defined(__NPU_ARCH__)
    return batchLaunchCount == before + 1 ? batchLaunchError : -2;
#else
    return 0;
#endif
}
