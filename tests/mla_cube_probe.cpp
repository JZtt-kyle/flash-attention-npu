// Device-only component check, never linked into the production extension.
// Build twice: the migrated Block and the unmodified pinned CANN Cube header.
#include <cstdint>
#include "acl/acl.h"
#include "runtime/rt_ffts.h"
#include "kernel_operator.h"

#ifdef MLA_CUBE_ORACLE
#define CUBE_PROBE_ENTRY cann_cube_probe
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
#include "fia_block_cube_nonquant_mla.h"
struct ProbeTypes {
    using queryType = half;
    using kvType = half;
    using outputType = half;
    static constexpr bool pageAttention = false;
    static constexpr FIA_LAYOUT layout = FIA_LAYOUT::BSH;
    static constexpr FIA_LAYOUT kvLayout = FIA_LAYOUT::BSH;
};
using Cube = FiaBlockCubeNonQuantMla<ProbeTypes>;
using Run = AttentionCommon::RunInfo;
#endif
#else
#define CUBE_PROBE_ENTRY mla_cube_probe
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
#include "mla/block/mla_cube.hpp"
using Cube = flash_attn::mla::Block::MlaCube;
using Run = flash_attn::mla::Block::RunInfo;
#endif
#endif

using namespace AscendC;

struct ProbeShape {
    uint32_t batch, sq, heads, sk, rounds;
};

// A synthetic Vector only consumes/produces the source Cube event protocol.
// P is supplied independently by the test; there is no softmax or MLA forward.
__global__ __aicore__ __mix__(1, 2) void CUBE_PROBE_ENTRY(uint64_t ffts, GM_ADDR qc, GM_ADDR qr, GM_ADDR kv, GM_ADDR kr,
                                                          GM_ADDR kvList, GM_ADDR p, GM_ADDR qk, GM_ADDR pv,
                                                          ProbeShape shape)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
    SetSyncBaseAddr(ffts);
    TPipe pipe;
    uint32_t core = GetBlockIdx();
    if ASCEND_IS_AIV {
        core /= GetSubBlockNum();
    }
    uint32_t cores = GetBlockNum();
    uint32_t count = (shape.batch - 1 - core) / cores + 1;
    uint32_t tasks = count * shape.rounds;
    uint32_t m = shape.sq * shape.heads;
    uint32_t ma = (m + 15) / 16 * 16;
    uint32_t na = (shape.sk + 31) / 32 * 32;
    uint32_t r1 = ma * na, r2 = ma * 512;

    if ASCEND_IS_AIC {
        Cube cube;
#ifdef MLA_CUBE_ORACLE
        AttentionCommon::ConstInfo params{};
        params.batchSize = shape.batch;
        params.gSize = shape.heads;
        params.qHeadNum = shape.heads;
        params.kvHeadNum = 1;
        params.qSeqSize = shape.sq;
        params.kvSeqSize = shape.sk;
        params.headDim = params.headDimAlign = 512;
        params.headDimRope = 64;
        params.ropeSplitMode = true;
        params.batchContinuous = true;
        params.preLoadNum = 2;
        params.nBufferMBaseSize = 256;
        params.mmResUbSize = r1;
        params.bmm2ResUbSize = r2;
        params.syncC1V1 = 7;
        params.syncV1C2 = 8;
        params.syncC2V1 = 4;
        params.syncV1NupdateC2 = 5;
        params.syncC2V2 = 9;
        cube.InitParams(params);
        cube.Init(qc, kvList, kvList, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                  nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                  nullptr, qr, kr, nullptr, nullptr, nullptr);
        cube.InitBuffers(&pipe);
#else
        flash_attn::mla::Block::CubeParams params{shape.sq, shape.heads, shape.sk, r1, r2};
        cube.Init(&pipe, params, qc, qr, kv, kr);
#endif
        cube.AllocEventID();
        GlobalTensor<float> qkGm, pvGm;
        GlobalTensor<half> pGm;
        qkGm.SetGlobalBuffer((__gm__ float*)qk);
        pvGm.SetGlobalBuffer((__gm__ float*)pv);
        pGm.SetGlobalBuffer((__gm__ half*)p);
        // The source order is QK(t), PV(t-1), with buffer iterators alive
        // across all tasks. Outputs expose both slots to check their offsets.
        for (uint32_t t = 0; t < tasks + 1; ++t) {
            if (t < tasks) {
                Run info{};
                info.bIdx = core + (t % count) * cores;
                info.loop = info.bn2IdxInCurCore = t;
                info.actMBaseSize = m;
                info.actualSingleProcessSInnerSize = shape.sk;
                info.actualSingleProcessSInnerSizeAlign = na;
                info.isFirstSInnerLoop = true;
                cube.InitMm1GlobalTensor(qkGm[uint64_t(info.bIdx) * 2 * r1]);
                cube.ComputeMm1(info);
            }
            if (t > 0) {
                Run info{};
                info.bIdx = core + ((t - 1) % count) * cores;
                info.loop = info.bn2IdxInCurCore = t - 1;
                info.actMBaseSize = m;
                info.actualSingleProcessSInnerSize = shape.sk;
                info.actualSingleProcessSInnerSizeAlign = na;
                info.isFirstSInnerLoop = true;
                cube.InitMm2GlobalTensor(pGm[uint64_t(info.bIdx) * 2 * r1], pvGm[uint64_t(info.bIdx) * 2 * r2]);
                cube.ComputeMm2(info);
            }
        }
        cube.FreeEventID();
    }
    if ASCEND_IS_AIV {
        for (uint32_t t = 0; t < tasks + 2; ++t) {
            if (t > 0 && t <= tasks) {
                CrossCoreWaitFlag(7);
                CrossCoreSetFlag<2, PIPE_MTE3>(8);
                CrossCoreWaitFlag(4);
                CrossCoreSetFlag<2, PIPE_MTE3>(5);
            }
            if (t > 1) {
                CrossCoreWaitFlag(9);
            }
        }
        CrossCoreWaitFlag(4);
    }
#endif
}

extern "C" int launch_cube(void* stream, void* qc, void* qr, void* kv, void* kr, void* kvList, void* p, void* qk,
                           void* pv, uint32_t batch, uint32_t sq, uint32_t heads, uint32_t sk, uint32_t cores,
                           uint32_t rounds)
{
    if (batch == 0 || sq == 0 || heads == 0 || uint64_t(sq) * heads > 256 || sk == 0 || sk > 512 || cores == 0 ||
        cores > batch || rounds == 0) {
        return -1;
    }
    uint64_t ffts = 0;
    uint32_t len = 0;
    auto error = rtGetC2cCtrlAddr(&ffts, &len);
    if (error != RT_ERROR_NONE) {
        return error;
    }
    error = aclrtSetOpExecuteTimeOut(15);
    if (error != ACL_SUCCESS) {
        return error;
    }
    CUBE_PROBE_ENTRY<<<cores, nullptr, stream>>>(ffts, (uint8_t*)qc, (uint8_t*)qr, (uint8_t*)kv, (uint8_t*)kr,
                                                 (uint8_t*)kvList, (uint8_t*)p, (uint8_t*)qk, (uint8_t*)pv,
                                                 ProbeShape{batch, sq, heads, sk, rounds});
    return 0;
}
