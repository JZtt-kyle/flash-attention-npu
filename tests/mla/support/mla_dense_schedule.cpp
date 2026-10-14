// Test-only bridge from real platform resources to the M04b1 static scheduler.
#include <algorithm>
#include "tiling/platform/platform_ascendc.h"
#include "mla/mla_schedule.h"
#include "mla/mla_dense_params.h"
using namespace flash_attn::mla;

extern "C" uint32_t dense_schedule_size()
{
    return sizeof(MlaDenseKernelSchedule);
}
extern "C" int prepare_dense(int64_t batch, int64_t sq, int64_t heads, int64_t sk, MlaDenseKernelSchedule* params,
                             uint64_t* details)
{
    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (!platform || !params || !details) {
        return -1;
    }
    MlaPlatformInfo p{};
    p.aicNum = platform->GetCoreNumAic();
    p.aivNum = platform->GetCoreNumAiv();
    if (p.aicNum == 0 || p.aicNum > kMlaMaxAic || p.aivNum % p.aicNum) {
        return -2;
    }
    using Mem = platform_ascendc::CoreMemType;
    platform->GetCoreMemSize(Mem::L1, p.localMemory.l1Bytes);
    platform->GetCoreMemSize(Mem::UB, p.localMemory.ubBytes);
    platform->GetCoreMemSize(Mem::L0_A, p.localMemory.l0aBytes);
    platform->GetCoreMemSize(Mem::L0_B, p.localMemory.l0bBytes);
    platform->GetCoreMemSize(Mem::L0_C, p.localMemory.l0cBytes);
    p.libApiWorkspaceBytes = platform->GetLibApiWorkSpaceSize();
    uint32_t ratio = p.aivNum / p.aicNum;
    for (uint32_t c = 1; c <= p.aicNum; ++c) {
        p.blockDims[c - 1] = platform->CalcTschBlockDim(c * ratio, c, c * ratio);
    }
    MlaSchedule s{};
    auto status = BuildDenseSchedule({batch, sq, heads, sk}, p, s);
    if (status != MlaScheduleStatus::Success) {
        return static_cast<int>(status);
    }
    static_assert(MlaDenseKernelSchedule::kEndpointCapacity == kMlaMaxAic);
    *params = {s.bSize,
               s.s1Size,
               s.gSize,
               s.s2Size,
               s.mBaseSize,
               s.s2BaseSize,
               s.usedCoreNum,
               s.blockDim,
               s.cvRatio,
               static_cast<uint32_t>(s.workspace.mm1ResSize),
               static_cast<uint32_t>(s.workspace.mm2ResSize)};
    std::copy(s.bN2End.begin(), s.bN2End.end(), params->bN2End);
    std::copy(s.gS1End.begin(), s.gS1End.end(), params->gS1End);
    std::copy(s.s2End.begin(), s.s2End.end(), params->s2End);
    uint64_t values[] = {s.workspace.userBytes,
                         s.workspace.kernelBytes,
                         p.aicNum,
                         p.aivNum,
                         p.localMemory.l1Bytes,
                         p.localMemory.ubBytes,
                         p.localMemory.l0aBytes,
                         p.localMemory.l0bBytes,
                         p.localMemory.l0cBytes,
                         s.workspace.libApiBytes,
                         s.workspace.qkOffsetBytes,
                         s.workspace.pOffsetBytes,
                         s.workspace.pvOffsetBytes,
                         s.workspace.vec2OffsetBytes,
                         s.scheduleMode,
                         s.fd.numOfFdHead};
    std::copy(std::begin(values), std::end(values), details);
    return 0;
}
