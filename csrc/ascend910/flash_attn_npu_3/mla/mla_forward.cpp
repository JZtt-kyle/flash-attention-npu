#include "mla_forward.h"
#include "mla_launch.h"
#include "mla_schedule.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "runtime/rt_ffts.h"
#include <algorithm>
#include <cstring>
#include <limits>

namespace flash_attn::mla {
std::vector<at::Tensor> DenseForward(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                     const at::Tensor& qv, float scale)
{
    const char* soc = aclrtGetSocName();
    TORCH_CHECK(soc && std::strncmp(soc, "Ascend910B", 10) == 0,
                "MLA compressed forward is not implemented yet on devices other than 910B");
    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    TORCH_CHECK(platform, "MLA: platform information is unavailable");
    MlaPlatformInfo p{};
    p.aicNum = platform->GetCoreNumAic();
    p.aivNum = platform->GetCoreNumAiv();
    TORCH_CHECK(p.aicNum > 0 && p.aicNum <= kMlaMaxAic && p.aivNum == 2 * p.aicNum,
                "MLA: this FP16 launch requires validated C1V2 resources");
    using Mem = platform_ascendc::CoreMemType;
    platform->GetCoreMemSize(Mem::L1, p.localMemory.l1Bytes);
    platform->GetCoreMemSize(Mem::UB, p.localMemory.ubBytes);
    platform->GetCoreMemSize(Mem::L0_A, p.localMemory.l0aBytes);
    platform->GetCoreMemSize(Mem::L0_B, p.localMemory.l0bBytes);
    platform->GetCoreMemSize(Mem::L0_C, p.localMemory.l0cBytes);
    p.libApiWorkspaceBytes = platform->GetLibApiWorkSpaceSize();
    for (uint32_t c = 1; c <= p.aicNum; ++c) {
        p.blockDims[c - 1] = platform->CalcTschBlockDim(c * 2, c, c * 2);
    }
    MlaSchedule s{};
    auto status = BuildDenseSchedule({q.size(0), q.size(1), q.size(2), k.size(1)}, p, s);
    TORCH_CHECK(status == MlaScheduleStatus::Success, "MLA: dense scheduling failed (status ", int(status), ")");
    TORCH_CHECK(s.cvRatio == 2 && s.scheduleMode == 1 && s.fd.numOfFdHead == 0, "MLA: unexpected dense schedule");
    TORCH_CHECK(s.workspace.totalBytes <= uint64_t(std::numeric_limits<int64_t>::max()) &&
                    s.workspace.mm1ResSize <= UINT32_MAX && s.workspace.mm2ResSize <= UINT32_MAX,
                "MLA: workspace or launch field overflow");

    auto out = at::empty({q.size(0), q.size(1), q.size(2), v.size(3)}, q.options());
    auto lse = at::empty({q.size(0), q.size(2), q.size(1)}, q.options().dtype(at::kFloat));
    auto workspace = at::empty({static_cast<int64_t>(s.workspace.totalBytes)}, q.options().dtype(at::kByte));
    for (const auto& t : {q, k, v, qv, out, lse, workspace}) {
        auto address = reinterpret_cast<uintptr_t>(t.data_ptr());
        TORCH_CHECK(address != 0 && uint64_t(t.nbytes()) <= std::numeric_limits<uintptr_t>::max() - address,
                    "MLA: tensor address overflow");
    }
    MlaLaunchArgs a{};
    uint32_t fftsLength = 0;
    TORCH_CHECK(rtGetC2cCtrlAddr(&a.ffts, &fftsLength) == RT_ERROR_NONE,
                "MLA: failed to obtain cross-core synchronization address");
    auto stream = c10_npu::getCurrentNPUStream();
    a.stream = stream.stream(false);
    for (const auto& t : {q, k, v, qv}) {
        c10_npu::NPUCachingAllocator::recordStream(t.storage().data_ptr(), stream);
    }
    a.qc = static_cast<uint8_t*>(qv.data_ptr());
    a.qr = static_cast<uint8_t*>(q.data_ptr());
    a.kv = static_cast<uint8_t*>(v.data_ptr());
    a.kr = static_cast<uint8_t*>(k.data_ptr());
    a.output = static_cast<uint8_t*>(out.data_ptr());
    a.lse = static_cast<uint8_t*>(lse.data_ptr());
    // Reserve libapi bytes separately, as in the source, even though this
    // dense specialization only consumes user workspace.
    a.workspace = static_cast<uint8_t*>(workspace.data_ptr()) + s.workspace.libApiBytes;
    a.scale = scale;
    a.schedule = {s.bSize,
                  s.s1Size,
                  s.gSize,
                  s.s2Size,
                  s.mBaseSize,
                  s.s2BaseSize,
                  s.usedCoreNum,
                  s.blockDim,
                  s.cvRatio,
                  uint32_t(s.workspace.mm1ResSize),
                  uint32_t(s.workspace.mm2ResSize)};
    static_assert(MlaDenseKernelSchedule::kEndpointCapacity == kMlaMaxAic);
    std::copy(s.bN2End.begin(), s.bN2End.end(), a.schedule.bN2End);
    std::copy(s.gS1End.begin(), s.gS1End.end(), a.schedule.gS1End);
    std::copy(s.s2End.begin(), s.s2End.end(), a.schedule.s2End);
    auto launch = [a, q, k, v, qv, out, lse, workspace]() -> int {
        auto error = LaunchDenseFp16(a);
        TORCH_CHECK(error == ACL_SUCCESS, "MLA: BATCH_MODE kernel launch failed (", error, ")");
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("ascendc_mla_dense", launch);
    return {out, lse, at::Tensor(), at::Tensor()};
}
} // namespace flash_attn::mla
