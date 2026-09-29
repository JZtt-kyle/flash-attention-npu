#include "mla_forward.h"
#include "mla_launch.h"
#include "mla_schedule.h"
#include "mla_metadata_args.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "runtime/rt_ffts.h"
#include <cstring>
#include <limits>
#include <unordered_map>

extern __global__ __aicpu__ uint32_t ComputeMlaDenseMetadata(void* args);

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
                "MLA: this dense launch requires validated C1V2 resources");
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
    auto metadata = at::empty({kMlaDenseMetadataWords}, q.options().dtype(at::kInt));
    for (const auto& t : {q, k, v, qv, out, lse, workspace, metadata}) {
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
    const uint32_t dtype = uint32_t(q.scalar_type() == at::kBFloat16 ? MlaMetadataDtype::Bf16 : MlaMetadataDtype::Fp16);
    // The host schedule currently supplies only launch/allocation bounds. The
    // Kernel receives all tiling and endpoints from the AICPU record, checked
    // against this independently supplied context before any output write.
    a.metadata = static_cast<uint8_t*>(metadata.data_ptr());
    a.context = {dtype,
                 s.bSize,
                 s.s1Size,
                 s.gSize,
                 s.s2Size,
                 p.aicNum,
                 p.aivNum,
                 s.blockDim,
                 s.workspace.userBytes,
                 s.workspace.libApiBytes,
                 p.localMemory};
    MlaMetadataArgs metaArgs{reinterpret_cast<uint64_t>(metadata.data_ptr()),
                             kMlaDenseMetadataWords,
                             dtype,
                             {q.size(0), q.size(1), q.size(2), k.size(1)},
                             p,
                             0,
                             0};
    auto aicpuStream = c10_npu::getNPUStreamFromPool();
    auto aicpuHandle = aicpuStream.stream(false);
    // Match the existing FA3 metadata event handshake. Each host submission
    // enqueues both record/wait pairs before a later call can reuse the events.
    // Keep the handles for the thread/device lifetime; no host synchronization
    // or destruction of a still-pending event is needed in the steady state.
    struct MetadataEvents {
        aclrtEvent inputReady = nullptr;
        aclrtEvent metadataDone = nullptr;
    };
    static thread_local std::unordered_map<c10::DeviceIndex, MetadataEvents> eventsByDevice;
    auto& events = eventsByDevice[stream.device_index()];
    if (events.inputReady == nullptr) {
        TORCH_CHECK(aclrtCreateEvent(&events.inputReady) == ACL_SUCCESS, "MLA: input event creation failed");
    }
    if (events.metadataDone == nullptr) {
        TORCH_CHECK(aclrtCreateEvent(&events.metadataDone) == ACL_SUCCESS, "MLA: metadata event creation failed");
    }
    c10_npu::NPUCachingAllocator::recordStream(metadata.storage().data_ptr(), aicpuStream);
    auto launch = [a, aicpuHandle, inputReady = events.inputReady, metadataDone = events.metadataDone, metaArgs, q, k,
                   v, qv, out, lse, workspace, metadata]() mutable -> int {
        TORCH_CHECK(aclrtRecordEvent(inputReady, a.stream) == ACL_SUCCESS, "MLA: input event recording failed");
        TORCH_CHECK(aclrtStreamWaitEvent(aicpuHandle, inputReady) == ACL_SUCCESS, "MLA: input event wait failed");
        ComputeMlaDenseMetadata<<<1, nullptr, aicpuHandle>>>(&metaArgs, sizeof(metaArgs));
        TORCH_CHECK(aclrtRecordEvent(metadataDone, aicpuHandle) == ACL_SUCCESS, "MLA: metadata event recording failed");
        TORCH_CHECK(aclrtStreamWaitEvent(a.stream, metadataDone) == ACL_SUCCESS, "MLA: metadata event wait failed");
        auto error = q.scalar_type() == at::kBFloat16 ? LaunchDenseBf16(a) : LaunchDenseFp16(a);
        TORCH_CHECK(error == ACL_SUCCESS, "MLA: BATCH_MODE kernel launch failed (", error, ")");
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("ascendc_mla_dense", launch);
    return {out, lse, at::Tensor(), at::Tensor()};
}
} // namespace flash_attn::mla
