#include "mla_forward.h"
#include "mla_launch.h"
#include "mla_schedule.h"
#include "mla_metadata_args.h"
#include "mla_validation.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "runtime/rt_ffts.h"
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>
#include <cmath>
#include <cstring>
#include <limits>
#include <unordered_map>

extern __global__ __aicpu__ uint32_t ComputeMlaDenseMetadata(void* args);

namespace flash_attn::mla {
namespace {
MlaPlatformInfo DensePlatform()
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
    return p;
}

struct MetadataSubmission {
    at::Tensor metadata;
    std::optional<at::Tensor> lengths;
    MlaMetadataArgs args;
    aclrtStream current, worker;
    aclrtEvent inputReady, metadataDone;

    void Enqueue()
    {
        TORCH_CHECK(aclrtRecordEvent(inputReady, current) == ACL_SUCCESS, "MLA: input event recording failed");
        TORCH_CHECK(aclrtStreamWaitEvent(worker, inputReady) == ACL_SUCCESS, "MLA: input event wait failed");
        ComputeMlaDenseMetadata<<<1, nullptr, worker>>>(&args, sizeof(args));
        TORCH_CHECK(aclrtRecordEvent(metadataDone, worker) == ACL_SUCCESS, "MLA: metadata event recording failed");
        TORCH_CHECK(aclrtStreamWaitEvent(current, metadataDone) == ACL_SUCCESS, "MLA: metadata event wait failed");
    }
};

MetadataSubmission PrepareMetadata(const MlaDenseShape& shape, const MlaPlatformInfo& platform, at::ScalarType dtype,
                                   const at::TensorOptions& options, const std::optional<at::Tensor>& lengths)
{
    auto metadata = at::empty({kMlaDenseMetadataWords}, options.dtype(at::kInt));
    auto stream = c10_npu::getCurrentNPUStream();
    auto worker = c10_npu::getNPUStreamFromPool();
    // The two record/wait pairs are submitted together before these per-host-
    // thread/device handles can be reused. No steady-state host wait is needed.
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
    c10_npu::NPUCachingAllocator::recordStream(metadata.storage().data_ptr(), worker);
    if (lengths && lengths->numel() != 0) {
        c10_npu::NPUCachingAllocator::recordStream(lengths->storage().data_ptr(), worker);
    }
    MlaMetadataArgs args{reinterpret_cast<uint64_t>(metadata.data_ptr()),
                         kMlaDenseMetadataWords,
                         uint32_t(dtype == at::kBFloat16 ? MlaMetadataDtype::Bf16 : MlaMetadataDtype::Fp16),
                         shape,
                         platform,
                         lengths ? reinterpret_cast<uint64_t>(lengths->data_ptr()) : 0,
                         lengths ? uint64_t(lengths->numel()) : 0};
    return {metadata,           lengths, args, stream.stream(false), worker.stream(false), events.inputReady,
            events.metadataDone};
}
} // namespace

at::Tensor DenseMetadata(int64_t batch, int64_t sq, int64_t heads, int64_t sk, at::ScalarType dtype,
                         const at::Tensor& cacheSeqlens)
{
    auto submission =
        PrepareMetadata({batch, sq, heads, sk}, DensePlatform(), dtype, cacheSeqlens.options(), cacheSeqlens);
    at_npu::native::OpCommand::RunOpApiV2("ascendc_mla_metadata", [submission]() mutable -> int {
        submission.Enqueue();
        return 0;
    });
    return submission.metadata;
}

at::Tensor DenseMetadataOp(int64_t batch, int64_t sq, int64_t sk, int64_t heads, at::ScalarType dtype,
                           const at::Tensor& cacheSeqlens)
{
    TORCH_CHECK(FitsDenseShape(batch, sq, heads, sk),
                "MLA scheduler metadata is not implemented yet outside the dense shape bounds");
    TORCH_CHECK(dtype == at::kHalf || dtype == at::kBFloat16, "MLA: qkv_dtype must be FP16 or BF16");
    CheckCacheLengths(cacheSeqlens, batch, cacheSeqlens.device());
    const c10::OptionalDeviceGuard deviceGuard(device_of(cacheSeqlens));
    return DenseMetadata(batch, sq, heads, sk, dtype, cacheSeqlens);
}

std::vector<at::Tensor> DenseForward(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                     const at::Tensor& qv, float scale, const std::optional<at::Tensor>& cacheSeqlens,
                                     const std::optional<at::Tensor>& schedulerMetadata)
{
    auto p = DensePlatform();
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
    std::optional<MetadataSubmission> producer;
    if (!schedulerMetadata) {
        producer = PrepareMetadata({q.size(0), q.size(1), q.size(2), k.size(1)}, p, q.scalar_type(), q.options(),
                                   cacheSeqlens);
    }
    auto metadata = schedulerMetadata ? *schedulerMetadata : producer->metadata;
    for (const auto& t : {q, k, v, qv, out, lse, workspace, metadata}) {
        auto address = reinterpret_cast<uintptr_t>(t.data_ptr());
        TORCH_CHECK((t.numel() == 0 || address != 0) &&
                        uint64_t(t.nbytes()) <= std::numeric_limits<uintptr_t>::max() - address,
                    "MLA: tensor address overflow");
    }
    MlaLaunchArgs a{};
    uint32_t fftsLength = 0;
    TORCH_CHECK(rtGetC2cCtrlAddr(&a.ffts, &fftsLength) == RT_ERROR_NONE,
                "MLA: failed to obtain cross-core synchronization address");
    auto stream = c10_npu::getCurrentNPUStream();
    a.stream = stream.stream(false);
    for (const auto& t : {q, k, v, qv, metadata}) {
        if (t.numel() != 0) {
            c10_npu::NPUCachingAllocator::recordStream(t.storage().data_ptr(), stream);
        }
    }
    if (cacheSeqlens && cacheSeqlens->numel() != 0) {
        c10_npu::NPUCachingAllocator::recordStream(cacheSeqlens->storage().data_ptr(), stream);
        a.cacheSeqlens = static_cast<uint8_t*>(cacheSeqlens->data_ptr());
    }
    a.qc = static_cast<uint8_t*>(qv.data_ptr());
    a.qr = static_cast<uint8_t*>(q.data_ptr());
    a.kv = static_cast<uint8_t*>(v.data_ptr());
    a.kr = static_cast<uint8_t*>(k.data_ptr());
    a.output = static_cast<uint8_t*>(out.data_ptr());
    a.lse = static_cast<uint8_t*>(lse.data_ptr());
    // Reserve libapi bytes separately, as in the source, even though this
    // dense specialization only consumes user workspace.
    a.workspace =
        workspace.numel() == 0 ? nullptr : static_cast<uint8_t*>(workspace.data_ptr()) + s.workspace.libApiBytes;
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
    auto launch = [a, producer, q, k, v, qv, out, lse, workspace, metadata, cacheSeqlens]() mutable -> int {
        if (producer) {
            producer->Enqueue();
        }
        auto error = q.scalar_type() == at::kBFloat16 ? LaunchDenseBf16(a) : LaunchDenseFp16(a);
        TORCH_CHECK(error == ACL_SUCCESS, "MLA: BATCH_MODE kernel launch failed (", error, ")");
        return 0;
    };
    at_npu::native::OpCommand::RunOpApiV2("ascendc_mla_dense", launch);
    if (k.size(1) == 0 && out.numel() != 0) {
        // TriDao FA3 d15f1531, hopper/flash_api.cpp::mha_fwd global-empty
        // branch. These stream-ordered fills follow metadata/length validation.
        out.zero_();
        lse.fill_(std::numeric_limits<float>::infinity());
    }
    return {out, lse, at::Tensor(), at::Tensor()};
}

// The public read-only cache path has already rejected unsupported options.
// Keep tensor/scale checks here as well: the dispatcher is independently callable.
std::tuple<at::Tensor, at::Tensor> DenseForwardOp(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                                  const at::Tensor& qv, std::optional<double> softmaxScale,
                                                  const std::optional<at::Tensor>& cacheSeqlens,
                                                  const std::optional<at::Tensor>& schedulerMetadata)
{
    CheckForwardBoundary(q, k, v, qv, false, false, false);
    TORCH_CHECK(!softmaxScale ||
                    (std::isfinite(*softmaxScale) && std::abs(*softmaxScale) <= std::numeric_limits<float>::max()),
                "MLA: softmax_scale must be finite in FP32");
    float scale = softmaxScale ? static_cast<float>(*softmaxScale) : 1.0f / std::sqrt(576.0f);
    if (cacheSeqlens) {
        CheckCacheLengths(*cacheSeqlens, q.size(0), q.device());
    }
    if (schedulerMetadata) {
        CheckDenseMetadataTensor(*schedulerMetadata, q.device());
    }
    const c10::OptionalDeviceGuard deviceGuard(device_of(q));
    auto result = DenseForward(q, k, v, qv, scale, cacheSeqlens, schedulerMetadata);
    return {result[0], result[1]};
}
} // namespace flash_attn::mla

TORCH_LIBRARY_FRAGMENT(flash_attn_npu_3, m)
{
    m.def("_mla_dense_metadata(int batch_size, int max_seqlen_q, int max_seqlen_k, int num_heads_q, "
          "ScalarType qkv_dtype, Tensor cache_seqlens) -> Tensor");
    m.def("_mla_dense_forward(Tensor q, Tensor k, Tensor v, Tensor qv, float? softmax_scale=None, "
          "Tensor? cache_seqlens=None, Tensor? scheduler_metadata=None) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(flash_attn_npu_3, CompositeExplicitAutograd, m)
{
    // Validate the sole tensor's device before selecting the NPU generator,
    // including when a direct caller supplies CPU lengths.
    m.impl("_mla_dense_metadata", &flash_attn::mla::DenseMetadataOp);
}

TORCH_LIBRARY_IMPL(flash_attn_npu_3, Autograd, m)
{
    // Lengths and metadata are int32. Let real/fake kernels validate their
    // dtype, including malformed floating tensors that require gradients.
    m.impl("_mla_dense_metadata", torch::CppFunction::makeFallthrough());
}

TORCH_LIBRARY_IMPL(flash_attn_npu_3, PrivateUse1, m)
{
    m.impl("_mla_dense_forward", &flash_attn::mla::DenseForwardOp);
}

TORCH_LIBRARY_IMPL(flash_attn_npu_3, AutogradPrivateUse1, m)
{
    // Preserve FakeTensor dispatch below Autograd. The real/fake kernels both
    // reject requires_grad before allocating or submitting forward-only MLA.
    m.impl("_mla_dense_forward", torch::CppFunction::makeFallthrough());
}
