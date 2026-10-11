#include "mla_forward.h"
#include "mla_launch.h"
#include "mla_schedule.h"
#include "mla_metadata_args.h"
#include "mla_tnd_metadata_args.h"
#include "mla_tnd_bsnd_metadata_args.h"
#include "mla_used_metadata_args.h"
#include "mla_paged_metadata_args.h"
#include "mla_validation.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "runtime/rt_ffts.h"
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <unordered_map>

extern __global__ __aicpu__ uint32_t ComputeMlaDenseMetadata(void* args);
extern __global__ __aicpu__ uint32_t ComputeMlaBsndKvLengthMetadata(void* args);
extern __global__ __aicpu__ uint32_t ComputeMlaTndMetadata(void* args);
extern __global__ __aicpu__ uint32_t ComputeMlaTndBsndMetadataFromPrefix(void* args);
extern __global__ __aicpu__ uint32_t ComputeMlaUsedMetadata(void* args);
extern __global__ __aicpu__ uint32_t ComputeMlaPagedMetadata(void* args);

namespace flash_attn::mla {
namespace {
MlaPlatformInfo DensePlatform(bool fullLaunch = false)
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
    if (fullLaunch) {
        // The number of working cores depends on device lengths. Launch all
        // platform cores without reading the generated record back to host.
        p.blockDims.fill(p.blockDims[p.aicNum - 1]);
    }
    return p;
}

template <typename Args>
struct MetadataSubmission {
    at::Tensor metadata;
    std::optional<at::Tensor> lengths, query, usedQuery, usedKv;
    Args args;
    aclrtStream current, worker;
    aclrtEvent inputReady, metadataDone;
    bool hasKvLengths;

    void Enqueue()
    {
        TORCH_CHECK(aclrtRecordEvent(inputReady, current) == ACL_SUCCESS, "MLA: input event recording failed");
        TORCH_CHECK(aclrtStreamWaitEvent(worker, inputReady) == ACL_SUCCESS, "MLA: input event wait failed");
        if constexpr (std::is_same_v<Args, MlaTndMetadataArgs>) {
            ComputeMlaTndMetadata<<<1, nullptr, worker>>>(&args, sizeof(args));
        } else if constexpr (std::is_same_v<Args, MlaTndBsndMetadataArgs>) {
            ComputeMlaTndBsndMetadataFromPrefix<<<1, nullptr, worker>>>(&args, sizeof(args));
        } else if constexpr (std::is_same_v<Args, MlaUsedMetadataArgs>) {
            ComputeMlaUsedMetadata<<<1, nullptr, worker>>>(&args, sizeof(args));
        } else if constexpr (std::is_same_v<Args, MlaPagedMetadataArgs>) {
            ComputeMlaPagedMetadata<<<1, nullptr, worker>>>(&args, sizeof(args));
        } else if (hasKvLengths) {
            ComputeMlaBsndKvLengthMetadata<<<1, nullptr, worker>>>(&args, sizeof(args));
        } else {
            ComputeMlaDenseMetadata<<<1, nullptr, worker>>>(&args, sizeof(args));
        }
        TORCH_CHECK(aclrtRecordEvent(metadataDone, worker) == ACL_SUCCESS, "MLA: metadata event recording failed");
        TORCH_CHECK(aclrtStreamWaitEvent(current, metadataDone) == ACL_SUCCESS, "MLA: metadata event wait failed");
    }
};

template <typename Args>
MetadataSubmission<Args> PrepareSubmission(const at::Tensor& metadata, const Args& args,
                                           const std::optional<at::Tensor>& lengths,
                                           const std::optional<at::Tensor>& query, bool hasKvLengths,
                                           const std::optional<at::Tensor>& usedQuery = std::nullopt,
                                           const std::optional<at::Tensor>& usedKv = std::nullopt)
{
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
    for (const auto* input : {&lengths, &query, &usedQuery, &usedKv}) {
        if (*input && (*input)->numel() != 0) {
            c10_npu::NPUCachingAllocator::recordStream((*input)->storage().data_ptr(), worker);
        }
    }
    return {metadata,
            lengths,
            query,
            usedQuery,
            usedKv,
            args,
            stream.stream(false),
            worker.stream(false),
            events.inputReady,
            events.metadataDone,
            hasKvLengths};
}

MetadataSubmission<MlaMetadataArgs> PrepareMetadata(const MlaDenseShape& shape, const MlaPlatformInfo& platform,
                                                    at::ScalarType dtype, const at::TensorOptions& options,
                                                    const std::optional<at::Tensor>& lengths, bool hasKvLengths)
{
    const uint32_t words = hasKvLengths ? CheckBsndMetadataWords(shape.batch) : kMlaDenseMetadataWords;
    auto metadata = at::empty({words}, options.dtype(at::kInt));
    MlaMetadataArgs args{reinterpret_cast<uint64_t>(metadata.data_ptr()),
                         words,
                         uint32_t(dtype == at::kBFloat16 ? MlaMetadataDtype::Bf16 : MlaMetadataDtype::Fp16),
                         shape,
                         platform,
                         lengths ? reinterpret_cast<uint64_t>(lengths->data_ptr()) : 0,
                         lengths ? uint64_t(lengths->numel()) : 0};
    return PrepareSubmission(metadata, args, lengths, std::nullopt, hasKvLengths);
}

MetadataSubmission<MlaTndBsndMetadataArgs> PrepareTndBsndMetadata(uint32_t words, int64_t batch, int64_t maxQ,
                                                                  int64_t sk, int64_t heads, at::ScalarType dtype,
                                                                  const MlaPlatformInfo& platform,
                                                                  const at::Tensor& lengths, const at::Tensor& cuQ)
{
    auto metadata = at::empty({words}, lengths.options().dtype(at::kInt));
    MlaTndBsndMetadataArgs args{reinterpret_cast<uint64_t>(metadata.data_ptr()),
                                words,
                                uint32_t(dtype == at::kBFloat16 ? MlaMetadataDtype::Bf16 : MlaMetadataDtype::Fp16),
                                {batch, 0, heads, maxQ, sk},
                                platform,
                                reinterpret_cast<uint64_t>(cuQ.data_ptr()),
                                uint64_t(cuQ.numel()),
                                reinterpret_cast<uint64_t>(lengths.data_ptr()),
                                uint64_t(lengths.numel())};
    return PrepareSubmission(metadata, args, lengths, cuQ, true);
}
} // namespace

at::Tensor DenseMetadata(int64_t batch, int64_t sq, int64_t heads, int64_t sk, at::ScalarType dtype,
                         const at::Tensor& cacheSeqlens)
{
    auto submission = PrepareMetadata({batch, sq, heads, sk}, DensePlatform(batch != 0 && sq != 0 && sk != 0), dtype,
                                      cacheSeqlens.options(), cacheSeqlens, true);
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

at::Tensor TndBsndMetadataOp(int64_t batch, int64_t maxQ, int64_t sk, int64_t heads, at::ScalarType dtype,
                             const at::Tensor& cacheSeqlens, const at::Tensor& cuQ)
{
    const auto words = CheckTndBsndMetadataInputs(batch, maxQ, sk, heads, dtype, cacheSeqlens, cuQ);
    const c10::OptionalDeviceGuard deviceGuard(device_of(cacheSeqlens));
    auto submission = PrepareTndBsndMetadata(words, batch, maxQ, sk, heads, dtype, DensePlatform(), cacheSeqlens, cuQ);
    at_npu::native::OpCommand::RunOpApiV2("ascendc_mla_tnd_bsnd_metadata", [submission]() mutable -> int {
        submission.Enqueue();
        return 0;
    });
    return submission.metadata;
}

at::Tensor PagedMetadataOp(int64_t batch, int64_t maxQ, int64_t capacity, int64_t heads, int64_t pageSize,
                           at::ScalarType dtype, const at::Tensor& cacheSeqlens, const std::optional<at::Tensor>& cuQ)
{
    const auto words = CheckPagedMetadataInputs(batch, maxQ, capacity, heads, pageSize, dtype, cacheSeqlens, cuQ);
    const c10::OptionalDeviceGuard deviceGuard(device_of(cacheSeqlens));
    auto metadata = at::empty({words}, cacheSeqlens.options().dtype(at::kInt));
    const MlaSequenceAllocation query{cuQ ? MlaSequenceLayout::Tnd : MlaSequenceLayout::Bsnd, cuQ ? 0 : batch * maxQ,
                                      maxQ};
    // AICPU derives the TND total from the prefix and reads all length values.
    // No physical pages, cache addresses or strides are part of generation.
    // Static capacity fixes blockDim, including device-only zero lengths, so
    // a future consumer need not read the record to choose its launch size.
    MlaPagedMetadataArgs args{reinterpret_cast<uint64_t>(metadata.data_ptr()),
                              words,
                              uint32_t(dtype == at::kBFloat16 ? MlaMetadataDtype::Bf16 : MlaMetadataDtype::Fp16),
                              {batch, heads, query, capacity, pageSize},
                              DensePlatform(batch != 0 && maxQ != 0 && capacity != 0),
                              {cuQ ? reinterpret_cast<uint64_t>(cuQ->data_ptr()) : 0, cuQ ? uint64_t(cuQ->numel()) : 0},
                              {},
                              {reinterpret_cast<uint64_t>(cacheSeqlens.data_ptr()), uint64_t(cacheSeqlens.numel())}};
    auto submission = PrepareSubmission(metadata, args, cacheSeqlens, cuQ, true);
    at_npu::native::OpCommand::RunOpApiV2("ascendc_mla_paged_metadata", [submission]() mutable -> int {
        submission.Enqueue();
        return 0;
    });
    return metadata;
}

std::vector<at::Tensor> DenseForward(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                     const at::Tensor& qv, float scale, const std::optional<at::Tensor>& cacheSeqlens,
                                     const std::optional<at::Tensor>& schedulerMetadata)
{
    const auto layouts = CheckBsndInputLayouts(q, k, v, qv);
    const bool hasKvLengths = cacheSeqlens.has_value() || schedulerMetadata.has_value();
    if (hasKvLengths) {
        CheckBsndMetadataWords(q.size(0));
    }
    const bool nonempty = q.size(0) != 0 && q.size(1) != 0 && k.size(1) != 0;
    auto p = DensePlatform(hasKvLengths && nonempty);
    MlaSchedule s{};
    auto status = BuildDenseSchedule({q.size(0), q.size(1), q.size(2), k.size(1)}, p, s);
    TORCH_CHECK(status == MlaScheduleStatus::Success, "MLA: dense scheduling failed (status ", int(status), ")");
    TORCH_CHECK(s.cvRatio == 2 && s.scheduleMode == 1 && s.fd.numOfFdHead == 0, "MLA: unexpected dense schedule");
    if (hasKvLengths && nonempty) {
        // CANN CalcNormalWorkspaceSize with the maximum M base (256). Unequal
        // lengths may increase M from the dense candidate. This is only an
        // allocation bound; AICPU supplies and the device validates real tiling.
        const uint64_t rows = std::min(uint64_t(s.s1Size) * s.gSize, uint64_t(256));
        const uint64_t m = (rows + 15) / 16 * 16;
        const uint64_t n = (std::min(s.s2Size, 512U) + 31) / 32 * 32;
        s.workspace.userBytes = uint64_t(2) * p.aicNum * (6 * m * n + 8 * m * 512 + 8 * 256);
        TORCH_CHECK(s.workspace.libApiBytes <= UINT64_MAX - s.workspace.userBytes, "MLA: workspace overflow");
        s.workspace.totalBytes = s.workspace.libApiBytes + s.workspace.userBytes;
    }
    TORCH_CHECK(s.workspace.totalBytes <= uint64_t(std::numeric_limits<int64_t>::max()) &&
                    s.workspace.mm1ResSize <= UINT32_MAX && s.workspace.mm2ResSize <= UINT32_MAX,
                "MLA: workspace or launch field overflow");

    auto out = at::empty({q.size(0), q.size(1), q.size(2), v.size(3)}, q.options());
    auto lse = at::empty({q.size(0), q.size(2), q.size(1)}, q.options().dtype(at::kFloat));
    auto workspace = at::empty({static_cast<int64_t>(s.workspace.totalBytes)}, q.options().dtype(at::kByte));
    std::optional<MetadataSubmission<MlaMetadataArgs>> producer;
    if (!schedulerMetadata) {
        producer = PrepareMetadata({q.size(0), q.size(1), q.size(2), k.size(1)}, p, q.scalar_type(), q.options(),
                                   cacheSeqlens, hasKvLengths);
    }
    auto metadata = schedulerMetadata ? *schedulerMetadata : producer->metadata;
    for (const auto& t : {q, k, v, qv, out, lse, workspace, metadata}) {
        auto address = reinterpret_cast<uintptr_t>(t.data_ptr());
        TORCH_CHECK((t.numel() == 0 || address != 0) &&
                        uint64_t(t.nbytes()) <= std::numeric_limits<uintptr_t>::max() - address,
                    "MLA: tensor address overflow");
    }
    MlaLaunchArgs a{};
    a.layouts = layouts;
    a.hasStrides = !q.is_contiguous() || !k.is_contiguous() || !v.is_contiguous() || !qv.is_contiguous();
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
    a.hasKvLengths = hasKvLengths;
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

namespace {
std::vector<at::Tensor> UsedForward(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& qv,
                                    const MlaUsedShape& shape, const std::optional<at::Tensor>& cuQ,
                                    const std::optional<at::Tensor>& cuK, const std::optional<at::Tensor>& usedQuery,
                                    const std::optional<at::Tensor>& usedKv, float scale)
{
    const uint32_t words = CheckUsedInputs(q, shape.batch, usedQuery, usedKv);
    const int64_t batch = shape.batch, sq = shape.query.maxLength, heads = shape.queryHeads, sk = shape.kv.maxLength;
    const bool tnd = shape.query.layout == MlaSequenceLayout::Tnd;
    const auto layouts = !tnd                                        ? CheckBsndInputLayouts(q, k, v, qv)
                         : shape.kv.layout == MlaSequenceLayout::Tnd ? CheckTndInputLayouts(q, k, v, qv)
                                                                     : CheckTndBsndInputLayouts(q, k, v, qv);
    const bool nonempty = batch != 0 && shape.query.totalTokens != 0 && shape.kv.totalTokens != 0;
    auto p = DensePlatform(nonempty);
    // Static allocation bounds only; source scheduling reads used on AICPU.
    const uint64_t m = (std::min(uint64_t(sq) * heads, uint64_t(256)) + 15) / 16 * 16;
    const uint64_t n = (std::min(uint64_t(sk), uint64_t(512)) + 31) / 32 * 32;
    const uint64_t userBytes = nonempty ? uint64_t(2) * p.aicNum * (6 * m * n + 8 * m * 512 + 8 * 256) : 0;
    const uint64_t libBytes = nonempty ? p.libApiWorkspaceBytes : 0;
    TORCH_CHECK(libBytes <= uint64_t(INT64_MAX) - userBytes, "MLA: workspace overflow");
    auto out = tnd ? at::empty({shape.query.totalTokens, heads, 512}, q.options())
                   : at::empty({batch, sq, heads, 512}, q.options());
    auto lse = tnd ? at::empty({heads, shape.query.totalTokens}, q.options().dtype(at::kFloat))
                   : at::empty({batch, heads, sq}, q.options().dtype(at::kFloat));
    auto workspace = at::empty({static_cast<int64_t>(userBytes + libBytes)}, q.options().dtype(at::kByte));
    auto metadata = at::empty({words}, q.options().dtype(at::kInt));
    const auto checkAddress = [](const at::Tensor& t) {
        const auto address = reinterpret_cast<uintptr_t>(t.data_ptr());
        TORCH_CHECK((t.numel() == 0 || address != 0) && uint64_t(t.nbytes()) <= UINTPTR_MAX - address,
                    "MLA: tensor address overflow");
    };
    for (const auto& t : {q, k, v, qv, out, lse, workspace, metadata}) {
        checkAddress(t);
    }
    for (const auto* input : {&cuQ, &cuK, &usedQuery, &usedKv}) {
        if (*input) {
            checkAddress(**input);
        }
    }
    const uint32_t dtype = uint32_t(q.scalar_type() == at::kBFloat16 ? MlaMetadataDtype::Bf16 : MlaMetadataDtype::Fp16);
    const auto descriptor = [&](const std::optional<at::Tensor>& t) {
        // Empty torch tensors may have null data_ptr. A nonnull zero-extent
        // descriptor preserves explicit presence for B=0; it is never read
        // and has no overlap with the metadata output's nonempty extent.
        return t ? MlaUsedMetadataArray{reinterpret_cast<uint64_t>(t->numel() == 0 ? metadata.data_ptr()
                                                                                   : t->data_ptr()),
                                        uint64_t(t->numel())}
                 : MlaUsedMetadataArray{};
    };
    MlaUsedMetadataArgs args{reinterpret_cast<uint64_t>(metadata.data_ptr()),
                             words,
                             dtype,
                             shape,
                             p,
                             descriptor(cuQ),
                             descriptor(cuK),
                             descriptor(usedQuery),
                             descriptor(usedKv)};
    auto producer = PrepareSubmission(metadata, args, cuK, cuQ, true, usedQuery, usedKv);
    auto stream = c10_npu::getCurrentNPUStream();
    for (const auto& t : {q, k, v, qv, metadata}) {
        if (t.numel() != 0) {
            c10_npu::NPUCachingAllocator::recordStream(t.storage().data_ptr(), stream);
        }
    }
    for (const auto* input : {&cuQ, &cuK, &usedQuery, &usedKv}) {
        if (*input && (*input)->numel() != 0) {
            c10_npu::NPUCachingAllocator::recordStream((*input)->storage().data_ptr(), stream);
        }
    }
    MlaUsedLaunchArgs a{};
    a.layouts = layouts;
    a.hasStrides = !q.is_contiguous() || !k.is_contiguous() || !v.is_contiguous() || !qv.is_contiguous();
    uint32_t fftsLength = 0;
    TORCH_CHECK(rtGetC2cCtrlAddr(&a.ffts, &fftsLength) == RT_ERROR_NONE,
                "MLA: failed to obtain cross-core synchronization address");
    a.stream = stream.stream(false);
    a.qc = static_cast<uint8_t*>(qv.data_ptr());
    a.qr = static_cast<uint8_t*>(q.data_ptr());
    a.kv = static_cast<uint8_t*>(v.data_ptr());
    a.kr = static_cast<uint8_t*>(k.data_ptr());
    a.output = static_cast<uint8_t*>(out.data_ptr());
    a.lse = static_cast<uint8_t*>(lse.data_ptr());
    a.workspace = workspace.numel() == 0 ? nullptr : static_cast<uint8_t*>(workspace.data_ptr()) + libBytes;
    a.metadata = static_cast<uint8_t*>(metadata.data_ptr());
    const auto allocation = [](const MlaSequenceAllocation& axis) {
        const auto total = uint64_t(axis.totalTokens);
        return MlaMetadataAllocation{
            uint32_t(axis.layout), {uint32_t(total), uint32_t(total >> 32)}, uint32_t(axis.maxLength)};
    };
    a.context = {dtype,
                 uint32_t(batch),
                 uint32_t(heads),
                 (usedQuery ? kMlaUsedQueryPresent : 0) | (usedKv ? kMlaUsedKvPresent : 0),
                 allocation(args.shape.query),
                 allocation(args.shape.kv),
                 p.aicNum,
                 p.aivNum,
                 p.blockDims[nonempty ? p.aicNum - 1 : 0],
                 userBytes,
                 libBytes,
                 p.localMemory};
    a.inputs = {args.cuQuery, args.cuKv, args.usedQuery, args.usedKv};
    a.scale = scale;
    at_npu::native::OpCommand::RunOpApiV2(
        tnd ? "ascendc_mla_tnd_used" : "ascendc_mla_bsnd_used",
        [a, producer, q, k, v, qv, out, lse, workspace, tnd]() mutable -> int {
            producer.Enqueue();
            auto error = tnd ? (q.scalar_type() == at::kBFloat16 ? LaunchTndUsedBf16(a) : LaunchTndUsedFp16(a))
                             : (q.scalar_type() == at::kBFloat16 ? LaunchBsndUsedBf16(a) : LaunchBsndUsedFp16(a));
            TORCH_CHECK(error == ACL_SUCCESS, "MLA: used BATCH_MODE kernel launch failed (", error, ")");
            return 0;
        });
    return {out, lse, at::Tensor(), at::Tensor()};
}
} // namespace

std::vector<at::Tensor> BsndUsedForward(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                        const at::Tensor& qv, const at::Tensor& usedQuery,
                                        const std::optional<at::Tensor>& usedKv, float scale)
{
    const int64_t batch = q.size(0), sq = q.size(1), heads = q.size(2), sk = k.size(1);
    return UsedForward(
        q, k, v, qv,
        {batch, heads, {MlaSequenceLayout::Bsnd, batch * sq, sq}, {MlaSequenceLayout::Bsnd, batch * sk, sk}},
        std::nullopt, std::nullopt, usedQuery, usedKv, scale);
}

std::vector<at::Tensor> TndForward(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& qv,
                                   const at::Tensor& cuQ, const at::Tensor& cuK, int64_t maxQ, int64_t maxK,
                                   float scale, const std::optional<at::Tensor>& usedQuery,
                                   const std::optional<at::Tensor>& usedKv)
{
    const uint32_t words = CheckTndInputs(q, k, cuQ, cuK, maxQ, maxK);
    const int64_t batch = cuQ.numel() - 1;
    if (usedQuery || usedKv) {
        return UsedForward(
            q, k, v, qv,
            {batch, q.size(1), {MlaSequenceLayout::Tnd, q.size(0), maxQ}, {MlaSequenceLayout::Tnd, k.size(0), maxK}},
            cuQ, cuK, usedQuery, usedKv, scale);
    }
    const auto layouts = CheckTndInputLayouts(q, k, v, qv);
    const bool nonempty = batch != 0 && q.size(0) != 0 && k.size(0) != 0;
    auto p = DensePlatform(nonempty);
    // Allocation/launch bounds use static capacities, never a host prefix read.
    // The generator supplies observed maxima and the complete source schedule.
    const uint64_t rows = std::min(uint64_t(maxQ) * q.size(1), uint64_t(256));
    const uint64_t m = (rows + 15) / 16 * 16;
    const uint64_t n = (std::min(uint64_t(maxK), uint64_t(512)) + 31) / 32 * 32;
    const uint64_t userBytes = nonempty ? uint64_t(2) * p.aicNum * (6 * m * n + 8 * m * 512 + 8 * 256) : 0;
    const uint64_t libBytes = nonempty ? p.libApiWorkspaceBytes : 0;
    TORCH_CHECK(libBytes <= uint64_t(INT64_MAX) - userBytes, "MLA: workspace overflow");
    auto out = at::empty({q.size(0), q.size(1), 512}, q.options());
    auto lse = at::empty({q.size(1), q.size(0)}, q.options().dtype(at::kFloat));
    auto workspace = at::empty({static_cast<int64_t>(userBytes + libBytes)}, q.options().dtype(at::kByte));
    auto metadata = at::empty({words}, q.options().dtype(at::kInt));
    for (const auto& t : {q, k, v, qv, cuQ, cuK, out, lse, workspace, metadata}) {
        const auto address = reinterpret_cast<uintptr_t>(t.data_ptr());
        TORCH_CHECK((t.numel() == 0 || address != 0) && uint64_t(t.nbytes()) <= UINTPTR_MAX - address,
                    "MLA: tensor address overflow");
    }
    const uint32_t dtype = uint32_t(q.scalar_type() == at::kBFloat16 ? MlaMetadataDtype::Bf16 : MlaMetadataDtype::Fp16);
    MlaTndMetadataArgs args{reinterpret_cast<uint64_t>(metadata.data_ptr()),
                            words,
                            dtype,
                            {batch, q.size(0), k.size(0), q.size(1), maxQ, maxK},
                            p,
                            reinterpret_cast<uint64_t>(cuQ.data_ptr()),
                            uint64_t(cuQ.numel()),
                            reinterpret_cast<uint64_t>(cuK.data_ptr()),
                            uint64_t(cuK.numel())};
    auto producer = PrepareSubmission(metadata, args, cuK, cuQ, true);
    auto stream = c10_npu::getCurrentNPUStream();
    for (const auto& t : {q, k, v, qv, cuQ, cuK, metadata}) {
        if (t.numel() != 0) {
            c10_npu::NPUCachingAllocator::recordStream(t.storage().data_ptr(), stream);
        }
    }
    MlaTndLaunchArgs a{};
    a.layouts = layouts;
    a.hasStrides = !q.is_contiguous() || !k.is_contiguous() || !v.is_contiguous() || !qv.is_contiguous();
    uint32_t fftsLength = 0;
    TORCH_CHECK(rtGetC2cCtrlAddr(&a.ffts, &fftsLength) == RT_ERROR_NONE,
                "MLA: failed to obtain cross-core synchronization address");
    a.stream = stream.stream(false);
    a.qc = static_cast<uint8_t*>(qv.data_ptr());
    a.qr = static_cast<uint8_t*>(q.data_ptr());
    a.kv = static_cast<uint8_t*>(v.data_ptr());
    a.kr = static_cast<uint8_t*>(k.data_ptr());
    a.output = static_cast<uint8_t*>(out.data_ptr());
    a.lse = static_cast<uint8_t*>(lse.data_ptr());
    a.workspace = workspace.numel() == 0 ? nullptr : static_cast<uint8_t*>(workspace.data_ptr()) + libBytes;
    a.metadata = static_cast<uint8_t*>(metadata.data_ptr());
    a.cuQuery = static_cast<uint8_t*>(cuQ.data_ptr());
    a.cuKv = static_cast<uint8_t*>(cuK.data_ptr());
    a.context = {dtype,
                 uint32_t(batch),
                 uint32_t(q.size(0)),
                 uint32_t(k.size(0)),
                 uint32_t(q.size(1)),
                 uint32_t(maxQ),
                 uint32_t(maxK),
                 p.aicNum,
                 p.aivNum,
                 p.blockDims[nonempty ? p.aicNum - 1 : 0],
                 userBytes,
                 libBytes,
                 p.localMemory};
    a.scale = scale;
    at_npu::native::OpCommand::RunOpApiV2(
        "ascendc_mla_tnd", [a, producer, q, k, v, qv, cuQ, cuK, out, lse, workspace]() mutable -> int {
            producer.Enqueue();
            auto error = q.scalar_type() == at::kBFloat16 ? LaunchTndBf16(a) : LaunchTndFp16(a);
            TORCH_CHECK(error == ACL_SUCCESS, "MLA: TND BATCH_MODE kernel launch failed (", error, ")");
            return 0;
        });
    return {out, lse, at::Tensor(), at::Tensor()};
}

std::vector<at::Tensor> TndBsndUsedForward(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                           const at::Tensor& qv, const at::Tensor& cuQ, int64_t maxQ,
                                           const at::Tensor& usedQuery, const std::optional<at::Tensor>& usedKv,
                                           float scale)
{
    CheckTndBsndForwardInputs(q, k, cuQ, maxQ, usedKv, std::nullopt);
    const int64_t batch = k.size(0), sk = k.size(1);
    // Q prefixes describe allocation; cache slots remain b*Sk regardless of
    // either used length. The TND used launcher reads both v5 physical starts.
    return UsedForward(
        q, k, v, qv,
        {batch, q.size(1), {MlaSequenceLayout::Tnd, q.size(0), maxQ}, {MlaSequenceLayout::Bsnd, batch * sk, sk}}, cuQ,
        std::nullopt, usedQuery, usedKv, scale);
}

std::vector<at::Tensor> TndBsndForward(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                       const at::Tensor& qv, const at::Tensor& cuQ, int64_t maxQ, float scale,
                                       const std::optional<at::Tensor>& cacheSeqlens,
                                       const std::optional<at::Tensor>& schedulerMetadata)
{
    const auto words = CheckTndBsndForwardInputs(q, k, cuQ, maxQ, cacheSeqlens, schedulerMetadata);
    const auto layouts = CheckTndBsndInputLayouts(q, k, v, qv);
    const int64_t batch = k.size(0), sk = k.size(1);
    const bool nonempty = batch != 0 && q.size(0) != 0 && sk != 0;
    auto p = DensePlatform();
    // Capacity bounds only. Q/KV values and the source schedule stay on device.
    const uint64_t rows = std::min(uint64_t(maxQ) * q.size(1), uint64_t(256));
    const uint64_t m = (rows + 15) / 16 * 16;
    const uint64_t n = (std::min(uint64_t(sk), uint64_t(512)) + 31) / 32 * 32;
    const uint64_t userBytes = nonempty ? uint64_t(2) * p.aicNum * (6 * m * n + 8 * m * 512 + 8 * 256) : 0;
    const uint64_t libBytes = nonempty ? p.libApiWorkspaceBytes : 0;
    TORCH_CHECK(libBytes <= uint64_t(INT64_MAX) - userBytes, "MLA: workspace overflow");
    auto out = at::empty({q.size(0), q.size(1), 512}, q.options());
    auto lse = at::empty({q.size(1), q.size(0)}, q.options().dtype(at::kFloat));
    auto workspace = at::empty({static_cast<int64_t>(userBytes + libBytes)}, q.options().dtype(at::kByte));
    // v4 always validates a concrete current length vector. None means full
    // physical capacity, including when a caller supplies an explicit record.
    auto lengths = cacheSeqlens ? *cacheSeqlens : at::full({batch}, sk, q.options().dtype(at::kInt));
    std::optional<MetadataSubmission<MlaTndBsndMetadataArgs>> producer;
    if (!schedulerMetadata) {
        producer = PrepareTndBsndMetadata(words, batch, maxQ, sk, q.size(1), q.scalar_type(), p, lengths, cuQ);
    }
    auto metadata = schedulerMetadata ? *schedulerMetadata : producer->metadata;
    for (const auto& t : {q, k, v, qv, cuQ, lengths, out, lse, workspace, metadata}) {
        const auto address = reinterpret_cast<uintptr_t>(t.data_ptr());
        TORCH_CHECK((t.numel() == 0 || address != 0) && uint64_t(t.nbytes()) <= UINTPTR_MAX - address,
                    "MLA: tensor address overflow");
    }
    auto stream = c10_npu::getCurrentNPUStream();
    for (const auto& t : {q, k, v, qv, cuQ, lengths, metadata}) {
        if (t.numel() != 0) {
            c10_npu::NPUCachingAllocator::recordStream(t.storage().data_ptr(), stream);
        }
    }
    MlaTndBsndLaunchArgs a{};
    a.layouts = layouts;
    a.hasStrides = !q.is_contiguous() || !k.is_contiguous() || !v.is_contiguous() || !qv.is_contiguous();
    uint32_t fftsLength = 0;
    TORCH_CHECK(rtGetC2cCtrlAddr(&a.ffts, &fftsLength) == RT_ERROR_NONE,
                "MLA: failed to obtain cross-core synchronization address");
    a.stream = stream.stream(false);
    a.qc = static_cast<uint8_t*>(qv.data_ptr());
    a.qr = static_cast<uint8_t*>(q.data_ptr());
    a.kv = static_cast<uint8_t*>(v.data_ptr());
    a.kr = static_cast<uint8_t*>(k.data_ptr());
    a.output = static_cast<uint8_t*>(out.data_ptr());
    a.lse = static_cast<uint8_t*>(lse.data_ptr());
    a.workspace = workspace.numel() == 0 ? nullptr : static_cast<uint8_t*>(workspace.data_ptr()) + libBytes;
    a.metadata = static_cast<uint8_t*>(metadata.data_ptr());
    a.cuQuery = static_cast<uint8_t*>(cuQ.data_ptr());
    a.cacheSeqlens = static_cast<uint8_t*>(lengths.data_ptr());
    a.context = {uint32_t(q.scalar_type() == at::kBFloat16 ? MlaMetadataDtype::Bf16 : MlaMetadataDtype::Fp16),
                 uint32_t(batch),
                 uint32_t(q.size(0)),
                 uint32_t(q.size(1)),
                 uint32_t(maxQ),
                 uint32_t(sk),
                 p.aicNum,
                 p.aivNum,
                 p.blockDims[nonempty ? p.aicNum - 1 : 0],
                 userBytes,
                 libBytes,
                 p.localMemory};
    a.scale = scale;
    at_npu::native::OpCommand::RunOpApiV2(
        "ascendc_mla_tnd_bsnd",
        [a, producer, q, k, v, qv, cuQ, lengths, out, lse, workspace, metadata]() mutable -> int {
            if (producer) {
                producer->Enqueue();
            }
            auto error = q.scalar_type() == at::kBFloat16 ? LaunchTndBsndBf16(a) : LaunchTndBsndFp16(a);
            TORCH_CHECK(error == ACL_SUCCESS, "MLA: mixed BATCH_MODE kernel launch failed (", error, ")");
            return 0;
        });
    return {out, lse, at::Tensor(), at::Tensor()};
}

std::tuple<at::Tensor, at::Tensor> TndBsndForwardOp(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                                    const at::Tensor& qv, const at::Tensor& cuQ, int64_t maxQ,
                                                    std::optional<double> softmaxScale,
                                                    const std::optional<at::Tensor>& cacheSeqlens,
                                                    const std::optional<at::Tensor>& schedulerMetadata)
{
    CheckForwardBoundary(q, k, v, qv, true, false, false);
    TORCH_CHECK(!softmaxScale ||
                    (std::isfinite(*softmaxScale) && std::abs(*softmaxScale) <= std::numeric_limits<float>::max()),
                "MLA: softmax_scale must be finite in FP32");
    const c10::OptionalDeviceGuard deviceGuard(device_of(q));
    auto result = TndBsndForward(q, k, v, qv, cuQ, maxQ, softmaxScale ? float(*softmaxScale) : 1.0f / std::sqrt(576.0f),
                                 cacheSeqlens, schedulerMetadata);
    return {result[0], result[1]};
}

std::tuple<at::Tensor, at::Tensor> TndForwardOp(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                                const at::Tensor& qv, const at::Tensor& cuQ, const at::Tensor& cuK,
                                                int64_t maxQ, int64_t maxK, std::optional<double> softmaxScale,
                                                const std::optional<at::Tensor>& usedQuery,
                                                const std::optional<at::Tensor>& usedKv)
{
    CheckForwardBoundary(q, k, v, qv, true, true, false);
    TORCH_CHECK(!softmaxScale ||
                    (std::isfinite(*softmaxScale) && std::abs(*softmaxScale) <= std::numeric_limits<float>::max()),
                "MLA: softmax_scale must be finite in FP32");
    const c10::OptionalDeviceGuard deviceGuard(device_of(q));
    auto result = TndForward(q, k, v, qv, cuQ, cuK, maxQ, maxK,
                             softmaxScale ? float(*softmaxScale) : 1.0f / std::sqrt(576.0f), usedQuery, usedKv);
    return {result[0], result[1]};
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
        CheckDenseMetadataTensor(*schedulerMetadata, q.device(), q.size(0));
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
    m.def("_mla_tnd_bsnd_metadata(int batch_size, int max_seqlen_q, int max_seqlen_k, int num_heads_q, "
          "ScalarType qkv_dtype, Tensor cache_seqlens, Tensor cu_seqlens_q) -> Tensor");
    m.def("_mla_paged_metadata(int batch_size, int max_seqlen_q, int max_seqlen_k, int num_heads_q, int page_size, "
          "ScalarType qkv_dtype, Tensor cache_seqlens, Tensor? cu_seqlens_q) -> Tensor");
    m.def("_mla_dense_forward(Tensor q, Tensor k, Tensor v, Tensor qv, float? softmax_scale=None, "
          "Tensor? cache_seqlens=None, Tensor? scheduler_metadata=None) -> (Tensor, Tensor)");
    m.def("_mla_tnd_forward(Tensor q, Tensor k, Tensor v, Tensor qv, Tensor cu_seqlens_q, Tensor cu_seqlens_k, "
          "int max_seqlen_q, int max_seqlen_k, float? softmax_scale=None, Tensor? seqused_q=None, "
          "Tensor? seqused_k=None) -> (Tensor, Tensor)");
    m.def(
        "_mla_tnd_bsnd_forward(Tensor q, Tensor k, Tensor v, Tensor qv, Tensor cu_seqlens_q, int max_seqlen_q, "
        "float? softmax_scale=None, Tensor? cache_seqlens=None, Tensor? scheduler_metadata=None) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(flash_attn_npu_3, CompositeExplicitAutograd, m)
{
    // Validate the sole tensor's device before selecting the NPU generator,
    // including when a direct caller supplies CPU lengths.
    m.impl("_mla_dense_metadata", &flash_attn::mla::DenseMetadataOp);
    m.impl("_mla_tnd_bsnd_metadata", &flash_attn::mla::TndBsndMetadataOp);
    m.impl("_mla_paged_metadata", &flash_attn::mla::PagedMetadataOp);
}

TORCH_LIBRARY_IMPL(flash_attn_npu_3, Autograd, m)
{
    // Lengths and metadata are int32. Let real/fake kernels validate their
    // dtype, including malformed floating tensors that require gradients.
    m.impl("_mla_dense_metadata", torch::CppFunction::makeFallthrough());
    m.impl("_mla_tnd_bsnd_metadata", torch::CppFunction::makeFallthrough());
    m.impl("_mla_paged_metadata", torch::CppFunction::makeFallthrough());
}

TORCH_LIBRARY_IMPL(flash_attn_npu_3, PrivateUse1, m)
{
    m.impl("_mla_dense_forward", &flash_attn::mla::DenseForwardOp);
    m.impl("_mla_tnd_forward", &flash_attn::mla::TndForwardOp);
    m.impl("_mla_tnd_bsnd_forward", &flash_attn::mla::TndBsndForwardOp);
}

TORCH_LIBRARY_IMPL(flash_attn_npu_3, AutogradPrivateUse1, m)
{
    // Preserve FakeTensor dispatch below Autograd. The real/fake kernels both
    // reject requires_grad before allocating or submitting forward-only MLA.
    m.impl("_mla_dense_forward", torch::CppFunction::makeFallthrough());
    m.impl("_mla_tnd_forward", torch::CppFunction::makeFallthrough());
    m.impl("_mla_tnd_bsnd_forward", torch::CppFunction::makeFallthrough());
}
