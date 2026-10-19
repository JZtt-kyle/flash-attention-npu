#include "mla_forward.h"
#include "mla_int8_launch.h"
#include "mla_gqa_launch.h"
#include "mla_gqa_schedule.h"
#include "mla_launch.h"
#include "mla_prefill_bsnd_launch.h"
#include "mla_schedule.h"
#include "mla_metadata_args.h"
#include "mla_tnd_metadata_args.h"
#include "mla_tnd_bsnd_metadata_args.h"
#include "mla_used_metadata_args.h"
#include "mla_causal_metadata.h"
#include "mla_window_metadata.h"
#include "mla_split_bsnd_metadata.h"
#include "mla_split_bsnd_launch.h"
#include "mla_split_tnd_launch.h"
#include "mla_split_tnd_plan.h"
#include "mla_split_mixed_plan.h"
#include "mla_split_mixed_launch.h"
#include "mla_split_kv_plan.h"
#include "mla_auto_split_kv_plan.h"
#include "mla_auto_split_bsnd_launch.h"
#include "mla_auto_split_tnd_launch.h"
#include "mla_auto_split_tnd_plan.h"
#include "mla_auto_split_mixed_plan.h"
#include "mla_auto_split_mixed_launch.h"
#include "mla_window_paged_metadata.h"
#include "mla_paged_metadata_args.h"
#include "mla_paged_launch.h"
#include "mla_paged_plan.h"
#include "mla_split_paged_bsnd_plan.h"
#include "mla_split_paged_bsnd_launch.h"
#include "mla_split_paged_tnd_launch.h"
#include "mla_split_paged_tnd_plan.h"
#include "mla_auto_split_paged_bsnd_launch.h"
#include "mla_auto_split_paged_bsnd_plan.h"
#include "mla_auto_split_paged_tnd_launch.h"
#include "mla_auto_split_paged_tnd_plan.h"
#include "mla_validation.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUGraphsUtils.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "runtime/rt_ffts.h"
#include <c10/core/DeviceGuard.h>
#include <torch/library.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <unordered_map>

extern __global__ __aicpu__ uint32_t ComputeMlaDenseMetadata(void* args);
extern __global__ __aicpu__ uint32_t ComputeMlaInt8Metadata(void* args);
extern __global__ __aicpu__ uint32_t ComputeMlaGqaMetadata(void* args);
extern __global__ __aicpu__ uint32_t ComputeMlaPrefillBsndMetadata(void* args);
extern __global__ __aicpu__ uint32_t ComputeMlaBsndKvLengthMetadata(void* args);
extern __global__ __aicpu__ uint32_t ComputeMlaTndMetadata(void* args);
extern __global__ __aicpu__ uint32_t ComputeMlaTndBsndMetadataFromPrefix(void* args);
extern __global__ __aicpu__ uint32_t ComputeMlaUsedMetadata(void* args);
extern __global__ __aicpu__ uint32_t ComputeMlaSplitBsndMetadata(void* args);
extern __global__ __aicpu__ uint32_t ComputeMlaSplitTndMetadata(void* args);
extern __global__ __aicpu__ uint32_t ComputeMlaSplitMixedMetadata(void* args);
extern __global__ __aicpu__ uint32_t ComputeMlaAutoSplitBsndMetadata(void* args);
extern __global__ __aicpu__ uint32_t ComputeMlaAutoSplitTndMetadata(void* args);
extern __global__ __aicpu__ uint32_t ComputeMlaAutoSplitMixedMetadata(void* args);
extern __global__ __aicpu__ uint32_t ComputeMlaCausalBsndMetadata(void* args);
extern __global__ __aicpu__ uint32_t ComputeMlaWindowBsndMetadata(void* args);
extern __global__ __aicpu__ uint32_t ComputeMlaWindowTndMetadata(void* args);
extern __global__ __aicpu__ uint32_t ComputeMlaWindowMixedMetadata(void* args);
extern __global__ __aicpu__ uint32_t ComputeMlaWindowMixedMetadataFromPrefix(void* args);
extern __global__ __aicpu__ uint32_t ComputeMlaCausalTndMetadata(void* args);
extern __global__ __aicpu__ uint32_t ComputeMlaCausalMixedMetadata(void* args);
extern __global__ __aicpu__ uint32_t ComputeMlaCausalMixedMetadataFromPrefix(void* args);
extern __global__ __aicpu__ uint32_t ComputeMlaPagedMetadata(void* args);
extern __global__ __aicpu__ uint32_t ComputeMlaSplitPagedBsndMetadata(void* args);
extern __global__ __aicpu__ uint32_t ComputeMlaSplitPagedTndMetadata(void* args);
extern __global__ __aicpu__ uint32_t ComputeMlaAutoSplitPagedBsndMetadata(void* args);
extern __global__ __aicpu__ uint32_t ComputeMlaAutoSplitPagedTndMetadata(void* args);
extern __global__ __aicpu__ uint32_t ComputeMlaWindowPagedBsndMetadata(void* args);
extern __global__ __aicpu__ uint32_t ComputeMlaWindowPagedTndMetadata(void* args);
extern __global__ __aicpu__ uint32_t ComputeMlaCausalPagedBsndMetadata(void* args);
extern __global__ __aicpu__ uint32_t ComputeMlaCausalPagedTndMetadata(void* args);

namespace flash_attn::mla {
namespace {
void RunCausalLaunch(const char* name, const std::function<int()>& launch)
{
    if (std::getenv("ASCEND_LAUNCH_BLOCKING") || std::getenv("TORCH_NPU_SANITIZER")) {
        at_npu::native::OpCommand::RunOpApiV2(name, launch);
        return;
    }
    // Drain pending host submissions (including length updates) before using
    // the ASC launcher directly. stream() does not wait for device completion;
    // subsequent kernels and allocator reuse remain ordered on this stream.
    TORCH_CHECK(c10_npu::getCurrentNPUStream().stream() != nullptr, "MLA: host submission queue failed");
    RECORD_FUNCTION(name, std::vector<c10::IValue>({}));
    TORCH_CHECK(launch() == ACL_SUCCESS, "MLA: causal kernel submission failed");
}

MlaPlatformInfo DensePlatform(bool fullLaunch = false)
{
    const char* soc = aclrtGetSocName();
    // CANN names the 910C family Ascend910_93; both families use the 2201
    // kernels. Keep resource validation below for each actual device/partition.
    TORCH_CHECK(soc && (std::strncmp(soc, "Ascend910B", 10) == 0 || std::strncmp(soc, "Ascend910_93", 12) == 0),
                "MLA forward requires an Ascend 910B/C device");
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

MlaPlatformInfo CausalPlatform(c10::DeviceIndex device, bool fullLaunch)
{
    // These chip resources are immutable. Keep each host thread/device's
    // validated descriptor separate; no tensor, length or schedule is cached.
    static thread_local std::unordered_map<c10::DeviceIndex, MlaPlatformInfo> platforms;
    auto found = platforms.find(device);
    if (found == platforms.end()) {
        found = platforms.emplace(device, DensePlatform()).first;
    }
    auto p = found->second;
    if (fullLaunch) {
        p.blockDims.fill(p.blockDims[p.aicNum - 1]);
    }
    return p;
}

std::shared_ptr<const MlaCausalFullSchedule>
CausalFullSchedule(const MlaUsedShape& shape, const MlaPlatformInfo& platform, c10::DeviceIndex device, uint32_t dtype)
{
    using Key = std::array<int64_t, 8>;
    const Key key{device,           dtype,           shape.batch,    shape.query.maxLength, shape.kv.maxLength,
                  shape.queryHeads, platform.aicNum, platform.aivNum};
    struct Entry {
        Key key;
        std::shared_ptr<const MlaCausalFullSchedule> value;
    };
    static thread_local std::vector<Entry> entries;
    static thread_local size_t next = 0;
    for (const auto& entry : entries) {
        if (entry.key == key) {
            return entry.value;
        }
    }
    auto candidate = std::make_shared<MlaCausalFullSchedule>();
    if (!BuildCausalFullSchedule(shape, platform, *candidate)) {
        candidate.reset();
    }
    // Only immutable host data is retained. Captures own the candidate even
    // after eviction; the AICPU receives a value copy and checks actual lengths.
    constexpr size_t capacity = 16;
    if (entries.size() < capacity) {
        entries.push_back({key, candidate});
    } else {
        entries[next] = {key, candidate};
        next = (next + 1) % capacity;
    }
    return candidate;
}

template <typename Args>
struct MetadataSubmission {
    at::Tensor metadata;
    std::optional<at::Tensor> lengths, query, usedQuery, usedKv;
    Args args;
    aclrtStream current, worker;
    aclrtEvent inputReady, metadataDone;
    bool hasKvLengths;
    bool causal = false;
    std::shared_ptr<const MlaCausalFullSchedule> fullSchedule;
    bool queryTotalFromPrefix = false;
    std::optional<MlaWindowConfig> window;
    int64_t splits = 1;
    bool hasSoftcap = false;

    void Enqueue()
    {
        if constexpr (std::is_same_v<Args, MlaUsedMetadataArgs>) {
            if (splits == 0) {
                if (args.shape.query.layout == MlaSequenceLayout::Tnd &&
                    args.shape.kv.layout == MlaSequenceLayout::Bsnd) {
                    MlaAutoSplitMixedMetadataArgs autoArgs{args, *window, queryTotalFromPrefix, uint32_t(hasSoftcap)};
                    ComputeMlaAutoSplitMixedMetadata<<<1, nullptr, current>>>(&autoArgs, sizeof(autoArgs));
                } else if (args.shape.query.layout == MlaSequenceLayout::Tnd) {
                    MlaAutoSplitTndMetadataArgs autoArgs{args, *window, uint32_t(hasSoftcap)};
                    ComputeMlaAutoSplitTndMetadata<<<1, nullptr, current>>>(&autoArgs, sizeof(autoArgs));
                } else {
                    MlaAutoSplitBsndMetadataArgs autoArgs{args, *window, uint32_t(hasSoftcap)};
                    ComputeMlaAutoSplitBsndMetadata<<<1, nullptr, current>>>(&autoArgs, sizeof(autoArgs));
                }
                return;
            }
            if (splits > 1 || hasSoftcap) {
                if (args.shape.query.layout == MlaSequenceLayout::Tnd &&
                    args.shape.kv.layout == MlaSequenceLayout::Bsnd) {
                    MlaSplitMixedMetadataArgs splitArgs{args, *window, splits, queryTotalFromPrefix,
                                                        uint32_t(hasSoftcap)};
                    ComputeMlaSplitMixedMetadata<<<1, nullptr, current>>>(&splitArgs, sizeof(splitArgs));
                } else if (args.shape.query.layout == MlaSequenceLayout::Tnd) {
                    MlaSplitTndMetadataArgs splitArgs{args, *window, splits, uint32_t(hasSoftcap)};
                    ComputeMlaSplitTndMetadata<<<1, nullptr, current>>>(&splitArgs, sizeof(splitArgs));
                } else {
                    MlaSplitBsndMetadataArgs splitArgs{args, *window, splits, uint32_t(hasSoftcap)};
                    ComputeMlaSplitBsndMetadata<<<1, nullptr, current>>>(&splitArgs, sizeof(splitArgs));
                }
                return;
            }
            if (window) {
                MlaWindowMetadataArgs windowArgs{args, *window};
                if (args.shape.query.layout == MlaSequenceLayout::Tnd) {
                    if (args.shape.kv.layout == MlaSequenceLayout::Bsnd) {
                        if (queryTotalFromPrefix) {
                            ComputeMlaWindowMixedMetadataFromPrefix<<<1, nullptr, current>>>(&windowArgs,
                                                                                             sizeof(windowArgs));
                        } else {
                            ComputeMlaWindowMixedMetadata<<<1, nullptr, current>>>(&windowArgs, sizeof(windowArgs));
                        }
                    } else {
                        ComputeMlaWindowTndMetadata<<<1, nullptr, current>>>(&windowArgs, sizeof(windowArgs));
                    }
                } else {
                    ComputeMlaWindowBsndMetadata<<<1, nullptr, current>>>(&windowArgs, sizeof(windowArgs));
                }
                return;
            }
            if (causal) {
                // v7/v8/v9 is followed by its consumer on the caller's stream.
                // The producer's DataStoreBarrier completes before that task.
                if (args.shape.query.layout == MlaSequenceLayout::Tnd) {
                    if (args.shape.kv.layout == MlaSequenceLayout::Bsnd) {
                        if (queryTotalFromPrefix) {
                            ComputeMlaCausalMixedMetadataFromPrefix<<<1, nullptr, current>>>(&args, sizeof(args));
                        } else {
                            ComputeMlaCausalMixedMetadata<<<1, nullptr, current>>>(&args, sizeof(args));
                        }
                    } else {
                        ComputeMlaCausalTndMetadata<<<1, nullptr, current>>>(&args, sizeof(args));
                    }
                } else {
                    MlaCausalMetadataArgs causalArgs{args, fullSchedule ? *fullSchedule : MlaCausalFullSchedule{}};
                    ComputeMlaCausalBsndMetadata<<<1, nullptr, current>>>(&causalArgs, sizeof(causalArgs));
                }
                return;
            }
        }
        if constexpr (std::is_same_v<Args, MlaPagedMetadataArgs>) {
            if (splits == 0) {
                if (args.shape.query.layout == MlaSequenceLayout::Tnd) {
                    MlaAutoSplitPagedTndMetadataArgs autoArgs{args, *window, queryTotalFromPrefix,
                                                              uint32_t(hasSoftcap)};
                    ComputeMlaAutoSplitPagedTndMetadata<<<1, nullptr, current>>>(&autoArgs, sizeof(autoArgs));
                } else {
                    MlaAutoSplitPagedBsndMetadataArgs autoArgs{args, *window, uint32_t(hasSoftcap)};
                    ComputeMlaAutoSplitPagedBsndMetadata<<<1, nullptr, current>>>(&autoArgs, sizeof(autoArgs));
                }
                return;
            }
            if (splits > 1 || hasSoftcap) {
                if (args.shape.query.layout == MlaSequenceLayout::Tnd) {
                    MlaSplitPagedTndMetadataArgs splitArgs{args, *window, splits, queryTotalFromPrefix,
                                                           uint32_t(hasSoftcap)};
                    ComputeMlaSplitPagedTndMetadata<<<1, nullptr, current>>>(&splitArgs, sizeof(splitArgs));
                } else {
                    MlaSplitPagedBsndMetadataArgs splitArgs{args, *window, splits, uint32_t(hasSoftcap)};
                    ComputeMlaSplitPagedBsndMetadata<<<1, nullptr, current>>>(&splitArgs, sizeof(splitArgs));
                }
                return;
            }
            if (window) {
                MlaWindowPagedMetadataArgs windowArgs{args, *window};
                if (args.shape.query.layout == MlaSequenceLayout::Tnd) {
                    ComputeMlaWindowPagedTndMetadata<<<1, nullptr, current>>>(&windowArgs, sizeof(windowArgs));
                } else {
                    ComputeMlaWindowPagedBsndMetadata<<<1, nullptr, current>>>(&windowArgs, sizeof(windowArgs));
                }
                return;
            }
            if (causal) {
                if (args.shape.query.layout == MlaSequenceLayout::Tnd) {
                    ComputeMlaCausalPagedTndMetadata<<<1, nullptr, current>>>(&args, sizeof(args));
                } else {
                    ComputeMlaCausalPagedBsndMetadata<<<1, nullptr, current>>>(&args, sizeof(args));
                }
                return;
            }
        }
        if (inputReady != nullptr) {
            TORCH_CHECK(aclrtRecordEvent(inputReady, current) == ACL_SUCCESS, "MLA: input event recording failed");
            TORCH_CHECK(aclrtStreamWaitEvent(worker, inputReady) == ACL_SUCCESS, "MLA: input event wait failed");
        }
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
        if (metadataDone != nullptr) {
            TORCH_CHECK(aclrtRecordEvent(metadataDone, worker) == ACL_SUCCESS, "MLA: metadata event recording failed");
            TORCH_CHECK(aclrtStreamWaitEvent(current, metadataDone) == ACL_SUCCESS, "MLA: metadata event wait failed");
        }
    }
};

template <typename Args>
MetadataSubmission<Args> PrepareSubmission(const at::Tensor& metadata, const Args& args,
                                           const std::optional<at::Tensor>& lengths,
                                           const std::optional<at::Tensor>& query, bool hasKvLengths,
                                           const std::optional<at::Tensor>& usedQuery = std::nullopt,
                                           const std::optional<at::Tensor>& usedKv = std::nullopt, bool causal = false)
{
    auto stream = c10_npu::getCurrentNPUStream();
    bool onCurrentStream = false;
    if constexpr (std::is_same_v<Args, MlaUsedMetadataArgs> || std::is_same_v<Args, MlaPagedMetadataArgs>) {
        onCurrentStream = causal;
    }
    // The eager worker uses ordinary reusable events, which cannot be recorded
    // during graph capture on 910B. Capture the same producer on the consumer
    // stream so replay reads current device lengths and retains its allocations
    // in the graph pool. No host length snapshot or graph-specific record exists.
    // Metadata allocation has already initialized this device's context.
    if (onCurrentStream || c10_npu::currentStreamCaptureStatusMayInitCtx() == c10_npu::CaptureStatus::Active) {
        c10_npu::NPUCachingAllocator::recordStream(metadata.storage().data_ptr(), stream);
        for (const auto* input : {&lengths, &query, &usedQuery, &usedKv}) {
            if (*input && (*input)->numel() != 0) {
                c10_npu::NPUCachingAllocator::recordStream((*input)->storage().data_ptr(), stream);
            }
        }
        return {metadata, lengths, query,        usedQuery, usedKv, args, stream.stream(false), stream.stream(false),
                nullptr,  nullptr, hasKvLengths, causal};
    }
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

template <typename View = MlaBsndTensorView>
View PagedTensorView(const at::Tensor& t)
{
    View view{};
    for (int64_t i = 0; i < t.dim(); ++i) {
        view.sizes[i] = t.size(i);
        view.strides[i] = t.stride(i);
    }
    view.storageOffset = t.storage_offset();
    view.storageBytes = t.storage().nbytes();
    return view;
}

// Validate the entire accessible span, not numel() for a strided view. Empty
// descriptors have no address requirement; Q-used presence is restored below.
MlaPagedMetadataArray PagedTensorArray(const at::Tensor& t, uint64_t span)
{
    if (span == 0) {
        return {};
    }
    const uint64_t elements = t.storage().nbytes() / t.element_size();
    TORCH_CHECK(t.storage_offset() >= 0 && uint64_t(t.storage_offset()) <= elements &&
                    span <= elements - uint64_t(t.storage_offset()),
                "MLA: paged tensor view exceeds storage bounds");
    const auto address = reinterpret_cast<uintptr_t>(t.data_ptr());
    TORCH_CHECK(address != 0 && address % t.element_size() == 0 && span <= (UINTPTR_MAX - address) / t.element_size(),
                "MLA: tensor address overflow");
    return {address, span};
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

namespace {
MlaWindowKernelParams CheckWindow(const MlaWindowConfig& config)
{
    MlaWindow normalized{};
    TORCH_CHECK(NormalizeWindow(config, normalized) == MlaCausalStatus::Success,
                "MLA: window bounds exceed the supported normalization domain");
    return {uint32_t(normalized.kind), normalized.left, normalized.right};
}
} // namespace

at::Tensor SplitBsndMetadataOp(int64_t batch, int64_t sq, int64_t sk, int64_t heads, at::ScalarType dtype,
                               const at::Tensor& cacheSeqlens, int64_t windowLeft, int64_t windowRight, bool causal,
                               int64_t splits, bool hasSoftcap)
{
    TORCH_CHECK(FitsDenseShape(batch, sq, heads, sk), "MLA: causal BSND shape bounds exceeded");
    TORCH_CHECK(dtype == at::kHalf || dtype == at::kBFloat16, "MLA: qkv_dtype must be FP16 or BF16");
    CheckCacheLengths(cacheSeqlens, batch, cacheSeqlens.device());
    CheckUsedInputs(cacheSeqlens, batch, std::nullopt, cacheSeqlens);
    TORCH_CHECK(splits == 0 || (splits >= (hasSoftcap ? 1 : 2) && splits <= 256),
                "MLA: split requires num_splits=0 or [2,256]");
    uint32_t words;
    TORCH_CHECK(splits == 0 ? AutoSplitBsndMetadataWords(batch, words) : SplitBsndMetadataWords(batch, words),
                "MLA: split metadata word count overflow");
    MlaWindowConfig config{sq, sk, windowLeft, windowRight, causal};
    CheckWindow(config);
    const c10::OptionalDeviceGuard deviceGuard(device_of(cacheSeqlens));
    auto metadata = at::empty({words}, cacheSeqlens.options());
    const auto address = reinterpret_cast<uint64_t>(batch == 0 ? metadata.data_ptr() : cacheSeqlens.data_ptr());
    MlaUsedMetadataArgs args{
        reinterpret_cast<uint64_t>(metadata.data_ptr()),
        words,
        uint32_t(dtype == at::kBFloat16 ? MlaMetadataDtype::Bf16 : MlaMetadataDtype::Fp16),
        {batch, heads, {MlaSequenceLayout::Bsnd, batch * sq, sq}, {MlaSequenceLayout::Bsnd, batch * sk, sk}},
        CausalPlatform(cacheSeqlens.device().index(), batch != 0 && sq != 0 && sk != 0),
        {},
        {},
        {},
        {address, uint64_t(batch)}};
    auto producer = PrepareSubmission(metadata, args, std::nullopt, std::nullopt, true, std::nullopt,
                                      std::optional<at::Tensor>(cacheSeqlens), true);
    producer.window = config;
    producer.splits = splits;
    producer.hasSoftcap = hasSoftcap;
    RunCausalLaunch("ascendc_mla_split_metadata", [producer]() mutable -> int {
        producer.Enqueue();
        return 0;
    });
    return metadata;
}

at::Tensor WindowBsndMetadataOp(int64_t batch, int64_t sq, int64_t sk, int64_t heads, at::ScalarType dtype,
                                const at::Tensor& cacheSeqlens, int64_t windowLeft, int64_t windowRight, bool causal)
{
    TORCH_CHECK(FitsDenseShape(batch, sq, heads, sk), "MLA: causal BSND shape bounds exceeded");
    TORCH_CHECK(dtype == at::kHalf || dtype == at::kBFloat16, "MLA: qkv_dtype must be FP16 or BF16");
    CheckCacheLengths(cacheSeqlens, batch, cacheSeqlens.device());
    CheckUsedInputs(cacheSeqlens, batch, std::nullopt, cacheSeqlens);
    uint32_t words;
    TORCH_CHECK(WindowMetadataWords(batch, words), "MLA: window metadata word count overflow");
    MlaWindowConfig config{sq, sk, windowLeft, windowRight, causal};
    CheckWindow(config);
    const c10::OptionalDeviceGuard deviceGuard(device_of(cacheSeqlens));
    auto metadata = at::empty({words}, cacheSeqlens.options());
    const auto address = reinterpret_cast<uint64_t>(batch == 0 ? metadata.data_ptr() : cacheSeqlens.data_ptr());
    MlaUsedMetadataArgs args{
        reinterpret_cast<uint64_t>(metadata.data_ptr()),
        words,
        uint32_t(dtype == at::kBFloat16 ? MlaMetadataDtype::Bf16 : MlaMetadataDtype::Fp16),
        {batch, heads, {MlaSequenceLayout::Bsnd, batch * sq, sq}, {MlaSequenceLayout::Bsnd, batch * sk, sk}},
        CausalPlatform(cacheSeqlens.device().index(), batch != 0 && sq != 0 && sk != 0),
        {},
        {},
        {},
        {address, uint64_t(batch)}};
    auto producer = PrepareSubmission(metadata, args, std::nullopt, std::nullopt, true, std::nullopt,
                                      std::optional<at::Tensor>(cacheSeqlens), true);
    producer.window = config;
    RunCausalLaunch("ascendc_mla_window_metadata", [producer]() mutable -> int {
        producer.Enqueue();
        return 0;
    });
    return metadata;
}

at::Tensor CausalBsndMetadataOp(int64_t batch, int64_t sq, int64_t sk, int64_t heads, at::ScalarType dtype,
                                const at::Tensor& cacheSeqlens)
{
    TORCH_CHECK(FitsDenseShape(batch, sq, heads, sk), "MLA: causal BSND shape bounds exceeded");
    TORCH_CHECK(dtype == at::kHalf || dtype == at::kBFloat16, "MLA: qkv_dtype must be FP16 or BF16");
    CheckCacheLengths(cacheSeqlens, batch, cacheSeqlens.device());
    const uint32_t words = CheckUsedInputs(cacheSeqlens, batch, std::nullopt, cacheSeqlens);
    const c10::OptionalDeviceGuard deviceGuard(device_of(cacheSeqlens));
    auto metadata = at::empty({words}, cacheSeqlens.options());
    const auto address = reinterpret_cast<uint64_t>(batch == 0 ? metadata.data_ptr() : cacheSeqlens.data_ptr());
    MlaUsedMetadataArgs args{
        reinterpret_cast<uint64_t>(metadata.data_ptr()),
        words,
        uint32_t(dtype == at::kBFloat16 ? MlaMetadataDtype::Bf16 : MlaMetadataDtype::Fp16),
        {batch, heads, {MlaSequenceLayout::Bsnd, batch * sq, sq}, {MlaSequenceLayout::Bsnd, batch * sk, sk}},
        CausalPlatform(cacheSeqlens.device().index(), batch != 0 && sq != 0 && sk != 0),
        {},
        {},
        {},
        {address, uint64_t(batch)}};
    auto producer = PrepareSubmission(metadata, args, std::nullopt, std::nullopt, true, std::nullopt,
                                      std::optional<at::Tensor>(cacheSeqlens), true);
    producer.fullSchedule = CausalFullSchedule(args.shape, args.platform, cacheSeqlens.device().index(), args.dtype);
    RunCausalLaunch("ascendc_mla_causal_metadata", [producer]() mutable -> int {
        producer.Enqueue();
        return 0;
    });
    return metadata;
}

at::Tensor CausalMixedMetadataOp(int64_t batch, int64_t maxQ, int64_t sk, int64_t heads, at::ScalarType dtype,
                                 const at::Tensor& cacheSeqlens, const at::Tensor& cuQ)
{
    CheckTndBsndMetadataInputs(batch, maxQ, sk, heads, dtype, cacheSeqlens, cuQ);
    const uint32_t words = CheckUsedInputs(cacheSeqlens, batch, std::nullopt, cacheSeqlens);
    const c10::OptionalDeviceGuard deviceGuard(device_of(cacheSeqlens));
    auto metadata = at::empty({words}, cacheSeqlens.options());
    const auto address = reinterpret_cast<uint64_t>(batch == 0 ? metadata.data_ptr() : cacheSeqlens.data_ptr());
    MlaUsedMetadataArgs args{
        reinterpret_cast<uint64_t>(metadata.data_ptr()),
        words,
        uint32_t(dtype == at::kBFloat16 ? MlaMetadataDtype::Bf16 : MlaMetadataDtype::Fp16),
        {batch, heads, {MlaSequenceLayout::Tnd, 0, maxQ}, {MlaSequenceLayout::Bsnd, batch * sk, sk}},
        CausalPlatform(cacheSeqlens.device().index(), false),
        {reinterpret_cast<uint64_t>(cuQ.data_ptr()), uint64_t(cuQ.numel())},
        {},
        {},
        {address, uint64_t(batch)}};
    auto producer = PrepareSubmission(metadata, args, std::nullopt, std::optional<at::Tensor>(cuQ), true, std::nullopt,
                                      std::optional<at::Tensor>(cacheSeqlens), true);
    producer.queryTotalFromPrefix = true;
    RunCausalLaunch("ascendc_mla_causal_mixed_metadata", [producer]() mutable -> int {
        producer.Enqueue();
        return 0;
    });
    return metadata;
}

at::Tensor WindowMixedMetadataOp(int64_t batch, int64_t maxQ, int64_t sk, int64_t heads, at::ScalarType dtype,
                                 const at::Tensor& cacheSeqlens, const at::Tensor& cuQ, int64_t windowLeft,
                                 int64_t windowRight, bool causal)
{
    CheckTndBsndMetadataInputs(batch, maxQ, sk, heads, dtype, cacheSeqlens, cuQ);
    CheckUsedInputs(cacheSeqlens, batch, std::nullopt, cacheSeqlens);
    uint32_t words;
    TORCH_CHECK(WindowMetadataWords(batch, words), "MLA: window metadata word count overflow");
    MlaWindowConfig config{maxQ, sk, windowLeft, windowRight, causal};
    CheckWindow(config);
    const c10::OptionalDeviceGuard deviceGuard(device_of(cacheSeqlens));
    auto metadata = at::empty({words}, cacheSeqlens.options());
    const auto address = reinterpret_cast<uint64_t>(batch == 0 ? metadata.data_ptr() : cacheSeqlens.data_ptr());
    MlaUsedMetadataArgs args{
        reinterpret_cast<uint64_t>(metadata.data_ptr()),
        words,
        uint32_t(dtype == at::kBFloat16 ? MlaMetadataDtype::Bf16 : MlaMetadataDtype::Fp16),
        {batch, heads, {MlaSequenceLayout::Tnd, 0, maxQ}, {MlaSequenceLayout::Bsnd, batch * sk, sk}},
        CausalPlatform(cacheSeqlens.device().index(), false),
        {reinterpret_cast<uint64_t>(cuQ.data_ptr()), uint64_t(cuQ.numel())},
        {},
        {},
        {address, uint64_t(batch)}};
    auto producer = PrepareSubmission(metadata, args, std::nullopt, std::optional<at::Tensor>(cuQ), true, std::nullopt,
                                      std::optional<at::Tensor>(cacheSeqlens), true);
    producer.queryTotalFromPrefix = true;
    producer.window = config;
    RunCausalLaunch("ascendc_mla_window_mixed_metadata", [producer]() mutable -> int {
        producer.Enqueue();
        return 0;
    });
    return metadata;
}

at::Tensor SplitMixedMetadataOp(int64_t batch, int64_t maxQ, int64_t sk, int64_t heads, at::ScalarType dtype,
                                const at::Tensor& cacheSeqlens, const at::Tensor& cuQ, int64_t windowLeft,
                                int64_t windowRight, bool causal, int64_t splits, bool hasSoftcap)
{
    TORCH_CHECK(splits == 0 || (splits >= (hasSoftcap ? 1 : 2) && splits <= 256),
                "MLA: mixed split requires 0 or [2,256], or 1 with softcap");
    CheckTndBsndMetadataInputs(batch, maxQ, sk, heads, dtype, cacheSeqlens, cuQ);
    CheckUsedInputs(cacheSeqlens, batch, std::nullopt, cacheSeqlens);
    uint32_t words;
    TORCH_CHECK(splits == 0 ? AutoSplitMixedMetadataWords(batch, words) : SplitMixedMetadataWords(batch, words),
                "MLA: split metadata word count overflow");
    MlaWindowConfig config{maxQ, sk, windowLeft, windowRight, causal};
    CheckWindow(config);
    const c10::OptionalDeviceGuard deviceGuard(device_of(cacheSeqlens));
    auto metadata = at::empty({words}, cacheSeqlens.options());
    const auto address = reinterpret_cast<uint64_t>(batch == 0 ? metadata.data_ptr() : cacheSeqlens.data_ptr());
    MlaUsedMetadataArgs args{
        reinterpret_cast<uint64_t>(metadata.data_ptr()),
        words,
        uint32_t(dtype == at::kBFloat16 ? MlaMetadataDtype::Bf16 : MlaMetadataDtype::Fp16),
        {batch, heads, {MlaSequenceLayout::Tnd, 0, maxQ}, {MlaSequenceLayout::Bsnd, batch * sk, sk}},
        CausalPlatform(cacheSeqlens.device().index(), false),
        {reinterpret_cast<uint64_t>(cuQ.data_ptr()), uint64_t(cuQ.numel())},
        {},
        {},
        {address, uint64_t(batch)}};
    auto producer = PrepareSubmission(metadata, args, std::nullopt, std::optional<at::Tensor>(cuQ), true, std::nullopt,
                                      std::optional<at::Tensor>(cacheSeqlens), true);
    producer.queryTotalFromPrefix = true;
    producer.window = config;
    producer.splits = splits;
    producer.hasSoftcap = hasSoftcap;
    RunCausalLaunch("ascendc_mla_split_mixed_metadata", [producer]() mutable -> int {
        producer.Enqueue();
        return 0;
    });
    return metadata;
}

at::Tensor PagedMetadataOp(int64_t batch, int64_t maxQ, int64_t capacity, int64_t heads, int64_t pageSize,
                           at::ScalarType dtype, const at::Tensor& cacheSeqlens, const std::optional<at::Tensor>& cuQ,
                           bool causal, int64_t windowLeft, int64_t windowRight, int64_t splits, bool hasSoftcap)
{
    auto words = CheckPagedMetadataInputs(batch, maxQ, capacity, heads, pageSize, dtype, cacheSeqlens, cuQ);
    TORCH_CHECK(splits >= 0 && splits <= 256, "MLA: paged num_splits must be in [0,256]");
    const bool isSplit = splits != 1 || hasSoftcap;
    const bool isWindow = windowLeft != -1 || windowRight != -1;
    MlaWindowConfig config{maxQ, capacity, windowLeft, windowRight, causal};
    if (isWindow || isSplit) {
        CheckWindow(config);
        TORCH_CHECK(
            splits == 0
                ? (cuQ ? AutoSplitPagedTndMetadataWords(batch, words) : AutoSplitPagedBsndMetadataWords(batch, words))
            : isSplit ? (cuQ ? SplitPagedTndMetadataWords(batch, words) : SplitPagedBsndMetadataWords(batch, words))
                      : WindowPagedMetadataWords(batch, words),
            "MLA: window paged metadata word count overflow");
    }
    const c10::OptionalDeviceGuard deviceGuard(device_of(cacheSeqlens));
    auto metadata = at::empty({words}, cacheSeqlens.options().dtype(at::kInt));
    const MlaSequenceAllocation query{cuQ ? MlaSequenceLayout::Tnd : MlaSequenceLayout::Bsnd, cuQ ? 0 : batch * maxQ,
                                      maxQ};
    // AICPU derives the TND total from the prefix and reads all length values.
    // No physical pages, cache addresses or strides are part of generation.
    // Static capacity fixes blockDim, including device-only zero lengths, so
    // forward consumers need not read the record to choose their launch size.
    MlaPagedMetadataArgs args{reinterpret_cast<uint64_t>(metadata.data_ptr()),
                              words,
                              uint32_t(dtype == at::kBFloat16 ? MlaMetadataDtype::Bf16 : MlaMetadataDtype::Fp16),
                              {batch, heads, query, capacity, pageSize},
                              (causal || isWindow || isSplit) ? CausalPlatform(cacheSeqlens.device().index(),
                                                                               batch != 0 && maxQ != 0 && capacity != 0)
                                                              : DensePlatform(batch != 0 && maxQ != 0 && capacity != 0),
                              {cuQ ? reinterpret_cast<uint64_t>(cuQ->data_ptr()) : 0, cuQ ? uint64_t(cuQ->numel()) : 0},
                              {},
                              {reinterpret_cast<uint64_t>(cacheSeqlens.data_ptr()), uint64_t(cacheSeqlens.numel())}};
    if (isSplit) {
        args.platform.blockDims.fill(batch && maxQ && capacity ? args.platform.aicNum : 1);
        if (splits == 0 && !cuQ) {
            MlaAutoSplitPagedBsndBudget budget{};
            TORCH_CHECK(BuildAutoSplitPagedBsndProductionBudget(args.shape, args.platform, budget) ==
                            MlaScheduleStatus::Success,
                        "MLA: paged auto split workspace or accumulator overflow");
        } else if (!cuQ) {
            MlaSplitPagedBsndBudget budget{};
            TORCH_CHECK(BuildSplitPagedBsndBudget(args.shape, args.platform, splits, budget) ==
                            MlaScheduleStatus::Success,
                        "MLA: paged split workspace or accumulator overflow");
        }
    }
    auto submission = PrepareSubmission(metadata, args, cacheSeqlens, cuQ, true, std::nullopt, std::nullopt,
                                        causal || isWindow || isSplit);
    if (isWindow || isSplit) {
        submission.window = config;
        submission.splits = splits;
        submission.hasSoftcap = hasSoftcap;
        submission.queryTotalFromPrefix = bool(cuQ);
    }
    auto enqueue = [submission]() mutable -> int {
        submission.Enqueue();
        return 0;
    };
    if (causal || isWindow || isSplit) {
        RunCausalLaunch(isSplit
                            ? (cuQ ? "ascendc_mla_split_paged_tnd_metadata" : "ascendc_mla_split_paged_bsnd_metadata")
                        : isWindow ? "ascendc_mla_window_paged_metadata"
                                   : "ascendc_mla_causal_paged_metadata",
                        enqueue);
    } else {
        at_npu::native::OpCommand::RunOpApiV2("ascendc_mla_paged_metadata", enqueue);
    }
    return metadata;
}

namespace {
std::vector<at::Tensor>
PagedForward(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& qv,
             const at::Tensor& pageTable, float scale, const std::optional<at::Tensor>& cacheSeqlens,
             const std::optional<at::Tensor>& usedQuery, const std::optional<at::Tensor>& schedulerMetadata,
             const std::optional<at::Tensor>& cuQ, int64_t maxQ, bool causal = false, int64_t windowLeft = -1,
             int64_t windowRight = -1, int64_t splits = 1, float softcap = 0.0f)
{
    const bool tnd = cuQ.has_value();
    TORCH_CHECK(splits >= 0 && splits <= 256, "MLA: paged num_splits must be in [0,256]");
    TORCH_CHECK(std::isfinite(softcap) && softcap >= 0, "MLA: softcap must be finite and non-negative in FP32");
    const bool isSplit = splits != 1 || softcap > 0;
    const bool isWindow = windowLeft != -1 || windowRight != -1;
    TORCH_CHECK(q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16,
                "MLA: inputs must have FP16 or BF16 dtype");
    TORCH_CHECK(q.device().type() == at::kPrivateUse1, "MLA: inputs must be NPU tensors");
    for (const auto& t : {q, k, v, qv}) {
        TORCH_CHECK(t.scalar_type() == q.scalar_type(), "MLA: all inputs must have the same dtype");
        TORCH_CHECK(t.device() == q.device(), "MLA: all inputs must be on the same NPU device");
        TORCH_CHECK(!t.requires_grad(), "MLA compressed forward is not implemented yet for gradients");
    }
    TORCH_CHECK(q.dim() == (tnd ? 3 : 4) && qv.dim() == q.dim() && k.dim() == 4 && v.dim() == 4,
                "MLA: paged forward requires matching BSND/TND Q and rank-4 KV");
    TORCH_CHECK(std::isfinite(scale), "MLA: softmax_scale must be finite in FP32");
    TORCH_CHECK(pageTable.dim() == 2 && pageTable.scalar_type() == at::kInt, "MLA: page_table must be rank-2 int32");
    TORCH_CHECK(pageTable.device() == q.device(), "MLA: page_table must be on the same NPU device");
    TORCH_CHECK(!usedQuery || !schedulerMetadata, "MLA: Q-used paged metadata cannot be explicitly reused");
    const c10::OptionalDeviceGuard deviceGuard(device_of(q));
    const MlaPageTableView table{{pageTable.size(0), pageTable.size(1)},
                                 {pageTable.stride(0), pageTable.stride(1)},
                                 pageTable.storage_offset(),
                                 pageTable.storage().nbytes()};
    const auto dtype = uint32_t(q.scalar_type() == at::kBFloat16 ? MlaMetadataDtype::Bf16 : MlaMetadataDtype::Fp16);
    const auto platform = (causal || isWindow || isSplit) ? CausalPlatform(q.device().index(), false) : DensePlatform();
    MlaPagedPlan plan{};
    const auto status = tnd ? BuildPagedTndPlan(PagedTensorView<MlaTndTensorView>(q),
                                                PagedTensorView<MlaTndTensorView>(qv), PagedTensorView(k),
                                                PagedTensorView(v), table, maxQ, platform, dtype, bool(usedQuery), plan)
                            : BuildPagedBsndPlan(PagedTensorView(q), PagedTensorView(qv), PagedTensorView(k),
                                                 PagedTensorView(v), table, platform, dtype, bool(usedQuery), plan);
    TORCH_CHECK(status == MlaPagedPlanStatus::Success,
                "MLA: paged shape, stride, storage or platform bounds exceeded (status ", int(status), ")");
    MlaWindowConfig config{plan.shape.query.maxLength, plan.shape.logicalKvCapacity, windowLeft, windowRight, causal};
    MlaWindowKernelParams window{};
    if (isWindow || isSplit) {
        window = CheckWindow(config);
        TORCH_CHECK(splits == 0 ? (tnd ? AutoSplitPagedTndMetadataWords(plan.shape.batch, plan.metadataWords)
                                       : AutoSplitPagedBsndMetadataWords(plan.shape.batch, plan.metadataWords))
                    : isSplit   ? (tnd ? SplitPagedTndMetadataWords(plan.shape.batch, plan.metadataWords)
                                       : SplitPagedBsndMetadataWords(plan.shape.batch, plan.metadataWords))
                                : WindowPagedMetadataWords(plan.shape.batch, plan.metadataWords),
                    "MLA: window paged metadata word count overflow");
    }
    if (isSplit) {
        if (tnd) {
            const uint32_t blocks = plan.shape.batch && q.size(0) && plan.shape.logicalKvCapacity ? platform.aicNum : 1;
            plan.platform.blockDims.fill(blocks);
            plan.context.blockDim = blocks;
            if (splits == 0) {
                MlaAutoSplitPagedTndBudget budget{};
                TORCH_CHECK(BuildAutoSplitPagedTndProductionBudget(plan.shape, plan.platform, budget) ==
                                MlaScheduleStatus::Success,
                            "MLA: TND paged auto split workspace or accumulator overflow");
                plan.totalWorkspaceBytes = budget.totalBytes;
                plan.context.workspaceBytes = budget.userBytes;
            } else {
                MlaSplitPagedTndBudget budget{};
                TORCH_CHECK(BuildSplitPagedTndBudget(plan.shape, plan.platform, splits, budget) ==
                                MlaScheduleStatus::Success,
                            "MLA: TND paged split workspace or accumulator overflow");
                plan.totalWorkspaceBytes = budget.totalBytes;
                plan.context.workspaceBytes = budget.userBytes;
            }
        } else if (splits == 0) {
            MlaAutoSplitPagedBsndBudget budget{};
            TORCH_CHECK(BuildAutoSplitPagedBsndProductionBudget(plan.shape, plan.platform, budget) ==
                            MlaScheduleStatus::Success,
                        "MLA: paged auto split workspace or accumulator overflow");
            plan.totalWorkspaceBytes = budget.totalBytes;
            plan.context.workspaceBytes = budget.userBytes;
        } else {
            MlaSplitPagedBsndBudget budget{};
            TORCH_CHECK(BuildSplitPagedBsndBudget(plan.shape, plan.platform, splits, budget) ==
                            MlaScheduleStatus::Success,
                        "MLA: paged split workspace or accumulator overflow");
            plan.totalWorkspaceBytes = budget.totalBytes;
            plan.context.workspaceBytes = budget.userBytes;
        }
    }
    if (!isSplit && tnd && q.size(0) == 0 && plan.shape.batch != 0 && maxQ != 0 && plan.shape.logicalKvCapacity != 0) {
        // Public v6 generation only knows B/maxQ/C on the host. Match its
        // full validation launch even when the actual TND allocation is empty;
        // keep the plan's zero workspace and validate every current KV page.
        const auto blocks = platform.blockDims[platform.aicNum - 1];
        TORCH_CHECK(blocks == platform.aicNum, "MLA: paged C1V2 launch bounds exceeded");
        plan.context.blockDim = blocks;
        plan.platform.blockDims.fill(blocks);
    }
    const auto qr = PagedTensorArray(q, plan.layouts.q.spanElements);
    const auto qc = PagedTensorArray(qv, plan.layouts.qv.spanElements);
    const MlaPagedCacheInputs cache{plan.cache, PagedTensorArray(pageTable, plan.cache.tableSpanElements),
                                    PagedTensorArray(k, plan.cache.k.spanElements),
                                    PagedTensorArray(v, plan.cache.v.spanElements)};
    const int64_t batch = plan.shape.batch;
    if (cuQ) {
        TORCH_CHECK(cuQ->scalar_type() == at::kInt && cuQ->dim() == 1 && cuQ->numel() == batch + 1 &&
                        cuQ->is_contiguous(),
                    "MLA: cu_seqlens_q must be contiguous int32[batch_size + 1]");
        TORCH_CHECK(cuQ->device() == q.device(), "MLA: cu_seqlens_q must be on the same NPU device");
        PagedTensorArray(*cuQ, cuQ->numel());
    }
    if (cacheSeqlens) {
        CheckCacheLengths(*cacheSeqlens, batch, q.device());
        PagedTensorArray(*cacheSeqlens, cacheSeqlens->numel());
    }
    if (usedQuery) {
        // Use v6's word bound from the plan, not the nonpaged v5 bound.
        TORCH_CHECK(usedQuery->scalar_type() == at::kInt && usedQuery->dim() == 1 && usedQuery->numel() == batch &&
                        usedQuery->is_contiguous(),
                    "MLA: seqused_q must be contiguous int32[batch_size]");
        TORCH_CHECK(usedQuery->device() == q.device(), "MLA: seqused_q must be on the same NPU device");
        PagedTensorArray(*usedQuery, usedQuery->numel());
    }
    if (schedulerMetadata) {
        TORCH_CHECK(schedulerMetadata->scalar_type() == at::kInt && schedulerMetadata->dim() == 1 &&
                        schedulerMetadata->numel() == plan.metadataWords && schedulerMetadata->is_contiguous(),
                    "MLA: scheduler_metadata must be contiguous int32 with the selected paged protocol size");
        TORCH_CHECK(schedulerMetadata->device() == q.device(),
                    "MLA: scheduler_metadata must be on the same NPU device");
        PagedTensorArray(*schedulerMetadata, plan.metadataWords);
    }

    auto out = tnd ? at::empty({q.size(0), q.size(1), 512}, q.options())
                   : at::empty({q.size(0), q.size(1), q.size(2), 512}, q.options());
    auto lse = tnd ? at::empty({q.size(1), q.size(0)}, q.options().dtype(at::kFloat))
                   : at::empty({q.size(0), q.size(2), q.size(1)}, q.options().dtype(at::kFloat));
    const int64_t splitCapacity = splits == 0 ? kMlaAutoSplitCapacity : splits;
    auto accum =
        isSplit
            ? (tnd ? at::empty({splitCapacity, q.size(1), q.size(0), 512}, q.options().dtype(at::kFloat))
                   : at::empty({splitCapacity, q.size(0), q.size(2), q.size(1), 512}, q.options().dtype(at::kFloat)))
            : at::Tensor();
    auto accumLse =
        isSplit ? (tnd ? at::empty({splitCapacity, q.size(1), q.size(0)}, q.options().dtype(at::kFloat))
                       : at::empty({splitCapacity, q.size(0), q.size(2), q.size(1)}, q.options().dtype(at::kFloat)))
                : at::Tensor();
    if (isSplit) {
        PagedTensorArray(accum, accum.numel());
        PagedTensorArray(accumLse, accumLse.numel());
    }
    auto workspace = at::empty({int64_t(plan.totalWorkspaceBytes)}, q.options().dtype(at::kByte));
    // None means logical capacity M*P, independent of physical pool size.
    auto lengths =
        cacheSeqlens ? *cacheSeqlens : at::full({batch}, plan.shape.logicalKvCapacity, q.options().dtype(at::kInt));
    auto metadata =
        schedulerMetadata ? *schedulerMetadata : at::empty({plan.metadataWords}, q.options().dtype(at::kInt));
    for (const auto& t : {out, lse, workspace, metadata}) {
        PagedTensorArray(t, t.numel());
    }
    auto used = usedQuery ? PagedTensorArray(*usedQuery, usedQuery->numel()) : MlaPagedMetadataArray{};
    if (usedQuery && usedQuery->numel() == 0) {
        // B=0 still carries Q-used presence, with a nonnull zero-count array
        // that is never read and does not overlap the metadata output extent.
        used.address = reinterpret_cast<uint64_t>(metadata.data_ptr());
    }
    const auto prefix = cuQ ? PagedTensorArray(*cuQ, cuQ->numel()) : MlaPagedMetadataArray{};
    MlaPagedMetadataArgs args{reinterpret_cast<uint64_t>(metadata.data_ptr()),
                              plan.metadataWords,
                              dtype,
                              plan.shape,
                              plan.platform,
                              prefix,
                              used,
                              PagedTensorArray(lengths, lengths.numel())};
    std::optional<MetadataSubmission<MlaPagedMetadataArgs>> producer;
    if (!schedulerMetadata) {
        producer = PrepareSubmission(metadata, args, lengths, cuQ, true, usedQuery, std::nullopt,
                                     causal || isWindow || isSplit);
        if (isWindow || isSplit) {
            producer->window = config;
            producer->splits = splits;
            producer->hasSoftcap = softcap > 0;
        }
    }
    auto stream = c10_npu::getCurrentNPUStream();
    for (const auto& t : {q, k, v, qv, pageTable, lengths, metadata}) {
        if (t.numel() != 0) {
            c10_npu::NPUCachingAllocator::recordStream(t.storage().data_ptr(), stream);
        }
    }
    if (usedQuery && usedQuery->numel() != 0) {
        c10_npu::NPUCachingAllocator::recordStream(usedQuery->storage().data_ptr(), stream);
    }
    if (cuQ) {
        c10_npu::NPUCachingAllocator::recordStream(cuQ->storage().data_ptr(), stream);
    }
    MlaPagedLaunchArgs a{};
    uint32_t fftsLength = 0;
    TORCH_CHECK(rtGetC2cCtrlAddr(&a.ffts, &fftsLength) == RT_ERROR_NONE,
                "MLA: failed to obtain cross-core synchronization address");
    a.stream = stream.stream(false);
    a.qc = reinterpret_cast<uint8_t*>(qc.address);
    a.qr = reinterpret_cast<uint8_t*>(qr.address);
    a.output = static_cast<uint8_t*>(out.data_ptr());
    a.lse = static_cast<uint8_t*>(lse.data_ptr());
    a.workspace =
        workspace.numel() == 0 ? nullptr : static_cast<uint8_t*>(workspace.data_ptr()) + plan.context.libApiBytes;
    a.metadata = static_cast<uint8_t*>(metadata.data_ptr());
    a.context = plan.context;
    a.inputs = {args.cuQuery, args.usedQuery, args.kvLengths};
    a.cache = cache;
    a.layouts = plan.layouts;
    a.scale = scale;
    a.causal = causal;
    a.isWindow = isWindow;
    a.window = window;
    auto enqueue = [a, producer, tnd, q, k, v, qv, pageTable, lengths, cuQ, usedQuery, metadata, out, lse, workspace,
                    isSplit, splits, accum, accumLse, softcap]() mutable -> int {
        if (producer) {
            producer->Enqueue();
        }
        if (isSplit) {
            const MlaSplitPagedBsndLaunchArgs splitArgs{a, uint32_t(splits), static_cast<uint8_t*>(accum.data_ptr()),
                                                        static_cast<uint8_t*>(accumLse.data_ptr()), softcap};
            const auto error =
                tnd ? (splits == 0 ? (q.scalar_type() == at::kBFloat16 ? LaunchAutoSplitPagedTndBf16(splitArgs)
                                                                       : LaunchAutoSplitPagedTndFp16(splitArgs))
                                   : (q.scalar_type() == at::kBFloat16 ? LaunchSplitPagedTndBf16(splitArgs)
                                                                       : LaunchSplitPagedTndFp16(splitArgs)))
                : splits == 0 ? (q.scalar_type() == at::kBFloat16 ? LaunchAutoSplitPagedBsndBf16(splitArgs)
                                                                  : LaunchAutoSplitPagedBsndFp16(splitArgs))
                              : (q.scalar_type() == at::kBFloat16 ? LaunchSplitPagedBsndBf16(splitArgs)
                                                                  : LaunchSplitPagedBsndFp16(splitArgs));
            TORCH_CHECK(error == ACL_SUCCESS, "MLA: paged split BATCH_MODE launch failed (", error, ")");
            return 0;
        }
        const auto error = tnd ? (q.scalar_type() == at::kBFloat16 ? LaunchPagedTndBf16(a) : LaunchPagedTndFp16(a))
                               : (q.scalar_type() == at::kBFloat16 ? LaunchPagedBsndBf16(a) : LaunchPagedBsndFp16(a));
        TORCH_CHECK(error == ACL_SUCCESS, "MLA: paged BATCH_MODE kernel launch failed (", error, ")");
        return 0;
    };
    if (causal || isWindow || isSplit) {
        RunCausalLaunch(isSplit    ? (tnd ? "ascendc_mla_split_paged_tnd" : "ascendc_mla_split_paged_bsnd")
                        : isWindow ? (tnd ? "ascendc_mla_window_paged_tnd" : "ascendc_mla_window_paged_bsnd")
                        : tnd      ? "ascendc_mla_causal_paged_tnd"
                                   : "ascendc_mla_causal_paged_bsnd",
                        enqueue);
    } else {
        at_npu::native::OpCommand::RunOpApiV2(tnd ? "ascendc_mla_paged_tnd" : "ascendc_mla_paged_bsnd", enqueue);
    }
    if (splits == 1 && softcap > 0) {
        accum = at::empty({0}, at::TensorOptions().device(q.device()));
        accumLse = at::empty({0}, at::TensorOptions().device(q.device()));
    }
    return {out, lse, accum, accumLse};
}
} // namespace

std::vector<at::Tensor> PagedBsndForward(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                         const at::Tensor& qv, const at::Tensor& pageTable, float scale,
                                         const std::optional<at::Tensor>& cacheSeqlens,
                                         const std::optional<at::Tensor>& usedQuery,
                                         const std::optional<at::Tensor>& schedulerMetadata, bool causal,
                                         int64_t windowLeft, int64_t windowRight, int64_t splits, float softcap)
{
    return PagedForward(q, k, v, qv, pageTable, scale, cacheSeqlens, usedQuery, schedulerMetadata, std::nullopt, 0,
                        causal, windowLeft, windowRight, splits, softcap);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
SplitPagedBsndForwardOp(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& qv,
                        const at::Tensor& pageTable, int64_t splits, int64_t windowLeft, int64_t windowRight,
                        bool causal, std::optional<double> scale, const std::optional<at::Tensor>& lengths,
                        const std::optional<at::Tensor>& used, const std::optional<at::Tensor>& metadata,
                        double softcap)
{
    TORCH_CHECK(std::isfinite(softcap) && softcap >= 0 && std::isfinite(float(softcap)) &&
                    (softcap == 0 || float(softcap) > 0),
                "MLA: softcap must be finite and non-negative in FP32");
    TORCH_CHECK(splits == 0 || (splits >= (softcap > 0 ? 1 : 2) && splits <= 256),
                "MLA: paged split requires num_splits=0 or [2,256], or 1 with softcap");
    TORCH_CHECK(!scale || (std::isfinite(*scale) && std::abs(*scale) <= std::numeric_limits<float>::max()),
                "MLA: softmax_scale must be finite in FP32");
    auto r = PagedBsndForward(q, k, v, qv, pageTable, scale ? float(*scale) : 1.0f / std::sqrt(576.0f), lengths, used,
                              metadata, causal, windowLeft, windowRight, splits, softcap);
    return {r[0], r[1], r[2], r[3]};
}

std::vector<at::Tensor> PagedTndForward(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                        const at::Tensor& qv, const at::Tensor& pageTable, const at::Tensor& cuQ,
                                        int64_t maxQ, float scale, const std::optional<at::Tensor>& cacheSeqlens,
                                        const std::optional<at::Tensor>& usedQuery,
                                        const std::optional<at::Tensor>& schedulerMetadata, bool causal,
                                        int64_t windowLeft, int64_t windowRight, int64_t splits, float softcap)
{
    return PagedForward(q, k, v, qv, pageTable, scale, cacheSeqlens, usedQuery, schedulerMetadata, cuQ, maxQ, causal,
                        windowLeft, windowRight, splits, softcap);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
SplitPagedTndForwardOp(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& qv,
                       const at::Tensor& pageTable, const at::Tensor& cuQ, int64_t maxQ, int64_t splits,
                       int64_t windowLeft, int64_t windowRight, bool causal, std::optional<double> scale,
                       const std::optional<at::Tensor>& lengths, const std::optional<at::Tensor>& used,
                       const std::optional<at::Tensor>& metadata, double softcap)
{
    TORCH_CHECK(std::isfinite(softcap) && softcap >= 0 && std::isfinite(float(softcap)) &&
                    (softcap == 0 || float(softcap) > 0),
                "MLA: softcap must be finite and non-negative in FP32");
    TORCH_CHECK(splits == 0 || (splits >= (softcap > 0 ? 1 : 2) && splits <= 256),
                "MLA: TND paged split requires num_splits=0 or [2,256], or 1 with softcap");
    TORCH_CHECK(!scale || (std::isfinite(*scale) && std::abs(*scale) <= std::numeric_limits<float>::max()),
                "MLA: softmax_scale must be finite in FP32");
    auto r = PagedTndForward(q, k, v, qv, pageTable, cuQ, maxQ, scale ? float(*scale) : 1.0f / std::sqrt(576.0f),
                             lengths, used, metadata, causal, windowLeft, windowRight, splits, softcap);
    return {r[0], r[1], r[2], r[3]};
}

std::tuple<at::Tensor, at::Tensor>
PagedTndForwardOp(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& qv,
                  const at::Tensor& pageTable, const at::Tensor& cuQ, int64_t maxQ, std::optional<double> softmaxScale,
                  const std::optional<at::Tensor>& cacheSeqlens, const std::optional<at::Tensor>& usedQuery,
                  const std::optional<at::Tensor>& schedulerMetadata, bool causal, int64_t windowLeft,
                  int64_t windowRight)
{
    TORCH_CHECK(!softmaxScale ||
                    (std::isfinite(*softmaxScale) && std::abs(*softmaxScale) <= std::numeric_limits<float>::max()),
                "MLA: softmax_scale must be finite in FP32");
    auto result = PagedTndForward(q, k, v, qv, pageTable, cuQ, maxQ,
                                  softmaxScale ? float(*softmaxScale) : 1.0f / std::sqrt(576.0f), cacheSeqlens,
                                  usedQuery, schedulerMetadata, causal, windowLeft, windowRight);
    return {result[0], result[1]};
}

std::tuple<at::Tensor, at::Tensor> PagedBsndForwardOp(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                                      const at::Tensor& qv, const at::Tensor& pageTable,
                                                      std::optional<double> softmaxScale,
                                                      const std::optional<at::Tensor>& cacheSeqlens,
                                                      const std::optional<at::Tensor>& usedQuery,
                                                      const std::optional<at::Tensor>& schedulerMetadata, bool causal,
                                                      int64_t windowLeft, int64_t windowRight)
{
    TORCH_CHECK(!softmaxScale ||
                    (std::isfinite(*softmaxScale) && std::abs(*softmaxScale) <= std::numeric_limits<float>::max()),
                "MLA: softmax_scale must be finite in FP32");
    auto result =
        PagedBsndForward(q, k, v, qv, pageTable, softmaxScale ? float(*softmaxScale) : 1.0f / std::sqrt(576.0f),
                         cacheSeqlens, usedQuery, schedulerMetadata, causal, windowLeft, windowRight);
    return {result[0], result[1]};
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
                                    const std::optional<at::Tensor>& usedKv, float scale, bool causal = false,
                                    const std::optional<at::Tensor>& schedulerMetadata = std::nullopt,
                                    const std::optional<MlaWindowConfig>& window = std::nullopt, int64_t splits = 1,
                                    float softcap = 0.0f)
{
    uint32_t words = CheckUsedInputs(q, shape.batch, usedQuery, usedKv);
    if (window) {
        TORCH_CHECK(WindowMetadataWords(shape.batch, words), "MLA: window metadata word count overflow");
    }
    if (splits != 1 || softcap > 0) {
        TORCH_CHECK(splits == 0 ? AutoSplitBsndMetadataWords(shape.batch, words)
                                : SplitBsndMetadataWords(shape.batch, words),
                    "MLA: split metadata word count overflow");
    }
    const int64_t batch = shape.batch, sq = shape.query.maxLength, heads = shape.queryHeads, sk = shape.kv.maxLength;
    const bool tnd = shape.query.layout == MlaSequenceLayout::Tnd;
    const auto layouts = !tnd                                        ? CheckBsndInputLayouts(q, k, v, qv)
                         : shape.kv.layout == MlaSequenceLayout::Tnd ? CheckTndInputLayouts(q, k, v, qv)
                                                                     : CheckTndBsndInputLayouts(q, k, v, qv);
    const bool nonempty = batch != 0 && shape.query.totalTokens != 0 && shape.kv.totalTokens != 0;
    auto p = (causal || window) ? CausalPlatform(q.device().index(), nonempty) : DensePlatform(nonempty);
    // Static allocation bounds only; source scheduling reads used on AICPU.
    const uint64_t m = (std::min(uint64_t(sq) * heads, uint64_t(256)) + 15) / 16 * 16;
    const uint64_t n = (std::min(uint64_t(sk), uint64_t(512)) + 31) / 32 * 32;
    uint64_t userBytes = nonempty ? uint64_t(2) * p.aicNum * (6 * m * n + 8 * m * 512 + 8 * 256) : 0;
    const uint64_t libBytes = nonempty ? p.libApiWorkspaceBytes : 0;
    if (splits > 1 || (splits == 1 && softcap > 0)) {
        MlaSplitKvPlanLayout layout{};
        const auto status =
            tnd && shape.kv.layout == MlaSequenceLayout::Bsnd
                ? BuildSplitMixedPlanLayout({{batch, shape.query.totalTokens, heads, sq, sk},
                                             256,
                                             512,
                                             splits,
                                             p.aicNum,
                                             p.blockDims[nonempty ? p.aicNum - 1 : 0],
                                             2,
                                             libBytes},
                                            layout)
            : tnd ? BuildSplitTndPlanLayout({{batch, shape.query.totalTokens, shape.kv.totalTokens, heads, sq, sk},
                                             256,
                                             512,
                                             splits,
                                             p.aicNum,
                                             p.blockDims[nonempty ? p.aicNum - 1 : 0],
                                             2,
                                             libBytes},
                                            layout)
                  : BuildSplitKvPlanLayout({batch, sq, sk, heads, 256, 512, splits, p.aicNum,
                                            p.blockDims[nonempty ? p.aicNum - 1 : 0], 2, libBytes},
                                           layout);
        TORCH_CHECK(status == MlaScheduleStatus::Success, "MLA: split workspace or accumulator overflow");
        userBytes = layout.userBytes;
    }
    if (splits == 0) {
        MlaAutoSplitKvLayout layout{};
        MlaAccumulatorLayout accumulator{};
        const auto status =
            tnd && shape.kv.layout == MlaSequenceLayout::Bsnd
                ? BuildAutoSplitMixedPlanLayout(
                      {{batch, shape.query.totalTokens, heads, sq, sk}, 256, 512, p.aicNum, 2, libBytes}, layout)
            : tnd ? BuildAutoSplitTndPlanLayout({{batch, shape.query.totalTokens, shape.kv.totalTokens, heads, sq, sk},
                                                 256,
                                                 512,
                                                 p.aicNum,
                                                 2,
                                                 libBytes},
                                                layout)
                  : BuildAutoSplitKvLayout({batch, sq, sk, heads, 256, 512, p.aicNum, 2, libBytes}, layout);
        TORCH_CHECK(
            status == MlaScheduleStatus::Success &&
                BuildSplitAccumulatorLayout({tnd ? MlaAccumulatorQueryLayout::Tnd : MlaAccumulatorQueryLayout::Bsnd,
                                             batch, sq, shape.query.totalTokens, heads, kMlaAutoSplitCapacity},
                                            accumulator) == MlaScheduleStatus::Success,
            "MLA: auto split workspace or accumulator overflow");
        userBytes = layout.userBytes + (nonempty ? uint64_t(p.aicNum) * 256 * 528 * 4 : 0);
    }
    TORCH_CHECK(libBytes <= uint64_t(INT64_MAX) - userBytes, "MLA: workspace overflow");
    auto out = tnd ? at::empty({shape.query.totalTokens, heads, 512}, q.options())
                   : at::empty({batch, sq, heads, 512}, q.options());
    auto lse = tnd ? at::empty({heads, shape.query.totalTokens}, q.options().dtype(at::kFloat))
                   : at::empty({batch, heads, sq}, q.options().dtype(at::kFloat));
    const int64_t capacity = splits == 0 ? kMlaAutoSplitCapacity : splits;
    auto accum = splits == 1 && softcap == 0 ? at::Tensor()
                 : tnd ? at::empty({capacity, heads, shape.query.totalTokens, 512}, q.options().dtype(at::kFloat))
                       : at::empty({capacity, batch, heads, sq, 512}, q.options().dtype(at::kFloat));
    auto accumLse = splits == 1 && softcap == 0 ? at::Tensor()
                    : tnd ? at::empty({capacity, heads, shape.query.totalTokens}, q.options().dtype(at::kFloat))
                          : at::empty({capacity, batch, heads, sq}, q.options().dtype(at::kFloat));
    auto workspace = at::empty({static_cast<int64_t>(userBytes + libBytes)}, q.options().dtype(at::kByte));
    auto metadata = schedulerMetadata ? *schedulerMetadata : at::empty({words}, q.options().dtype(at::kInt));
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
    std::optional<MetadataSubmission<MlaUsedMetadataArgs>> producer;
    if (!schedulerMetadata) {
        producer = PrepareSubmission(metadata, args, cuK, cuQ, true, usedQuery, usedKv, causal || window.has_value());
        producer->window = window;
        producer->splits = splits;
        producer->hasSoftcap = softcap > 0;
        if (causal && !tnd) {
            producer->fullSchedule = CausalFullSchedule(shape, p, q.device().index(), dtype);
        }
    }
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
    a.causal = causal;
    a.isWindow = window.has_value();
    if (window) {
        a.window = CheckWindow(*window);
    }
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
    a.softcap = softcap;
    const auto launch = [a, producer, q, k, v, qv, out, lse, workspace, metadata, cuQ, cuK, usedQuery, usedKv, tnd,
                         splits, accum, accumLse, softcap]() mutable -> int {
        if (producer) {
            producer->Enqueue();
        }
        aclError error;
        if (splits != 1 || softcap > 0) {
            MlaSplitBsndLaunchArgs splitArgs{a, uint32_t(splits), static_cast<uint8_t*>(accum.data_ptr()),
                                             static_cast<uint8_t*>(accumLse.data_ptr())};
            if (tnd && a.context.kv.layout == uint32_t(MlaSequenceLayout::Bsnd)) {
                MlaSplitMixedLaunchArgs mixedArgs{a, uint32_t(splits), static_cast<uint8_t*>(accum.data_ptr()),
                                                  static_cast<uint8_t*>(accumLse.data_ptr())};
                error = splits == 0 ? (q.scalar_type() == at::kBFloat16 ? LaunchAutoSplitMixedBf16(mixedArgs)
                                                                        : LaunchAutoSplitMixedFp16(mixedArgs))
                                    : (q.scalar_type() == at::kBFloat16 ? LaunchSplitMixedBf16(mixedArgs)
                                                                        : LaunchSplitMixedFp16(mixedArgs));
            } else if (tnd) {
                MlaSplitTndLaunchArgs tndArgs{a, uint32_t(splits), static_cast<uint8_t*>(accum.data_ptr()),
                                              static_cast<uint8_t*>(accumLse.data_ptr())};
                error = splits == 0 ? (q.scalar_type() == at::kBFloat16 ? LaunchAutoSplitTndBf16(tndArgs)
                                                                        : LaunchAutoSplitTndFp16(tndArgs))
                                    : (q.scalar_type() == at::kBFloat16 ? LaunchSplitTndBf16(tndArgs)
                                                                        : LaunchSplitTndFp16(tndArgs));
            } else
                error = splits == 0 ? (q.scalar_type() == at::kBFloat16 ? LaunchAutoSplitBsndBf16(splitArgs)
                                                                        : LaunchAutoSplitBsndFp16(splitArgs))
                                    : (q.scalar_type() == at::kBFloat16 ? LaunchSplitBsndBf16(splitArgs)
                                                                        : LaunchSplitBsndFp16(splitArgs));
        } else
            error = tnd ? (q.scalar_type() == at::kBFloat16 ? LaunchTndUsedBf16(a) : LaunchTndUsedFp16(a))
                        : (q.scalar_type() == at::kBFloat16 ? LaunchBsndUsedBf16(a) : LaunchBsndUsedFp16(a));
        TORCH_CHECK(error == ACL_SUCCESS, "MLA: used BATCH_MODE kernel launch failed (", error, ")");
        return 0;
    };
    if (causal || window) {
        RunCausalLaunch(tnd ? "ascendc_mla_tnd_used" : "ascendc_mla_bsnd_used", launch);
    } else {
        at_npu::native::OpCommand::RunOpApiV2(tnd ? "ascendc_mla_tnd_used" : "ascendc_mla_bsnd_used", launch);
    }
    if ((splits > 1 || softcap > 0) && !nonempty) {
        // Stream-ordered after both consumers validate even an empty record.
        out.zero_();
        lse.fill_(std::numeric_limits<float>::infinity());
        accum.zero_();
        accumLse.fill_(-std::numeric_limits<float>::infinity());
    }
    return {out, lse, accum, accumLse};
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

std::vector<at::Tensor> SplitBsndForward(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                         const at::Tensor& qv, float scale, int64_t windowLeft, int64_t windowRight,
                                         bool causal, int64_t splits, const std::optional<at::Tensor>& cacheSeqlens,
                                         const std::optional<at::Tensor>& usedQuery,
                                         const std::optional<at::Tensor>& schedulerMetadata, float softcap)
{
    TORCH_CHECK(std::isfinite(softcap) && softcap >= 0, "MLA: softcap must be finite and non-negative in FP32");
    CheckForwardBoundary(q, k, v, qv, false, false, false);
    TORCH_CHECK(std::isfinite(scale), "MLA: softmax_scale must be finite in FP32");
    TORCH_CHECK(!usedQuery || !schedulerMetadata,
                "MLA compressed forward is not implemented yet for Q-used with explicit metadata");
    const int64_t batch = q.size(0), sq = q.size(1), heads = q.size(2), sk = k.size(1);
    CheckUsedInputs(q, batch, usedQuery, cacheSeqlens);
    TORCH_CHECK(splits == 0 || (splits >= (softcap > 0 ? 1 : 2) && splits <= 256),
                "MLA: split requires num_splits=0 or [2,256]");
    uint32_t words;
    TORCH_CHECK(splits == 0 ? AutoSplitBsndMetadataWords(batch, words) : SplitBsndMetadataWords(batch, words),
                "MLA: split metadata word count overflow");
    MlaWindowConfig config{sq, sk, windowLeft, windowRight, causal};
    CheckWindow(config);
    if (schedulerMetadata) {
        TORCH_CHECK(schedulerMetadata->scalar_type() == at::kInt && schedulerMetadata->dim() == 1 &&
                        schedulerMetadata->numel() == words && schedulerMetadata->is_contiguous(),
                    "MLA: split scheduler_metadata must be contiguous int32 with the selected protocol size");
        TORCH_CHECK(schedulerMetadata->device() == q.device(),
                    "MLA: scheduler_metadata must be on the same NPU device");
    }
    const c10::OptionalDeviceGuard deviceGuard(device_of(q));
    // Explicit public records carry KV lengths. Only their None-length reuse
    // needs a matching array; private records can use the existing absent-KV
    // semantics to derive full capacity without a separate fill task.
    auto lengths = cacheSeqlens;
    if (schedulerMetadata && !lengths) {
        lengths = at::full({batch}, sk, q.options().dtype(at::kInt));
    }
    auto result = UsedForward(
        q, k, v, qv,
        {batch, heads, {MlaSequenceLayout::Bsnd, batch * sq, sq}, {MlaSequenceLayout::Bsnd, batch * sk, sk}},
        std::nullopt, std::nullopt, usedQuery, lengths, scale, false, schedulerMetadata, config, splits, softcap);
    if (splits == 1) {
        result[2] = at::empty({0}, at::TensorOptions().device(q.device()));
        result[3] = at::empty({0}, at::TensorOptions().device(q.device()));
    }
    return result;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
SplitBsndForwardOp(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& qv, int64_t splits,
                   int64_t left, int64_t right, bool causal, std::optional<double> scale,
                   const std::optional<at::Tensor>& lengths, const std::optional<at::Tensor>& usedQ,
                   const std::optional<at::Tensor>& metadata, double softcap)
{
    TORCH_CHECK(std::isfinite(softcap) && softcap >= 0 && std::isfinite(float(softcap)) &&
                    (softcap == 0 || float(softcap) > 0),
                "MLA: softcap must be finite and non-negative in FP32");
    auto r = SplitBsndForward(q, k, v, qv, scale ? float(*scale) : 1.0f / std::sqrt(576.0f), left, right, causal,
                              splits, lengths, usedQ, metadata, float(softcap));
    return {r[0], r[1], r[2], r[3]};
}

std::vector<at::Tensor> WindowBsndForward(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                          const at::Tensor& qv, float scale, int64_t windowLeft, int64_t windowRight,
                                          bool causal, const std::optional<at::Tensor>& cacheSeqlens,
                                          const std::optional<at::Tensor>& usedQuery,
                                          const std::optional<at::Tensor>& schedulerMetadata)
{
    CheckForwardBoundary(q, k, v, qv, false, false, false);
    TORCH_CHECK(std::isfinite(scale), "MLA: softmax_scale must be finite in FP32");
    TORCH_CHECK(!usedQuery || !schedulerMetadata,
                "MLA compressed forward is not implemented yet for Q-used with explicit metadata");
    const int64_t batch = q.size(0), sq = q.size(1), heads = q.size(2), sk = k.size(1);
    CheckUsedInputs(q, batch, usedQuery, cacheSeqlens);
    uint32_t words;
    TORCH_CHECK(WindowMetadataWords(batch, words), "MLA: window metadata word count overflow");
    MlaWindowConfig config{sq, sk, windowLeft, windowRight, causal};
    CheckWindow(config);
    if (schedulerMetadata) {
        TORCH_CHECK(schedulerMetadata->scalar_type() == at::kInt && schedulerMetadata->dim() == 1 &&
                        schedulerMetadata->numel() == words && schedulerMetadata->is_contiguous(),
                    "MLA: window scheduler_metadata must be contiguous int32[413 + 8 * batch_size]");
        TORCH_CHECK(schedulerMetadata->device() == q.device(),
                    "MLA: scheduler_metadata must be on the same NPU device");
    }
    const c10::OptionalDeviceGuard deviceGuard(device_of(q));
    // Explicit public records carry KV lengths. Only their None-length reuse
    // needs a matching array; private records can use the existing absent-KV
    // semantics to derive full capacity without a separate fill task.
    auto lengths = cacheSeqlens;
    if (schedulerMetadata && !lengths) {
        lengths = at::full({batch}, sk, q.options().dtype(at::kInt));
    }
    return UsedForward(
        q, k, v, qv,
        {batch, heads, {MlaSequenceLayout::Bsnd, batch * sq, sq}, {MlaSequenceLayout::Bsnd, batch * sk, sk}},
        std::nullopt, std::nullopt, usedQuery, lengths, scale, false, schedulerMetadata, config);
}

std::vector<at::Tensor> CausalBsndForward(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                          const at::Tensor& qv, float scale,
                                          const std::optional<at::Tensor>& cacheSeqlens,
                                          const std::optional<at::Tensor>& usedQuery,
                                          const std::optional<at::Tensor>& schedulerMetadata)
{
    CheckForwardBoundary(q, k, v, qv, false, false, false);
    TORCH_CHECK(std::isfinite(scale), "MLA: softmax_scale must be finite in FP32");
    TORCH_CHECK(!usedQuery || !schedulerMetadata,
                "MLA compressed forward is not implemented yet for Q-used with explicit metadata");
    const int64_t batch = q.size(0), sq = q.size(1), heads = q.size(2), sk = k.size(1);
    const uint32_t words = CheckUsedInputs(q, batch, usedQuery, cacheSeqlens);
    if (schedulerMetadata) {
        TORCH_CHECK(schedulerMetadata->scalar_type() == at::kInt && schedulerMetadata->dim() == 1 &&
                        schedulerMetadata->numel() == words && schedulerMetadata->is_contiguous(),
                    "MLA: causal scheduler_metadata must be contiguous int32[410 + 8 * batch_size]");
        TORCH_CHECK(schedulerMetadata->device() == q.device(),
                    "MLA: scheduler_metadata must be on the same NPU device");
    }
    const c10::OptionalDeviceGuard deviceGuard(device_of(q));
    // Explicit public records carry KV lengths. Only their None-length reuse
    // needs a matching array; private records can use the existing absent-KV
    // semantics to derive full capacity without a separate fill task.
    auto lengths = cacheSeqlens;
    if (schedulerMetadata && !lengths) {
        lengths = at::full({batch}, sk, q.options().dtype(at::kInt));
    }
    return UsedForward(
        q, k, v, qv,
        {batch, heads, {MlaSequenceLayout::Bsnd, batch * sq, sq}, {MlaSequenceLayout::Bsnd, batch * sk, sk}},
        std::nullopt, std::nullopt, usedQuery, lengths, scale, true, schedulerMetadata);
}

std::vector<at::Tensor> CausalMixedForward(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                           const at::Tensor& qv, const at::Tensor& cuQ, int64_t maxQ, float scale,
                                           const std::optional<at::Tensor>& cacheSeqlens,
                                           const std::optional<at::Tensor>& usedQuery,
                                           const std::optional<at::Tensor>& schedulerMetadata)
{
    CheckForwardBoundary(q, k, v, qv, true, false, false);
    TORCH_CHECK(std::isfinite(scale), "MLA: softmax_scale must be finite in FP32");
    CheckTndBsndForwardInputs(q, k, cuQ, maxQ, cacheSeqlens, std::nullopt);
    TORCH_CHECK(!usedQuery || !schedulerMetadata,
                "MLA compressed forward is not implemented yet for Q-used with explicit metadata");
    const int64_t batch = k.size(0), sk = k.size(1), heads = q.size(1);
    const uint32_t words = CheckUsedInputs(q, batch, usedQuery, cacheSeqlens);
    if (schedulerMetadata) {
        TORCH_CHECK(schedulerMetadata->scalar_type() == at::kInt && schedulerMetadata->dim() == 1 &&
                        schedulerMetadata->numel() == words && schedulerMetadata->is_contiguous(),
                    "MLA: causal scheduler_metadata must be contiguous int32[410 + 8 * batch_size]");
        TORCH_CHECK(schedulerMetadata->device() == q.device(),
                    "MLA: scheduler_metadata must be on the same NPU device");
    }
    const c10::OptionalDeviceGuard deviceGuard(device_of(q));
    auto lengths = cacheSeqlens;
    if (schedulerMetadata && !lengths) {
        lengths = at::full({batch}, sk, q.options().dtype(at::kInt));
    }
    return UsedForward(
        q, k, v, qv,
        {batch, heads, {MlaSequenceLayout::Tnd, q.size(0), maxQ}, {MlaSequenceLayout::Bsnd, batch * sk, sk}}, cuQ,
        std::nullopt, usedQuery, lengths, scale, true, schedulerMetadata);
}

std::tuple<at::Tensor, at::Tensor> CausalMixedForwardOp(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                                        const at::Tensor& qv, const at::Tensor& cuQ, int64_t maxQ,
                                                        std::optional<double> softmaxScale,
                                                        const std::optional<at::Tensor>& cacheSeqlens,
                                                        const std::optional<at::Tensor>& usedQuery,
                                                        const std::optional<at::Tensor>& schedulerMetadata)
{
    TORCH_CHECK(!softmaxScale ||
                    (std::isfinite(*softmaxScale) && std::abs(*softmaxScale) <= std::numeric_limits<float>::max()),
                "MLA: softmax_scale must be finite in FP32");
    auto result =
        CausalMixedForward(q, k, v, qv, cuQ, maxQ, softmaxScale ? float(*softmaxScale) : 1.0f / std::sqrt(576.0f),
                           cacheSeqlens, usedQuery, schedulerMetadata);
    return {result[0], result[1]};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
CausalMixedCacheOp(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& qv,
                   const at::Tensor& cuQ, int64_t maxQ, std::optional<double> softmaxScale,
                   const std::optional<at::Tensor>& cacheSeqlens, const std::optional<at::Tensor>& schedulerMetadata)
{
    auto result =
        CausalMixedForwardOp(q, k, v, qv, cuQ, maxQ, softmaxScale, cacheSeqlens, std::nullopt, schedulerMetadata);
    const auto options = at::TensorOptions().device(q.device());
    return {std::get<0>(result), std::get<1>(result), at::empty({0}, options), at::empty({0}, options)};
}

std::vector<at::Tensor> SplitMixedForward(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                          const at::Tensor& qv, const at::Tensor& cuQ, int64_t maxQ, float scale,
                                          int64_t windowLeft, int64_t windowRight, bool causal, int64_t splits,
                                          const std::optional<at::Tensor>& cacheSeqlens,
                                          const std::optional<at::Tensor>& usedQuery,
                                          const std::optional<at::Tensor>& schedulerMetadata, float softcap)
{
    TORCH_CHECK(splits == 0 || (splits >= (softcap > 0 ? 1 : 2) && splits <= 256),
                "MLA: mixed split requires 0 or [2,256], or 1 with softcap");
    TORCH_CHECK(std::isfinite(softcap) && softcap >= 0, "MLA: softcap must be finite and non-negative in FP32");
    CheckForwardBoundary(q, k, v, qv, true, false, false);
    TORCH_CHECK(std::isfinite(scale), "MLA: softmax_scale must be finite in FP32");
    CheckTndBsndForwardInputs(q, k, cuQ, maxQ, cacheSeqlens, std::nullopt);
    const int64_t batch = k.size(0), sk = k.size(1), heads = q.size(1);
    CheckUsedInputs(q, batch, usedQuery, cacheSeqlens);
    TORCH_CHECK(!usedQuery || !schedulerMetadata,
                "MLA compressed forward is not implemented yet for Q-used with explicit metadata");
    uint32_t words;
    TORCH_CHECK(splits == 0 ? AutoSplitMixedMetadataWords(batch, words) : SplitMixedMetadataWords(batch, words),
                "MLA: split metadata word count overflow");
    MlaWindowConfig config{maxQ, sk, windowLeft, windowRight, causal};
    CheckWindow(config);
    if (schedulerMetadata) {
        TORCH_CHECK(schedulerMetadata->scalar_type() == at::kInt && schedulerMetadata->dim() == 1 &&
                        schedulerMetadata->numel() == words && schedulerMetadata->is_contiguous(),
                    "MLA: split scheduler_metadata must be contiguous int32 with the selected protocol size");
        TORCH_CHECK(schedulerMetadata->device() == q.device(),
                    "MLA: scheduler_metadata must be on the same NPU device");
    }
    const c10::OptionalDeviceGuard deviceGuard(device_of(q));
    auto lengths = cacheSeqlens;
    if (schedulerMetadata && !lengths) {
        lengths = at::full({batch}, sk, q.options().dtype(at::kInt));
    }
    auto result = UsedForward(
        q, k, v, qv,
        {batch, heads, {MlaSequenceLayout::Tnd, q.size(0), maxQ}, {MlaSequenceLayout::Bsnd, batch * sk, sk}}, cuQ,
        std::nullopt, usedQuery, lengths, scale, false, schedulerMetadata, config, splits, softcap);
    if (splits == 1) {
        result[2] = at::empty({0}, at::TensorOptions().device(q.device()));
        result[3] = at::empty({0}, at::TensorOptions().device(q.device()));
    }
    return result;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
SplitMixedForwardOp(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& qv,
                    const at::Tensor& cuQ, int64_t maxQ, int64_t splits, int64_t left, int64_t right, bool causal,
                    std::optional<double> scale, const std::optional<at::Tensor>& lengths,
                    const std::optional<at::Tensor>& usedQ, const std::optional<at::Tensor>& metadata, double softcap)
{
    TORCH_CHECK(!scale || (std::isfinite(*scale) && std::abs(*scale) <= std::numeric_limits<float>::max()),
                "MLA: softmax_scale must be finite in FP32");
    TORCH_CHECK(std::isfinite(softcap) && softcap >= 0 && std::isfinite(float(softcap)) &&
                    (softcap == 0 || float(softcap) > 0),
                "MLA: softcap must be finite and non-negative in FP32");
    auto r = SplitMixedForward(q, k, v, qv, cuQ, maxQ, scale ? float(*scale) : 1.0f / std::sqrt(576.0f), left, right,
                               causal, splits, lengths, usedQ, metadata, float(softcap));
    return {r[0], r[1], r[2], r[3]};
}

std::vector<at::Tensor> WindowMixedForward(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                           const at::Tensor& qv, const at::Tensor& cuQ, int64_t maxQ, float scale,
                                           int64_t windowLeft, int64_t windowRight, bool causal,
                                           const std::optional<at::Tensor>& cacheSeqlens,
                                           const std::optional<at::Tensor>& usedQuery,
                                           const std::optional<at::Tensor>& schedulerMetadata)
{
    CheckForwardBoundary(q, k, v, qv, true, false, false);
    TORCH_CHECK(std::isfinite(scale), "MLA: softmax_scale must be finite in FP32");
    CheckTndBsndForwardInputs(q, k, cuQ, maxQ, cacheSeqlens, std::nullopt);
    TORCH_CHECK(!usedQuery || !schedulerMetadata,
                "MLA compressed forward is not implemented yet for Q-used with explicit metadata");
    const int64_t batch = k.size(0), sk = k.size(1), heads = q.size(1);
    CheckUsedInputs(q, batch, usedQuery, cacheSeqlens);
    uint32_t words;
    TORCH_CHECK(WindowMetadataWords(batch, words), "MLA: window metadata word count overflow");
    MlaWindowConfig config{maxQ, sk, windowLeft, windowRight, causal};
    CheckWindow(config);
    if (schedulerMetadata) {
        TORCH_CHECK(schedulerMetadata->scalar_type() == at::kInt && schedulerMetadata->dim() == 1 &&
                        schedulerMetadata->numel() == words && schedulerMetadata->is_contiguous(),
                    "MLA: window scheduler_metadata must be contiguous int32[413 + 8 * batch_size]");
        TORCH_CHECK(schedulerMetadata->device() == q.device(),
                    "MLA: scheduler_metadata must be on the same NPU device");
    }
    const c10::OptionalDeviceGuard deviceGuard(device_of(q));
    auto lengths = cacheSeqlens;
    if (schedulerMetadata && !lengths) {
        lengths = at::full({batch}, sk, q.options().dtype(at::kInt));
    }
    return UsedForward(
        q, k, v, qv,
        {batch, heads, {MlaSequenceLayout::Tnd, q.size(0), maxQ}, {MlaSequenceLayout::Bsnd, batch * sk, sk}}, cuQ,
        std::nullopt, usedQuery, lengths, scale, false, schedulerMetadata, config);
}

std::tuple<at::Tensor, at::Tensor>
WindowMixedForwardOp(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& qv,
                     const at::Tensor& cuQ, int64_t maxQ, int64_t windowLeft, int64_t windowRight, bool causal,
                     std::optional<double> softmaxScale, const std::optional<at::Tensor>& cacheSeqlens,
                     const std::optional<at::Tensor>& usedQuery, const std::optional<at::Tensor>& schedulerMetadata)
{
    TORCH_CHECK(!softmaxScale ||
                    (std::isfinite(*softmaxScale) && std::abs(*softmaxScale) <= std::numeric_limits<float>::max()),
                "MLA: softmax_scale must be finite in FP32");
    auto result =
        WindowMixedForward(q, k, v, qv, cuQ, maxQ, softmaxScale ? float(*softmaxScale) : 1.0f / std::sqrt(576.0f),
                           windowLeft, windowRight, causal, cacheSeqlens, usedQuery, schedulerMetadata);
    return {result[0], result[1]};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
WindowMixedCacheOp(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& qv,
                   const at::Tensor& cuQ, int64_t maxQ, int64_t windowLeft, int64_t windowRight, bool causal,
                   std::optional<double> softmaxScale, const std::optional<at::Tensor>& cacheSeqlens,
                   const std::optional<at::Tensor>& schedulerMetadata)
{
    auto result = WindowMixedForwardOp(q, k, v, qv, cuQ, maxQ, windowLeft, windowRight, causal, softmaxScale,
                                       cacheSeqlens, std::nullopt, schedulerMetadata);
    const auto options = at::TensorOptions().device(q.device());
    return {std::get<0>(result), std::get<1>(result), at::empty({0}, options), at::empty({0}, options)};
}

std::tuple<at::Tensor, at::Tensor> WindowBsndForwardOp(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                                       const at::Tensor& qv, int64_t windowLeft, int64_t windowRight,
                                                       bool causal, std::optional<double> softmaxScale,
                                                       const std::optional<at::Tensor>& cacheSeqlens,
                                                       const std::optional<at::Tensor>& usedQuery,
                                                       const std::optional<at::Tensor>& schedulerMetadata)
{
    TORCH_CHECK(!softmaxScale ||
                    (std::isfinite(*softmaxScale) && std::abs(*softmaxScale) <= std::numeric_limits<float>::max()),
                "MLA: softmax_scale must be finite in FP32");
    auto result = WindowBsndForward(q, k, v, qv, softmaxScale ? float(*softmaxScale) : 1.0f / std::sqrt(576.0f),
                                    windowLeft, windowRight, causal, cacheSeqlens, usedQuery, schedulerMetadata);
    return {result[0], result[1]};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
WindowBsndCacheOp(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& qv,
                  int64_t windowLeft, int64_t windowRight, bool causal, std::optional<double> softmaxScale,
                  const std::optional<at::Tensor>& cacheSeqlens, const std::optional<at::Tensor>& schedulerMetadata)
{
    auto result = WindowBsndForwardOp(q, k, v, qv, windowLeft, windowRight, causal, softmaxScale, cacheSeqlens,
                                      std::nullopt, schedulerMetadata);
    // Match torch.empty(0, device=q.device): each accumulator is independent
    // and uses the current default floating dtype, not the input dtype.
    const auto options = at::TensorOptions().device(q.device());
    return {std::get<0>(result), std::get<1>(result), at::empty({0}, options), at::empty({0}, options)};
}

std::tuple<at::Tensor, at::Tensor> CausalBsndForwardOp(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                                       const at::Tensor& qv, std::optional<double> softmaxScale,
                                                       const std::optional<at::Tensor>& cacheSeqlens,
                                                       const std::optional<at::Tensor>& usedQuery,
                                                       const std::optional<at::Tensor>& schedulerMetadata)
{
    TORCH_CHECK(!softmaxScale ||
                    (std::isfinite(*softmaxScale) && std::abs(*softmaxScale) <= std::numeric_limits<float>::max()),
                "MLA: softmax_scale must be finite in FP32");
    auto result = CausalBsndForward(q, k, v, qv, softmaxScale ? float(*softmaxScale) : 1.0f / std::sqrt(576.0f),
                                    cacheSeqlens, usedQuery, schedulerMetadata);
    return {result[0], result[1]};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
CausalBsndCacheOp(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& qv,
                  std::optional<double> softmaxScale, const std::optional<at::Tensor>& cacheSeqlens,
                  const std::optional<at::Tensor>& schedulerMetadata)
{
    auto result = CausalBsndForwardOp(q, k, v, qv, softmaxScale, cacheSeqlens, std::nullopt, schedulerMetadata);
    // Match torch.empty(0, device=q.device): each accumulator is independent
    // and uses the current default floating dtype, not the input dtype.
    const auto options = at::TensorOptions().device(q.device());
    return {std::get<0>(result), std::get<1>(result), at::empty({0}, options), at::empty({0}, options)};
}

namespace {
MlaUsedShape CheckSplitTnd(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& qv,
                           const at::Tensor& cuQ, const at::Tensor& cuK, int64_t maxQ, int64_t maxK, int64_t splits,
                           const std::optional<at::Tensor>& usedQ, const std::optional<at::Tensor>& usedK,
                           bool hasSoftcap)
{
    CheckForwardBoundary(q, k, v, qv, true, true, false);
    CheckTndInputs(q, k, cuQ, cuK, maxQ, maxK);
    CheckUsedInputs(q, cuQ.numel() - 1, usedQ, usedK);
    TORCH_CHECK(splits == 0 || (splits >= (hasSoftcap ? 1 : 2) && splits <= 256),
                "MLA: TND split requires 0 or [2,256], or 1 with softcap");
    return {cuQ.numel() - 1,
            q.size(1),
            {MlaSequenceLayout::Tnd, q.size(0), maxQ},
            {MlaSequenceLayout::Tnd, k.size(0), maxK}};
}
} // namespace

std::vector<at::Tensor> SplitTndForward(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                        const at::Tensor& qv, const at::Tensor& cuQ, const at::Tensor& cuK,
                                        int64_t maxQ, int64_t maxK, float scale, int64_t left, int64_t right,
                                        bool causal, int64_t splits, const std::optional<at::Tensor>& usedQ,
                                        const std::optional<at::Tensor>& usedK,
                                        const std::optional<at::Tensor>& metadata, float softcap)
{
    const auto shape = CheckSplitTnd(q, k, v, qv, cuQ, cuK, maxQ, maxK, splits, usedQ, usedK, softcap > 0);
    TORCH_CHECK(std::isfinite(softcap) && softcap >= 0, "MLA: softcap must be finite and non-negative in FP32");
    TORCH_CHECK(std::isfinite(scale), "MLA: softmax_scale must be finite in FP32");
    MlaWindowConfig window{maxQ, maxK, left, right, causal};
    CheckWindow(window);
    uint32_t words;
    TORCH_CHECK(splits == 0 ? AutoSplitTndMetadataWords(shape.batch, words) : SplitTndMetadataWords(shape.batch, words),
                "MLA: split metadata word count overflow");
    if (metadata) {
        TORCH_CHECK(metadata->scalar_type() == at::kInt && metadata->dim() == 1 && metadata->numel() == words &&
                        metadata->is_contiguous() && metadata->device() == q.device(),
                    "MLA: TND split metadata must be same-device contiguous int32 with the selected protocol size");
    }
    const c10::OptionalDeviceGuard guard(device_of(q));
    auto result =
        UsedForward(q, k, v, qv, shape, cuQ, cuK, usedQ, usedK, scale, false, metadata, window, splits, softcap);
    if (splits == 1) {
        result[2] = at::empty({0}, at::TensorOptions().device(q.device()));
        result[3] = at::empty({0}, at::TensorOptions().device(q.device()));
    }
    return result;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
SplitTndForwardOp(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& qv,
                  const at::Tensor& cuQ, const at::Tensor& cuK, int64_t maxQ, int64_t maxK, int64_t splits,
                  int64_t left, int64_t right, bool causal, std::optional<double> scale,
                  const std::optional<at::Tensor>& usedQ, const std::optional<at::Tensor>& usedK,
                  const std::optional<at::Tensor>& metadata, double softcap)
{
    TORCH_CHECK(!scale || (std::isfinite(*scale) && std::abs(*scale) <= std::numeric_limits<float>::max()),
                "MLA: softmax_scale must be finite in FP32");
    TORCH_CHECK(std::isfinite(softcap) && softcap >= 0 && std::isfinite(float(softcap)) &&
                    (softcap == 0 || float(softcap) > 0),
                "MLA: softcap must be finite and non-negative in FP32");
    auto r = SplitTndForward(q, k, v, qv, cuQ, cuK, maxQ, maxK, scale ? float(*scale) : 1.0f / std::sqrt(576.0f), left,
                             right, causal, splits, usedQ, usedK, metadata, float(softcap));
    return {r[0], r[1], r[2], r[3]};
}

// Internal producer for controlled reuse/testing. Public varlen still creates
// its own record per call, as its fixed FA3 signature has no metadata argument.
at::Tensor SplitTndMetadataOp(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& qv,
                              const at::Tensor& cuQ, const at::Tensor& cuK, int64_t maxQ, int64_t maxK, int64_t splits,
                              int64_t left, int64_t right, bool causal, const std::optional<at::Tensor>& usedQ,
                              const std::optional<at::Tensor>& usedK, bool hasSoftcap)
{
    const auto shape = CheckSplitTnd(q, k, v, qv, cuQ, cuK, maxQ, maxK, splits, usedQ, usedK, hasSoftcap);
    MlaWindowConfig window{maxQ, maxK, left, right, causal};
    CheckWindow(window);
    const c10::OptionalDeviceGuard guard(device_of(q));
    uint32_t words;
    TORCH_CHECK(splits == 0 ? AutoSplitTndMetadataWords(shape.batch, words) : SplitTndMetadataWords(shape.batch, words),
                "MLA: split metadata word count overflow");
    auto metadata = at::empty({words}, q.options().dtype(at::kInt));
    const auto arr = [&](const std::optional<at::Tensor>& t) {
        return t ? MlaUsedMetadataArray{reinterpret_cast<uint64_t>(t->numel() ? t->data_ptr() : metadata.data_ptr()),
                                        uint64_t(t->numel())}
                 : MlaUsedMetadataArray{};
    };
    MlaUsedMetadataArgs args{
        reinterpret_cast<uint64_t>(metadata.data_ptr()),
        words,
        uint32_t(q.scalar_type() == at::kBFloat16 ? MlaMetadataDtype::Bf16 : MlaMetadataDtype::Fp16),
        shape,
        CausalPlatform(q.device().index(), shape.batch && q.size(0) && k.size(0)),
        arr(cuQ),
        arr(cuK),
        arr(usedQ),
        arr(usedK)};
    auto producer = PrepareSubmission(metadata, args, cuK, cuQ, true, usedQ, usedK, true);
    producer.window = window;
    producer.splits = splits;
    producer.hasSoftcap = hasSoftcap;
    RunCausalLaunch("ascendc_mla_split_tnd_metadata", [producer]() mutable -> int {
        producer.Enqueue();
        return 0;
    });
    return metadata;
}

std::vector<at::Tensor> WindowTndForward(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                         const at::Tensor& qv, const at::Tensor& cuQ, const at::Tensor& cuK,
                                         int64_t maxQ, int64_t maxK, float scale, int64_t windowLeft,
                                         int64_t windowRight, bool causal, const std::optional<at::Tensor>& usedQuery,
                                         const std::optional<at::Tensor>& usedKv)
{
    CheckTndInputs(q, k, cuQ, cuK, maxQ, maxK);
    MlaWindowConfig config{maxQ, maxK, windowLeft, windowRight, causal};
    CheckWindow(config);
    return UsedForward(q, k, v, qv,
                       {cuQ.numel() - 1,
                        q.size(1),
                        {MlaSequenceLayout::Tnd, q.size(0), maxQ},
                        {MlaSequenceLayout::Tnd, k.size(0), maxK}},
                       cuQ, cuK, usedQuery, usedKv, scale, false, std::nullopt, config);
}

std::vector<at::Tensor> TndForward(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& qv,
                                   const at::Tensor& cuQ, const at::Tensor& cuK, int64_t maxQ, int64_t maxK,
                                   float scale, const std::optional<at::Tensor>& usedQuery,
                                   const std::optional<at::Tensor>& usedKv, bool causal)
{
    const uint32_t words = CheckTndInputs(q, k, cuQ, cuK, maxQ, maxK);
    const int64_t batch = cuQ.numel() - 1;
    if (causal || usedQuery || usedKv) {
        return UsedForward(
            q, k, v, qv,
            {batch, q.size(1), {MlaSequenceLayout::Tnd, q.size(0), maxQ}, {MlaSequenceLayout::Tnd, k.size(0), maxK}},
            cuQ, cuK, usedQuery, usedKv, scale, causal);
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

std::tuple<at::Tensor, at::Tensor>
WindowTndForwardOp(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& qv,
                   const at::Tensor& cuQ, const at::Tensor& cuK, int64_t maxQ, int64_t maxK, int64_t windowLeft,
                   int64_t windowRight, bool causal, std::optional<double> softmaxScale,
                   const std::optional<at::Tensor>& usedQuery, const std::optional<at::Tensor>& usedKv)
{
    CheckForwardBoundary(q, k, v, qv, true, true, false);
    TORCH_CHECK(!softmaxScale ||
                    (std::isfinite(*softmaxScale) && std::abs(*softmaxScale) <= std::numeric_limits<float>::max()),
                "MLA: softmax_scale must be finite in FP32");
    const c10::OptionalDeviceGuard deviceGuard(device_of(q));
    auto result = WindowTndForward(q, k, v, qv, cuQ, cuK, maxQ, maxK,
                                   softmaxScale ? float(*softmaxScale) : 1.0f / std::sqrt(576.0f), windowLeft,
                                   windowRight, causal, usedQuery, usedKv);
    return {result[0], result[1]};
}

template <bool IsCausal = false>
std::tuple<at::Tensor, at::Tensor>
TndForwardOp(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& qv, const at::Tensor& cuQ,
             const at::Tensor& cuK, int64_t maxQ, int64_t maxK, std::optional<double> softmaxScale,
             const std::optional<at::Tensor>& usedQuery, const std::optional<at::Tensor>& usedKv)
{
    CheckForwardBoundary(q, k, v, qv, true, true, false);
    TORCH_CHECK(!softmaxScale ||
                    (std::isfinite(*softmaxScale) && std::abs(*softmaxScale) <= std::numeric_limits<float>::max()),
                "MLA: softmax_scale must be finite in FP32");
    const c10::OptionalDeviceGuard deviceGuard(device_of(q));
    auto result =
        TndForward(q, k, v, qv, cuQ, cuK, maxQ, maxK, softmaxScale ? float(*softmaxScale) : 1.0f / std::sqrt(576.0f),
                   usedQuery, usedKv, IsCausal);
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
namespace {
MlaPrefillBsndContext PrefillContext(int64_t batch, int64_t sq, int64_t sk, int64_t heads, int64_t kvHeads,
                                     at::ScalarType dtype, int64_t splits, int64_t left, int64_t right, bool causal,
                                     uint32_t flags, const MlaPlatformInfo& platform, uint32_t tnd = 0,
                                     uint32_t totalQ = 0, uint32_t totalK = 0, int64_t pageSize = 0,
                                     bool hasSoftcap = false)
{
    TORCH_CHECK(FitsDenseShape(tnd ? 0 : batch, sq, heads, sk) && kvHeads > 0 && heads % kvHeads == 0 && splits >= 0 &&
                    splits <= 256,
                "MLA: prefill BSND shape/split bounds exceeded");
    TORCH_CHECK(dtype == at::kHalf || dtype == at::kBFloat16, "MLA: qkv_dtype must be FP16 or BF16");
    TORCH_CHECK(pageSize == 0 ||
                    (tnd != 1 && pageSize >= 16 && pageSize <= 1024 && pageSize % 16 == 0 && sk % pageSize == 0),
                "MLA: prefill paged capacity/page size invalid");
    MlaWindow window{};
    TORCH_CHECK(NormalizeWindow({sq, sk, left, right, causal}, window) == MlaCausalStatus::Success,
                "MLA: prefill window bounds exceeded");
    MlaPrefillBsndContext c{uint32_t(batch),
                            uint32_t(sq),
                            uint32_t(sk),
                            uint32_t(heads),
                            uint32_t(kvHeads),
                            uint32_t(splits),
                            uint32_t(dtype == at::kHalf ? MlaMetadataDtype::Fp16 : MlaMetadataDtype::Bf16),
                            platform.aicNum,
                            flags,
                            {uint32_t(window.kind), window.left, window.right},
                            0,
                            uint32_t(tnd),
                            totalQ,
                            totalK,
                            uint32_t(pageSize)};
    c.hasSoftcap = hasSoftcap;
    MlaPrefillBsndLayout l{};
    TORCH_CHECK(PrefillBsndLayout(c, l), "MLA: prefill metadata/workspace/accumulator overflow");
    TORCH_CHECK(l.bytes <= uint64_t(INT64_MAX) - platform.libApiWorkspaceBytes, "MLA: workspace overflow");
    c.workspaceBytes = l.bytes;
    return c;
}
uint64_t PrefillPointer(const at::Tensor& tensor)
{
    return reinterpret_cast<uint64_t>(tensor.data_ptr());
}
MlaPrefillBsndMetadataArgs PrefillMetadataArgs(const MlaPrefillBsndContext& c, const at::Tensor& metadata,
                                               const std::optional<at::Tensor>& usedQ,
                                               const std::optional<at::Tensor>& usedK)
{
    return {c, usedQ ? uint64_t(usedQ->data_ptr()) : 0, usedK ? uint64_t(usedK->data_ptr()) : 0,
            uint64_t(metadata.data_ptr()), uint32_t(metadata.numel())};
}
void RecordPrefill(const at::Tensor& tensor, const c10_npu::NPUStream& stream)
{
    if (tensor.numel())
        c10_npu::NPUCachingAllocator::recordStream(tensor.storage().data_ptr(), stream);
}
} // namespace

namespace {
MlaGqaContext GqaContext(int64_t batch, int64_t sq, int64_t sk, int64_t heads, int64_t kvHeads, int64_t valueDim,
                         at::ScalarType dtype, int64_t splits, int64_t left, int64_t right, bool causal, uint32_t flags,
                         const MlaPlatformInfo& platform, uint32_t layout, int64_t totalQ, int64_t totalK,
                         int64_t pageSize, bool hasSoftcap, std::optional<bool> pack, int64_t margin)
{
    TORCH_CHECK(FitsDenseShape(0, sq, heads, sk) && batch >= 0 && uint64_t(batch) <= UINT32_MAX && kvHeads > 0 &&
                    heads % kvHeads == 0 && splits >= 0 && splits <= 256 && totalQ >= 0 && totalQ <= INT32_MAX &&
                    totalK >= 0 && totalK <= INT32_MAX && pageSize >= 0 && pageSize <= 1024 &&
                    (valueDim == 128 || valueDim == 512),
                "MLA: GQA shape/split bounds exceeded");
    TORCH_CHECK(dtype == at::kHalf || dtype == at::kBFloat16, "MLA: qkv_dtype must be FP16 or BF16");
    MlaGqaResources resources{};
    TORCH_CHECK(ResolveGqaResources(heads, kvHeads, {pack ? int64_t(*pack) : -1, margin}, platform, resources) ==
                    MlaScheduleStatus::Success,
                "MLA: invalid pack_gqa or sm_margin; sm_margin must leave at least one physical AIC");
    TORCH_CHECK(resources.platform.blockDims[resources.platform.aicNum - 1] == resources.platform.aicNum,
                "MLA: unsupported GQA C1V2 blockDim mapping");
    MlaWindow window{};
    TORCH_CHECK(NormalizeWindow({sq, sk, left, right, causal}, window) == MlaCausalStatus::Success,
                "MLA: GQA window bounds exceeded");
    MlaGqaContext c{uint32_t(batch),
                    uint32_t(sq),
                    uint32_t(sk),
                    uint32_t(heads),
                    uint32_t(kvHeads),
                    uint32_t(splits),
                    uint32_t(dtype == at::kHalf ? MlaMetadataDtype::Fp16 : MlaMetadataDtype::Bf16),
                    platform.aicNum,
                    uint32_t(margin),
                    uint32_t(resources.packed),
                    layout,
                    uint32_t(totalQ),
                    uint32_t(totalK),
                    uint32_t(pageSize),
                    flags,
                    uint32_t(valueDim),
                    uint32_t(hasSoftcap),
                    {uint32_t(window.kind), window.left, window.right}};
    MlaGqaLayout l{};
    TORCH_CHECK(GqaLayout(c, l), "MLA: GQA metadata/workspace/accumulator bounds exceeded");
    TORCH_CHECK(l.bytes <= uint64_t(INT64_MAX) - platform.libApiWorkspaceBytes, "MLA: GQA workspace overflow");
    // Reuse the accepted C1V2 resource check. Platform capacities exclude a
    // runtime reserve, so nominal chip capacities are not valid lower bounds.
    MlaSchedule resourceCheck{};
    TORCH_CHECK(BuildDenseSchedule({1, 1, 1, 1}, platform, resourceCheck) == MlaScheduleStatus::Success,
                "MLA: invalid GQA C1V2 resources");
    c.workspaceBytes = l.bytes;
    return c;
}
void CheckGqaRecord(const at::Tensor& record, c10::Device device, uint32_t words)
{
    TORCH_CHECK(record.scalar_type() == at::kInt && record.device() == device && record.dim() == 1 &&
                    record.numel() == words && record.is_contiguous(),
                "MLA: GQA scheduler_metadata must be same-device contiguous int32 with matching word count");
}
} // namespace

at::Tensor GqaMetadataOp(int64_t batch, int64_t sq, int64_t sk, int64_t heads, int64_t kvHeads, int64_t valueDim,
                         at::ScalarType dtype, const at::Tensor& lengths, const std::optional<at::Tensor>& cuQ,
                         int64_t splits, int64_t left, int64_t right, bool causal, int64_t pageSize, bool hasSoftcap,
                         std::optional<bool> pack, int64_t margin, const std::optional<at::Tensor>& usedQ)
{
    CheckCacheLengths(lengths, batch, lengths.device());
    TORCH_CHECK(!usedQ || (cuQ && valueDim == 128), "MLA: Q-used metadata requires prefill TND Q/cache");
    CheckUsedInputs(lengths, batch, usedQ, std::nullopt);
    if (cuQ)
        CheckMetadataQueryPrefix(*cuQ, batch, lengths.device());
    const c10::OptionalDeviceGuard guard(device_of(lengths));
    const auto platform = CausalPlatform(lengths.device().index(), false);
    const auto c = GqaContext(batch, sq, sk, heads, kvHeads, valueDim, dtype, splits, left, right, causal,
                              2 | uint32_t(bool(usedQ)), platform, pageSize ? (cuQ ? 4 : 3) : (cuQ ? 2 : 0), 0, 0,
                              pageSize, hasSoftcap, pack, margin);
    MlaGqaLayout l{};
    GqaLayout(c, l);
    auto metadata = at::empty({l.words}, lengths.options());
    MlaGqaMetadataArgs args{c,
                            cuQ ? PrefillPointer(*cuQ) : 0,
                            0,
                            usedQ ? PrefillPointer(*usedQ) : 0,
                            PrefillPointer(lengths),
                            PrefillPointer(metadata),
                            l.words,
                            uint32_t(bool(cuQ))};
    auto stream = c10_npu::getCurrentNPUStream();
    RecordPrefill(metadata, stream);
    RecordPrefill(lengths, stream);
    if (usedQ)
        RecordPrefill(*usedQ, stream);
    if (cuQ)
        RecordPrefill(*cuQ, stream);
    RunCausalLaunch("ascendc_mla_gqa_metadata", [metadata, lengths, cuQ, usedQ, args, stream]() mutable {
        ComputeMlaGqaMetadata<<<1, nullptr, stream.stream(false)>>>(&args, sizeof(args));
        return 0;
    });
    return metadata;
}

std::vector<at::Tensor> GqaForward(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                   const std::optional<at::Tensor>& qv, int64_t splits, int64_t left, int64_t right,
                                   bool causal, std::optional<double> scaleOption,
                                   const std::optional<at::Tensor>& usedQ, const std::optional<at::Tensor>& usedK,
                                   const std::optional<at::Tensor>& supplied, const std::optional<at::Tensor>& cuQ,
                                   const std::optional<at::Tensor>& cuK, int64_t maxQ, int64_t maxK,
                                   const std::optional<at::Tensor>& pageTable, double softcap, std::optional<bool> pack,
                                   int64_t margin)
{
    const bool tnd = bool(cuQ);
    TORCH_CHECK(CheckForwardBoundary(q, k, v, qv, tnd, bool(cuK), bool(pageTable)),
                "MLA: GQA entry requires compressed 64/512 or prefill 192/128 tensors");
    const float scale = scaleOption.value_or(1.0 / std::sqrt(qv ? 576.0 : 192.0));
    TORCH_CHECK(std::isfinite(scale), "MLA: softmax_scale must be finite in FP32");
    TORCH_CHECK(std::isfinite(float(softcap)) && softcap >= 0 && (!softcap || float(softcap) > 0),
                "MLA: softcap must be finite and non-negative in FP32");
    TORCH_CHECK(!usedQ || !supplied || (tnd && (!qv || cuK)),
                "MLA: Q-used with explicit metadata is not implemented yet");
    if (cuK)
        CheckTndInputs(q, k, *cuQ, *cuK, maxQ, maxK);
    else if (tnd && !pageTable)
        CheckTndBsndForwardInputs(q, k, *cuQ, maxQ, usedK, std::nullopt);
    else if (tnd) {
        TORCH_CHECK(cuQ->dim() == 1 && cuQ->numel() >= 1, "MLA: cu_seqlens_q must have shape [batch_size + 1]");
        CheckMetadataQueryPrefix(*cuQ, cuQ->numel() - 1, q.device());
    }
    const int64_t batch = tnd ? cuQ->numel() - 1 : q.size(0);
    CheckUsedInputs(q, batch, usedQ, usedK);
    MlaPagedCacheLayout paged{};
    if (pageTable) {
        TORCH_CHECK(pageTable->dim() == 2 && pageTable->size(0) == batch && pageTable->scalar_type() == at::kInt &&
                        pageTable->device() == q.device(),
                    "MLA: page_table must be same-device int32[batch_size, max_pages]");
        TORCH_CHECK(BuildPagedCacheLayout(PagedTensorView<MlaPageTableView>(*pageTable), PagedTensorView(k),
                                          PagedTensorView(v), paged, !qv) == MlaLayoutStatus::Success,
                    "MLA: paged cache exceeds shape, stride, storage or signed byte bounds");
    }
    const int64_t kvCapacity = pageTable ? int64_t(paged.logicalCapacity) : cuK ? maxK : k.size(1);
    const c10::OptionalDeviceGuard guard(device_of(q));
    const auto currentK = !cuK && supplied && !usedK
                              ? std::optional<at::Tensor>(at::full({batch}, kvCapacity, q.options().dtype(at::kInt)))
                              : usedK;
    const auto platform = CausalPlatform(q.device().index(), false);
    const auto c =
        GqaContext(batch, tnd ? maxQ : q.size(1), kvCapacity, q.size(-2), k.size(-2), v.size(-1), q.scalar_type(),
                   splits, left, right, causal, uint32_t(bool(usedQ)) | (uint32_t(bool(currentK)) << 1), platform,
                   pageTable ? (tnd ? 4 : 3)
                   : cuK     ? 1
                   : tnd     ? 2
                             : 0,
                   tnd ? q.size(0) : 0, cuK ? k.size(0) : 0, pageTable ? k.size(1) : 0, softcap > 0, pack, margin);
    MlaGqaLayout l{};
    GqaLayout(c, l);
    if (supplied)
        CheckGqaRecord(*supplied, q.device(), l.words);
    const int64_t b = c.batch, sq = c.sq, h = c.heads, dim = c.valueDim, capacity = splits ? splits : 26;
    auto out = tnd ? at::empty({c.totalQ, h, dim}, q.options()) : at::empty({b, sq, h, dim}, q.options());
    auto lse = tnd ? at::empty({h, c.totalQ}, q.options().dtype(at::kFloat))
                   : at::empty({b, h, sq}, q.options().dtype(at::kFloat));
    const auto emptyOptions = at::TensorOptions().device(q.device());
    auto accum = splits == 1 ? at::empty({0}, emptyOptions)
                 : tnd       ? at::empty({capacity, h, c.totalQ, dim}, q.options().dtype(at::kFloat))
                             : at::empty({capacity, b, h, sq, dim}, q.options().dtype(at::kFloat));
    auto accumLse = splits == 1 ? at::empty({0}, emptyOptions)
                    : tnd       ? at::empty({capacity, h, c.totalQ}, q.options().dtype(at::kFloat))
                                : at::empty({capacity, b, h, sq}, q.options().dtype(at::kFloat));
    auto workspace = at::empty({int64_t(l.bytes + platform.libApiWorkspaceBytes)}, q.options().dtype(at::kByte));
    auto metadata = supplied ? *supplied : at::empty({l.words}, q.options().dtype(at::kInt));
    MlaGqaMetadataArgs args{c,
                            cuQ ? PrefillPointer(*cuQ) : 0,
                            cuK ? PrefillPointer(*cuK) : 0,
                            usedQ ? PrefillPointer(*usedQ) : 0,
                            currentK ? PrefillPointer(*currentK) : 0,
                            PrefillPointer(metadata),
                            l.words};
    auto stream = c10_npu::getCurrentNPUStream();
    for (const auto& tensor : {q, k, v, out, lse, accum, accumLse, workspace, metadata})
        RecordPrefill(tensor, stream);
    for (const auto& tensor : {qv, cuQ, cuK, usedQ, currentK, pageTable})
        if (tensor)
            RecordPrefill(*tensor, stream);
    MlaGqaLaunchArgs a{};
    a.stream = stream.stream(false);
    uint32_t fftsLength = 0;
    TORCH_CHECK(rtGetC2cCtrlAddr(&a.ffts, &fftsLength) == RT_ERROR_NONE, "MLA: FFTS address unavailable");
    a.q = PrefillPointer(q);
    a.qv = qv ? PrefillPointer(*qv) : 0;
    a.k = PrefillPointer(k);
    a.v = PrefillPointer(v);
    a.out = PrefillPointer(out);
    a.lse = PrefillPointer(lse);
    a.accum = PrefillPointer(accum);
    a.accumLse = PrefillPointer(accumLse);
    a.workspace = PrefillPointer(workspace) + platform.libApiWorkspaceBytes;
    a.metadata = PrefillPointer(metadata);
    a.cuQuery = args.cuQuery;
    a.cuKv = args.cuKv;
    a.usedQuery = args.usedQuery;
    a.usedKv = args.usedKv;
    a.context = c;
    a.inputs = {(tnd ? CheckTndTensorLayout(q) : CheckBsndTensorLayout(q)),
                qv ? (tnd ? CheckTndTensorLayout(*qv) : CheckBsndTensorLayout(*qv)) : MlaBsndTensorLayout{},
                (cuK ? CheckTndTensorLayout(k) : CheckBsndTensorLayout(k)),
                (cuK ? CheckTndTensorLayout(v) : CheckBsndTensorLayout(v))};
    if (pageTable)
        a.cache = {paged, PagedTensorArray(*pageTable, paged.tableSpanElements),
                   PagedTensorArray(k, paged.k.spanElements), PagedTensorArray(v, paged.v.spanElements)};
    a.scale = scale;
    a.softcap = softcap;
    RunCausalLaunch("ascendc_mla_gqa", [a, args, supplied, q, qv, k, v, out, lse, accum, accumLse, workspace, metadata,
                                        usedQ, currentK, cuQ, cuK, pageTable]() mutable {
        if (!supplied)
            ComputeMlaGqaMetadata<<<1, nullptr, a.stream>>>(&args, sizeof(args));
        const auto error = q.scalar_type() == at::kHalf ? LaunchGqaFp16(a) : LaunchGqaBf16(a);
        TORCH_CHECK(error == ACL_SUCCESS, "MLA: GQA BATCH_MODE launch failed (", error, ")");
        return 0;
    });
    return {out, lse, accum, accumLse};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
GqaForwardOp(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const std::optional<at::Tensor>& qv,
             int64_t splits, int64_t left, int64_t right, bool causal, std::optional<double> scale,
             const std::optional<at::Tensor>& usedQ, const std::optional<at::Tensor>& usedK,
             const std::optional<at::Tensor>& metadata, const std::optional<at::Tensor>& cuQ,
             const std::optional<at::Tensor>& cuK, int64_t maxQ, int64_t maxK, const std::optional<at::Tensor>& pages,
             double cap, std::optional<bool> pack, int64_t margin)
{
    const auto r = GqaForward(q, k, v, qv, splits, left, right, causal, scale, usedQ, usedK, metadata, cuQ, cuK, maxQ,
                              maxK, pages, cap, pack, margin);
    return {r[0], r[1], r[2], r[3]};
}

at::Tensor PrefillBsndMetadataOp(int64_t batch, int64_t sq, int64_t sk, int64_t heads, int64_t kvHeads,
                                 at::ScalarType dtype, const at::Tensor& lengths, int64_t splits, int64_t left,
                                 int64_t right, bool causal, int64_t pageSize, bool hasSoftcap)
{
    CheckCacheLengths(lengths, batch, lengths.device());
    const c10::OptionalDeviceGuard guard(device_of(lengths));
    const auto platform = CausalPlatform(lengths.device().index(), true);
    const auto c = PrefillContext(batch, sq, sk, heads, kvHeads, dtype, splits, left, right, causal, 2, platform, 0, 0,
                                  0, pageSize, hasSoftcap);
    MlaPrefillBsndLayout l{};
    PrefillBsndLayout(c, l);
    auto metadata = at::empty({l.words}, lengths.options());
    auto args = PrefillMetadataArgs(c, metadata, std::nullopt, lengths);
    auto stream = c10_npu::getCurrentNPUStream();
    RecordPrefill(lengths, stream);
    RecordPrefill(metadata, stream);
    RunCausalLaunch("ascendc_mla_prefill_metadata", [metadata, lengths, args, stream]() mutable {
        ComputeMlaPrefillBsndMetadata<<<1, nullptr, stream.stream(false)>>>(&args, sizeof(args));
        return 0;
    });
    return metadata;
}

std::vector<at::Tensor>
PrefillBsndForward(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, float scale, int64_t splits,
                   int64_t left, int64_t right, bool causal, const std::optional<at::Tensor>& usedQ,
                   const std::optional<at::Tensor>& usedK, const std::optional<at::Tensor>& supplied,
                   const std::optional<at::Tensor>& cuQ, const std::optional<at::Tensor>& cuK, int64_t maxQ,
                   int64_t maxK, const std::optional<at::Tensor>& pageTable, float softcap)
{
    TORCH_CHECK(std::isfinite(softcap) && softcap >= 0, "MLA: softcap must be finite and non-negative in FP32");
    const bool tnd = bool(cuQ);
    CheckForwardBoundary(q, k, v, std::nullopt, tnd, bool(cuK), bool(pageTable));
    TORCH_CHECK(q.size(-1) == 192 && v.size(-1) == 128, "MLA: prefill requires 192/128 tensors");
    TORCH_CHECK(std::isfinite(scale), "MLA: softmax_scale must be finite in FP32");
    TORCH_CHECK(tnd || !usedQ || !supplied, "MLA: prefill Q-used with explicit metadata is not implemented yet");
    if (cuK)
        CheckTndInputs(q, k, *cuQ, *cuK, maxQ, maxK);
    else if (tnd && !pageTable)
        CheckTndBsndForwardInputs(q, k, *cuQ, maxQ, usedK, std::nullopt);
    else if (tnd) {
        TORCH_CHECK(cuQ->dim() == 1 && cuQ->numel() >= 1, "MLA: cu_seqlens_q must have shape [batch_size + 1]");
        CheckMetadataQueryPrefix(*cuQ, cuQ->numel() - 1, q.device());
        TORCH_CHECK(q.size(0) <= INT32_MAX && uint64_t(q.size(0)) * q.size(1) <= kMlaMaxBatchRows,
                    "MLA: compact TND shape bounds exceeded");
    }
    const int64_t batch = tnd ? cuQ->numel() - 1 : q.size(0);
    CheckUsedInputs(q, batch, usedQ, usedK);
    MlaPagedCacheLayout paged{};
    if (pageTable) {
        TORCH_CHECK(pageTable->dim() == 2 && pageTable->size(0) == batch && pageTable->scalar_type() == at::kInt &&
                        pageTable->device() == q.device(),
                    "MLA: page_table must be same-device int32[batch_size, max_pages]");
        TORCH_CHECK(BuildPagedCacheLayout(PagedTensorView<MlaPageTableView>(*pageTable), PagedTensorView(k),
                                          PagedTensorView(v), paged, true) == MlaLayoutStatus::Success,
                    "MLA: paged cache exceeds shape, stride, storage or signed byte bounds");
    }
    const int64_t kvCapacity = pageTable ? int64_t(paged.logicalCapacity) : k.size(1);
    const c10::OptionalDeviceGuard guard(device_of(q));
    // Explicit metadata snapshots must still be checked against this call's
    // full capacity when cache_seqlens is omitted.
    const auto currentK = !cuK && supplied && !usedK
                              ? std::optional<at::Tensor>(at::full({batch}, kvCapacity, q.options().dtype(at::kInt)))
                              : usedK;
    auto platform = CausalPlatform(q.device().index(), true);
    auto c = PrefillContext(batch, tnd ? maxQ : q.size(1), cuK ? maxK : kvCapacity, q.size(-2), k.size(-2),
                            q.scalar_type(), splits, left, right, causal,
                            uint32_t(bool(usedQ)) | (uint32_t(bool(currentK)) << 1), platform, tnd ? (cuK ? 1 : 2) : 0,
                            tnd ? q.size(0) : 0, cuK ? k.size(0) : 0, pageTable ? k.size(1) : 0, softcap > 0);
    MlaPrefillBsndLayout l{};
    PrefillBsndLayout(c, l);
    if (supplied) {
        TORCH_CHECK(supplied->scalar_type() == at::kInt && supplied->device() == q.device() && supplied->dim() == 1 &&
                        supplied->numel() == l.words && supplied->is_contiguous(),
                    "MLA: prefill scheduler_metadata must be same-device contiguous int32 with matching word count");
    }
    const int64_t b = c.batch, sq = c.sq, h = c.heads, capacity = splits == 0 ? 26 : splits;
    // Reuse the validated TND initializer for PA. One disjoint device launch
    // initializes all outputs before attention, including unused query slots.
    auto out = tnd         ? at::empty({c.totalQ, h, 128}, q.options())
               : pageTable ? at::empty({b, sq, h, 128}, q.options())
                           : at::zeros({b, sq, h, 128}, q.options());
    auto lse = tnd ? at::empty({h, c.totalQ}, q.options().dtype(at::kFloat))
               : pageTable
                   ? at::empty({b, h, sq}, q.options().dtype(at::kFloat))
                   : at::full({b, h, sq}, -std::numeric_limits<float>::infinity(), q.options().dtype(at::kFloat));
    const auto emptyOptions = at::TensorOptions().device(q.device());
    auto accum = splits == 1 ? at::empty({0}, emptyOptions)
                 : tnd       ? at::empty({capacity, h, c.totalQ, 128}, q.options().dtype(at::kFloat))
                 : pageTable ? at::empty({capacity, b, h, sq, 128}, q.options().dtype(at::kFloat))
                             : at::zeros({capacity, b, h, sq, 128}, q.options().dtype(at::kFloat));
    auto accumLse = splits == 1 ? at::empty({0}, emptyOptions)
                    : tnd       ? at::empty({capacity, h, c.totalQ}, q.options().dtype(at::kFloat))
                    : pageTable ? at::empty({capacity, b, h, sq}, q.options().dtype(at::kFloat))
                                : at::full({capacity, b, h, sq}, -std::numeric_limits<float>::infinity(),
                                           q.options().dtype(at::kFloat));
    auto workspace = at::empty({int64_t(l.bytes + platform.libApiWorkspaceBytes)}, q.options().dtype(at::kByte));
    auto metadata = supplied ? *supplied : at::empty({l.words}, q.options().dtype(at::kInt));
    auto args = PrefillMetadataArgs(c, metadata, usedQ, currentK);
    args.cuQuery = tnd ? PrefillPointer(*cuQ) : 0;
    args.cuKv = cuK ? PrefillPointer(*cuK) : 0;
    auto stream = c10_npu::getCurrentNPUStream();
    for (const auto& t : {q, k, v, out, lse, accum, accumLse, workspace, metadata})
        RecordPrefill(t, stream);
    if (tnd) {
        RecordPrefill(*cuQ, stream);
        if (cuK)
            RecordPrefill(*cuK, stream);
    }
    if (usedQ)
        RecordPrefill(*usedQ, stream);
    if (currentK)
        RecordPrefill(*currentK, stream);
    MlaPrefillBsndLaunchArgs a{};
    a.stream = stream.stream(false);
    uint32_t fftsLength = 0;
    TORCH_CHECK(rtGetC2cCtrlAddr(&a.ffts, &fftsLength) == RT_ERROR_NONE, "MLA: FFTS address unavailable");
    a.q = PrefillPointer(q);
    a.k = PrefillPointer(k);
    a.v = PrefillPointer(v);
    a.out = PrefillPointer(out);
    a.lse = PrefillPointer(lse);
    a.accum = PrefillPointer(accum);
    a.accumLse = PrefillPointer(accumLse);
    a.workspace = PrefillPointer(workspace) + platform.libApiWorkspaceBytes;
    a.metadata = PrefillPointer(metadata);
    a.usedQuery = usedQ ? PrefillPointer(*usedQ) : 0;
    a.usedKv = currentK ? PrefillPointer(*currentK) : 0;
    a.context = c;
    a.cuQuery = args.cuQuery;
    a.cuKv = args.cuKv;
    if (pageTable) {
        a.cache = {paged, PagedTensorArray(*pageTable, paged.tableSpanElements),
                   PagedTensorArray(k, paged.k.spanElements), PagedTensorArray(v, paged.v.spanElements)};
        RecordPrefill(*pageTable, stream);
    }
    a.params = {c.batch,
                c.sq,
                c.sk,
                c.heads,
                c.kvHeads,
                l.m,
                l.n,
                c.cores,
                scale,
                c.window,
                (tnd ? CheckTndTensorLayout(q) : CheckBsndTensorLayout(q)),
                (cuK ? CheckTndTensorLayout(k) : CheckBsndTensorLayout(k)),
                (cuK ? CheckTndTensorLayout(v) : CheckBsndTensorLayout(v))};
    a.params.softcap = softcap;
    RunCausalLaunch("ascendc_mla_prefill_bsnd", [a, args, supplied, q, k, v, out, lse, accum, accumLse, workspace,
                                                 metadata, usedQ, currentK, cuQ, cuK, pageTable]() mutable {
        if (!supplied)
            ComputeMlaPrefillBsndMetadata<<<1, nullptr, a.stream>>>(&args, sizeof(args));
        const auto error = q.scalar_type() == at::kHalf ? LaunchPrefillBsndFp16(a) : LaunchPrefillBsndBf16(a);
        TORCH_CHECK(error == ACL_SUCCESS, "MLA: prefill BATCH_MODE launch failed (", error, ")");
        return 0;
    });
    if (!tnd && !c.pageSize && !c.sk)
        lse.fill_(std::numeric_limits<float>::infinity());
    return {out, lse, accum, accumLse};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
PrefillBsndForwardOp(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, int64_t splits, int64_t left,
                     int64_t right, bool causal, std::optional<double> scale, const std::optional<at::Tensor>& usedK,
                     const std::optional<at::Tensor>& usedQ, const std::optional<at::Tensor>& metadata,
                     const std::optional<at::Tensor>& pageTable, double softcap)
{
    TORCH_CHECK(!scale || (std::isfinite(*scale) && std::abs(*scale) <= std::numeric_limits<float>::max()),
                "MLA: softmax_scale must be finite in FP32");
    TORCH_CHECK(std::isfinite(softcap) && softcap >= 0 && std::isfinite(float(softcap)) &&
                    (softcap == 0 || float(softcap) > 0),
                "MLA: softcap must be finite and non-negative in FP32");
    auto r = PrefillBsndForward(q, k, v, scale ? float(*scale) : 1.0f / std::sqrt(192.0f), splits, left, right, causal,
                                usedQ, usedK, metadata, std::nullopt, std::nullopt, 0, 0, pageTable, float(softcap));
    return {r[0], r[1], r[2], r[3]};
}
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
PrefillTndForwardOp(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& cuQ,
                    const at::Tensor& cuK, int64_t maxQ, int64_t maxK, int64_t splits, int64_t left, int64_t right,
                    bool causal, std::optional<double> scale, const std::optional<at::Tensor>& usedQ,
                    const std::optional<at::Tensor>& usedK, const std::optional<at::Tensor>& metadata, double softcap)
{
    TORCH_CHECK(!scale || (std::isfinite(*scale) && std::abs(*scale) <= std::numeric_limits<float>::max()),
                "MLA: softmax_scale must be finite in FP32");
    TORCH_CHECK(std::isfinite(softcap) && softcap >= 0 && std::isfinite(float(softcap)) &&
                    (softcap == 0 || float(softcap) > 0),
                "MLA: softcap must be finite and non-negative in FP32");
    auto r = PrefillBsndForward(q, k, v, scale ? float(*scale) : 1.0f / std::sqrt(192.0f), splits, left, right, causal,
                                usedQ, usedK, metadata, cuQ, cuK, maxQ, maxK, std::nullopt, float(softcap));
    return {r[0], r[1], r[2], r[3]};
}
at::Tensor PrefillTndMetadataOp(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& cuQ,
                                const at::Tensor& cuK, int64_t maxQ, int64_t maxK, int64_t splits, int64_t left,
                                int64_t right, bool causal, const std::optional<at::Tensor>& usedQ,
                                const std::optional<at::Tensor>& usedK, bool hasSoftcap)
{
    CheckForwardBoundary(q, k, v, std::nullopt, true, true, false);
    TORCH_CHECK(q.size(-1) == 192 && v.size(-1) == 128, "MLA: prefill requires 192/128 tensors");
    CheckTndInputs(q, k, cuQ, cuK, maxQ, maxK);
    const auto batch = cuQ.numel() - 1;
    CheckUsedInputs(q, batch, usedQ, usedK);
    const c10::OptionalDeviceGuard guard(device_of(q));
    auto c = PrefillContext(batch, maxQ, maxK, q.size(1), k.size(1), q.scalar_type(), splits, left, right, causal,
                            uint32_t(bool(usedQ)) | (uint32_t(bool(usedK)) << 1),
                            CausalPlatform(q.device().index(), true), true, q.size(0), k.size(0), 0, hasSoftcap);
    MlaPrefillBsndLayout l{};
    PrefillBsndLayout(c, l);
    auto metadata = at::empty({l.words}, q.options().dtype(at::kInt));
    auto args = PrefillMetadataArgs(c, metadata, usedQ, usedK);
    args.cuQuery = PrefillPointer(cuQ);
    args.cuKv = PrefillPointer(cuK);
    auto stream = c10_npu::getCurrentNPUStream();
    for (const auto& t : {cuQ, cuK, metadata})
        RecordPrefill(t, stream);
    if (usedQ)
        RecordPrefill(*usedQ, stream);
    if (usedK)
        RecordPrefill(*usedK, stream);
    RunCausalLaunch("ascendc_mla_prefill_tnd_metadata", [metadata, cuQ, cuK, usedQ, usedK, args, stream]() mutable {
        ComputeMlaPrefillBsndMetadata<<<1, nullptr, stream.stream(false)>>>(&args, sizeof(args));
        return 0;
    });
    return metadata;
}

at::Tensor PrefillMixedMetadataOp(int64_t batch, int64_t maxQ, int64_t sk, int64_t heads, int64_t kvHeads,
                                  at::ScalarType dtype, const at::Tensor& lengths, const at::Tensor& cuQ,
                                  int64_t splits, int64_t left, int64_t right, bool causal,
                                  const std::optional<at::Tensor>& usedQ, int64_t pageSize, bool hasSoftcap)
{
    CheckTndBsndMetadataInputs(batch, maxQ, sk, heads, dtype, lengths, cuQ);
    CheckUsedInputs(lengths, batch, usedQ, lengths);
    const c10::OptionalDeviceGuard guard(device_of(lengths));
    auto c =
        PrefillContext(batch, maxQ, sk, heads, kvHeads, dtype, splits, left, right, causal, 2 | uint32_t(bool(usedQ)),
                       CausalPlatform(lengths.device().index(), true), 2, 0, 0, pageSize, hasSoftcap);
    MlaPrefillBsndLayout l{};
    PrefillBsndLayout(c, l);
    auto metadata = at::empty({l.words}, lengths.options());
    auto args = PrefillMetadataArgs(c, metadata, usedQ, lengths);
    args.cuQuery = PrefillPointer(cuQ);
    args.inferTotalQ = 1;
    auto stream = c10_npu::getCurrentNPUStream();
    for (const auto& t : {cuQ, lengths, metadata})
        RecordPrefill(t, stream);
    if (usedQ)
        RecordPrefill(*usedQ, stream);
    RunCausalLaunch("ascendc_mla_prefill_mixed_metadata", [metadata, cuQ, lengths, usedQ, args, stream]() mutable {
        ComputeMlaPrefillBsndMetadata<<<1, nullptr, stream.stream(false)>>>(&args, sizeof(args));
        return 0;
    });
    return metadata;
}
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
PrefillMixedForwardOp(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& cuQ,
                      int64_t maxQ, int64_t splits, int64_t left, int64_t right, bool causal,
                      std::optional<double> scale, const std::optional<at::Tensor>& usedK,
                      const std::optional<at::Tensor>& usedQ, const std::optional<at::Tensor>& metadata,
                      const std::optional<at::Tensor>& pageTable, double softcap)
{
    TORCH_CHECK(!scale || (std::isfinite(*scale) && std::abs(*scale) <= std::numeric_limits<float>::max()),
                "MLA: softmax_scale must be finite in FP32");
    TORCH_CHECK(std::isfinite(softcap) && softcap >= 0 && std::isfinite(float(softcap)) &&
                    (softcap == 0 || float(softcap) > 0),
                "MLA: softcap must be finite and non-negative in FP32");
    auto result = PrefillBsndForward(q, k, v, scale.value_or(1.0 / std::sqrt(192.0)), splits, left, right, causal,
                                     usedQ, usedK, metadata, cuQ, std::nullopt, maxQ, 0, pageTable, float(softcap));
    return {result[0], result[1], result[2], result[3]};
}

namespace {
Int8::Config Int8Config(int64_t batch, int64_t sq, int64_t heads, int64_t capacity, int64_t splits, bool causal,
                        c10::DeviceIndex device)
{
    TORCH_CHECK(batch >= 0 && uint64_t(batch) <= UINT32_MAX && sq >= 0 && sq <= 32 && heads > 0 && heads <= 128 &&
                    capacity >= 0 && capacity <= INT32_MAX && splits >= 0 && splits <= 1,
                "MLA INT8: BSND shape or split bounds exceeded");
    auto platform = CausalPlatform(device, true);
    TORCH_CHECK(platform.blockDims[platform.aicNum - 1] == platform.aicNum,
                "MLA INT8: unsupported C1V2 blockDim mapping");
    Int8::Config c{uint32_t(batch),
                   uint32_t(sq),
                   uint32_t(heads),
                   uint32_t(capacity),
                   platform.aicNum,
                   uint32_t(splits),
                   uint32_t(causal && sq > 1),
                   0};
    TORCH_CHECK(Int8::Valid(c), "MLA INT8: expected BSND Q<=32, power-of-two Hq<=128, page=128 and multi-Q causal");
    return c;
}

void CheckInt8Storage(const at::Tensor& tensor, c10::Device device, at::ScalarType dtype, bool lastContiguous = true)
{
    TORCH_CHECK(tensor.device() == device && device.type() == c10::DeviceType::PrivateUse1 &&
                    tensor.scalar_type() == dtype && !tensor.requires_grad(),
                "MLA INT8: wrong dtype/device or unsupported gradients");
    TORCH_CHECK(tensor.dim() > 0 && (!lastContiguous || tensor.stride(-1) == 1),
                "MLA INT8: contiguous last dimension required");
    unsigned __int128 span = 1;
    for (int i = 0; i < tensor.dim(); ++i) {
        TORCH_CHECK(tensor.stride(i) >= 0, "MLA INT8: negative stride");
        if (tensor.size(i))
            span += uint64_t(tensor.size(i) - 1) * static_cast<unsigned __int128>(tensor.stride(i));
    }
    TORCH_CHECK(tensor.storage_offset() >= 0, "MLA INT8: negative storage offset");
    if (!tensor.numel())
        return;
    const auto end = (span + uint64_t(tensor.storage_offset())) * tensor.element_size();
    TORCH_CHECK(end <= INT64_MAX && end <= tensor.storage().nbytes(), "MLA INT8: view exceeds storage/byte bounds");
}

at::Tensor Int8Mask(const at::Tensor& q)
{
    // Fixed source mask only; never cache user tensors, lengths or scales.
    // Stream-local ownership preserves initialization ordering. A first call
    // inside capture owns its own mask instead of publishing unexecuted data.
    static thread_local std::unordered_map<int, std::unordered_map<uint64_t, at::Tensor>> deviceMasks;
    auto& masks = deviceMasks[q.device().index()];
    uint64_t stream = uint64_t(c10_npu::getCurrentNPUStream().stream(false));
    auto found = masks.find(stream);
    if (found != masks.end())
        return found->second;
    auto mask = at::ones({2048, 2048}, q.options().dtype(at::kBool)).triu(1);
    if (c10_npu::currentStreamCaptureStatusMayInitCtx() != c10_npu::CaptureStatus::Active)
        masks.emplace(stream, mask);
    return mask;
}
} // namespace

at::Tensor Int8MetadataOp(int64_t batch, int64_t sq, int64_t heads, int64_t capacity, const at::Tensor& lengths,
                          int64_t splits, bool causal)
{
    TORCH_CHECK(lengths.device().type() == c10::DeviceType::PrivateUse1, "MLA INT8: lengths must be NPU tensors");
    CheckCacheLengths(lengths, batch, lengths.device());
    const c10::OptionalDeviceGuard guard(device_of(lengths));
    auto c = Int8Config(batch, sq, heads, capacity, splits, causal, lengths.device().index());
    auto metadata = at::empty({Int8::kFixedWords + int64_t(c.batch)}, lengths.options());
    Int8::MetadataArgs args{c, PrefillPointer(lengths), PrefillPointer(metadata)};
    auto stream = c10_npu::getCurrentNPUStream();
    RecordPrefill(lengths, stream);
    RecordPrefill(metadata, stream);
    RunCausalLaunch("ascendc_mla_int8_metadata", [args, lengths, metadata, stream]() mutable {
        ComputeMlaInt8Metadata<<<1, nullptr, stream.stream(false)>>>(&args, sizeof(args));
        return 0;
    });
    return metadata;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
Int8ForwardOp(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& qv,
              const at::Tensor& table, const at::Tensor& qs, const at::Tensor& ks, const at::Tensor& vs,
              const std::optional<at::Tensor>& lengths, const std::optional<at::Tensor>& metadata,
              std::optional<double> scale, bool causal, int64_t splits, bool partials)
{
    TORCH_CHECK(q.dim() == 4 && qv.dim() == 4 && k.dim() == 4 && v.dim() == 4 && table.dim() == 2,
                "MLA INT8: BSND Q and ND PA cache required; TND is not connected");
    const auto device = q.device();
    for (const auto& t : {q, k})
        CheckInt8Storage(t, device, at::kBFloat16);
    for (const auto& t : {qv, v})
        CheckInt8Storage(t, device, at::kChar);
    for (const auto& t : {qs, ks, vs})
        CheckInt8Storage(t, device, at::kFloat, false);
    CheckInt8Storage(table, device, at::kInt);
    const int64_t b = q.size(0), sq = q.size(1), h = q.size(2);
    TORCH_CHECK(q.size(3) == 64 && qv.sizes() == at::IntArrayRef({b, sq, h, 512}) &&
                    qs.sizes() == at::IntArrayRef({b, sq, h}) && ks.sizes() == at::IntArrayRef({1}) &&
                    vs.sizes() == at::IntArrayRef({1}) && table.size(0) == b,
                "MLA INT8: invalid Q/content/scale/page-table shapes");
    MlaPagedCacheLayout cache{};
    TORCH_CHECK(BuildInt8PagedCacheLayout(PagedTensorView<MlaPageTableView>(table), PagedTensorView(k),
                                          PagedTensorView(v), cache) == MlaLayoutStatus::Success,
                "MLA INT8: invalid ND PA cache layout/storage");
    TORCH_CHECK(!scale || (std::isfinite(*scale) && std::isfinite(float(*scale))),
                "MLA INT8: softmax_scale must be finite in FP32");
    if (lengths)
        CheckCacheLengths(*lengths, b, device);
    const c10::OptionalDeviceGuard guard(device_of(q));
    auto c = Int8Config(b, sq, h, cache.logicalCapacity, splits, causal, device.index());
    if (metadata)
        CheckGqaRecord(*metadata, device, Int8::kFixedWords + c.batch);
    auto query = q.contiguous(), content = qv.contiguous(), qscale = qs.contiguous();
    auto mask = sq > 1 ? Int8Mask(q) : at::empty({0}, q.options().dtype(at::kBool));
    auto out = at::empty({b, sq, h, 512}, q.options());
    auto lse = at::empty({b, h, sq}, q.options().dtype(at::kFloat));
    auto accum = partials && splits == 0 ? at::zeros({26, b, h, sq, 512}, q.options().dtype(at::kFloat))
                                         : at::empty({0}, q.options().dtype(at::kFloat));
    auto accumLse = partials && splits == 0 ? at::full({26, b, h, sq}, -INFINITY, q.options().dtype(at::kFloat))
                                            : at::empty({0}, q.options().dtype(at::kFloat));
    // Full launch remains independent of current device lengths. Numerical
    // workspace uses the source's GetBlockNum(), plus bounded balanced FD.
    const uint64_t rows = std::min<uint64_t>(sq * h, 128);
    const uint64_t bytes =
        c.cores * 2ULL * (rows * 512 * 9) + 4ULL * (c.cores * 2 * 128 * 512 + 2 * c.cores * 2 * 128 * 8);
    auto workspace = at::empty({int64_t(bytes + 512)}, q.options().dtype(at::kByte));
    auto stream = c10_npu::getCurrentNPUStream();
    auto owned = std::vector<at::Tensor>{q,  qv,    query, content, k,   v,     qs,       qscale,   ks,
                                         vs, table, mask,  out,     lse, accum, accumLse, workspace};
    if (lengths)
        owned.push_back(*lengths);
    if (metadata)
        owned.push_back(*metadata);
    for (const auto& t : owned)
        RecordPrefill(t, stream);
    Int8::LaunchArgs a{};
    a.stream = stream.stream(false);
    uint32_t fftsLength = 0;
    TORCH_CHECK(rtGetC2cCtrlAddr(&a.ffts, &fftsLength) == RT_ERROR_NONE, "MLA INT8: FFTS unavailable");
    a.q = PrefillPointer(query);
    a.qv = PrefillPointer(content);
    a.k = PrefillPointer(k);
    a.v = PrefillPointer(v);
    a.qs = PrefillPointer(qscale);
    a.ks = PrefillPointer(ks);
    a.vs = PrefillPointer(vs);
    a.table = PrefillPointer(table);
    a.mask = sq > 1 ? PrefillPointer(mask) : 0;
    a.lengths = lengths ? PrefillPointer(*lengths) : 0;
    a.metadata = metadata ? PrefillPointer(*metadata) : 0;
    a.out = PrefillPointer(out);
    a.lse = PrefillPointer(lse);
    a.accum = accum.numel() ? PrefillPointer(accum) : 0;
    a.accumLse = accumLse.numel() ? PrefillPointer(accumLse) : 0;
    a.workspace = PrefillPointer(workspace);
    a.config = c;
    a.cache = cache;
    a.scale = scale ? float(*scale) : 1.0f / 24.0f;
    RunCausalLaunch("ascendc_mla_int8_bsnd", [a, owned]() { return Int8::Launch(a); });
    return {out, lse, accum, accumLse};
}
} // namespace flash_attn::mla

TORCH_LIBRARY_FRAGMENT(flash_attn_npu_3, m)
{
    m.def("_mla_int8_metadata(int batch, int sq, int heads, int capacity, Tensor lengths, int splits, bool causal) -> "
          "Tensor");
    m.def("_mla_int8_bsnd(Tensor q, Tensor k, Tensor v, Tensor qv, Tensor table, Tensor qs, Tensor ks, Tensor vs, "
          "Tensor? lengths, Tensor? metadata, float? scale, bool causal, int splits, bool partials) -> (Tensor, "
          "Tensor, Tensor, Tensor)");
    m.def("_mla_gqa_forward(Tensor q, Tensor k, Tensor v, Tensor? qv, int num_splits, int window_left, "
          "int window_right, bool causal, float? softmax_scale, Tensor? seqused_q, Tensor? seqused_k, "
          "Tensor? scheduler_metadata, Tensor? cu_seqlens_q, Tensor? cu_seqlens_k, int max_seqlen_q, "
          "int max_seqlen_k, Tensor? page_table, float softcap, bool? pack_gqa, int sm_margin) -> "
          "(Tensor, Tensor, Tensor, Tensor)");
    m.def("_mla_gqa_metadata(int batch_size, int max_seqlen_q, int max_seqlen_k, int num_heads_q, int num_heads_kv, "
          "int headdim_v, ScalarType qkv_dtype, Tensor cache_seqlens, Tensor? cu_seqlens_q, int num_splits, "
          "int window_left, int window_right, bool causal, int page_size, bool has_softcap, bool? pack_gqa, "
          "int sm_margin, Tensor? seqused_q=None) -> Tensor");
    m.def("_mla_prefill_mixed_forward(Tensor q, Tensor k, Tensor v, Tensor cu_seqlens_q, int max_seqlen_q, "
          "int num_splits=1, int window_left=-1, int window_right=-1, bool causal=False, "
          "float? softmax_scale=None, Tensor? cache_seqlens=None, Tensor? seqused_q=None, "
          "Tensor? scheduler_metadata=None, Tensor? page_table=None, float softcap=0.) -> (Tensor, Tensor, Tensor, "
          "Tensor)");
    m.def("_mla_prefill_mixed_metadata(int batch_size, int max_seqlen_q, int max_seqlen_k, int num_heads_q, "
          "int num_heads_kv, ScalarType qkv_dtype, Tensor cache_seqlens, Tensor cu_seqlens_q, "
          "int num_splits=1, int window_left=-1, int window_right=-1, bool causal=False, "
          "Tensor? seqused_q=None, int page_size=0, bool has_softcap=False) -> Tensor");
    m.def("_mla_prefill_tnd_forward(Tensor q, Tensor k, Tensor v, Tensor cu_seqlens_q, Tensor cu_seqlens_k, "
          "int max_seqlen_q, int max_seqlen_k, int num_splits=1, int window_left=-1, int window_right=-1, "
          "bool causal=False, float? softmax_scale=None, Tensor? seqused_q=None, Tensor? seqused_k=None, "
          "Tensor? scheduler_metadata=None, float softcap=0.) -> (Tensor, Tensor, Tensor, Tensor)");
    m.def("_mla_prefill_tnd_metadata(Tensor q, Tensor k, Tensor v, Tensor cu_seqlens_q, Tensor cu_seqlens_k, "
          "int max_seqlen_q, int max_seqlen_k, int num_splits=1, int window_left=-1, int window_right=-1, "
          "bool causal=False, Tensor? seqused_q=None, Tensor? seqused_k=None, bool has_softcap=False) -> Tensor");
    m.def("_mla_prefill_bsnd_forward(Tensor q, Tensor k, Tensor v, int num_splits=1, int window_left=-1, "
          "int window_right=-1, bool causal=False, float? softmax_scale=None, Tensor? cache_seqlens=None, "
          "Tensor? seqused_q=None, Tensor? scheduler_metadata=None, Tensor? page_table=None, float softcap=0.) -> "
          "(Tensor, Tensor, "
          "Tensor, Tensor)");
    m.def("_mla_prefill_bsnd_metadata(int batch_size, int max_seqlen_q, int max_seqlen_k, int num_heads_q, "
          "int num_heads_kv, ScalarType qkv_dtype, Tensor cache_seqlens, int num_splits=1, int window_left=-1, "
          "int window_right=-1, bool causal=False, int page_size=0, bool has_softcap=False) -> Tensor");
    m.def("_mla_split_paged_tnd_forward(Tensor q, Tensor k, Tensor v, Tensor qv, Tensor page_table, "
          "Tensor cu_seqlens_q, int max_seqlen_q, int num_splits, int window_left=-1, int window_right=-1, "
          "bool causal=False, float? softmax_scale=None, Tensor? cache_seqlens=None, Tensor? seqused_q=None, "
          "Tensor? scheduler_metadata=None, float softcap=0.) -> (Tensor, Tensor, Tensor, Tensor)");
    m.def("_mla_split_paged_bsnd_forward(Tensor q, Tensor k, Tensor v, Tensor qv, Tensor page_table, int num_splits, "
          "int window_left=-1, int window_right=-1, bool causal=False, float? softmax_scale=None, "
          "Tensor? cache_seqlens=None, Tensor? seqused_q=None, Tensor? scheduler_metadata=None, float softcap=0.) -> "
          "(Tensor, Tensor, "
          "Tensor, Tensor)");
    m.def("_mla_split_mixed_forward(Tensor q, Tensor k, Tensor v, Tensor qv, Tensor cu_seqlens_q, int max_seqlen_q, "
          "int num_splits, int window_left=-1, int window_right=-1, bool causal=False, float? softmax_scale=None, "
          "Tensor? cache_seqlens=None, Tensor? seqused_q=None, Tensor? scheduler_metadata=None, float softcap=0.) -> "
          "(Tensor, Tensor, "
          "Tensor, Tensor)");
    m.def("_mla_split_mixed_metadata(int batch_size, int max_seqlen_q, int max_seqlen_k, int num_heads_q, "
          "ScalarType qkv_dtype, Tensor cache_seqlens, Tensor cu_seqlens_q, int window_left, int window_right, bool "
          "causal, int num_splits, bool has_softcap=False) -> Tensor");
    m.def("_mla_split_tnd_forward(Tensor q, Tensor k, Tensor v, Tensor qv, Tensor cu_seqlens_q, Tensor cu_seqlens_k, "
          "int max_seqlen_q, int max_seqlen_k, int num_splits, int window_left=-1, int window_right=-1, bool "
          "causal=False, "
          "float? softmax_scale=None, Tensor? seqused_q=None, Tensor? seqused_k=None, Tensor? scheduler_metadata=None, "
          "float softcap=0.) "
          "-> (Tensor, Tensor, Tensor, Tensor)");
    m.def("_mla_split_tnd_metadata(Tensor q, Tensor k, Tensor v, Tensor qv, Tensor cu_seqlens_q, Tensor cu_seqlens_k, "
          "int max_seqlen_q, int max_seqlen_k, int num_splits, int window_left=-1, int window_right=-1, bool "
          "causal=False, "
          "Tensor? seqused_q=None, Tensor? seqused_k=None, bool has_softcap=False) -> Tensor");
    m.def("_mla_split_bsnd_forward(Tensor q, Tensor k, Tensor v, Tensor qv, int num_splits, "
          "int window_left=-1, int window_right=-1, bool causal=False, float? softmax_scale=None, "
          "Tensor? cache_seqlens=None, Tensor? seqused_q=None, Tensor? scheduler_metadata=None, float softcap=0.) -> "
          "(Tensor, Tensor, "
          "Tensor, Tensor)");
    m.def("_mla_split_bsnd_metadata(int batch_size, int max_seqlen_q, int max_seqlen_k, int num_heads_q, "
          "ScalarType qkv_dtype, Tensor cache_seqlens, int window_left, int window_right, bool causal, int num_splits, "
          "bool has_softcap=False) "
          "-> Tensor");
    m.def("_mla_window_mixed_metadata(int batch_size, int max_seqlen_q, int max_seqlen_k, int num_heads_q, "
          "ScalarType qkv_dtype, Tensor cache_seqlens, Tensor cu_seqlens_q, int window_left, int window_right, bool "
          "causal) -> Tensor");
    m.def("_mla_window_mixed_forward(Tensor q, Tensor k, Tensor v, Tensor qv, Tensor cu_seqlens_q, int max_seqlen_q, "
          "int window_left, int window_right, bool causal, "
          "float? softmax_scale=None, Tensor? cache_seqlens=None, Tensor? seqused_q=None, Tensor? "
          "scheduler_metadata=None) -> (Tensor, Tensor)");
    m.def("_mla_window_mixed_cache(Tensor q, Tensor k, Tensor v, Tensor qv, Tensor cu_seqlens_q, int max_seqlen_q, int "
          "window_left, int window_right, bool causal, "
          "float? softmax_scale=None, Tensor? cache_seqlens=None, Tensor? scheduler_metadata=None) -> (Tensor, Tensor, "
          "Tensor, Tensor)");
    m.def("_mla_causal_mixed_metadata(int batch_size, int max_seqlen_q, int max_seqlen_k, int num_heads_q, "
          "ScalarType qkv_dtype, Tensor cache_seqlens, Tensor cu_seqlens_q) -> Tensor");
    m.def("_mla_causal_mixed_forward(Tensor q, Tensor k, Tensor v, Tensor qv, Tensor cu_seqlens_q, int max_seqlen_q, "
          "float? softmax_scale=None, Tensor? cache_seqlens=None, Tensor? seqused_q=None, Tensor? "
          "scheduler_metadata=None) -> (Tensor, Tensor)");
    m.def("_mla_causal_mixed_cache(Tensor q, Tensor k, Tensor v, Tensor qv, Tensor cu_seqlens_q, int max_seqlen_q, "
          "float? softmax_scale=None, Tensor? cache_seqlens=None, Tensor? scheduler_metadata=None) -> (Tensor, Tensor, "
          "Tensor, Tensor)");
    m.def("_mla_window_bsnd_metadata(int batch_size, int max_seqlen_q, int max_seqlen_k, int num_heads_q, "
          "ScalarType qkv_dtype, Tensor cache_seqlens, int window_left, int window_right, bool causal) -> Tensor");
    m.def("_mla_window_bsnd_forward(Tensor q, Tensor k, Tensor v, Tensor qv, int window_left, int window_right, bool "
          "causal, float? softmax_scale=None, "
          "Tensor? cache_seqlens=None, Tensor? seqused_q=None, Tensor? scheduler_metadata=None) -> (Tensor, Tensor)");
    m.def("_mla_window_bsnd_cache(Tensor q, Tensor k, Tensor v, Tensor qv, int window_left, int window_right, bool "
          "causal, float? softmax_scale=None, "
          "Tensor? cache_seqlens=None, Tensor? scheduler_metadata=None) -> (Tensor, Tensor, Tensor, Tensor)");
    m.def("_mla_causal_bsnd_metadata(int batch_size, int max_seqlen_q, int max_seqlen_k, int num_heads_q, "
          "ScalarType qkv_dtype, Tensor cache_seqlens) -> Tensor");
    m.def("_mla_causal_bsnd_forward(Tensor q, Tensor k, Tensor v, Tensor qv, float? softmax_scale=None, "
          "Tensor? cache_seqlens=None, Tensor? seqused_q=None, Tensor? scheduler_metadata=None) -> (Tensor, Tensor)");
    m.def("_mla_causal_bsnd_cache(Tensor q, Tensor k, Tensor v, Tensor qv, float? softmax_scale=None, "
          "Tensor? cache_seqlens=None, Tensor? scheduler_metadata=None) -> (Tensor, Tensor, Tensor, Tensor)");
    m.def("_mla_dense_metadata(int batch_size, int max_seqlen_q, int max_seqlen_k, int num_heads_q, "
          "ScalarType qkv_dtype, Tensor cache_seqlens) -> Tensor");
    m.def("_mla_tnd_bsnd_metadata(int batch_size, int max_seqlen_q, int max_seqlen_k, int num_heads_q, "
          "ScalarType qkv_dtype, Tensor cache_seqlens, Tensor cu_seqlens_q) -> Tensor");
    m.def("_mla_paged_metadata(int batch_size, int max_seqlen_q, int max_seqlen_k, int num_heads_q, int page_size, "
          "ScalarType qkv_dtype, Tensor cache_seqlens, Tensor? cu_seqlens_q, bool causal=False, "
          "int window_left=-1, int window_right=-1, int num_splits=1, bool has_softcap=False) -> Tensor");
    m.def("_mla_dense_forward(Tensor q, Tensor k, Tensor v, Tensor qv, float? softmax_scale=None, "
          "Tensor? cache_seqlens=None, Tensor? scheduler_metadata=None) -> (Tensor, Tensor)");
    m.def("_mla_paged_bsnd_forward(Tensor q, Tensor k, Tensor v, Tensor qv, Tensor page_table, "
          "float? softmax_scale=None, Tensor? cache_seqlens=None, Tensor? seqused_q=None, "
          "Tensor? scheduler_metadata=None, bool causal=False, int window_left=-1, int window_right=-1) -> (Tensor, "
          "Tensor)");
    m.def("_mla_paged_tnd_forward(Tensor q, Tensor k, Tensor v, Tensor qv, Tensor page_table, "
          "Tensor cu_seqlens_q, int max_seqlen_q, float? softmax_scale=None, Tensor? cache_seqlens=None, "
          "Tensor? seqused_q=None, Tensor? scheduler_metadata=None, bool causal=False, int window_left=-1, "
          "int window_right=-1) -> (Tensor, Tensor)");
    m.def("_mla_tnd_forward(Tensor q, Tensor k, Tensor v, Tensor qv, Tensor cu_seqlens_q, Tensor cu_seqlens_k, "
          "int max_seqlen_q, int max_seqlen_k, float? softmax_scale=None, Tensor? seqused_q=None, "
          "Tensor? seqused_k=None) -> (Tensor, Tensor)");
    m.def("_mla_window_tnd_forward(Tensor q, Tensor k, Tensor v, Tensor qv, Tensor cu_seqlens_q, Tensor cu_seqlens_k, "
          "int max_seqlen_q, int max_seqlen_k, int window_left, int window_right, bool causal, "
          "float? softmax_scale=None, Tensor? seqused_q=None, Tensor? seqused_k=None) -> (Tensor, Tensor)");
    m.def("_mla_causal_tnd_forward(Tensor q, Tensor k, Tensor v, Tensor qv, Tensor cu_seqlens_q, Tensor cu_seqlens_k, "
          "int max_seqlen_q, int max_seqlen_k, float? softmax_scale=None, Tensor? seqused_q=None, "
          "Tensor? seqused_k=None) -> (Tensor, Tensor)");
    m.def(
        "_mla_tnd_bsnd_forward(Tensor q, Tensor k, Tensor v, Tensor qv, Tensor cu_seqlens_q, int max_seqlen_q, "
        "float? softmax_scale=None, Tensor? cache_seqlens=None, Tensor? scheduler_metadata=None) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(flash_attn_npu_3, CompositeExplicitAutograd, m)
{
    m.impl("_mla_int8_metadata", &flash_attn::mla::Int8MetadataOp);
    m.impl("_mla_int8_bsnd", &flash_attn::mla::Int8ForwardOp);
    m.impl("_mla_gqa_metadata", &flash_attn::mla::GqaMetadataOp);
    m.impl("_mla_prefill_bsnd_metadata", &flash_attn::mla::PrefillBsndMetadataOp);
    m.impl("_mla_split_bsnd_metadata", &flash_attn::mla::SplitBsndMetadataOp);
    m.impl("_mla_window_mixed_metadata", &flash_attn::mla::WindowMixedMetadataOp);
    m.impl("_mla_causal_mixed_metadata", &flash_attn::mla::CausalMixedMetadataOp);
    m.impl("_mla_window_bsnd_metadata", &flash_attn::mla::WindowBsndMetadataOp);
    m.impl("_mla_causal_bsnd_metadata", &flash_attn::mla::CausalBsndMetadataOp);
    // Validate the sole tensor's device before selecting the NPU generator,
    // including when a direct caller supplies CPU lengths.
    m.impl("_mla_dense_metadata", &flash_attn::mla::DenseMetadataOp);
    m.impl("_mla_tnd_bsnd_metadata", &flash_attn::mla::TndBsndMetadataOp);
    m.impl("_mla_paged_metadata", &flash_attn::mla::PagedMetadataOp);
}

TORCH_LIBRARY_IMPL(flash_attn_npu_3, Autograd, m)
{
    m.impl("_mla_int8_metadata", torch::CppFunction::makeFallthrough());
    m.impl("_mla_int8_bsnd", torch::CppFunction::makeFallthrough());
    m.impl("_mla_gqa_metadata", torch::CppFunction::makeFallthrough());
    m.impl("_mla_gqa_forward", torch::CppFunction::makeFallthrough());
    m.impl("_mla_prefill_bsnd_metadata", torch::CppFunction::makeFallthrough());
    m.impl("_mla_split_bsnd_metadata", torch::CppFunction::makeFallthrough());
    m.impl("_mla_window_mixed_metadata", torch::CppFunction::makeFallthrough());
    m.impl("_mla_causal_mixed_metadata", torch::CppFunction::makeFallthrough());
    m.impl("_mla_window_bsnd_metadata", torch::CppFunction::makeFallthrough());
    m.impl("_mla_causal_bsnd_metadata", torch::CppFunction::makeFallthrough());
    // Lengths and metadata are int32. Let real/fake kernels validate their
    // dtype, including malformed floating tensors that require gradients.
    m.impl("_mla_dense_metadata", torch::CppFunction::makeFallthrough());
    m.impl("_mla_tnd_bsnd_metadata", torch::CppFunction::makeFallthrough());
    m.impl("_mla_paged_metadata", torch::CppFunction::makeFallthrough());
}

TORCH_LIBRARY_IMPL(flash_attn_npu_3, PrivateUse1, m)
{
    m.impl("_mla_gqa_forward", &flash_attn::mla::GqaForwardOp);
    m.impl("_mla_prefill_mixed_forward", &flash_attn::mla::PrefillMixedForwardOp);
    m.impl("_mla_prefill_mixed_metadata", &flash_attn::mla::PrefillMixedMetadataOp);
    m.impl("_mla_prefill_tnd_forward", &flash_attn::mla::PrefillTndForwardOp);
    m.impl("_mla_prefill_tnd_metadata", &flash_attn::mla::PrefillTndMetadataOp);
    m.impl("_mla_prefill_bsnd_forward", &flash_attn::mla::PrefillBsndForwardOp);
    m.impl("_mla_split_paged_bsnd_forward", &flash_attn::mla::SplitPagedBsndForwardOp);
    m.impl("_mla_split_paged_tnd_forward", &flash_attn::mla::SplitPagedTndForwardOp);
    m.impl("_mla_split_mixed_forward", &flash_attn::mla::SplitMixedForwardOp);
    m.impl("_mla_split_mixed_metadata", &flash_attn::mla::SplitMixedMetadataOp);
    m.impl("_mla_split_tnd_forward", &flash_attn::mla::SplitTndForwardOp);
    m.impl("_mla_split_tnd_metadata", &flash_attn::mla::SplitTndMetadataOp);
    m.impl("_mla_split_bsnd_forward", &flash_attn::mla::SplitBsndForwardOp);
    m.impl("_mla_window_mixed_forward", &flash_attn::mla::WindowMixedForwardOp);
    m.impl("_mla_causal_mixed_forward", &flash_attn::mla::CausalMixedForwardOp);
    m.impl("_mla_window_mixed_cache", &flash_attn::mla::WindowMixedCacheOp);
    m.impl("_mla_causal_mixed_cache", &flash_attn::mla::CausalMixedCacheOp);
    m.impl("_mla_window_bsnd_forward", &flash_attn::mla::WindowBsndForwardOp);
    m.impl("_mla_causal_bsnd_forward", &flash_attn::mla::CausalBsndForwardOp);
    m.impl("_mla_window_bsnd_cache", &flash_attn::mla::WindowBsndCacheOp);
    m.impl("_mla_causal_bsnd_cache", &flash_attn::mla::CausalBsndCacheOp);
    m.impl("_mla_dense_forward", &flash_attn::mla::DenseForwardOp);
    m.impl("_mla_tnd_forward", &flash_attn::mla::TndForwardOp<false>);
    m.impl("_mla_window_tnd_forward", &flash_attn::mla::WindowTndForwardOp);
    m.impl("_mla_causal_tnd_forward", &flash_attn::mla::TndForwardOp<true>);
    m.impl("_mla_tnd_bsnd_forward", &flash_attn::mla::TndBsndForwardOp);
    m.impl("_mla_paged_bsnd_forward", &flash_attn::mla::PagedBsndForwardOp);
    m.impl("_mla_paged_tnd_forward", &flash_attn::mla::PagedTndForwardOp);
}

TORCH_LIBRARY_IMPL(flash_attn_npu_3, AutogradPrivateUse1, m)
{
    m.impl("_mla_prefill_mixed_forward", torch::CppFunction::makeFallthrough());
    m.impl("_mla_prefill_mixed_metadata", torch::CppFunction::makeFallthrough());
    m.impl("_mla_prefill_tnd_forward", torch::CppFunction::makeFallthrough());
    m.impl("_mla_prefill_tnd_metadata", torch::CppFunction::makeFallthrough());
    m.impl("_mla_prefill_bsnd_forward", torch::CppFunction::makeFallthrough());
    m.impl("_mla_split_paged_bsnd_forward", torch::CppFunction::makeFallthrough());
    m.impl("_mla_split_paged_tnd_forward", torch::CppFunction::makeFallthrough());
    m.impl("_mla_split_mixed_forward", torch::CppFunction::makeFallthrough());
    m.impl("_mla_split_tnd_forward", torch::CppFunction::makeFallthrough());
    m.impl("_mla_split_tnd_metadata", torch::CppFunction::makeFallthrough());
    m.impl("_mla_split_bsnd_forward", torch::CppFunction::makeFallthrough());
    m.impl("_mla_window_mixed_forward", torch::CppFunction::makeFallthrough());
    m.impl("_mla_causal_mixed_forward", torch::CppFunction::makeFallthrough());
    m.impl("_mla_window_mixed_cache", torch::CppFunction::makeFallthrough());
    m.impl("_mla_causal_mixed_cache", torch::CppFunction::makeFallthrough());
    m.impl("_mla_window_bsnd_forward", torch::CppFunction::makeFallthrough());
    m.impl("_mla_causal_bsnd_forward", torch::CppFunction::makeFallthrough());
    m.impl("_mla_window_bsnd_cache", torch::CppFunction::makeFallthrough());
    m.impl("_mla_causal_bsnd_cache", torch::CppFunction::makeFallthrough());
    // Preserve FakeTensor dispatch below Autograd. The real/fake kernels both
    // reject requires_grad before allocating or submitting forward-only MLA.
    m.impl("_mla_dense_forward", torch::CppFunction::makeFallthrough());
    m.impl("_mla_tnd_forward", torch::CppFunction::makeFallthrough());
    m.impl("_mla_window_tnd_forward", torch::CppFunction::makeFallthrough());
    m.impl("_mla_causal_tnd_forward", torch::CppFunction::makeFallthrough());
    m.impl("_mla_tnd_bsnd_forward", torch::CppFunction::makeFallthrough());
    m.impl("_mla_paged_bsnd_forward", torch::CppFunction::makeFallthrough());
    m.impl("_mla_paged_tnd_forward", torch::CppFunction::makeFallthrough());
}
