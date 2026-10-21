#include "mla_layout.h"
#include "mla_metadata.h"
#include <algorithm>
#include "mla_forward.h"
#include "kernel_operator.h"
#include "mla_dispatch.h"
#include "mla_split.h"
#include "mla_tiling.h"
#include "mla_validation.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUEvent.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "runtime/rt_ffts.h"
#include <c10/core/DeviceGuard.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <unordered_map>

// Static host preparation for compressed BSND/TND MLA paged attention.
// No tensor data, addresses, torch types or device runtime calls enter this API.

namespace flash_attn::mla {

struct MlaPagedPlan {
    MlaPagedShape shape;
    // Pass this fixed launch policy to the AICPU generator, including when
    // device-only lengths turn a nonempty allocation into an empty schedule.
    MlaPlatformInfo platform;
    MlaPagedMetadataContext context;
    MlaBsndInputLayouts layouts; // TND Q/qv: batchStride=0, rowStride=tokenStride.
    MlaPagedCacheLayout cache;
    uint32_t metadataWords;
    uint64_t totalWorkspaceBytes; // context.workspaceBytes + context.libApiBytes.
};

using MlaPagedBsndPlan = MlaPagedPlan;
using MlaPagedTndPlan = MlaPagedPlan;

enum class MlaPagedPlanStatus {
    Success,
    InvalidShape,
    InvalidLayout,
    UnsupportedDtype,
    InvalidPlatform,
    InsufficientLocalMemory,
    WorkspaceOverflow,
};

} // namespace flash_attn::mla

/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Workspace bounds derived from CANN ops-transformer
// ca18aee610729ca5270a44e2d84c751bc687fbf0,
// attention/fused_infer_attention_score/op_host/arch22/fia_tiling_nonquant_mla.cpp:
// CalcInnerSize, CalcMmResSize and CalcNormalWorkspaceSize (LICENSE.cann).
// Changes: bound all dynamic lengths with allocated geometry and M<=256;
// prepare independent v6 invocation context without reading a schedule.

namespace flash_attn::mla {
namespace {

MlaPagedPlanStatus ValidateGeometry(int64_t batch, int64_t maxQuery, int64_t heads, bool tnd,
                                    const MlaPlatformInfo& platform, uint32_t dtype)
{
    using Status = MlaPagedPlanStatus;
    if (dtype != uint32_t(MlaMetadataDtype::Fp16) && dtype != uint32_t(MlaMetadataDtype::Bf16)) {
        return Status::UnsupportedDtype;
    }
    if (batch < 0 || batch > (UINT32_MAX - kMlaPagedMetadataPrefixWords) / kMlaPagedMetadataSequenceWords) {
        return Status::InvalidShape;
    }
    if (platform.aicNum == 0 || platform.aicNum > kMlaMaxAic || platform.aivNum != 2 * platform.aicNum) {
        return Status::InvalidPlatform;
    }
    // Reuse static shape, platform and local-memory checks. sk=0 guarantees
    // this call never runs SplitCore or allocates per-batch scheduling arrays.
    // TND's total allocation is checked separately, not as B*maxQuery.
    MlaSchedule validation{};
    const auto status = BuildDenseSchedule({tnd ? 0 : batch, maxQuery, heads, 0}, platform, validation);
    if (status == MlaScheduleStatus::InvalidPlatform) {
        return Status::InvalidPlatform;
    }
    if (status == MlaScheduleStatus::InsufficientLocalMemory) {
        return Status::InsufficientLocalMemory;
    }
    if (status != MlaScheduleStatus::Success) {
        return Status::InvalidShape;
    }
    return Status::Success;
}

MlaPagedPlanStatus PrepareResources(const MlaPlatformInfo& platform, uint32_t dtype, bool hasUsedQuery,
                                    MlaPagedPlan& result)
{
    using Status = MlaPagedPlanStatus;
    const auto& query = result.shape.query;
    const uint64_t batch = result.shape.batch, heads = result.shape.queryHeads, totalQ = query.totalTokens;
    const uint64_t capacity = result.cache.logicalCapacity, page = result.cache.pageSize;
    if (capacity > kMlaMaxKvLength) {
        return Status::InvalidShape;
    }
    const bool nonempty = batch != 0 && totalQ != 0 && capacity != 0;
    const uint32_t blocks = platform.blockDims[nonempty ? platform.aicNum - 1 : 0];
    // C1V2 needs room for every possible working Cube, while static empties
    // use the single validation core. Never infer blockDim from device data.
    if (blocks != (nonempty ? platform.aicNum : 1U)) {
        return Status::InvalidPlatform;
    }
    result.shape.logicalKvCapacity = capacity;
    result.shape.pageSize = page;
    result.platform = platform;
    result.platform.blockDims.fill(blocks);
    result.layouts.k = result.cache.k;
    result.layouts.v = result.cache.v;
    result.metadataWords = kMlaPagedMetadataPrefixWords + kMlaPagedMetadataSequenceWords * batch;

    // Source PA alignment precedes short-S2 truncation. Q-used and unequal
    // KV lengths can select M=256 even when dense/equal lengths select less.
    const uint64_t maxRows = std::min(uint64_t(query.maxLength), totalQ) * heads;
    const uint64_t m = (std::min(maxRows, uint64_t(256)) + 15) / 16 * 16;
    const uint64_t baseN = page < 512 && 512 % page != 0 ? 512 / page * page : 512;
    const uint64_t n = (std::min(capacity, baseN) + 31) / 32 * 32;
    const uint64_t userBytes = nonempty ? uint64_t(2) * platform.aicNum * (6 * m * n + 8 * m * 512 + 8 * 256) : 0;
    const uint64_t libBytes = nonempty ? platform.libApiWorkspaceBytes : 0;
    if (libBytes > uint64_t(INT64_MAX) - userBytes) {
        return Status::WorkspaceOverflow;
    }
    result.totalWorkspaceBytes = userBytes + libBytes;
    result.context = {dtype,
                      uint32_t(batch),
                      uint32_t(heads),
                      hasUsedQuery ? kMlaPagedUsedQueryPresent : 0,
                      {uint32_t(query.layout), {uint32_t(totalQ), uint32_t(totalQ >> 32)}, uint32_t(query.maxLength)},
                      uint32_t(page),
                      uint32_t(capacity),
                      platform.aicNum,
                      platform.aivNum,
                      blocks,
                      userBytes,
                      libBytes,
                      platform.localMemory};
    return Status::Success;
}

} // namespace

// Validate the full allocation and bound workspace without reading device lengths.
MlaPagedPlanStatus BuildPagedBsndPlan(const MlaBsndTensorView& q, const MlaBsndTensorView& qv,
                                      const MlaBsndTensorView& k, const MlaBsndTensorView& v,
                                      const MlaPageTableView& table, const MlaPlatformInfo& platform, uint32_t dtype,
                                      bool hasUsedQuery, MlaPagedBsndPlan& output)
{
    using Status = MlaPagedPlanStatus;
    const int64_t batch = q.sizes[0], sq = q.sizes[1], heads = q.sizes[2];
    auto status = ValidateGeometry(batch, sq, heads, false, platform, dtype);
    if (status != Status::Success) {
        return status;
    }
    if (q.sizes[3] != 64 || qv.sizes[3] != 512 || qv.sizes[0] != batch || qv.sizes[1] != sq || qv.sizes[2] != heads ||
        table.sizes[0] != batch) {
        return Status::InvalidShape;
    }
    MlaPagedBsndPlan result{};
    if (BuildBsndTensorLayout(q, result.layouts.q) != MlaLayoutStatus::Success ||
        BuildBsndTensorLayout(qv, result.layouts.qv) != MlaLayoutStatus::Success ||
        BuildPagedCacheLayout(table, k, v, result.cache) != MlaLayoutStatus::Success) {
        return Status::InvalidLayout;
    }
    result.shape = {batch, heads, {MlaSequenceLayout::Bsnd, batch * sq, sq}, 0, 0};
    status = PrepareResources(platform, dtype, hasUsedQuery, result);
    if (status == Status::Success) {
        output = result;
    }
    return status;
}

// TND addresses use allocation prefixes; used lengths only restrict attention.
MlaPagedPlanStatus BuildPagedTndPlan(const MlaTndTensorView& q, const MlaTndTensorView& qv, const MlaBsndTensorView& k,
                                     const MlaBsndTensorView& v, const MlaPageTableView& table, int64_t maxQueryLength,
                                     const MlaPlatformInfo& platform, uint32_t dtype, bool hasUsedQuery,
                                     MlaPagedTndPlan& output)
{
    using Status = MlaPagedPlanStatus;
    const int64_t batch = table.sizes[0], totalQ = q.sizes[0], heads = q.sizes[1];
    auto status = ValidateGeometry(batch, maxQueryLength, heads, true, platform, dtype);
    if (status != Status::Success) {
        return status;
    }
    // Validated B/maxQ bound their product below INT64_MAX. Prefix entries
    // are int32, while output addresses are checked against actual total Q.
    if (q.sizes[2] != 64 || qv.sizes[2] != 512 || qv.sizes[0] != totalQ || qv.sizes[1] != heads || totalQ < 0 ||
        totalQ > INT32_MAX || uint64_t(totalQ) > kMlaMaxBatchRows / heads || totalQ > batch * maxQueryLength) {
        return Status::InvalidShape;
    }
    MlaPagedTndPlan result{};
    MlaTndTensorLayout qLayout{}, qvLayout{};
    if (BuildTndTensorLayout(q, qLayout) != MlaLayoutStatus::Success ||
        BuildTndTensorLayout(qv, qvLayout) != MlaLayoutStatus::Success ||
        BuildPagedCacheLayout(table, k, v, result.cache) != MlaLayoutStatus::Success) {
        return Status::InvalidLayout;
    }
    result.layouts.q = {0, qLayout.tokenStride, qLayout.headStride, qLayout.spanElements};
    result.layouts.qv = {0, qvLayout.tokenStride, qvLayout.headStride, qvLayout.spanElements};
    result.shape = {batch, heads, {MlaSequenceLayout::Tnd, totalQ, maxQueryLength}, 0, 0};
    status = PrepareResources(platform, dtype, hasUsedQuery, result);
    if (status == Status::Success) {
        output = result;
    }
    return status;
}

} // namespace flash_attn::mla

extern __global__ __aicpu__ uint32_t ComputeMlaDenseMetadata(void* args);
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
    // Only immutable host data is retained. Submissions own the candidate even
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
    if (onCurrentStream) {
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
void SetPagedSplitBudget(MlaPagedPlan& plan, int64_t splits)
{
    const bool tnd = plan.shape.query.layout == MlaSequenceLayout::Tnd;
    if (tnd) {
        const uint32_t blocks =
            plan.shape.batch && plan.shape.query.totalTokens && plan.shape.logicalKvCapacity ? plan.platform.aicNum : 1;
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
        TORCH_CHECK(BuildSplitPagedBsndBudget(plan.shape, plan.platform, splits, budget) == MlaScheduleStatus::Success,
                    "MLA: paged split workspace or accumulator overflow");
        plan.totalWorkspaceBytes = budget.totalBytes;
        plan.context.workspaceBytes = budget.userBytes;
    }
}

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
    if (isSplit)
        SetPagedSplitBudget(plan, splits);
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
                tnd           ? (splits == 0
                                     ? (q.scalar_type() == at::kBFloat16 ? LaunchAutoSplitPaged<bfloat16_t, true>(splitArgs)
                                                                         : LaunchAutoSplitPaged<half, true>(splitArgs))
                                     : (q.scalar_type() == at::kBFloat16 ? LaunchSplitPagedTnd<bfloat16_t>(splitArgs)
                                                                         : LaunchSplitPagedTnd<half>(splitArgs)))
                : splits == 0 ? (q.scalar_type() == at::kBFloat16 ? LaunchAutoSplitPaged<bfloat16_t, false>(splitArgs)
                                                                  : LaunchAutoSplitPaged<half, false>(splitArgs))
                              : (q.scalar_type() == at::kBFloat16 ? LaunchSplitPagedBsnd<bfloat16_t>(splitArgs)
                                                                  : LaunchSplitPagedBsnd<half>(splitArgs));
            TORCH_CHECK(error == ACL_SUCCESS, "MLA: paged split BATCH_MODE launch failed (", error, ")");
            return 0;
        }
        const auto error =
            tnd ? (q.scalar_type() == at::kBFloat16 ? LaunchPaged<bfloat16_t, true>(a) : LaunchPaged<half, true>(a))
                : (q.scalar_type() == at::kBFloat16 ? LaunchPaged<bfloat16_t, false>(a) : LaunchPaged<half, false>(a));
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
        auto error = q.scalar_type() == at::kBFloat16 ? LaunchDense<bfloat16_t>(a) : LaunchDense<half>(a);
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
std::pair<uint64_t, uint64_t> UsedWorkspaceBytes(const MlaUsedShape& shape, const MlaPlatformInfo& p, int64_t splits,
                                                 bool hasSoftcap)
{
    const int64_t batch = shape.batch, sq = shape.query.maxLength, heads = shape.queryHeads, sk = shape.kv.maxLength;
    const bool tnd = shape.query.layout == MlaSequenceLayout::Tnd;
    const bool nonempty = batch != 0 && shape.query.totalTokens != 0 && shape.kv.totalTokens != 0;
    // Static allocation bounds only; source scheduling reads used on AICPU.
    const uint64_t m = (std::min(uint64_t(sq) * heads, uint64_t(256)) + 15) / 16 * 16;
    const uint64_t n = (std::min(uint64_t(sk), uint64_t(512)) + 31) / 32 * 32;
    uint64_t userBytes = nonempty ? uint64_t(2) * p.aicNum * (6 * m * n + 8 * m * 512 + 8 * 256) : 0;
    const uint64_t libBytes = nonempty ? p.libApiWorkspaceBytes : 0;
    if (splits > 1 || (splits == 1 && hasSoftcap)) {
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
    return {userBytes, libBytes};
}

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
    const auto [userBytes, libBytes] = UsedWorkspaceBytes(shape, p, splits, softcap > 0);
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
                error = splits == 0
                            ? (q.scalar_type() == at::kBFloat16 ? LaunchAutoSplit<bfloat16_t, true, true>(mixedArgs)
                                                                : LaunchAutoSplit<half, true, true>(mixedArgs))
                            : (q.scalar_type() == at::kBFloat16 ? LaunchSplitTnd<bfloat16_t, true>(mixedArgs)
                                                                : LaunchSplitTnd<half, true>(mixedArgs));
            } else if (tnd) {
                MlaSplitTndLaunchArgs tndArgs{a, uint32_t(splits), static_cast<uint8_t*>(accum.data_ptr()),
                                              static_cast<uint8_t*>(accumLse.data_ptr())};
                error = splits == 0
                            ? (q.scalar_type() == at::kBFloat16 ? LaunchAutoSplit<bfloat16_t, true, false>(tndArgs)
                                                                : LaunchAutoSplit<half, true, false>(tndArgs))
                            : (q.scalar_type() == at::kBFloat16 ? LaunchSplitTnd<bfloat16_t, false>(tndArgs)
                                                                : LaunchSplitTnd<half, false>(tndArgs));
            } else
                error = splits == 0
                            ? (q.scalar_type() == at::kBFloat16 ? LaunchAutoSplit<bfloat16_t, false, false>(splitArgs)
                                                                : LaunchAutoSplit<half, false, false>(splitArgs))
                            : (q.scalar_type() == at::kBFloat16 ? LaunchSplitBsnd<bfloat16_t>(splitArgs)
                                                                : LaunchSplitBsnd<half>(splitArgs));
        } else
            error = tnd ? (q.scalar_type() == at::kBFloat16 ? LaunchTndUsed<bfloat16_t>(a) : LaunchTndUsed<half>(a))
                        : (q.scalar_type() == at::kBFloat16 ? LaunchBsndUsed<bfloat16_t>(a) : LaunchBsndUsed<half>(a));
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
            auto error = q.scalar_type() == at::kBFloat16 ? LaunchTnd<bfloat16_t>(a) : LaunchTnd<half>(a);
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
            auto error = q.scalar_type() == at::kBFloat16 ? LaunchTndBsnd<bfloat16_t>(a) : LaunchTndBsnd<half>(a);
            TORCH_CHECK(error == ACL_SUCCESS, "MLA: mixed BATCH_MODE kernel launch failed (", error, ")");
            return 0;
        });
    return {out, lse, at::Tensor(), at::Tensor()};
}

// The public read-only cache path has already rejected unsupported options.
// Keep tensor/scale checks here as well: the dispatcher is independently callable.

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
        const auto error = q.scalar_type() == at::kHalf ? LaunchGqa<half>(a) : LaunchGqa<bfloat16_t>(a);
        TORCH_CHECK(error == ACL_SUCCESS, "MLA: GQA BATCH_MODE launch failed (", error, ")");
        return 0;
    });
    return {out, lse, accum, accumLse};
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
        const auto error = q.scalar_type() == at::kHalf ? LaunchPrefillBsnd<half>(a) : LaunchPrefillBsnd<bfloat16_t>(a);
        TORCH_CHECK(error == ACL_SUCCESS, "MLA: prefill BATCH_MODE launch failed (", error, ")");
        return 0;
    });
    if (!tnd && !c.pageSize && !c.sk)
        lse.fill_(std::numeric_limits<float>::infinity());
    return {out, lse, accum, accumLse};
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

namespace {
// Meta inference checks allocation bounds against the supported core envelope;
// actual launches still query and validate the selected device's resources.
MlaPlatformInfo ShapePlatform()
{
    MlaPlatformInfo p{};
    p.aicNum = kMlaMaxAic;
    p.aivNum = 2 * p.aicNum;
    p.localMemory = {512 * 1024, 256 * 1024, 64 * 1024, 64 * 1024, 128 * 1024};
    for (uint32_t i = 0; i < p.aicNum; ++i)
        p.blockDims[i] = i + 1;
    return p;
}

uint32_t ShapeMetadataWords(int64_t batch, int64_t sq, int64_t sk, int64_t heads, int64_t kvHeads, int64_t dim,
                            int64_t valueDim, at::ScalarType dtype, int64_t splits, int64_t left, int64_t right,
                            bool causal, int64_t page, uint32_t layout, int64_t totalQ, int64_t totalK,
                            std::optional<bool> pack, int64_t margin, bool hasSoftcap, bool usedQ = false)
{
    TORCH_CHECK((dim == 64 && valueDim == 512 && kvHeads == 1) || (dim == 192 && valueDim == 128),
                "MLA: expected compressed 64/512 or prefill 192/128 dimensions");
    TORCH_CHECK(dtype == at::kHalf || dtype == at::kBFloat16, "MLA: qkv_dtype must be FP16 or BF16");
    TORCH_CHECK(batch >= 0 && uint64_t(batch) <= UINT32_MAX && FitsDenseShape(layout ? 0 : batch, sq, heads, sk) &&
                    kvHeads > 0 && heads % kvHeads == 0 && splits >= 0 && splits <= 256 && totalQ >= 0 &&
                    totalQ <= INT32_MAX && totalK >= 0 && totalK <= INT32_MAX,
                "MLA: shape bounds or split bounds exceeded");
    TORCH_CHECK(page == 0 || (page >= 16 && page <= 1024 && page % 16 == 0 && sk % page == 0),
                "MLA: invalid page size or capacity");
    CheckWindow({sq, sk, left, right, causal});
    if (margin != 0 || (pack && !*pack && heads != kvHeads)) {
        auto c = GqaContext(batch, sq, sk, heads, kvHeads, valueDim, dtype, splits, left, right, causal, 0,
                            ShapePlatform(), layout, totalQ, totalK, page, hasSoftcap, pack, margin);
        MlaGqaLayout l{};
        TORCH_CHECK(GqaLayout(c, l), "MLA: GQA metadata/workspace overflow");
        return l.words;
    }
    if (dim == 192) {
        auto c = PrefillContext(batch, sq, sk, heads, kvHeads, dtype, splits, left, right, causal, 0, ShapePlatform(),
                                layout == 1                    ? 1
                                : (layout == 2 || layout == 4) ? 2
                                                               : 0,
                                totalQ, totalK, page, hasSoftcap);
        MlaPrefillBsndLayout l{};
        TORCH_CHECK(PrefillBsndLayout(c, l), "MLA: prefill metadata/workspace overflow");
        return l.words;
    }
    const bool split = splits != 1 || hasSoftcap;
    const bool window = left != -1 || right != -1;
    uint32_t words = 0;
    bool valid = true;
    if (page) {
        valid = splits == 0 ? (layout == 4 ? AutoSplitPagedTndMetadataWords(batch, words)
                                           : AutoSplitPagedBsndMetadataWords(batch, words))
                : split     ? (layout == 4 ? SplitPagedTndMetadataWords(batch, words)
                                           : SplitPagedBsndMetadataWords(batch, words))
                : window    ? WindowPagedMetadataWords(batch, words)
                            : batch <= (UINT32_MAX - kMlaPagedMetadataPrefixWords) / kMlaPagedMetadataSequenceWords;
        if (!split && !window)
            words = kMlaPagedMetadataPrefixWords + batch * kMlaPagedMetadataSequenceWords;
    } else if (split) {
        valid = splits == 0   ? AutoSplitBsndMetadataWords(batch, words)
                : layout == 1 ? SplitTndMetadataWords(batch, words)
                : layout == 2 ? SplitMixedMetadataWords(batch, words)
                              : SplitBsndMetadataWords(batch, words);
    } else if (window) {
        valid = WindowMetadataWords(batch, words);
    } else if (causal || usedQ) {
        valid = batch <= (UINT32_MAX - kMlaUsedMetadataPrefixWords) / kMlaUsedMetadataSequenceWords;
        words = kMlaUsedMetadataPrefixWords + batch * kMlaUsedMetadataSequenceWords;
    } else if (layout == 1 || layout == 2) {
        const auto prefix = layout == 1 ? kMlaTndMetadataPrefixWords : kMlaTndBsndMetadataPrefixWords;
        const auto sequence = layout == 1 ? kMlaTndMetadataSequenceWords : kMlaTndBsndMetadataSequenceWords;
        valid = batch <= (UINT32_MAX - prefix) / sequence;
        words = prefix + batch * sequence;
    } else {
        words = CheckBsndMetadataWords(batch);
    }
    TORCH_CHECK(valid, "MLA: metadata word count overflow");
    return words;
}

std::vector<at::Tensor> ForwardMeta(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                    const std::optional<at::Tensor>& qv, const std::optional<at::Tensor>& cuQ,
                                    const std::optional<at::Tensor>& cuK, const std::optional<at::Tensor>& usedQ,
                                    const std::optional<at::Tensor>& usedK, std::optional<int64_t> maxQ,
                                    std::optional<int64_t> maxK, const std::optional<at::Tensor>& table,
                                    std::optional<float> scale, bool causal, int64_t left, int64_t right, float softcap,
                                    const std::optional<at::Tensor>& metadata, int64_t splits, std::optional<bool> pack,
                                    int64_t margin)
{
    TORCH_CHECK(!scale || std::isfinite(*scale), "MLA: softmax_scale must be finite in FP32");
    TORCH_CHECK(std::isfinite(softcap) && softcap >= 0, "MLA: softcap must be finite and non-negative in FP32");
    TORCH_CHECK(!cuQ || (maxQ && (!cuK || maxK)), "MLA: TND requires max_seqlen_q/k");
    for (const auto& tensor : {q, k, v})
        tensor.dim() == 3 ? CheckTndTensorLayout(tensor) : CheckBsndTensorLayout(tensor);
    if (qv)
        qv->dim() == 3 ? CheckTndTensorLayout(*qv) : CheckBsndTensorLayout(*qv);
    const int64_t batch = cuQ ? cuQ->numel() - 1 : q.size(0);
    const int64_t sq = cuQ ? *maxQ : q.size(1), heads = q.size(-2);
    if (cuQ)
        CheckMetadataQueryPrefix(*cuQ, batch, q.device());
    if (cuK)
        CheckTndInputs(q, k, *cuQ, *cuK, *maxQ, *maxK);
    else if (cuQ && !table)
        CheckTndBsndForwardInputs(q, k, *cuQ, *maxQ, usedK, std::nullopt);
    CheckUsedInputs(q, batch, usedQ, usedK);
    MlaPagedCacheLayout cache{};
    if (table) {
        TORCH_CHECK(table->dim() == 2 && table->size(0) == batch && table->scalar_type() == at::kInt &&
                        table->device() == q.device(),
                    "MLA: page_table must be same-device int32[batch_size, max_pages]");
        TORCH_CHECK(BuildPagedCacheLayout(PagedTensorView<MlaPageTableView>(*table), PagedTensorView(k),
                                          PagedTensorView(v), cache, !qv) == MlaLayoutStatus::Success,
                    "MLA: paged cache shape, stride or storage bounds exceeded");
        TORCH_CHECK(!cuQ || q.size(0) <= batch * sq, "MLA: TND paged query allocation bounds exceeded");
    }
    const int64_t sk = table ? cache.logicalCapacity : cuK ? *maxK : k.size(1);
    const uint32_t layout = table ? (cuQ ? 4 : 3) : cuK ? 1 : cuQ ? 2 : 0;
    const auto words =
        ShapeMetadataWords(batch, sq, sk, heads, k.size(-2), q.size(-1), v.size(-1), q.scalar_type(), splits, left,
                           right, causal, table ? k.size(1) : 0, layout, cuQ ? q.size(0) : 0, cuK ? k.size(0) : 0, pack,
                           margin, softcap > 0, bool(usedQ));
    const bool gqa = margin != 0 || (pack && !*pack && heads != k.size(-2));
    if (qv && !gqa) {
        const auto platform = ShapePlatform();
        if (table) {
            MlaPagedPlan plan{};
            const auto dtype =
                uint32_t(q.scalar_type() == at::kBFloat16 ? MlaMetadataDtype::Bf16 : MlaMetadataDtype::Fp16);
            const auto tableView = PagedTensorView<MlaPageTableView>(*table);
            const auto status =
                cuQ ? BuildPagedTndPlan(PagedTensorView<MlaTndTensorView>(q), PagedTensorView<MlaTndTensorView>(*qv),
                                        PagedTensorView(k), PagedTensorView(v), tableView, sq, platform, dtype,
                                        bool(usedQ), plan)
                    : BuildPagedBsndPlan(PagedTensorView(q), PagedTensorView(*qv), PagedTensorView(k),
                                         PagedTensorView(v), tableView, platform, dtype, bool(usedQ), plan);
            TORCH_CHECK(status == MlaPagedPlanStatus::Success, "MLA: paged shape, stride or storage bounds exceeded");
            if (splits != 1 || softcap > 0)
                SetPagedSplitBudget(plan, splits);
        } else {
            const MlaUsedShape shape{
                batch,
                heads,
                {cuQ ? MlaSequenceLayout::Tnd : MlaSequenceLayout::Bsnd, cuQ ? q.size(0) : batch * sq, sq},
                {cuK ? MlaSequenceLayout::Tnd : MlaSequenceLayout::Bsnd, cuK ? k.size(0) : batch * sk, sk}};
            UsedWorkspaceBytes(shape, platform, splits, softcap > 0);
        }
    }
    if (metadata) {
        TORCH_CHECK(!usedQ || (cuQ && (!qv || (gqa && cuK))), "MLA: Q-used explicit metadata is unsupported");
        TORCH_CHECK(!qv || !cuK || gqa || splits != 1 || softcap > 0,
                    "MLA: compact TND explicit metadata is unsupported");
        CheckGqaRecord(*metadata, q.device(), words);
    }
    auto shape = q.sizes().vec();
    shape.back() = v.size(-1);
    auto lseShape = cuQ ? std::vector<int64_t>{heads, q.size(0)} : std::vector<int64_t>{batch, heads, sq};
    auto out = at::empty(shape, q.options());
    auto lse = at::empty(lseShape, q.options().dtype(at::kFloat));
    if (splits == 1)
        return {out, lse, at::empty({0}, at::TensorOptions().device(q.device())),
                at::empty({0}, at::TensorOptions().device(q.device()))};
    const int64_t capacity = splits ? splits : kMlaMaxAic;
    unsigned __int128 bytes =
        uint64_t(capacity) * static_cast<unsigned __int128>(heads) * (cuQ ? q.size(0) : batch * sq);
    TORCH_CHECK(bytes * v.size(-1) * sizeof(float) <= INT64_MAX, "MLA: accumulator byte bounds exceeded");
    lseShape.insert(lseShape.begin(), capacity);
    auto accumShape = lseShape;
    accumShape.push_back(v.size(-1));
    return {out, lse, at::empty(accumShape, q.options().dtype(at::kFloat)),
            at::empty(lseShape, q.options().dtype(at::kFloat))};
}
} // namespace

at::Tensor SchedulerMetadata(int64_t batch, int64_t sq, int64_t sk, int64_t heads, int64_t kvHeads, int64_t dim,
                             int64_t valueDim, at::ScalarType dtype, const at::Tensor& lengths,
                             const std::optional<at::Tensor>& cuQ, std::optional<int64_t> page, int64_t left,
                             int64_t right, bool causal, bool hasSoftcap, int64_t splits, std::optional<bool> pack,
                             int64_t margin)
{
    CheckCacheLengths(lengths, batch, lengths.device());
    if (cuQ)
        CheckMetadataQueryPrefix(*cuQ, batch, lengths.device());
    TORCH_CHECK(!page || *page > 0, "MLA: page_size must be positive");
    const auto layout = page ? (cuQ ? 4U : 3U) : cuQ ? 2U : 0U;
    const auto words = ShapeMetadataWords(batch, sq, sk, heads, kvHeads, dim, valueDim, dtype, splits, left, right,
                                          causal, page.value_or(0), layout, 0, 0, pack, margin, hasSoftcap);
    if (lengths.is_meta())
        return at::empty({words}, lengths.options().dtype(at::kInt));
    if (margin != 0 || (pack && !*pack && heads != kvHeads))
        return GqaMetadataOp(batch, sq, sk, heads, kvHeads, valueDim, dtype, lengths, cuQ, splits, left, right, causal,
                             page.value_or(0), hasSoftcap, pack, margin);
    if (dim == 192) {
        if (cuQ)
            return PrefillMixedMetadataOp(batch, sq, sk, heads, kvHeads, dtype, lengths, *cuQ, splits, left, right,
                                          causal, std::nullopt, page.value_or(0), hasSoftcap);
        return PrefillBsndMetadataOp(batch, sq, sk, heads, kvHeads, dtype, lengths, splits, left, right, causal,
                                     page.value_or(0), hasSoftcap);
    }
    if (page)
        return PagedMetadataOp(batch, sq, sk, heads, *page, dtype, lengths, cuQ, causal, left, right, splits,
                               hasSoftcap);
    if (splits != 1 || hasSoftcap) {
        if (cuQ)
            return SplitMixedMetadataOp(batch, sq, sk, heads, dtype, lengths, *cuQ, left, right, causal, splits,
                                        hasSoftcap);
        return SplitBsndMetadataOp(batch, sq, sk, heads, dtype, lengths, left, right, causal, splits, hasSoftcap);
    }
    if (left != -1 || right != -1) {
        if (cuQ)
            return WindowMixedMetadataOp(batch, sq, sk, heads, dtype, lengths, *cuQ, left, right, causal);
        return WindowBsndMetadataOp(batch, sq, sk, heads, dtype, lengths, left, right, causal);
    }
    if (cuQ)
        return causal ? CausalMixedMetadataOp(batch, sq, sk, heads, dtype, lengths, *cuQ)
                      : TndBsndMetadataOp(batch, sq, sk, heads, dtype, lengths, *cuQ);
    return causal ? CausalBsndMetadataOp(batch, sq, sk, heads, dtype, lengths)
                  : DenseMetadataOp(batch, sq, sk, heads, dtype, lengths);
}

std::vector<at::Tensor> Forward(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                                const std::optional<at::Tensor>& qv, const std::optional<at::Tensor>& cuQ,
                                const std::optional<at::Tensor>& cuK, const std::optional<at::Tensor>& usedQ,
                                const std::optional<at::Tensor>& usedK, std::optional<int64_t> maxQ,
                                std::optional<int64_t> maxK, const std::optional<at::Tensor>& pageTable,
                                std::optional<float> softmaxScale, bool causal, int64_t left, int64_t right,
                                float softcap, const std::optional<at::Tensor>& metadata, int64_t splits,
                                std::optional<bool> pack, int64_t margin)
{
    if (q.is_meta())
        return ForwardMeta(q, k, v, qv, cuQ, cuK, usedQ, usedK, maxQ, maxK, pageTable, softmaxScale, causal, left,
                           right, softcap, metadata, splits, pack, margin);

    TORCH_CHECK(std::isfinite(softcap) && softcap >= 0, "MLA: softcap must be finite and non-negative in FP32");
    const float scale = softmaxScale.value_or(1.0f / std::sqrt(qv ? 576.0f : 192.0f));
    TORCH_CHECK(std::isfinite(scale), "MLA: softmax_scale must be finite in FP32");
    if (margin != 0 || (pack.has_value() && !*pack && q.size(-2) != k.size(-2))) {
        TORCH_CHECK(!cuQ || (maxQ && (!cuK || maxK)), "MLA: compact TND requires max_seqlen_q/k");
        return GqaForward(q, k, v, qv, splits, left, right, causal, scale, usedQ, usedK, metadata, cuQ, cuK,
                          maxQ.value_or(0), maxK.value_or(0), pageTable, softcap, pack, margin);
    }
    if (!qv) {
        TORCH_CHECK(!cuQ || (maxQ && (!cuK || maxK)), "MLA: compact TND requires max_seqlen_q/k");
        return PrefillBsndForward(q, k, v, scale, splits, left, right, causal, usedQ, usedK, metadata, cuQ, cuK,
                                  maxQ.value_or(0), maxK.value_or(0), pageTable, softcap);
    }
    // Paged kernels accept mask and split options directly for both query layouts.
    if (pageTable) {
        if (cuQ) {
            TORCH_CHECK(maxQ, "MLA: TND paged forward requires max_seqlen_q");
            return PagedTndForward(q, k, v, *qv, *pageTable, *cuQ, *maxQ, scale, usedK, usedQ, metadata, causal, left,
                                   right, splits, softcap);
        }
        return PagedBsndForward(q, k, v, *qv, *pageTable, scale, usedK, usedQ, metadata, causal, left, right, splits,
                                softcap);
    }
    const bool split = splits != 1 || softcap > 0;
    const bool window = left != -1 || right != -1;
    if (cuQ) {
        TORCH_CHECK(maxQ, "MLA: TND forward requires max_seqlen_q");
        if (cuK) {
            TORCH_CHECK(maxK, "MLA: compact TND requires max_seqlen_q/k");
            if (split) {
                return SplitTndForward(q, k, v, *qv, *cuQ, *cuK, *maxQ, *maxK, scale, left, right, causal, splits,
                                       usedQ, usedK, metadata, softcap);
            }
            TORCH_CHECK(!metadata, "MLA compressed forward is not implemented yet for TND explicit metadata");
            if (window) {
                return WindowTndForward(q, k, v, *qv, *cuQ, *cuK, *maxQ, *maxK, scale, left, right, causal, usedQ,
                                        usedK);
            }
            return TndForward(q, k, v, *qv, *cuQ, *cuK, *maxQ, *maxK, scale, usedQ, usedK, causal);
        }
        if (split) {
            return SplitMixedForward(q, k, v, *qv, *cuQ, *maxQ, scale, left, right, causal, splits, usedK, usedQ,
                                     metadata, softcap);
        }
        if (window) {
            return WindowMixedForward(q, k, v, *qv, *cuQ, *maxQ, scale, left, right, causal, usedK, usedQ, metadata);
        }
        if (causal) {
            return CausalMixedForward(q, k, v, *qv, *cuQ, *maxQ, scale, usedK, usedQ, metadata);
        }
        if (usedQ) {
            TORCH_CHECK(!metadata, "MLA compressed forward is not implemented yet for Q-used with explicit metadata");
            return TndBsndUsedForward(q, k, v, *qv, *cuQ, *maxQ, *usedQ, usedK, scale);
        }
        return TndBsndForward(q, k, v, *qv, *cuQ, *maxQ, scale, usedK, metadata);
    }
    if (split) {
        return SplitBsndForward(q, k, v, *qv, scale, left, right, causal, splits, usedK, usedQ, metadata, softcap);
    }
    if (window) {
        return WindowBsndForward(q, k, v, *qv, scale, left, right, causal, usedK, usedQ, metadata);
    }
    if (causal) {
        return CausalBsndForward(q, k, v, *qv, scale, usedK, usedQ, metadata);
    }
    if (usedQ) {
        TORCH_CHECK(!metadata, "MLA compressed forward is not implemented yet for Q-used with explicit metadata");
        return BsndUsedForward(q, k, v, *qv, *usedQ, usedK, scale);
    }
    if (usedK) {
        CheckCacheLengths(*usedK, q.size(0), q.device());
    }
    if (metadata) {
        CheckDenseMetadataTensor(*metadata, q.device(), q.size(0));
    }
    return DenseForward(q, k, v, *qv, scale, usedK, metadata);
}

// Adapt the existing public inference arguments without changing ordinary FA3.
std::optional<std::vector<at::Tensor>>
TryForward(at::Tensor q, at::Tensor k, at::Tensor v, std::optional<at::Tensor> k_new_, std::optional<at::Tensor> v_new_,
           std::optional<at::Tensor> q_v_, std::optional<at::Tensor> out_, std::optional<at::Tensor> cu_seqlens_q_,
           std::optional<at::Tensor> cu_seqlens_k_, std::optional<at::Tensor> cu_seqlens_k_new_,
           std::optional<at::Tensor> seqused_q_, std::optional<at::Tensor> seqused_k_,
           std::optional<int64_t> max_seqlen_q_, std::optional<int64_t> max_seqlen_k_,
           std::optional<at::Tensor> page_table_, std::optional<at::Tensor> kv_batch_idx_,
           std::optional<at::Tensor> leftpad_k_, std::optional<at::Tensor> rotary_cos_,
           std::optional<at::Tensor> rotary_sin_, std::optional<at::Tensor> seqlens_rotary_,
           std::optional<at::Tensor> q_descale_, std::optional<at::Tensor> k_descale_,
           std::optional<at::Tensor> v_descale_, std::optional<double> softmax_scale_, bool is_causal,
           int64_t window_size_left, int64_t window_size_right, int64_t attention_chunk, double softcap_option,
           bool is_rotary_interleaved, std::optional<at::Tensor> scheduler_metadata_, int64_t num_splits,
           std::optional<bool> pack_gqa_, int64_t sm_margin)
{
    const float softcap = static_cast<float>(softcap_option);
    const bool isMla = flash_attn::mla::CheckForwardBoundary(q, k, v, q_v_, cu_seqlens_q_.has_value(),
                                                             cu_seqlens_k_.has_value(), page_table_.has_value());
    if (!isMla)
        return std::nullopt;
    const c10::OptionalDeviceGuard device_guard(device_of(q));
    TORCH_CHECK(std::isfinite(softcap_option) && std::isfinite(softcap) && softcap_option >= 0 &&
                    (softcap_option == 0 || softcap > 0),
                "MLA: softcap must be finite and non-negative in FP32 without positive underflow");
    TORCH_CHECK(!k_new_ && !v_new_ && !out_ && !cu_seqlens_k_new_ && (cu_seqlens_q_ || !max_seqlen_q_) &&
                    (cu_seqlens_k_ || !max_seqlen_k_) && !kv_batch_idx_ && !leftpad_k_ && !rotary_cos_ &&
                    !rotary_sin_ && !seqlens_rotary_ && !q_descale_ && !k_descale_ && !v_descale_ &&
                    is_rotary_interleaved,
                "MLA: k/v, out, cu_seqlens_k_new, cache_batch_idx, cache_leftpad, rotary_cos, rotary_sin, "
                "rotary_seqlens, q_descale/k_descale/v_descale and length/rotary options are unsupported");
    TORCH_CHECK(attention_chunk == 0 &&
                    (num_splits == 1 || (num_splits >= 0 && num_splits <= 256 && (!cu_seqlens_k_ || cu_seqlens_q_))),
                "MLA compressed forward is not implemented yet for mask, split, softcap or scheduling options");
    auto result =
        flash_attn::mla::Forward(q, k, v, q_v_, cu_seqlens_q_, cu_seqlens_k_, seqused_q_, seqused_k_, max_seqlen_q_,
                                 max_seqlen_k_, page_table_, softmax_scale_, is_causal, window_size_left,
                                 window_size_right, softcap, scheduler_metadata_, num_splits, pack_gqa_, sm_margin);
    // Eager and Meta must both return tensors for the public accumulator slots.
    for (size_t i = 2; i < result.size(); ++i) {
        if (!result[i].defined())
            result[i] = at::empty({0}, at::TensorOptions().device(q.device()));
    }
    return result;
}

at::Tensor SchedulerMetadataOp(int64_t batch_size, int64_t max_seqlen_q, int64_t max_seqlen_k, int64_t num_heads_q,
                               int64_t num_heads_kv, int64_t headdim, int64_t headdim_v, at::ScalarType qkv_dtype,
                               at::Tensor cache_seqlens, std::optional<at::Tensor> seqlens_q,
                               std::optional<at::Tensor> cu_seqlens_k, std::optional<at::Tensor> cu_seqlens_k_new,
                               std::optional<at::Tensor> cache_leftpad, std::optional<int64_t> page_size,
                               int64_t max_seqlen_k_new, bool causal, int64_t window_size_left,
                               int64_t window_size_right, int64_t attention_chunk, bool has_softcap, int64_t num_splits,
                               std::optional<bool> pack_gqa, int64_t sm_margin)
{
    const auto dtype = qkv_dtype;
    TORCH_CHECK(headdim != headdim_v, "MLA: expected compressed or prefill dimensions");
    TORCH_CHECK(!cu_seqlens_k && !cu_seqlens_k_new && !cache_leftpad && max_seqlen_k_new == 0 && attention_chunk == 0,
                "MLA: metadata append, leftpad and attention_chunk are unsupported");
    return flash_attn::mla::SchedulerMetadata(batch_size, max_seqlen_q, max_seqlen_k, num_heads_q, num_heads_kv,
                                              headdim, headdim_v, dtype, cache_seqlens, seqlens_q, page_size,
                                              window_size_left, window_size_right, causal, has_softcap, num_splits,
                                              pack_gqa, sm_margin);
}

} // namespace flash_attn::mla
