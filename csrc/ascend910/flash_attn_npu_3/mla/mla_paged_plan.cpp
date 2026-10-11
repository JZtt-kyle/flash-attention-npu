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
#include "mla_paged_plan.h"

#include <algorithm>

namespace flash_attn::mla {

MlaPagedPlanStatus BuildPagedBsndPlan(const MlaBsndTensorView& q, const MlaBsndTensorView& qv,
                                      const MlaBsndTensorView& k, const MlaBsndTensorView& v,
                                      const MlaPageTableView& table, const MlaPlatformInfo& platform, uint32_t dtype,
                                      bool hasUsedQuery, MlaPagedBsndPlan& output)
{
    using Status = MlaPagedPlanStatus;
    if (dtype != uint32_t(MlaMetadataDtype::Fp16) && dtype != uint32_t(MlaMetadataDtype::Bf16)) {
        return Status::UnsupportedDtype;
    }
    const int64_t batch = q.sizes[0], sq = q.sizes[1], heads = q.sizes[2];
    if (q.sizes[3] != 64 || qv.sizes[3] != 512 || qv.sizes[0] != batch || qv.sizes[1] != sq || qv.sizes[2] != heads ||
        table.sizes[0] != batch || batch < 0 ||
        batch > (UINT32_MAX - kMlaPagedMetadataPrefixWords) / kMlaPagedMetadataSequenceWords) {
        return Status::InvalidShape;
    }
    if (platform.aicNum == 0 || platform.aicNum > kMlaMaxAic || platform.aivNum != 2 * platform.aicNum) {
        return Status::InvalidPlatform;
    }
    // Reuse static shape, platform and local-memory checks. sk=0 guarantees
    // this call never runs SplitCore or allocates per-batch scheduling arrays.
    MlaSchedule validation{};
    const auto status = BuildDenseSchedule({batch, sq, heads, 0}, platform, validation);
    if (status == MlaScheduleStatus::InvalidPlatform) {
        return Status::InvalidPlatform;
    }
    if (status == MlaScheduleStatus::InsufficientLocalMemory) {
        return Status::InsufficientLocalMemory;
    }
    if (status != MlaScheduleStatus::Success) {
        return Status::InvalidShape;
    }
    MlaPagedBsndPlan result{};
    if (BuildBsndTensorLayout(q, result.layouts.q) != MlaLayoutStatus::Success ||
        BuildBsndTensorLayout(qv, result.layouts.qv) != MlaLayoutStatus::Success ||
        BuildPagedCacheLayout(table, k, v, result.cache) != MlaLayoutStatus::Success) {
        return Status::InvalidLayout;
    }
    const uint64_t capacity = result.cache.logicalCapacity, page = result.cache.pageSize;
    if (capacity > kMlaMaxKvLength) {
        return Status::InvalidShape;
    }
    const bool nonempty = batch != 0 && sq != 0 && capacity != 0;
    const uint32_t blocks = platform.blockDims[nonempty ? platform.aicNum - 1 : 0];
    // C1V2 needs room for every possible working Cube, while static empties
    // use the single validation core. Never infer blockDim from device data.
    if (blocks != (nonempty ? platform.aicNum : 1U)) {
        return Status::InvalidPlatform;
    }
    const uint64_t totalQ = uint64_t(batch) * sq;
    result.shape = {batch, heads, {MlaSequenceLayout::Bsnd, int64_t(totalQ), sq}, int64_t(capacity), int64_t(page)};
    result.platform = platform;
    result.platform.blockDims.fill(blocks);
    result.layouts.k = result.cache.k;
    result.layouts.v = result.cache.v;
    result.metadataWords = kMlaPagedMetadataPrefixWords + kMlaPagedMetadataSequenceWords * batch;

    // Source PA alignment precedes short-S2 truncation. Q-used and unequal
    // KV lengths can select M=256 even when dense/equal lengths select less.
    const uint64_t m = (std::min(uint64_t(sq) * heads, uint64_t(256)) + 15) / 16 * 16;
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
                      {0, {uint32_t(totalQ), uint32_t(totalQ >> 32)}, uint32_t(sq)},
                      uint32_t(page),
                      uint32_t(capacity),
                      platform.aicNum,
                      platform.aivNum,
                      blocks,
                      userBytes,
                      libBytes,
                      platform.localMemory};
    output = result;
    return Status::Success;
}

} // namespace flash_attn::mla
