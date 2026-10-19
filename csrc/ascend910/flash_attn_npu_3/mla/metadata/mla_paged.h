#pragma once
#include "mla_dense.h"
#include "../tiling/mla_sequence.h"
#include <type_traits>
#include "../mla_layout.h"

// Private PA protocol shared by CPU/AICPU generation and device consumers.
// Public metadata and BSND/TND paged forward use the same v6 record.
// Scheduling retains its CANN source/license in tiling/mla_sequence.cpp,
// tiling/mla_schedule.cpp and detail/cann_split_core.cpp (see LICENSE.cann).
namespace flash_attn::mla {

constexpr uint32_t kMlaPagedMetadataVersion = 6;
constexpr uint32_t kMlaMetadataPaged = 6;
constexpr uint32_t kMlaCausalPagedBsndMetadataVersion = 10;
constexpr uint32_t kMlaMetadataCausalPagedBsnd = 10;
constexpr uint32_t kMlaCausalPagedTndMetadataVersion = 11;
constexpr uint32_t kMlaMetadataCausalPagedTnd = 11;
constexpr uint32_t kMlaPagedUsedQueryPresent = 1;

enum class MlaPagedMetadataStatus : uint32_t {
    Success = 0,
    UnsupportedDtype = 1,
    ScheduleFailure = 2,
    InvalidLengths = 3
};

// Tokens in the allocated Q view, never used tokens or byte addresses.
// layout: 0=BSND, 1=TND. maxLength: BSND capacity or TND prefix bound.
struct MlaPagedMetadataQuery {
    uint32_t layout;
    MlaMetadataWide totalTokens;
    uint32_t maxLength;
};

// Independent invocation geometry/resources, not inferred from the record.
// Length descriptors are MlaPagedInputs, also supplied by the invocation.
// workspaceBytes excludes the libapi reservation. Physical pool/page-table
// geometry and all strides belong to separate invocation/address validation.
struct MlaPagedMetadataContext {
    uint32_t dtype, batch, heads, usedFlags;
    MlaPagedMetadataQuery query;
    uint32_t pageSize, logicalKvCapacity;
    uint32_t aicNum, aivNum, blockDim;
    uint64_t workspaceBytes, libApiBytes;
    MlaLocalMemory localMemory;
};
static_assert(std::is_standard_layout_v<MlaPagedMetadataContext> &&
              std::is_trivially_copyable_v<MlaPagedMetadataContext>);

// 401 common words, Q allocation (4), Q-used presence (1), P (1), C (1).
// schedule.s2Size is the source PA S2=ceil(max(KV length)/P)*P, NOT C.
// Followed by B five-word spans: queryStart.low/high, queryCapacity,
// queryLength, kvLength. No KV starts, page IDs, pool size, strides or scale.
struct MlaPagedMetadataPrefix {
    uint32_t magic, version, kind, wordCount;
    uint32_t dtype, status, scheduleStatus, reserved;
    MlaMetadataSchedule schedule;
    MlaPagedMetadataQuery query;
    uint32_t usedFlags, pageSize, logicalKvCapacity;
};
constexpr uint32_t kMlaPagedMetadataPrefixWords = 408;
constexpr uint32_t kMlaPagedMetadataSequenceWords = 5;
static_assert(sizeof(MlaPagedMetadataPrefix) == kMlaPagedMetadataPrefixWords * sizeof(int32_t));
static_assert(alignof(MlaPagedMetadataPrefix) == alignof(int32_t));
static_assert(offsetof(MlaPagedMetadataPrefix, schedule) == 8 * sizeof(int32_t));
static_assert(offsetof(MlaPagedMetadataPrefix, query) == 401 * sizeof(int32_t));
static_assert(offsetof(MlaPagedMetadataPrefix, usedFlags) == 405 * sizeof(int32_t));
static_assert(offsetof(MlaPagedMetadataPrefix, pageSize) == 406 * sizeof(int32_t));
static_assert(offsetof(MlaPagedMetadataPrefix, logicalKvCapacity) == 407 * sizeof(int32_t));
static_assert(std::is_standard_layout_v<MlaPagedMetadataPrefix> &&
              std::is_trivially_copyable_v<MlaPagedMetadataPrefix>);
static_assert(uint32_t(MlaSequenceLayout::Bsnd) == 0 && uint32_t(MlaSequenceLayout::Tnd) == 1);

// Exact static allocation bound; false leaves words unchanged.
bool PagedMetadataWords(int64_t batch, uint32_t& words);

// CPU/AICPU serialization of BuildPagedSchedule's owned snapshot.
// False: unsafe output size/address or any input/output overlap; no writes.
// True: complete success/failure record; failure clears every payload word
// after the eight-word header. Validate descriptors before reading values.
// KV lengths are mandatory even when Q/used is zero; null/count=0 only at B=0.
// Read-only inputs may alias each other and must remain readable/stable until
// completion. A present, zero-count Q-used array at B=0 retains its flag.
// fromQueryPrefix: for TND only, derive totalTokens from the device prefix tail
// after descriptor/overlap validation; ignore the supplied total. BSND keeps
// its explicit allocation. This does not change the wire or the CPU default.
bool BuildPagedMetadata(const MlaPagedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype, uint32_t* output,
                        uint32_t outputWords, const MlaPagedInputs& inputs, bool fromQueryPrefix = false);

using MlaPagedScheduleBuilder = MlaScheduleStatus (*)(const MlaPagedShape&, const MlaPlatformInfo&,
                                                      const MlaPagedInputs&, MlaPagedSchedule&, const void*);
// Internal serialization adapter; retains all descriptor/overlap guarantees.
bool BuildPagedMetadataWithSchedule(const MlaPagedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype,
                                    uint32_t* output, uint32_t outputWords, const MlaPagedInputs& inputs,
                                    bool fromQueryPrefix, MlaPagedScheduleBuilder builder,
                                    const void* context = nullptr, uint32_t tailWords = 0);
bool BuildCausalPagedBsndMetadata(const MlaPagedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype,
                                  uint32_t* output, uint32_t outputWords, const MlaPagedInputs& inputs);

bool BuildCausalPagedTndMetadata(const MlaPagedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype,
                                 uint32_t* output, uint32_t outputWords, const MlaPagedInputs& inputs,
                                 bool fromQueryPrefix = false);

} // namespace flash_attn::mla

// Private host/AICPU launch ABI. Addresses are never serialized in v6 records.
namespace flash_attn::mla {
struct MlaPagedMetadataArray {
    uint64_t address, count;
};
struct MlaPagedMetadataArgs {
    uint64_t outputAddr;
    uint32_t outputWords, dtype;
    MlaPagedShape shape;
    MlaPlatformInfo platform;
    MlaPagedMetadataArray cuQuery, usedQuery, kvLengths;
};
static_assert(std::is_standard_layout_v<MlaPagedMetadataArgs> && std::is_trivially_copyable_v<MlaPagedMetadataArgs>);

inline MlaPagedInputs PagedMetadataInputs(const MlaPagedMetadataArgs& args)
{
    const auto array = [](const MlaPagedMetadataArray& input) {
        return MlaLengthArray{reinterpret_cast<const int32_t*>(input.address), input.count};
    };
    return {array(args.cuQuery), array(args.usedQuery), array(args.kvLengths)};
}
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

// Invocation-only PA descriptors shared by host preparation and device
// validation. Addresses/counts start at data_ptr(), after storage offset.
namespace flash_attn::mla {
struct MlaPagedMetadataInputs {
    MlaPagedMetadataArray cuQuery, usedQuery, kvLengths;
};

// Host validates dtype/device and builds the complete physical layout. The
// device independently checks geometry and spans before reading any data.
// table counts int32 entries; k/v count FP16/BF16 elements, including padding.
struct MlaPagedCacheInputs {
    MlaPagedCacheLayout layout;
    MlaPagedMetadataArray table, k, v;
};
static_assert(std::is_standard_layout_v<MlaPagedCacheInputs> && std::is_trivially_copyable_v<MlaPagedCacheInputs>);
} // namespace flash_attn::mla
