/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include "mla_tiling.h"
#include "mla_layout.h"
#include "mla_params.h"

// Private NPU record shared by the public FA3 metadata API and dense MLA kernels.
// The scheduler and field semantics follow the fixed CANN implementation.
namespace flash_attn::mla {

// Supplied by the caller, never taken from the record being validated. The
// workspace pointer passed to MlaDense starts AFTER the libapi reservation.
// No output pointer, scale or mutable per-call state belongs in metadata.
struct MlaDenseMetadataContext {
    uint32_t dtype, batch, sq, heads, sk;
    uint32_t aicNum, aivNum, blockDim;
    uint64_t workspaceBytes, libApiBytes;
    MlaLocalMemory localMemory;
};
static_assert(std::is_standard_layout_v<MlaDenseMetadataContext> &&
              std::is_trivially_copyable_v<MlaDenseMetadataContext>);

constexpr uint32_t kMlaMetadataMagic = 0x4d4c4131; // MLA1
constexpr uint32_t kMlaMetadataVersion = 1;
constexpr uint32_t kMlaMetadataDenseSharedKV = 1; // BSND, 512+64, Hkv=1, no mask/split.
enum class MlaMetadataDtype : uint32_t {
    Fp16 = 1,
    Bf16 = 2
};
enum class MlaMetadataStatus : uint32_t {
    Success = 0,
    UnsupportedDtype = 1,
    ScheduleFailure = 2,
    InvalidCacheLengths = 3,
    UnsupportedCacheLength = 4
};

// Every wire field has int32-tensor alignment, including 64-bit quantities.
// Preserve all bits as low/high words; never narrow workspace sizes or require
// an 8-byte aligned storage offset. Signed maxCost uses its two's-complement bits.
struct MlaMetadataWide {
    uint32_t low, high;
};

struct MlaMetadataWorkspace {
    MlaMetadataWide mm1ResSize, mm2ResSize, fdAccumOutSize, fdLogSumExpSize;
    MlaMetadataWide qkOffsetBytes, pOffsetBytes, pvOffsetBytes, vec2OffsetBytes;
    MlaMetadataWide kernelBytes, userBytes, libApiBytes, totalBytes;
};

struct MlaMetadataLocalMemory {
    MlaMetadataWide l1Bytes, ubBytes, l0aBytes, l0bBytes, l0cBytes;
};

struct MlaMetadataSchedule {
    uint32_t bSize, n2Size, gSize, s1Size, s2Size, headDim, headDimRope;
    uint32_t mBaseSize, s2BaseSize, aicNum, aivNum, cvRatio, usedCoreNum, blockDim, scheduleMode;
    uint32_t bN2End[kMlaMaxAic], gS1End[kMlaMaxAic], s2End[kMlaMaxAic];
    MlaMetadataWide maxCost;
    // Retain even the inactive source FD fields for exact schedule comparison.
    uint32_t numOfFdHead, maxS2SplitNum, usedVecNumOfFd, gS1BaseSizeOfFd;
    uint32_t bN2IdxOfFdHead[kMlaMaxAic], gS1IdxOfFdHead[kMlaMaxAic], s2SplitNumOfFdHead[kMlaMaxAic];
    uint32_t s2SplitStartIdxOfCore[kMlaMaxAic], gS1SplitNumOfFdHead[kMlaMaxAic], gS1LastPartSizeOfFdHead[kMlaMaxAic];
    uint32_t gS1IdxEndOfFdHead[kMlaMaxAic * 2], gS1IdxEndOfFdHeadSplit[kMlaMaxAic * 2];
    MlaMetadataWorkspace workspace;
    MlaMetadataLocalMemory requiredLocalMemory;
};

struct MlaDenseMetadata {
    uint32_t magic, version, kind, wordCount;
    uint32_t dtype, status, scheduleStatus, reserved;
    MlaMetadataSchedule schedule;
};

constexpr uint32_t kMlaDenseMetadataWords = 401;
static_assert(sizeof(MlaDenseMetadata) == kMlaDenseMetadataWords * sizeof(int32_t));
static_assert(alignof(MlaDenseMetadata) == alignof(int32_t));
static_assert(offsetof(MlaDenseMetadata, schedule) == 8 * sizeof(int32_t));
static_assert(std::is_standard_layout_v<MlaDenseMetadata> && std::is_trivially_copyable_v<MlaDenseMetadata>);

// Shared wire encoding only; each protocol supplies its own header and checks.
MlaMetadataSchedule SerializeMlaSchedule(const MlaSchedule& schedule);

// Deterministic full record, also on failure: a non-success header and zero
// payload replace any previous schedule. No scale, addresses or mutable scratch.
// Optional lengths must all equal shape.kvLength, including zero capacity.
// Shorter, nonuniform or over-capacity sequences remain unsupported.
// (nullptr, 0) uses full capacity. A non-null buffer must have B aligned int32
// entries, readable on the executing processor (host tests or AICPU).
void BuildDenseMetadata(const MlaDenseShape& shape, const MlaPlatformInfo& platform, uint32_t dtype,
                        MlaDenseMetadata& output, const int32_t* cacheSeqlens = nullptr,
                        uint64_t cacheSeqlensCount = 0);

} // namespace flash_attn::mla

// Private host/AICPU launch ABI. Addresses never belong in serialized records.
namespace flash_attn::mla {
struct MlaMetadataArgs {
    uint64_t outputAddr;
    uint32_t outputWords;
    uint32_t dtype;
    MlaDenseShape shape;
    MlaPlatformInfo platform;
    // (0, 0) means implicit full capacity for the internal dense producer.
    // Otherwise the caller supplies B contiguous int32 lengths on this device,
    // retains their storage and orders writes before the AICPU launch.
    uint64_t cacheSeqlensAddr;
    uint64_t cacheSeqlensCount;
};
static_assert(std::is_standard_layout_v<MlaMetadataArgs> && std::is_trivially_copyable_v<MlaMetadataArgs>);
} // namespace flash_attn::mla

// Private BSND actual-KV-length protocol with a dedicated validation consumer;
// never pass this record to the v1 dense Kernel.
namespace flash_attn::mla {

constexpr uint32_t kMlaBsndKvLengthMetadataVersion = 2;
constexpr uint32_t kMlaMetadataBsndKvLengths = 2;

// Same fixed schedule encoding as v1, followed immediately by B int32 lengths.
// wordCount includes the snapshot; batch comes from the independent static
// context. Implicit full capacity also materializes B lengths. No addresses,
// scale, strides or mutable scratch. Only 4-byte alignment is required.
struct MlaBsndKvLengthMetadataPrefix {
    uint32_t magic, version, kind, wordCount;
    uint32_t dtype, status, scheduleStatus, reserved;
    MlaMetadataSchedule schedule;
};
constexpr uint32_t kMlaBsndKvLengthMetadataPrefixWords = 401;
static_assert(sizeof(MlaBsndKvLengthMetadataPrefix) == kMlaBsndKvLengthMetadataPrefixWords * sizeof(int32_t));
static_assert(alignof(MlaBsndKvLengthMetadataPrefix) == alignof(int32_t));
static_assert(offsetof(MlaBsndKvLengthMetadataPrefix, schedule) == 8 * sizeof(int32_t));
static_assert(std::is_standard_layout_v<MlaBsndKvLengthMetadataPrefix> &&
              std::is_trivially_copyable_v<MlaBsndKvLengthMetadataPrefix>);

// Static allocation bound, with no device-value read. False leaves words intact.
bool BsndKvLengthMetadataWords(int64_t batch, uint32_t& words);

// False: invalid output size/address or overlapping input/output; no writes.
// True: a complete success or failure record was written. On dtype, descriptor
// or scheduling failure the entire schedule AND snapshot are zeroed. Values
// outside [0, capacity] are ScheduleFailure/UnsupportedShape, as in b1.
// Both buffers must belong to the executing processor (CPU tests or AICPU),
// and the caller must keep lengths immutable until generation has completed.
bool BuildBsndKvLengthMetadata(const MlaDenseShape& shape, const MlaPlatformInfo& platform, uint32_t dtype,
                               uint32_t* output, uint32_t outputWords, const int32_t* cacheSeqlens = nullptr,
                               uint64_t cacheSeqlensCount = 0);

} // namespace flash_attn::mla

// Private compact-TND protocol, generated and validated independently of
// attention. Never pass this record to a v1/v2 Kernel.
namespace flash_attn::mla {

struct MlaTndShape;
constexpr uint32_t kMlaTndMetadataVersion = 3;
constexpr uint32_t kMlaMetadataCompactTnd = 3;

// Static context shared by the production host launcher and device consumer.
struct MlaTndMetadataContext {
    uint32_t dtype, batch, totalQuery, totalKv, heads, maxQueryLength, maxKvLength;
    uint32_t aicNum, aivNum, blockDim;
    uint64_t workspaceBytes, libApiBytes;
    MlaLocalMemory localMemory;
};

enum class MlaTndMetadataStatus : uint32_t {
    Success = 0,
    UnsupportedDtype = 1,
    ScheduleFailure = 2,
    InvalidPrefixes = 3
};

// First 401 words retain the existing schedule encoding. s1Size/s2Size are
// observed maxima; the four extra words bind the caller's totals/upper bounds.
// Followed by B (queryStart, kvStart, queryLength, kvLength) token spans.
// No pointers, scale, strides or mutable scratch; only int32 alignment.
struct MlaTndMetadataPrefix {
    uint32_t magic, version, kind, wordCount;
    uint32_t dtype, status, scheduleStatus, reserved;
    MlaMetadataSchedule schedule;
    uint32_t totalQuery, totalKv, maxQueryLength, maxKvLength;
};
constexpr uint32_t kMlaTndMetadataPrefixWords = 405;
constexpr uint32_t kMlaTndMetadataSequenceWords = 4;
static_assert(sizeof(MlaTndMetadataPrefix) == kMlaTndMetadataPrefixWords * sizeof(int32_t));
static_assert(alignof(MlaTndMetadataPrefix) == alignof(int32_t));
static_assert(offsetof(MlaTndMetadataPrefix, schedule) == 8 * sizeof(int32_t));
static_assert(offsetof(MlaTndMetadataPrefix, totalQuery) == 401 * sizeof(int32_t));
static_assert(std::is_standard_layout_v<MlaTndMetadataPrefix> && std::is_trivially_copyable_v<MlaTndMetadataPrefix>);

// Static allocation bound only; false leaves words unchanged.
bool TndMetadataWords(int64_t batch, uint32_t& words);

// Both int32[B+1] prefixes are required, including B=0. Their values and the
// complete schedule are validated by BuildTndSchedule before serialization.
// False: unsafe output size/address or input/output overlap, with no writes.
// True: success or failure record written. Failure zeroes everything after
// the 8-word header, including totals, upper bounds and all token spans.
// Inputs must be readable on CPU/AICPU and immutable until generation ends.
// The two read-only inputs may alias; neither may overlap the output.
bool BuildTndMetadata(const MlaTndShape& shape, const MlaPlatformInfo& platform, uint32_t dtype, uint32_t* output,
                      uint32_t outputWords, const int32_t* cuQuery, uint64_t queryCount, const int32_t* cuKv,
                      uint64_t kvCount);

} // namespace flash_attn::mla

// Private host/AICPU launch ABI; addresses are never serialized in v3 records.
namespace flash_attn::mla {
struct MlaTndMetadataArgs {
    uint64_t outputAddr;
    uint32_t outputWords, dtype;
    MlaTndShape shape;
    MlaPlatformInfo platform;
    uint64_t cuQueryAddr, queryCount;
    uint64_t cuKvAddr, kvCount;
};
static_assert(std::is_standard_layout_v<MlaTndMetadataArgs> && std::is_trivially_copyable_v<MlaTndMetadataArgs>);
} // namespace flash_attn::mla

// Private TND-Q/BSND-KV protocol with independent generation and validation.
// Never pass this record to a v1/v2/v3 Kernel.
namespace flash_attn::mla {

struct MlaTndBsndShape;
constexpr uint32_t kMlaTndBsndMetadataVersion = 4;
constexpr uint32_t kMlaMetadataTndBsnd = 4;

// Plain launch context shared by host and device, without consumer code.
struct MlaTndBsndMetadataContext {
    uint32_t dtype, batch, totalQuery, heads, maxQueryLength, kvCapacity;
    uint32_t aicNum, aivNum, blockDim;
    uint64_t workspaceBytes, libApiBytes;
    MlaLocalMemory localMemory;
};

enum class MlaTndBsndMetadataStatus : uint32_t {
    Success = 0,
    UnsupportedDtype = 1,
    ScheduleFailure = 2,
    InvalidLengths = 3
};

// First 401 words retain the common schedule encoding: s1Size is the observed
// Q maximum, s2Size is physical KV capacity. The extra five words bind the Q
// total/upper bound and physical KV capacity, including 64-bit B*kvCapacity.
// Followed by B five-word spans: queryStart, kvStart.low, kvStart.high,
// queryLength, kvLength. All offsets/capacities count tokens, never bytes.
// No pointers, scale, strides or mutable scratch; only int32 alignment.
struct MlaTndBsndMetadataPrefix {
    uint32_t magic, version, kind, wordCount;
    uint32_t dtype, status, scheduleStatus, reserved;
    MlaMetadataSchedule schedule;
    uint32_t totalQuery, maxQueryLength, kvCapacity;
    MlaMetadataWide totalKvCapacity;
};
constexpr uint32_t kMlaTndBsndMetadataPrefixWords = 406;
constexpr uint32_t kMlaTndBsndMetadataSequenceWords = 5;
static_assert(sizeof(MlaTndBsndMetadataPrefix) == kMlaTndBsndMetadataPrefixWords * sizeof(int32_t));
static_assert(alignof(MlaTndBsndMetadataPrefix) == alignof(int32_t));
static_assert(offsetof(MlaTndBsndMetadataPrefix, schedule) == 8 * sizeof(int32_t));
static_assert(offsetof(MlaTndBsndMetadataPrefix, totalQuery) == 401 * sizeof(int32_t));
static_assert(std::is_standard_layout_v<MlaTndBsndMetadataPrefix> &&
              std::is_trivially_copyable_v<MlaTndBsndMetadataPrefix>);

// Static allocation bound only; false leaves words unchanged.
bool TndBsndMetadataWords(int64_t batch, uint32_t& words);

// Required Q prefix int32[B+1] (including [0] for B=0) and KV lengths int32[B].
// Null KV is legal only for B=0/count=0; it never means full capacity.
// False: unsafe output size/address or input/output overlap, with no writes.
// True: success or failure record written. Failure zeroes everything after
// the eight-word header, including capacity, upper bound and token spans.
// Read-only inputs may alias each other, must be readable on CPU/AICPU and
// remain immutable until generation ends. Neither may overlap the output.
// The public generator has no total-Q argument: fromQueryPrefix derives it
// after descriptor checks and selects full-core launch for nonempty capacity.
// Explicit-total callers retain the supplied shape and block-dimension table.
bool BuildTndBsndMetadata(const MlaTndBsndShape& shape, const MlaPlatformInfo& platform, uint32_t dtype,
                          uint32_t* output, uint32_t outputWords, const int32_t* cuQuery, uint64_t queryCount,
                          const int32_t* kvLengths, uint64_t kvCount, bool fromQueryPrefix = false);

} // namespace flash_attn::mla

// Private host/AICPU launch ABI; addresses are never serialized in v4 records.
namespace flash_attn::mla {
struct MlaTndBsndMetadataArgs {
    uint64_t outputAddr;
    uint32_t outputWords, dtype;
    MlaTndBsndShape shape;
    MlaPlatformInfo platform;
    uint64_t cuQueryAddr, queryCount;
    uint64_t kvLengthsAddr, kvCount;
};
static_assert(std::is_standard_layout_v<MlaTndBsndMetadataArgs> &&
              std::is_trivially_copyable_v<MlaTndBsndMetadataArgs>);
} // namespace flash_attn::mla

// Private host/AICPU launch ABI. Addresses are never serialized in v5 records.
namespace flash_attn::mla {
struct MlaUsedMetadataArray {
    uint64_t address, count;
};
struct MlaUsedMetadataArgs {
    uint64_t outputAddr;
    uint32_t outputWords, dtype;
    MlaUsedShape shape;
    MlaPlatformInfo platform;
    MlaUsedMetadataArray cuQuery, cuKv, usedQuery, usedKv;
};
static_assert(std::is_standard_layout_v<MlaUsedMetadataArgs> && std::is_trivially_copyable_v<MlaUsedMetadataArgs>);

inline MlaUsedInputs UsedMetadataInputs(const MlaUsedMetadataArgs& args)
{
    const auto array = [](const MlaUsedMetadataArray& input) {
        return MlaLengthArray{reinterpret_cast<const int32_t*>(input.address), input.count};
    };
    return {array(args.cuQuery), array(args.cuKv), array(args.usedQuery), array(args.usedKv)};
}
} // namespace flash_attn::mla

// Private used-length protocol; v1-v4 consumers must reject this record.
// Scheduling retains its CANN source/license in
// mla_tiling.cpp, mla_tiling.cpp and cann_split_core.cpp.
namespace flash_attn::mla {

constexpr uint32_t kMlaUsedMetadataVersion = 5;
constexpr uint32_t kMlaMetadataUsed = 5;
constexpr uint32_t kMlaUsedQueryPresent = 1;
constexpr uint32_t kMlaUsedKvPresent = 2;

enum class MlaUsedMetadataStatus : uint32_t {
    Success = 0,
    UnsupportedDtype = 1,
    ScheduleFailure = 2,
    InvalidLengths = 3
};

// Layout 0=BSND, 1=TND. Totals/starts count physical tokens, never used tokens
// or bytes. For BSND maxLength is capacity; for TND it is the prefix bound.
struct MlaMetadataAllocation {
    uint32_t layout;
    MlaMetadataWide totalTokens;
    uint32_t maxLength;
};

// Plain launch ABI shared by host code and the device consumer. Keep the
// device validation functions in mla_metadata_consumer.h.
struct MlaUsedMetadataContext {
    uint32_t dtype, batch, heads, usedFlags;
    MlaMetadataAllocation query, kv;
    uint32_t aicNum, aivNum, blockDim;
    uint64_t workspaceBytes, libApiBytes;
    MlaLocalMemory localMemory;
};

// Descriptor addresses belong to this invocation, never to the record.
// Read-only inputs may alias; all inputs must remain stable until completion.
struct MlaUsedMetadataInputs {
    MlaUsedMetadataArray cuQuery, cuKv, usedQuery, usedKv;
};

// Common 401-word schedule, two four-word allocations, then presence flags.
// Explicit used and absent used can select different source base tiles even
// when all effective lengths match; retain their identities in the record.
// Followed by B eight-word spans: queryStart.low/high, kvStart.low/high,
// queryCapacity, kvCapacity, queryLength, kvLength. No pointers or scale.
struct MlaUsedMetadataPrefix {
    uint32_t magic, version, kind, wordCount;
    uint32_t dtype, status, scheduleStatus, reserved;
    MlaMetadataSchedule schedule;
    MlaMetadataAllocation query, kv;
    uint32_t usedFlags;
};
constexpr uint32_t kMlaUsedMetadataPrefixWords = 410;
constexpr uint32_t kMlaUsedMetadataSequenceWords = 8;
static_assert(sizeof(MlaUsedMetadataPrefix) == kMlaUsedMetadataPrefixWords * sizeof(int32_t));
static_assert(alignof(MlaUsedMetadataPrefix) == alignof(int32_t));
static_assert(offsetof(MlaUsedMetadataPrefix, schedule) == 8 * sizeof(int32_t));
static_assert(offsetof(MlaUsedMetadataPrefix, query) == 401 * sizeof(int32_t));
static_assert(offsetof(MlaUsedMetadataPrefix, usedFlags) == 409 * sizeof(int32_t));
static_assert(std::is_standard_layout_v<MlaUsedMetadataPrefix> && std::is_trivially_copyable_v<MlaUsedMetadataPrefix>);
static_assert(uint32_t(MlaSequenceLayout::Bsnd) == 0 && uint32_t(MlaSequenceLayout::Tnd) == 1);

// Static allocation bound only; false leaves words unchanged.
bool UsedMetadataWords(int64_t batch, uint32_t& words);

// Array semantics are those of ParseUsedLengths, including empty allocations.
// False: unsafe output size/address or any input/output overlap; no writes.
// True: complete success/failure record. Failure zeroes all words after the
// eight-word header. Input descriptors are checked before reading values;
// read-only inputs may alias each other, but never the output. All inputs must
// be readable/stable on the executing CPU/AICPU until generation completes.
bool BuildUsedMetadata(const MlaUsedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype, uint32_t* output,
                       uint32_t outputWords, const MlaUsedInputs& inputs);

// Share descriptor validation and serialization with a different partitioner.
// The builder must return a complete schedule, using the same owned lengths.
using MlaUsedScheduleBuilder = MlaScheduleStatus (*)(const MlaUsedShape&, const MlaPlatformInfo&, const MlaUsedInputs&,
                                                     MlaUsedSchedule&, const void*);
bool BuildUsedMetadataWithSchedule(const MlaUsedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype,
                                   uint32_t* output, uint32_t outputWords, const MlaUsedInputs& inputs,
                                   MlaUsedScheduleBuilder builder, const void* context = nullptr,
                                   bool queryTotalFromPrefix = false, uint32_t trailingWords = 0);

} // namespace flash_attn::mla

// Private PA protocol shared by CPU/AICPU generation and device consumers.
// Public metadata and BSND/TND paged forward use the same v6 record.
// Scheduling retains its CANN source/license in mla_tiling.cpp,
// mla_tiling.cpp and cann_split_core.cpp (see LICENSE.cann).
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

// BSND right-down causal protocol. The v5 allocation/used wire layout is
// shared, but version/kind are distinct: a no-mask consumer must reject v7.
namespace flash_attn::mla {
constexpr uint32_t kMlaCausalBsndMetadataVersion = 7;
constexpr uint32_t kMlaMetadataCausalBsnd = 7;
constexpr uint32_t kMlaCausalTndMetadataVersion = 8;
constexpr uint32_t kMlaMetadataCausalTnd = 8;
constexpr uint32_t kMlaCausalMixedMetadataVersion = 9;
constexpr uint32_t kMlaMetadataCausalMixed = 9;

// Immutable host-generated candidate for full allocation lengths. The AICPU
// producer validates current inputs and geometry before using this partition.
// No pointer, device length value, tensor or serialized metadata is cached.
struct MlaCausalFullSchedule {
    uint32_t enabled = 0;
    uint32_t batch, queryCapacity, kvCapacity, heads;
    uint32_t aicNum, cvRatio, mBaseSize, s2BaseSize, usedCoreNum;
    std::array<uint32_t, kMlaMaxAic> bN2End, gS1End, s2End;
    int64_t maxCost;
};
static_assert(std::is_standard_layout_v<MlaCausalFullSchedule> && std::is_trivially_copyable_v<MlaCausalFullSchedule>);

struct MlaCausalMetadataArgs {
    MlaUsedMetadataArgs input;
    MlaCausalFullSchedule full;
};
static_assert(std::is_standard_layout_v<MlaCausalMetadataArgs> && std::is_trivially_copyable_v<MlaCausalMetadataArgs>);

// A failed static candidate is simply unavailable; actual used lengths can
// still yield a valid schedule. The normal producer remains the fallback.
bool BuildCausalFullSchedule(const MlaUsedShape& shape, const MlaPlatformInfo& platform, MlaCausalFullSchedule& output);

// Same buffer/descriptor/failure guarantees as BuildUsedMetadata. Only BSND
// Q/KV are supported. Effective Sq/Sk come from the serialized length snapshot,
// never from capacities. Empty allocations retain validation-only scheduling.
bool BuildCausalBsndMetadata(const MlaUsedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype,
                             uint32_t* output, uint32_t outputWords, const MlaUsedInputs& inputs,
                             const MlaCausalFullSchedule* full = nullptr);
// TND Q/KV retain both allocation prefixes and optional used arrays. No
// rectangular full-length candidate is valid for these device-owned spans.
bool BuildCausalTndMetadata(const MlaUsedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype,
                            uint32_t* output, uint32_t outputWords, const MlaUsedInputs& inputs);
// TND Q + BSND KV. Public metadata derives Q allocation size from its prefix
// after descriptor validation; forward generation uses the tensor's size.
bool BuildCausalMixedMetadata(const MlaUsedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype,
                              uint32_t* output, uint32_t outputWords, const MlaUsedInputs& inputs,
                              bool queryTotalFromPrefix = false);
} // namespace flash_attn::mla

// Nonpaged window protocols; source/license: mla_tiling.cpp and LICENSE.cann / LICENSE.fa3.
namespace flash_attn::mla {
constexpr uint32_t kMlaWindowBsndMetadataVersion = 12;
constexpr uint32_t kMlaMetadataWindowBsnd = 12;
constexpr uint32_t kMlaWindowTndMetadataVersion = 13;
constexpr uint32_t kMlaMetadataWindowTnd = 13;
constexpr uint32_t kMlaWindowMixedMetadataVersion = 14;
constexpr uint32_t kMlaMetadataWindowMixed = 14;
// Three signed int32 tail words: normalized kind, left, right. Allocations in
// the used prefix retain the call-level bounds independently of used lengths.
constexpr uint32_t kMlaWindowMetadataTailWords = 3;
struct MlaWindowMetadataArgs {
    MlaUsedMetadataArgs input;
    MlaWindowConfig window;
};
static_assert(std::is_standard_layout_v<MlaWindowMetadataArgs> && std::is_trivially_copyable_v<MlaWindowMetadataArgs>);
inline bool WindowMetadataWords(int64_t batch, uint32_t& words)
{
    if (batch < 0 || uint64_t(batch) > (UINT32_MAX - kMlaUsedMetadataPrefixWords - kMlaWindowMetadataTailWords) /
                                           kMlaUsedMetadataSequenceWords) {
        return false;
    }
    words = kMlaUsedMetadataPrefixWords + kMlaUsedMetadataSequenceWords * uint32_t(batch) + kMlaWindowMetadataTailWords;
    return true;
}
bool BuildWindowBsndMetadata(const MlaUsedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype,
                             uint32_t* output, uint32_t outputWords, const MlaUsedInputs& inputs,
                             const MlaWindowConfig& window);
bool BuildWindowTndMetadata(const MlaUsedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype,
                            uint32_t* output, uint32_t outputWords, const MlaUsedInputs& inputs,
                            const MlaWindowConfig& window);
bool BuildWindowMixedMetadata(const MlaUsedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype,
                              uint32_t* output, uint32_t outputWords, const MlaUsedInputs& inputs,
                              const MlaWindowConfig& window, bool queryTotalFromPrefix = false);
} // namespace flash_attn::mla

// Private BSND PA window protocol; pinned sources/licenses: mla_tiling.cpp.
namespace flash_attn::mla {
constexpr uint32_t kMlaWindowPagedBsndMetadataVersion = 15;
constexpr uint32_t kMlaMetadataWindowPagedBsnd = 15;
constexpr uint32_t kMlaWindowPagedTndMetadataVersion = 16;
constexpr uint32_t kMlaMetadataWindowPagedTnd = 16;
constexpr uint32_t kMlaWindowPagedMetadataTailWords = 3;
struct MlaWindowPagedMetadataArgs {
    MlaPagedMetadataArgs input;
    MlaWindowConfig window;
};
static_assert(std::is_standard_layout_v<MlaWindowPagedMetadataArgs> &&
              std::is_trivially_copyable_v<MlaWindowPagedMetadataArgs>);
inline bool WindowPagedMetadataWords(int64_t batch, uint32_t& words)
{
    if (batch < 0 || uint64_t(batch) > (UINT32_MAX - kMlaPagedMetadataPrefixWords - kMlaWindowPagedMetadataTailWords) /
                                           kMlaPagedMetadataSequenceWords) {
        return false;
    }
    words = kMlaPagedMetadataPrefixWords + uint32_t(batch) * kMlaPagedMetadataSequenceWords +
            kMlaWindowPagedMetadataTailWords;
    return true;
}
bool BuildWindowPagedBsndMetadata(const MlaPagedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype,
                                  uint32_t* output, uint32_t outputWords, const MlaPagedInputs& inputs,
                                  const MlaWindowConfig& window);
bool BuildWindowPagedTndMetadata(const MlaPagedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype,
                                 uint32_t* output, uint32_t outputWords, const MlaPagedInputs& inputs,
                                 const MlaWindowConfig& window, bool fromQueryPrefix = false);
} // namespace flash_attn::mla

// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.

// BSND explicit-split adapter for the pinned CANN/FA3 plan in mla_split.h.
namespace flash_attn::mla {
struct MlaSplitBsndContext {
    MlaUsedMetadataContext used;
    MlaWindowKernelParams window;
    uint32_t splits;
    uint32_t hasSoftcap = 0;
};
constexpr uint32_t kMlaSplitBsndVersion = 17;
constexpr uint32_t kMlaSoftcapSplitBsndVersion = 32;
// v5 allocation/used prefix and spans, then window(kind,left,right), splits,
// and B+1 low/high uint64 compact M prefixes. No addresses or CPU objects.
inline bool SplitBsndMetadataWords(int64_t batch, uint32_t& words)
{
    if (batch < 0 || uint64_t(batch) > (UINT32_MAX - 416) / 10)
        return false;
    words = 416 + 10 * uint32_t(batch);
    return true;
}
struct MlaSplitBsndMetadataArgs {
    MlaUsedMetadataArgs input;
    MlaWindowConfig window;
    int64_t splits;
    uint32_t hasSoftcap = 0;
};
bool BuildSplitBsndMetadata(const MlaUsedShape&, const MlaPlatformInfo&, uint32_t dtype, uint32_t* output,
                            uint32_t words, const MlaUsedInputs&, const MlaWindowConfig&, int64_t splits,
                            uint32_t hasSoftcap = 0);
} // namespace flash_attn::mla

// TND explicit-split adapter for the pinned CANN/FA3 plan in mla_split.h.
namespace flash_attn::mla {
struct MlaSplitTndContext {
    MlaUsedMetadataContext used;
    MlaWindowKernelParams window;
    uint32_t splits;
    uint32_t hasSoftcap = 0;
};
constexpr uint32_t kMlaSplitTndVersion = 19;
constexpr uint32_t kMlaSoftcapSplitTndVersion = 35;
// v5 allocation/used prefix and spans, then window(kind,left,right), splits,
// and B+1 low/high uint64 compact M prefixes. No addresses or CPU objects.
inline bool SplitTndMetadataWords(int64_t batch, uint32_t& words)
{
    if (batch < 0 || uint64_t(batch) > (UINT32_MAX - 416) / 10)
        return false;
    words = 416 + 10 * uint32_t(batch);
    return true;
}
struct MlaSplitTndMetadataArgs {
    MlaUsedMetadataArgs input;
    MlaWindowConfig window;
    int64_t splits;
    uint32_t hasSoftcap = 0;
};
bool BuildSplitTndMetadata(const MlaUsedShape&, const MlaPlatformInfo&, uint32_t dtype, uint32_t* output,
                           uint32_t words, const MlaUsedInputs&, const MlaWindowConfig&, int64_t splits,
                           uint32_t hasSoftcap = 0);
} // namespace flash_attn::mla

// Mixed TND Q / BSND KV explicit-split adapter for the pinned CANN/FA3 plan in mla_split.h.
namespace flash_attn::mla {
struct MlaSplitMixedContext {
    MlaUsedMetadataContext used;
    MlaWindowKernelParams window;
    uint32_t splits;
    uint32_t hasSoftcap = 0;
};
constexpr uint32_t kMlaSplitMixedVersion = 21;
constexpr uint32_t kMlaSoftcapSplitMixedVersion = 38;
// v5 allocation/used prefix and spans, then window(kind,left,right), splits,
// and B+1 low/high uint64 compact M prefixes. No addresses or CPU objects.
inline bool SplitMixedMetadataWords(int64_t batch, uint32_t& words)
{
    if (batch < 0 || uint64_t(batch) > (UINT32_MAX - 416) / 10)
        return false;
    words = 416 + 10 * uint32_t(batch);
    return true;
}
struct MlaSplitMixedMetadataArgs {
    MlaUsedMetadataArgs input;
    MlaWindowConfig window;
    int64_t splits;
    bool queryTotalFromPrefix = false;
    uint32_t hasSoftcap = 0;
};
bool BuildSplitMixedMetadata(const MlaUsedShape&, const MlaPlatformInfo&, uint32_t dtype, uint32_t* output,
                             uint32_t words, const MlaUsedInputs&, const MlaWindowConfig&, int64_t splits,
                             bool queryTotalFromPrefix = false, uint32_t hasSoftcap = 0);
} // namespace flash_attn::mla

// BSND PA explicit-split protocol; source attribution in the plan header.
namespace flash_attn::mla {
constexpr uint32_t kMlaSplitPagedBsndVersion = 23;
// Same record fields; bind the enabled feature, never the positive cap value.
constexpr uint32_t kMlaSoftcapSplitPagedBsndVersion = 41;
struct MlaSplitPagedBsndContext {
    MlaPagedMetadataContext paged;
    MlaWindowKernelParams window;
    uint32_t splits;
    uint32_t hasSoftcap = 0;
};
// v6 logical PA snapshot, window(kind,left,right), S, B+1 uint64 M prefixes.
// Does not bind page IDs, physical cache, addresses, strides or scale.
inline bool SplitPagedBsndMetadataWords(int64_t batch, uint32_t& words)
{
    if (batch < 0 || uint64_t(batch) > (UINT32_MAX - 414) / 7)
        return false;
    words = 414 + 7 * uint32_t(batch);
    return true;
}
struct MlaSplitPagedBsndMetadataArgs {
    MlaPagedMetadataArgs input;
    MlaWindowConfig window;
    int64_t splits;
    uint32_t hasSoftcap = 0;
};
// False leaves output untouched (unsafe descriptor/overlap); true writes a
// complete record, including a zero-payload failure record. No device reads.
bool BuildSplitPagedBsndMetadata(const MlaPagedShape&, const MlaPlatformInfo&, uint32_t dtype, uint32_t* output,
                                 uint32_t words, const MlaPagedInputs&, const MlaWindowConfig&, int64_t splits,
                                 uint32_t hasSoftcap = 0);
} // namespace flash_attn::mla

// TND PA explicit-split protocol; source attribution in the plan header.
namespace flash_attn::mla {
constexpr uint32_t kMlaSplitPagedTndVersion = 25;
// Same fields; bind softcap activation independently of the positive cap value.
constexpr uint32_t kMlaSoftcapSplitPagedTndVersion = 44;
struct MlaSplitPagedTndContext {
    MlaPagedMetadataContext paged;
    MlaWindowKernelParams window;
    uint32_t splits;
    uint32_t hasSoftcap = 0;
};
// v6 logical PA snapshot, window(kind,left,right), S, B+1 uint64 M prefixes.
// Does not bind page IDs, physical cache, addresses, strides or scale.
inline bool SplitPagedTndMetadataWords(int64_t batch, uint32_t& words)
{
    if (batch < 0 || uint64_t(batch) > (UINT32_MAX - 414) / 7)
        return false;
    words = 414 + 7 * uint32_t(batch);
    return true;
}
struct MlaSplitPagedTndMetadataArgs {
    MlaPagedMetadataArgs input;
    MlaWindowConfig window;
    int64_t splits;
    bool queryTotalFromPrefix = false;
    uint32_t hasSoftcap = 0;
};
// False leaves output untouched (unsafe descriptor/overlap); true writes a
// complete record, including a zero-payload failure record. No device reads.
bool BuildSplitPagedTndMetadata(const MlaPagedShape&, const MlaPlatformInfo&, uint32_t dtype, uint32_t* output,
                                uint32_t words, const MlaPagedInputs&, const MlaWindowConfig&, int64_t splits,
                                bool queryTotalFromPrefix = false, uint32_t hasSoftcap = 0);
} // namespace flash_attn::mla

// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.

// Native partition provenance: mla_split.h. This is a local wire
// protocol, not a CANN ABI; the production adapter defines the FA3 capacity.
namespace flash_attn::mla {
constexpr uint32_t kMlaAutoSplitBsndVersion = 18;
constexpr uint32_t kMlaSoftcapAutoSplitBsndVersion = 33;
static_assert(kMlaMaxAic == 26, "Changing endpoint capacity requires a new wire version");
// v5 geometry/lengths remain independently validated. The tail contains window
// (3), native usedCoreNum (1), AIC endpoints (3*26), FD scalars (4), FD arrays
// (6*26+2*52), then B+1 low/high M prefixes. No addresses, costs or CPU objects.
constexpr uint32_t kMlaAutoSplitTailPrefix = 346;
constexpr uint32_t kMlaAutoSplitFixedWords = 758;
inline bool AutoSplitBsndMetadataWords(int64_t batch, uint32_t& words)
{
    if (batch < 0 || uint64_t(batch) > (UINT32_MAX - kMlaAutoSplitFixedWords) / 10)
        return false;
    words = kMlaAutoSplitFixedWords + 10 * uint32_t(batch);
    return true;
}
struct MlaAutoSplitBsndContext {
    MlaUsedMetadataContext used;
    MlaWindowKernelParams window;
    uint32_t hasSoftcap = 0;
};
struct MlaAutoSplitBsndMetadataArgs {
    MlaUsedMetadataArgs input;
    MlaWindowConfig window;
    uint32_t hasSoftcap = 0;
};
// Static public allocation capacity; native per-M partitions remain variable.
constexpr uint32_t kMlaAutoSplitCapacity = kMlaMaxAic;
bool BuildAutoSplitBsndMetadata(const MlaUsedShape&, const MlaPlatformInfo&, uint32_t dtype, uint32_t* output,
                                uint32_t words, const MlaUsedInputs&, const MlaWindowConfig&, uint32_t hasSoftcap = 0);
} // namespace flash_attn::mla

// TND native partition provenance: mla_split.h.
// Layout-specific protocol shared by the production AICPU and AICore stages.
namespace flash_attn::mla {
constexpr uint32_t kMlaAutoSplitTndVersion = 20;
constexpr uint32_t kMlaSoftcapAutoSplitTndVersion = 36;
// Same tail schema as v18, distinct identity and TND allocation geometry.
inline bool AutoSplitTndMetadataWords(int64_t batch, uint32_t& words)
{
    if (batch < 0 || uint64_t(batch) > (UINT32_MAX - kMlaAutoSplitFixedWords) / 10)
        return false;
    words = kMlaAutoSplitFixedWords + 10 * uint32_t(batch);
    return true;
}
struct MlaAutoSplitTndContext {
    MlaUsedMetadataContext used;
    MlaWindowKernelParams window;
    uint32_t hasSoftcap = 0;
};
struct MlaAutoSplitTndMetadataArgs {
    MlaUsedMetadataArgs input;
    MlaWindowConfig window;
    uint32_t hasSoftcap = 0;
};
bool BuildAutoSplitTndMetadata(const MlaUsedShape&, const MlaPlatformInfo&, uint32_t dtype, uint32_t* output,
                               uint32_t words, const MlaUsedInputs&, const MlaWindowConfig&, uint32_t hasSoftcap = 0);
} // namespace flash_attn::mla

// mixed native partition provenance: mla_split.h.
// Shared CPU/AICPU producer and independent CPU/AICore consumer protocol.
namespace flash_attn::mla {
constexpr uint32_t kMlaAutoSplitMixedVersion = 22;
constexpr uint32_t kMlaSoftcapAutoSplitMixedVersion = 39;
// Same tail schema as v18, distinct identity and mixed allocation geometry.
inline bool AutoSplitMixedMetadataWords(int64_t batch, uint32_t& words)
{
    if (batch < 0 || uint64_t(batch) > (UINT32_MAX - kMlaAutoSplitFixedWords) / 10)
        return false;
    words = kMlaAutoSplitFixedWords + 10 * uint32_t(batch);
    return true;
}
struct MlaAutoSplitMixedContext {
    MlaUsedMetadataContext used;
    MlaWindowKernelParams window;
    uint32_t hasSoftcap = 0;
};
struct MlaAutoSplitMixedMetadataArgs {
    MlaUsedMetadataArgs input;
    MlaWindowConfig window;
    bool queryTotalFromPrefix = false;
    uint32_t hasSoftcap = 0;
};
bool BuildAutoSplitMixedMetadata(const MlaUsedShape&, const MlaPlatformInfo&, uint32_t dtype, uint32_t* output,
                                 uint32_t words, const MlaUsedInputs&, const MlaWindowConfig&,
                                 bool queryTotalFromPrefix = false, uint32_t hasSoftcap = 0);
} // namespace flash_attn::mla

// PA source geometry and native AIC/FD provenance: mla_split.h.
namespace flash_attn::mla {
constexpr uint32_t kMlaAutoSplitPagedBsndVersion = 24;
// Same record fields; bind the enabled feature, never the positive cap value.
constexpr uint32_t kMlaSoftcapAutoSplitPagedBsndVersion = 42;
constexpr uint32_t kMlaAutoSplitPagedBsndFixedWords = 756;
// v6 logical PA snapshot + native AIC/FD/Vector tail + B+1 uint64 M prefixes.
// Physical pages, cache addresses, strides and scale remain invocation state.
inline bool AutoSplitPagedBsndMetadataWords(int64_t batch, uint32_t& words)
{
    if (batch < 0 || uint64_t(batch) > (UINT32_MAX - kMlaAutoSplitPagedBsndFixedWords) / 7)
        return false;
    words = kMlaAutoSplitPagedBsndFixedWords + 7 * uint32_t(batch);
    return true;
}
struct MlaAutoSplitPagedBsndContext {
    MlaPagedMetadataContext paged;
    MlaWindowKernelParams window;
    uint32_t hasSoftcap = 0;
};
struct MlaAutoSplitPagedBsndMetadataArgs {
    MlaPagedMetadataArgs input;
    MlaWindowConfig window;
    uint32_t hasSoftcap = 0;
};
// False leaves output untouched for unsafe output/overlap. True writes a
// complete success or zero-payload failure record, including empty inputs.
bool BuildAutoSplitPagedBsndMetadata(const MlaPagedShape&, const MlaPlatformInfo&, uint32_t dtype, uint32_t* output,
                                     uint32_t words, const MlaPagedInputs&, const MlaWindowConfig&,
                                     uint32_t hasSoftcap = 0);
} // namespace flash_attn::mla

// PA source geometry and native AIC/FD provenance: mla_split.h.
namespace flash_attn::mla {
constexpr uint32_t kMlaAutoSplitPagedTndVersion = 26;
// Same fields; bind softcap activation independently of the positive cap value.
constexpr uint32_t kMlaSoftcapAutoSplitPagedTndVersion = 45;
constexpr uint32_t kMlaAutoSplitPagedTndFixedWords = 756;
// v26: v6 compact Q allocation/used PA snapshot + native AIC/FD/Vector tail + B+1 uint64 M prefixes.
// Physical pages, cache addresses, strides and scale remain invocation state.
inline bool AutoSplitPagedTndMetadataWords(int64_t batch, uint32_t& words)
{
    if (batch < 0 || uint64_t(batch) > (UINT32_MAX - kMlaAutoSplitPagedTndFixedWords) / 7)
        return false;
    words = kMlaAutoSplitPagedTndFixedWords + 7 * uint32_t(batch);
    return true;
}
struct MlaAutoSplitPagedTndContext {
    MlaPagedMetadataContext paged;
    MlaWindowKernelParams window;
    uint32_t hasSoftcap = 0;
};
struct MlaAutoSplitPagedTndMetadataArgs {
    MlaPagedMetadataArgs input;
    MlaWindowConfig window;
    bool queryTotalFromPrefix = false;
    uint32_t hasSoftcap = 0;
};
// False leaves output untouched for unsafe output/overlap. True writes a
// complete success or zero-payload failure record, including empty inputs.
bool BuildAutoSplitPagedTndMetadata(const MlaPagedShape&, const MlaPlatformInfo&, uint32_t dtype, uint32_t* output,
                                    uint32_t words, const MlaPagedInputs&, const MlaWindowConfig&,
                                    bool queryTotalFromPrefix = false, uint32_t hasSoftcap = 0);
} // namespace flash_attn::mla

// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
// BSND/TND 192/128 adapter for ca18aee's fia_tiling_nonquant.cpp and SplitCore.

#if defined(__NPU_ARCH__) && defined(__aicore__)
#define MLA_PREFILL_INLINE __aicore__ inline
#else
#define MLA_PREFILL_INLINE inline
#endif
namespace flash_attn::mla {
constexpr uint32_t kMlaPrefillBsndVersion = 27;
constexpr uint32_t kMlaPrefillTndVersion = 28;
constexpr uint32_t kMlaPrefillMixedVersion = 29;
constexpr uint32_t kMlaPrefillPagedBsndVersion = 30;
constexpr uint32_t kMlaPrefillPagedTndVersion = 31;
constexpr uint32_t kMlaSoftcapPrefillBsndVersion = 34;
constexpr uint32_t kMlaSoftcapPrefillTndVersion = 37;
constexpr uint32_t kMlaSoftcapPrefillMixedVersion = 40;
constexpr uint32_t kMlaSoftcapPrefillPagedBsndVersion = 43;
constexpr uint32_t kMlaSoftcapPrefillPagedTndVersion = 46;
// v27 has 20 header words and B pairs of effective lengths; v28 has 24 header
// words and B allocation/used spans. v29 shares those spans; v30 has a 24-word
// header and B effective-length pairs. All retain the 346-word native partition
// tail and B*Hkv+1 uint64 M prefixes, without device addresses. v31 combines
// v29 allocation spans with page size in header word 21 (no TND KV total).
struct MlaPrefillBsndContext {
    uint32_t batch, sq, sk, heads, kvHeads, splits, dtype, cores, usedFlags;
    MlaWindowKernelParams window;
    uint64_t workspaceBytes;
    // Layout: 0 = v27 BSND, 1 = v28 TND, 2 = v29 TND Q / BSND KV.
    // v29 retains v28 spans, but KV spans describe rectangular cache slots.
    uint32_t tnd = 0, totalQ = 0, totalK = 0;
    // v30/v31: BSND/TND Q and independent 192/128 ND paged KV.
    uint32_t pageSize = 0;
    uint32_t hasSoftcap = 0;
};
struct MlaPrefillBsndLayout {
    uint32_t words, m, n, capacity;
    uint64_t groups, scratch, outOffset, sumOffset, maxOffset, bytes;
};
MLA_PREFILL_INLINE uint32_t PrefillSequenceWord(const MlaPrefillBsndContext& c, uint32_t b)
{
    return c.tnd ? 24 + 8 * b : (c.pageSize ? 24 : 20) + 2 * b;
}
MLA_PREFILL_INLINE uint32_t PrefillLengthWord(const MlaPrefillBsndContext& c, uint32_t b)
{
    return PrefillSequenceWord(c, b) + (c.tnd ? 6 : 0);
}
MLA_PREFILL_INLINE uint32_t PrefillTailWord(const MlaPrefillBsndContext& c)
{
    return PrefillSequenceWord(c, c.batch);
}
MLA_PREFILL_INLINE bool PrefillBsndLayout(const MlaPrefillBsndContext& c, MlaPrefillBsndLayout& out)
{
    if (c.hasSoftcap > 1 || !c.heads || !c.kvHeads || c.heads % c.kvHeads || c.splits > 256 || !c.cores ||
        c.cores > 26 || c.tnd > 2 ||
        (c.pageSize && (c.tnd == 1 || c.pageSize < 16 || c.pageSize > 1024 || c.pageSize % 16 || c.sk % c.pageSize)) ||
        c.totalQ > INT32_MAX || c.totalK > INT32_MAX || (c.tnd && uint64_t(c.totalQ) * c.heads > kMlaMaxBatchRows) ||
        ((!c.tnd && c.totalQ) || (c.tnd != 1 && c.totalK)) || c.usedFlags > 3 || c.window.kind > 2 ||
        c.window.left < -1 || c.window.right < -1 || c.window.left > INT32_MAX || c.window.right > INT32_MAX ||
        c.sq > kMlaMaxQueryLength || c.sk > kMlaMaxKvLength || c.heads > kMlaMaxQueryHeads ||
        uint64_t(c.sq) * c.heads > kMlaMaxDenseRows ||
        (!c.tnd && uint64_t(c.batch) * c.sq > kMlaMaxBatchRows / c.heads) ||
        uint64_t(c.batch) * c.kvHeads > (UINT32_MAX - 758) / 10 ||
        uint64_t(c.batch) * ((c.tnd ? 8 : 2) + uint64_t(2) * c.kvHeads) > UINT32_MAX - 372)
        return false;
    const uint32_t g = c.heads / c.kvHeads;
    MlaPrefillBsndLayout r{};
    r.words = PrefillTailWord(c) + 348 + 2 * c.batch * c.kvHeads;
    // Exact CalcInnerSize/CalcMBaseSize rules from fixed CANN, including PA alignment.
    r.n = !c.tnd && c.sq <= 16 ? (g < 5 ? 8192 : g < 10 ? 4096 : 2048) : 512;
    if (c.window.kind && r.n > 1024)
        r.n = 1024;
    if (c.pageSize && r.n > c.pageSize && r.n % c.pageSize)
        r.n = r.n / c.pageSize * c.pageSize;
    if (r.n > c.sk)
        r.n = c.sk;
    if (!r.n)
        r.n = 32; // Empty-capacity validation launch has no numeric task.
    const uint32_t aligned = (r.n + 31) / 32 * 32;
    r.m = aligned <= 512 ? 512 : aligned <= 1024 ? 256 : aligned <= 2048 ? 128 : aligned <= 4096 ? 64 : 32;
    r.capacity = c.splits ? c.splits : c.cores;
    const uint64_t publicCapacity = c.splits == 0 ? 26 : c.splits;
    const uint64_t strideRows =
        (c.tnd ? uint64_t(c.totalQ ? c.totalQ : 1) : uint64_t(c.batch ? c.batch : 1) * (c.sq ? c.sq : 1)) * c.heads;
    if (strideRows > uint64_t(INT64_MAX) / 512 / publicCapacity)
        return false;
    if (c.batch && c.sq && c.sk && (!c.tnd || (c.totalQ && (c.tnd == 2 || c.totalK)))) {
        r.groups = uint64_t(c.batch) * c.kvHeads * ((uint64_t(c.sq) * g + r.m - 1) / r.m);
        if (c.tnd) {
            const uint64_t compact = (uint64_t(c.totalQ) * g / r.m + c.batch) * c.kvHeads;
            if (compact < r.groups)
                r.groups = compact;
        }
        r.scratch = uint64_t(c.cores) * (12 * uint64_t(r.m) * aligned + 16 * uint64_t(r.m) * 128);
        r.outOffset = r.scratch;
        const uint64_t rows = r.groups * r.capacity * r.m;
        if (rows > (uint64_t(INT64_MAX) - r.scratch) / (144 * 4))
            return false;
        r.sumOffset = r.outOffset + rows * 128 * 4;
        r.maxOffset = r.sumOffset + rows * 8 * 4;
        r.bytes = c.splits == 1 ? r.scratch : r.maxOffset + rows * 8 * 4;
    }
    out = r;
    return true;
}
struct MlaPrefillBsndMetadataArgs {
    MlaPrefillBsndContext context;
    uint64_t usedQuery, usedKv, output;
    uint32_t words;
    uint64_t cuQuery = 0, cuKv = 0;
    // Public mixed metadata receives only the device Q prefix, not a Q tensor.
    uint32_t inferTotalQ = 0;
};
// Unsafe output/overlap returns false without writes. Other failures publish
// a nonzero status and zero payload; successful snapshots own their lengths.
bool BuildPrefillBsndMetadata(const MlaPrefillBsndMetadataArgs&);
} // namespace flash_attn::mla
#undef MLA_PREFILL_INLINE

// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
// GQA scheduling adapter; fixed source/provenance in mla_split.h.

#if defined(__NPU_ARCH__) && defined(__aicore__)
#define MLA_GQA_INLINE __aicore__ inline
#else
#define MLA_GQA_INLINE inline
#endif
namespace flash_attn::mla {
constexpr uint32_t kMlaGqaVersion = 47;
// Physical tensor heads and logical scheduling groups are deliberately distinct.
// Layout: BSND=0, TND Q/KV=1, mixed=2, BSND PA=3, TND PA=4.
struct MlaGqaContext {
    uint32_t batch, sq, sk, heads, kvHeads, splits, dtype;
    uint32_t physicalCores, margin, packed, layout, totalQ, totalK, pageSize, usedFlags, valueDim, hasSoftcap;
    MlaWindowKernelParams window;
    uint64_t workspaceBytes = 0;
};
struct MlaGqaLayout {
    uint32_t words, m, n, cores, headGroups, g, capacity;
    uint64_t groups, scratch, outOffset, sumOffset, maxOffset, bytes;
};
MLA_GQA_INLINE bool GqaTnd(const MlaGqaContext& c)
{
    return c.layout == 1 || c.layout == 2 || c.layout == 4;
}
MLA_GQA_INLINE bool GqaEmptyKv(const MlaGqaContext& c)
{
    return !c.sk || (c.layout == 1 && !c.totalK);
}
MLA_GQA_INLINE uint32_t GqaTail(const MlaGqaContext& c)
{
    return 32 + 8 * c.batch;
}
MLA_GQA_INLINE bool GqaLayout(const MlaGqaContext& c, MlaGqaLayout& out)
{
    if (!c.kvHeads || !c.heads || c.heads % c.kvHeads || c.packed > 1 || (c.packed && c.heads == c.kvHeads) ||
        !c.physicalCores || c.physicalCores > 26 || c.margin >= c.physicalCores || c.layout > 4 ||
        (c.valueDim != 128 && c.valueDim != 512) || (c.valueDim == 512 && c.kvHeads != 1) ||
        ((c.layout >= 3) != bool(c.pageSize)))
        return false;
    MlaGqaLayout l{};
    l.cores = c.physicalCores - c.margin;
    l.g = c.packed ? c.heads / c.kvHeads : 1;
    l.headGroups = c.heads / l.g;
    // Reuse the existing static geometry, overflow checks and fixed generic
    // tiling rules. Compressed attention retains the source's 256x512 ceiling.
    MlaPrefillBsndContext p{c.batch,      c.sq,     c.sk,       c.heads,
                            l.headGroups, c.splits, c.dtype,    l.cores,
                            c.usedFlags,  c.window, 0,          GqaTnd(c) ? (c.layout == 1 ? 1U : 2U) : 0U,
                            c.totalQ,     c.totalK, c.pageSize, c.hasSoftcap};
    MlaPrefillBsndLayout base{};
    if (!PrefillBsndLayout(p, base) || uint64_t(c.batch) * (8 + uint64_t(2) * l.headGroups) > UINT32_MAX - 380)
        return false;
    l.m = c.valueDim == 512 ? 256 : base.m;
    l.n = c.valueDim == 512 ? (c.sk < 512 ? c.sk : 512) : base.n;
    if (c.valueDim == 512 && c.pageSize && l.n > c.pageSize && l.n % c.pageSize)
        l.n = l.n / c.pageSize * c.pageSize;
    if (!l.n)
        l.n = 32;
    l.words = GqaTail(c) + 348 + 2 * c.batch * l.headGroups;
    l.capacity = c.splits ? c.splits : l.cores;
    l.groups = uint64_t(c.batch) * l.headGroups * ((uint64_t(c.sq) * l.g + l.m - 1) / l.m);
    if (GqaTnd(c)) {
        const uint64_t compact = (uint64_t(c.totalQ) * l.g / l.m + c.batch) * l.headGroups;
        if (compact < l.groups)
            l.groups = compact;
        if (!c.totalQ)
            l.groups = 0;
    }
    if (l.groups && !GqaEmptyKv(c)) {
        l.scratch = uint64_t(l.cores) *
                    (12 * uint64_t(l.m) * ((l.n + 31) / 32 * 32) + 16 * uint64_t(l.m) * c.valueDim + 16 * l.m);
        l.outOffset = l.scratch;
        const uint64_t rows = l.groups * l.capacity * l.m;
        if (rows > (uint64_t(INT64_MAX) - l.scratch) / ((c.valueDim + 16) * 4))
            return false;
        l.sumOffset = l.outOffset + rows * c.valueDim * 4;
        l.maxOffset = l.sumOffset + rows * 8 * 4;
        l.bytes = l.maxOffset + rows * 8 * 4;
    }
    out = l;
    return true;
}
MLA_GQA_INLINE uint32_t GqaHeader(const MlaGqaContext& c, const MlaGqaLayout& l, uint32_t i)
{
    const uint32_t fields[32] = {kMlaMetadataMagic,
                                 kMlaGqaVersion,
                                 kMlaGqaVersion,
                                 l.words,
                                 c.dtype,
                                 0,
                                 c.batch,
                                 c.sq,
                                 c.sk,
                                 c.heads,
                                 c.kvHeads,
                                 c.splits,
                                 c.physicalCores,
                                 c.margin,
                                 c.packed,
                                 c.layout,
                                 c.totalQ,
                                 c.totalK,
                                 c.pageSize,
                                 c.usedFlags,
                                 c.valueDim,
                                 c.hasSoftcap,
                                 c.window.kind,
                                 uint32_t(c.window.left),
                                 uint32_t(c.window.right),
                                 l.m,
                                 l.n,
                                 l.cores,
                                 l.headGroups,
                                 0,
                                 0,
                                 0};
    return fields[i];
}
struct MlaGqaMetadataArgs {
    MlaGqaContext context;
    uint64_t cuQuery, cuKv, usedQuery, usedKv, output;
    uint32_t words, inferTotalQ = 0;
};
bool BuildGqaMetadata(const MlaGqaMetadataArgs&);
} // namespace flash_attn::mla
#undef MLA_GQA_INLINE
