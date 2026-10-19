#pragma once
#include "mla_dense.h"
#include "../tiling/mla_sequence.h"
#include <type_traits>

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
// tiling/mla_sequence.cpp, tiling/mla_schedule.cpp and detail/cann_split_core.cpp.
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
// device validation functions in metadata/mla_sequence_consumer.h.
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
