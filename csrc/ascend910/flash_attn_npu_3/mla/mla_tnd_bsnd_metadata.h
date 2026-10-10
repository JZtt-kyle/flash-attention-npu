// Private TND-Q/BSND-KV protocol with independent generation and validation.
// No attention entry yet. Never pass this record to a v1/v2/v3 Kernel.
#pragma once
#include "mla_metadata.h"

namespace flash_attn::mla {

struct MlaTndBsndShape;
constexpr uint32_t kMlaTndBsndMetadataVersion = 4;
constexpr uint32_t kMlaMetadataTndBsnd = 4;

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
bool BuildTndBsndMetadata(const MlaTndBsndShape& shape, const MlaPlatformInfo& platform, uint32_t dtype,
                          uint32_t* output, uint32_t outputWords, const int32_t* cuQuery, uint64_t queryCount,
                          const int32_t* kvLengths, uint64_t kvCount);

} // namespace flash_attn::mla
