// Private compact-TND protocol, generated and validated independently of
// attention. Never pass this record to a v1/v2 Kernel.
#pragma once
#include "mla_metadata.h"

namespace flash_attn::mla {

struct MlaTndShape;
constexpr uint32_t kMlaTndMetadataVersion = 3;
constexpr uint32_t kMlaMetadataCompactTnd = 3;

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
