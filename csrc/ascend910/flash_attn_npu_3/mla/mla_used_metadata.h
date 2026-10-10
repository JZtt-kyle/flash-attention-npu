// Private used-length protocol. Independent generation only; v1-v4 consumers
// must reject this record. Scheduling retains its CANN source/license in
// mla_tnd_schedule.cpp, mla_schedule.cpp and detail/cann_split_core.cpp.
#pragma once
#include "mla_metadata.h"
#include "mla_used_schedule.h"

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

} // namespace flash_attn::mla
