// Private BSND actual-KV-length protocol with a dedicated validation consumer;
// never pass this record to the v1 dense Kernel.
#pragma once
#include "mla_metadata.h"

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
