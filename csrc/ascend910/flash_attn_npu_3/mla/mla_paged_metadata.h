// Private PA protocol shared by CPU/AICPU generation and device consumers.
// Public metadata and BSND/TND paged forward use the same v6 record.
// Scheduling retains its CANN source/license in mla_tnd_schedule.cpp,
// mla_schedule.cpp and detail/cann_split_core.cpp (see LICENSE.cann).
#pragma once
#include "mla_metadata.h"
#include "mla_paged_schedule.h"

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
