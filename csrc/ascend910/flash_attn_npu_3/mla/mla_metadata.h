// Private NPU metadata for the CANN-derived dense MLA schedule. This is not a
// public entry point; the FA3 metadata API is connected in a separate step.
#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include "mla_schedule.h"

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
