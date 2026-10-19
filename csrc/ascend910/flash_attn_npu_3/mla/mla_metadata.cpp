#include "mla_metadata.h"
#include <algorithm>
#include <limits>
#include "mla_tiling.h"
#include <cstring>
#include <new>
#include <utility>
#include "mla_split.h"
#include "cann_split_core.h"
#include <vector>

// Serialization only. The source-derived algorithm and its copyright/license
// remain in mla_tiling.cpp and cann_split_core.cpp (LICENSE.cann).

namespace flash_attn::mla {
namespace {
MlaMetadataStatus CheckCacheLengths(const MlaDenseShape& shape, const int32_t* lengths, uint64_t count)
{
    if (lengths == nullptr && count == 0) {
        return MlaMetadataStatus::Success;
    }
    const auto address = reinterpret_cast<uintptr_t>(lengths);
    if (lengths == nullptr || address % alignof(int32_t) != 0 || count != uint64_t(shape.batch) ||
        count > (std::numeric_limits<uintptr_t>::max() - address) / sizeof(int32_t)) {
        return MlaMetadataStatus::InvalidCacheLengths;
    }
    for (uint64_t b = 0; b < count; ++b) {
        if (lengths[b] != shape.kvLength) {
            return MlaMetadataStatus::UnsupportedCacheLength;
        }
    }
    return MlaMetadataStatus::Success;
}

MlaMetadataWide Wide(uint64_t value)
{
    return {static_cast<uint32_t>(value), static_cast<uint32_t>(value >> 32)};
}

} // namespace

MlaMetadataSchedule SerializeMlaSchedule(const MlaSchedule& s)
{
    MlaMetadataSchedule out{};
    out.bSize = s.bSize;
    out.n2Size = s.n2Size;
    out.gSize = s.gSize;
    out.s1Size = s.s1Size;
    out.s2Size = s.s2Size;
    out.headDim = s.headDim;
    out.headDimRope = s.headDimRope;
    out.mBaseSize = s.mBaseSize;
    out.s2BaseSize = s.s2BaseSize;
    out.aicNum = s.aicNum;
    out.aivNum = s.aivNum;
    out.cvRatio = s.cvRatio;
    out.usedCoreNum = s.usedCoreNum;
    out.blockDim = s.blockDim;
    out.scheduleMode = s.scheduleMode;
    std::copy(s.bN2End.begin(), s.bN2End.end(), out.bN2End);
    std::copy(s.gS1End.begin(), s.gS1End.end(), out.gS1End);
    std::copy(s.s2End.begin(), s.s2End.end(), out.s2End);
    out.maxCost = Wide(static_cast<uint64_t>(s.maxCost));
    out.numOfFdHead = s.fd.numOfFdHead;
    out.maxS2SplitNum = s.fd.maxS2SplitNum;
    out.usedVecNumOfFd = s.fd.usedVecNumOfFd;
    out.gS1BaseSizeOfFd = s.fd.gS1BaseSizeOfFd;
    std::copy(s.fd.bN2IdxOfFdHead.begin(), s.fd.bN2IdxOfFdHead.end(), out.bN2IdxOfFdHead);
    std::copy(s.fd.gS1IdxOfFdHead.begin(), s.fd.gS1IdxOfFdHead.end(), out.gS1IdxOfFdHead);
    std::copy(s.fd.s2SplitNumOfFdHead.begin(), s.fd.s2SplitNumOfFdHead.end(), out.s2SplitNumOfFdHead);
    std::copy(s.fd.s2SplitStartIdxOfCore.begin(), s.fd.s2SplitStartIdxOfCore.end(), out.s2SplitStartIdxOfCore);
    std::copy(s.fd.gS1SplitNumOfFdHead.begin(), s.fd.gS1SplitNumOfFdHead.end(), out.gS1SplitNumOfFdHead);
    std::copy(s.fd.gS1LastPartSizeOfFdHead.begin(), s.fd.gS1LastPartSizeOfFdHead.end(), out.gS1LastPartSizeOfFdHead);
    std::copy(s.fd.gS1IdxEndOfFdHead.begin(), s.fd.gS1IdxEndOfFdHead.end(), out.gS1IdxEndOfFdHead);
    std::copy(s.fd.gS1IdxEndOfFdHeadSplit.begin(), s.fd.gS1IdxEndOfFdHeadSplit.end(), out.gS1IdxEndOfFdHeadSplit);
    const auto& w = s.workspace;
    out.workspace = {Wide(w.mm1ResSize),    Wide(w.mm2ResSize),   Wide(w.fdAccumOutSize), Wide(w.fdLogSumExpSize),
                     Wide(w.qkOffsetBytes), Wide(w.pOffsetBytes), Wide(w.pvOffsetBytes),  Wide(w.vec2OffsetBytes),
                     Wide(w.kernelBytes),   Wide(w.userBytes),    Wide(w.libApiBytes),    Wide(w.totalBytes)};
    const auto& m = s.requiredLocalMemory;
    out.requiredLocalMemory = {Wide(m.l1Bytes), Wide(m.ubBytes), Wide(m.l0aBytes), Wide(m.l0bBytes), Wide(m.l0cBytes)};
    return out;
}
void BuildDenseMetadata(const MlaDenseShape& shape, const MlaPlatformInfo& platform, uint32_t dtype,
                        MlaDenseMetadata& output, const int32_t* cacheSeqlens, uint64_t cacheSeqlensCount)
{
    MlaDenseMetadata metadata{};
    metadata.magic = kMlaMetadataMagic;
    metadata.version = kMlaMetadataVersion;
    metadata.kind = kMlaMetadataDenseSharedKV;
    metadata.wordCount = kMlaDenseMetadataWords;
    metadata.dtype = dtype;
    if (dtype != static_cast<uint32_t>(MlaMetadataDtype::Fp16) &&
        dtype != static_cast<uint32_t>(MlaMetadataDtype::Bf16)) {
        metadata.status = static_cast<uint32_t>(MlaMetadataStatus::UnsupportedDtype);
    } else {
        MlaSchedule schedule{};
        auto status = BuildDenseSchedule(shape, platform, schedule);
        metadata.scheduleStatus = static_cast<uint32_t>(status);
        if (status == MlaScheduleStatus::Success) {
            // Validate static scheduling bounds before dereferencing lengths.
            // Failure leaves the entire payload zero, including after reuse.
            metadata.status = static_cast<uint32_t>(CheckCacheLengths(shape, cacheSeqlens, cacheSeqlensCount));
            if (metadata.status == static_cast<uint32_t>(MlaMetadataStatus::Success)) {
                metadata.schedule = SerializeMlaSchedule(schedule);
            }
        } else {
            metadata.status = static_cast<uint32_t>(MlaMetadataStatus::ScheduleFailure);
        }
    }
    output = metadata;
}
} // namespace flash_attn::mla

// Serialization only. Scheduling and its CANN source/license remain in
// mla_tiling.cpp, mla_tiling.cpp and cann_split_core.cpp.

namespace flash_attn::mla {

bool BsndKvLengthMetadataWords(int64_t batch, uint32_t& words)
{
    if (batch < 0 || uint64_t(batch) > UINT32_MAX - kMlaBsndKvLengthMetadataPrefixWords) {
        return false;
    }
    words = kMlaBsndKvLengthMetadataPrefixWords + static_cast<uint32_t>(batch);
    return true;
}

bool BuildBsndKvLengthMetadata(const MlaDenseShape& shape, const MlaPlatformInfo& platform, uint32_t dtype,
                               uint32_t* output, uint32_t outputWords, const int32_t* cacheSeqlens,
                               uint64_t cacheSeqlensCount)
{
    uint32_t expectedWords;
    const uintptr_t outAddress = reinterpret_cast<uintptr_t>(output);
    constexpr auto maxAddress = std::numeric_limits<uintptr_t>::max();
    if (!BsndKvLengthMetadataWords(shape.batch, expectedWords) || outputWords != expectedWords || output == nullptr ||
        outAddress % alignof(int32_t) != 0 || outputWords > (maxAddress - outAddress) / sizeof(int32_t)) {
        return false;
    }
    const uintptr_t lengthAddress = reinterpret_cast<uintptr_t>(cacheSeqlens);
    const bool implicit = cacheSeqlens == nullptr && cacheSeqlensCount == 0;
    const bool validLengths = implicit || (cacheSeqlens != nullptr && lengthAddress % alignof(int32_t) == 0 &&
                                           cacheSeqlensCount == uint64_t(shape.batch) &&
                                           cacheSeqlensCount <= (maxAddress - lengthAddress) / sizeof(int32_t));
    const size_t outputBytes = size_t(outputWords) * sizeof(int32_t);
    if (cacheSeqlens != nullptr && cacheSeqlensCount != 0 &&
        cacheSeqlensCount <= (maxAddress - lengthAddress) / sizeof(int32_t) &&
        outAddress < lengthAddress + cacheSeqlensCount * sizeof(int32_t) && lengthAddress < outAddress + outputBytes) {
        return false;
    }

    MlaBsndKvLengthMetadataPrefix prefix{};
    prefix.magic = kMlaMetadataMagic;
    prefix.version = kMlaBsndKvLengthMetadataVersion;
    prefix.kind = kMlaMetadataBsndKvLengths;
    prefix.wordCount = outputWords;
    prefix.dtype = dtype;
    if (dtype != uint32_t(MlaMetadataDtype::Fp16) && dtype != uint32_t(MlaMetadataDtype::Bf16)) {
        prefix.status = uint32_t(MlaMetadataStatus::UnsupportedDtype);
    } else if (!validLengths) {
        prefix.status = uint32_t(MlaMetadataStatus::InvalidCacheLengths);
    } else {
        MlaSchedule schedule{};
        // b1 validates all static fields/resources before dereferencing lengths,
        // then checks every value before allocating the source int64 vector.
        const auto status = BuildBsndKvLengthSchedule(shape, platform, cacheSeqlens, cacheSeqlensCount, schedule);
        prefix.scheduleStatus = uint32_t(status);
        if (status == MlaScheduleStatus::Success) {
            prefix.schedule = SerializeMlaSchedule(schedule);
        } else {
            prefix.status = uint32_t(MlaMetadataStatus::ScheduleFailure);
        }
    }
    std::memset(output, 0, outputBytes);
    if (prefix.status == uint32_t(MlaMetadataStatus::Success)) {
        for (uint32_t b = 0; b < uint32_t(shape.batch); ++b) {
            output[kMlaBsndKvLengthMetadataPrefixWords + b] =
                implicit ? uint32_t(shape.kvLength) : uint32_t(cacheSeqlens[b]);
        }
    }
    std::memcpy(output, &prefix, sizeof(prefix));
    return true;
}

} // namespace flash_attn::mla

// Serialization only. Scheduling and its CANN source/license remain in
// mla_tiling.cpp, mla_tiling.cpp and cann_split_core.cpp.

namespace flash_attn::mla {

bool TndMetadataWords(int64_t batch, uint32_t& words)
{
    if (batch < 0 || uint64_t(batch) > (UINT32_MAX - kMlaTndMetadataPrefixWords) / kMlaTndMetadataSequenceWords) {
        return false;
    }
    words = kMlaTndMetadataPrefixWords + uint32_t(batch) * kMlaTndMetadataSequenceWords;
    return true;
}

bool BuildTndMetadata(const MlaTndShape& shape, const MlaPlatformInfo& platform, uint32_t dtype, uint32_t* output,
                      uint32_t outputWords, const int32_t* cuQuery, uint64_t queryCount, const int32_t* cuKv,
                      uint64_t kvCount)
{
    uint32_t expectedWords;
    const uintptr_t outAddress = reinterpret_cast<uintptr_t>(output);
    constexpr auto maxAddress = std::numeric_limits<uintptr_t>::max();
    if (!TndMetadataWords(shape.batch, expectedWords) || outputWords != expectedWords || output == nullptr ||
        outAddress % alignof(int32_t) != 0 || outputWords > (maxAddress - outAddress) / sizeof(int32_t)) {
        return false;
    }
    const size_t outputBytes = size_t(outputWords) * sizeof(int32_t);
    const auto overlaps = [&](const int32_t* input, uint64_t count) {
        const auto address = reinterpret_cast<uintptr_t>(input);
        // Subtraction avoids wrapping even for a malformed, enormous count.
        return input != nullptr && count != 0 && address < outAddress + outputBytes &&
               (address >= outAddress || count > (outAddress - address) / sizeof(int32_t));
    };
    if (overlaps(cuQuery, queryCount) || overlaps(cuKv, kvCount)) {
        return false;
    }
    const auto validDescriptor = [&](const int32_t* input, uint64_t count) {
        const auto address = reinterpret_cast<uintptr_t>(input);
        return input != nullptr && address % alignof(int32_t) == 0 && count == uint64_t(shape.batch) + 1 &&
               count <= (maxAddress - address) / sizeof(int32_t);
    };
    MlaTndMetadataPrefix prefix{};
    prefix.magic = kMlaMetadataMagic;
    prefix.version = kMlaTndMetadataVersion;
    prefix.kind = kMlaMetadataCompactTnd;
    prefix.wordCount = outputWords;
    prefix.dtype = dtype;
    MlaTndSchedule result;
    if (dtype != uint32_t(MlaMetadataDtype::Fp16) && dtype != uint32_t(MlaMetadataDtype::Bf16)) {
        prefix.status = uint32_t(MlaTndMetadataStatus::UnsupportedDtype);
    } else if (!validDescriptor(cuQuery, queryCount) || !validDescriptor(cuKv, kvCount)) {
        prefix.status = uint32_t(MlaTndMetadataStatus::InvalidPrefixes);
    } else {
        const auto status = BuildTndSchedule(shape, platform, cuQuery, queryCount, cuKv, kvCount, result);
        prefix.scheduleStatus = uint32_t(status);
        if (status == MlaScheduleStatus::Success) {
            prefix.schedule = SerializeMlaSchedule(result.schedule);
            prefix.totalQuery = uint32_t(shape.totalQuery);
            prefix.totalKv = uint32_t(shape.totalKv);
            prefix.maxQueryLength = uint32_t(shape.maxQueryLength);
            prefix.maxKvLength = uint32_t(shape.maxKvLength);
        } else {
            prefix.status = uint32_t(MlaTndMetadataStatus::ScheduleFailure);
        }
    }
    std::memset(output, 0, outputBytes);
    if (prefix.status == uint32_t(MlaTndMetadataStatus::Success)) {
        // Serialize the owned spans used by the scheduler, never reread inputs.
        uint32_t* span = output + kMlaTndMetadataPrefixWords;
        for (const auto& seq : result.lengths.sequences) {
            *span++ = seq.queryStart;
            *span++ = seq.kvStart;
            *span++ = seq.queryLength;
            *span++ = seq.kvLength;
        }
    }
    std::memcpy(output, &prefix, sizeof(prefix));
    return true;
}

bool TndBsndMetadataWords(int64_t batch, uint32_t& words)
{
    if (batch < 0 ||
        uint64_t(batch) > (UINT32_MAX - kMlaTndBsndMetadataPrefixWords) / kMlaTndBsndMetadataSequenceWords) {
        return false;
    }
    words = kMlaTndBsndMetadataPrefixWords + uint32_t(batch) * kMlaTndBsndMetadataSequenceWords;
    return true;
}

bool BuildTndBsndMetadata(const MlaTndBsndShape& inputShape, const MlaPlatformInfo& inputPlatform, uint32_t dtype,
                          uint32_t* output, uint32_t outputWords, const int32_t* cuQuery, uint64_t queryCount,
                          const int32_t* kvLengths, uint64_t kvCount, bool fromQueryPrefix)
{
    auto shape = inputShape;
    auto platform = inputPlatform;
    uint32_t expectedWords;
    const uintptr_t outAddress = reinterpret_cast<uintptr_t>(output);
    constexpr auto maxAddress = std::numeric_limits<uintptr_t>::max();
    if (!TndBsndMetadataWords(shape.batch, expectedWords) || outputWords != expectedWords || output == nullptr ||
        outAddress % alignof(int32_t) != 0 || outputWords > (maxAddress - outAddress) / sizeof(int32_t)) {
        return false;
    }
    const size_t outputBytes = size_t(outputWords) * sizeof(int32_t);
    const auto overlaps = [&](const int32_t* input, uint64_t count) {
        const auto address = reinterpret_cast<uintptr_t>(input);
        // Subtraction avoids wrapping even for a malformed, enormous count.
        return input != nullptr && count != 0 && address < outAddress + outputBytes &&
               (address >= outAddress || count > (outAddress - address) / sizeof(int32_t));
    };
    if (overlaps(cuQuery, queryCount) || overlaps(kvLengths, kvCount)) {
        return false;
    }
    const auto validDescriptor = [&](const int32_t* input, uint64_t count, uint64_t expected) {
        const auto address = reinterpret_cast<uintptr_t>(input);
        return count == expected && (input != nullptr || expected == 0) && address % alignof(int32_t) == 0 &&
               count <= (maxAddress - address) / sizeof(int32_t);
    };
    MlaTndBsndMetadataPrefix prefix{};
    prefix.magic = kMlaMetadataMagic;
    prefix.version = kMlaTndBsndMetadataVersion;
    prefix.kind = kMlaMetadataTndBsnd;
    prefix.wordCount = outputWords;
    prefix.dtype = dtype;
    MlaTndBsndSchedule result;
    if (dtype != uint32_t(MlaMetadataDtype::Fp16) && dtype != uint32_t(MlaMetadataDtype::Bf16)) {
        prefix.status = uint32_t(MlaTndBsndMetadataStatus::UnsupportedDtype);
    } else if (!validDescriptor(cuQuery, queryCount, uint64_t(shape.batch) + 1) ||
               !validDescriptor(kvLengths, kvCount, uint64_t(shape.batch))) {
        prefix.status = uint32_t(MlaTndBsndMetadataStatus::InvalidLengths);
    } else {
        if (fromQueryPrefix) {
            // Only AICPU reads device values. The parser still checks initial
            // zero, monotonicity, max/total address bounds and all KV lengths.
            shape.totalQuery = cuQuery[shape.batch];
            if (shape.batch != 0 && shape.totalQuery > 0 && shape.kvCapacity > 0 && platform.aicNum > 0 &&
                platform.aicNum <= kMlaMaxAic) {
                platform.blockDims.fill(platform.blockDims[platform.aicNum - 1]);
            }
        }
        const auto status = BuildTndBsndSchedule(shape, platform, cuQuery, queryCount, kvLengths, kvCount, result);
        prefix.scheduleStatus = uint32_t(status);
        if (status == MlaScheduleStatus::Success) {
            prefix.schedule = SerializeMlaSchedule(result.schedule);
            prefix.totalQuery = uint32_t(shape.totalQuery);
            prefix.maxQueryLength = uint32_t(shape.maxQueryLength);
            prefix.kvCapacity = uint32_t(shape.kvCapacity);
            const uint64_t totalKvCapacity = uint64_t(shape.batch) * uint64_t(shape.kvCapacity);
            prefix.totalKvCapacity = {uint32_t(totalKvCapacity), uint32_t(totalKvCapacity >> 32)};
        } else {
            prefix.status = uint32_t(MlaTndBsndMetadataStatus::ScheduleFailure);
        }
    }
    std::memset(output, 0, outputBytes);
    if (prefix.status == uint32_t(MlaTndBsndMetadataStatus::Success)) {
        // Serialize the owned spans used by the scheduler, never reread inputs.
        uint32_t* span = output + kMlaTndBsndMetadataPrefixWords;
        for (const auto& seq : result.lengths.sequences) {
            *span++ = seq.queryStart;
            *span++ = uint32_t(seq.kvStart);
            *span++ = uint32_t(seq.kvStart >> 32);
            *span++ = seq.queryLength;
            *span++ = seq.kvLength;
        }
    }
    std::memcpy(output, &prefix, sizeof(prefix));
    return true;
}

} // namespace flash_attn::mla

// Serialization only; reuse the CANN-derived allocation/used scheduler.

namespace flash_attn::mla {

bool UsedMetadataWords(int64_t batch, uint32_t& words)
{
    if (batch < 0 || uint64_t(batch) > (UINT32_MAX - kMlaUsedMetadataPrefixWords) / kMlaUsedMetadataSequenceWords) {
        return false;
    }
    words = kMlaUsedMetadataPrefixWords + uint32_t(batch) * kMlaUsedMetadataSequenceWords;
    return true;
}

bool BuildUsedMetadataWithSchedule(const MlaUsedShape& inputShape, const MlaPlatformInfo& inputPlatform, uint32_t dtype,
                                   uint32_t* output, uint32_t outputWords, const MlaUsedInputs& inputs,
                                   MlaUsedScheduleBuilder builder, const void* context, bool queryTotalFromPrefix,
                                   uint32_t trailingWords)
{
    auto shape = inputShape;
    auto platform = inputPlatform;
    uint32_t expectedWords;
    const uintptr_t outAddress = reinterpret_cast<uintptr_t>(output);
    constexpr auto maxAddress = std::numeric_limits<uintptr_t>::max();
    if (!UsedMetadataWords(shape.batch, expectedWords) || trailingWords > UINT32_MAX - expectedWords ||
        outputWords != expectedWords + trailingWords || output == nullptr || outAddress % alignof(int32_t) != 0 ||
        outputWords > (maxAddress - outAddress) / sizeof(int32_t)) {
        return false;
    }
    const size_t outputBytes = size_t(outputWords) * sizeof(int32_t);
    for (const auto& array : {inputs.cuQuery, inputs.cuKv, inputs.usedQuery, inputs.usedKv}) {
        const auto address = reinterpret_cast<uintptr_t>(array.data);
        // Subtraction avoids wrapping even for an invalid enormous count.
        if (array.data != nullptr && array.count != 0 && address < outAddress + outputBytes &&
            (address >= outAddress || array.count > (outAddress - address) / sizeof(int32_t))) {
            return false;
        }
    }
    const auto validDescriptor = [&](const MlaLengthArray& array) {
        const auto address = reinterpret_cast<uintptr_t>(array.data);
        return (array.data != nullptr || array.count == 0) && address % alignof(int32_t) == 0 &&
               array.count <= (maxAddress - address) / sizeof(int32_t);
    };
    const auto validAxis = [&](const MlaSequenceAllocation& allocation, const MlaLengthArray& prefix,
                               const MlaLengthArray& used) {
        if (!validDescriptor(prefix) || !validDescriptor(used) || (used.data && used.count != uint64_t(shape.batch))) {
            return false;
        }
        if (allocation.layout == MlaSequenceLayout::Tnd) {
            return prefix.data != nullptr && prefix.count == uint64_t(shape.batch) + 1;
        }
        if (allocation.layout == MlaSequenceLayout::Bsnd) {
            return prefix.data == nullptr && prefix.count == 0;
        }
        return true; // The scheduler rejects unknown layouts before any loads.
    };
    MlaUsedMetadataPrefix prefix{};
    prefix.magic = kMlaMetadataMagic;
    prefix.version = kMlaUsedMetadataVersion;
    prefix.kind = kMlaMetadataUsed;
    prefix.wordCount = outputWords;
    prefix.dtype = dtype;
    MlaUsedSchedule result;
    if (dtype != uint32_t(MlaMetadataDtype::Fp16) && dtype != uint32_t(MlaMetadataDtype::Bf16)) {
        prefix.status = uint32_t(MlaUsedMetadataStatus::UnsupportedDtype);
    } else if (!validAxis(shape.query, inputs.cuQuery, inputs.usedQuery) ||
               !validAxis(shape.kv, inputs.cuKv, inputs.usedKv) ||
               (queryTotalFromPrefix && shape.query.layout != MlaSequenceLayout::Tnd)) {
        prefix.status = uint32_t(MlaUsedMetadataStatus::InvalidLengths);
    } else {
        if (queryTotalFromPrefix) {
            // Only the AICPU producer reads device-owned prefix values. All
            // descriptors and output overlap were checked before this load.
            shape.query.totalTokens = inputs.cuQuery.data[shape.batch];
            if (shape.batch != 0 && shape.query.totalTokens > 0 && shape.kv.totalTokens > 0 && platform.aicNum > 0 &&
                platform.aicNum <= kMlaMaxAic) {
                platform.blockDims.fill(platform.blockDims[platform.aicNum - 1]);
            }
        }
        const auto status = builder(shape, platform, inputs, result, context);
        prefix.scheduleStatus = uint32_t(status);
        if (status == MlaScheduleStatus::Success) {
            prefix.schedule = SerializeMlaSchedule(result.schedule);
            const auto allocation = [](const MlaSequenceAllocation& axis) {
                const auto total = uint64_t(axis.totalTokens);
                return MlaMetadataAllocation{
                    uint32_t(axis.layout), {uint32_t(total), uint32_t(total >> 32)}, uint32_t(axis.maxLength)};
            };
            prefix.query = allocation(shape.query);
            prefix.kv = allocation(shape.kv);
            prefix.usedFlags =
                (inputs.usedQuery.data ? kMlaUsedQueryPresent : 0) | (inputs.usedKv.data ? kMlaUsedKvPresent : 0);
        } else {
            prefix.status = uint32_t(MlaUsedMetadataStatus::ScheduleFailure);
        }
    }
    std::memset(output, 0, outputBytes);
    if (prefix.status == uint32_t(MlaUsedMetadataStatus::Success)) {
        uint32_t* span = output + kMlaUsedMetadataPrefixWords;
        // Serialize the owned spans used by scheduling; never reread inputs.
        for (const auto& seq : result.lengths.sequences) {
            *span++ = uint32_t(seq.queryStart);
            *span++ = uint32_t(seq.queryStart >> 32);
            *span++ = uint32_t(seq.kvStart);
            *span++ = uint32_t(seq.kvStart >> 32);
            *span++ = seq.queryCapacity;
            *span++ = seq.kvCapacity;
            *span++ = seq.queryLength;
            *span++ = seq.kvLength;
        }
    }
    std::memcpy(output, &prefix, sizeof(prefix));
    return true;
}

bool BuildUsedMetadata(const MlaUsedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype, uint32_t* output,
                       uint32_t outputWords, const MlaUsedInputs& inputs)
{
    const auto builder = [](const MlaUsedShape& shape, const MlaPlatformInfo& platform, const MlaUsedInputs& inputs,
                            MlaUsedSchedule& result,
                            const void*) { return BuildUsedSchedule(shape, platform, inputs, result); };
    return BuildUsedMetadataWithSchedule(shape, platform, dtype, output, outputWords, inputs, builder);
}

} // namespace flash_attn::mla

// Serialization only; reuse the CANN-derived PA scheduler and common encoding.

namespace flash_attn::mla {

bool PagedMetadataWords(int64_t batch, uint32_t& words)
{
    if (batch < 0 || uint64_t(batch) > (UINT32_MAX - kMlaPagedMetadataPrefixWords) / kMlaPagedMetadataSequenceWords) {
        return false;
    }
    words = kMlaPagedMetadataPrefixWords + uint32_t(batch) * kMlaPagedMetadataSequenceWords;
    return true;
}

bool BuildPagedMetadataWithSchedule(const MlaPagedShape& inputShape, const MlaPlatformInfo& platform, uint32_t dtype,
                                    uint32_t* output, uint32_t outputWords, const MlaPagedInputs& inputs,
                                    bool fromQueryPrefix, MlaPagedScheduleBuilder builder, const void* context,
                                    uint32_t tailWords)
{
    auto shape = inputShape;
    uint32_t expectedWords;
    const uintptr_t outAddress = reinterpret_cast<uintptr_t>(output);
    constexpr auto maxAddress = std::numeric_limits<uintptr_t>::max();
    if (!PagedMetadataWords(shape.batch, expectedWords) || tailWords > UINT32_MAX - expectedWords ||
        outputWords != expectedWords + tailWords || output == nullptr || outAddress % alignof(int32_t) != 0 ||
        outputWords > (maxAddress - outAddress) / sizeof(int32_t)) {
        return false;
    }
    const size_t outputBytes = size_t(outputWords) * sizeof(int32_t);
    for (const auto& array : {inputs.cuQuery, inputs.usedQuery, inputs.kvLengths}) {
        const auto address = reinterpret_cast<uintptr_t>(array.data);
        // Check overlap before writing even a failure header. Subtraction
        // avoids wrapping invalid input ranges with enormous counts.
        if (array.data != nullptr && array.count != 0 && address < outAddress + outputBytes &&
            (address >= outAddress || array.count > (outAddress - address) / sizeof(int32_t))) {
            return false;
        }
    }
    const auto validDescriptor = [&](const MlaLengthArray& array) {
        const auto address = reinterpret_cast<uintptr_t>(array.data);
        return (array.data != nullptr || array.count == 0) && address % alignof(int32_t) == 0 &&
               array.count <= (maxAddress - address) / sizeof(int32_t);
    };
    const bool validLengths =
        validDescriptor(inputs.cuQuery) && validDescriptor(inputs.usedQuery) && validDescriptor(inputs.kvLengths) &&
        (shape.query.layout != MlaSequenceLayout::Tnd ||
         (inputs.cuQuery.data != nullptr && inputs.cuQuery.count == uint64_t(shape.batch) + 1)) &&
        (shape.query.layout != MlaSequenceLayout::Bsnd ||
         (inputs.cuQuery.data == nullptr && inputs.cuQuery.count == 0)) &&
        (inputs.usedQuery.data == nullptr || inputs.usedQuery.count == uint64_t(shape.batch)) &&
        inputs.kvLengths.count == uint64_t(shape.batch) && (shape.batch == 0 || inputs.kvLengths.data != nullptr);
    MlaPagedMetadataPrefix prefix{};
    prefix.magic = kMlaMetadataMagic;
    prefix.version = kMlaPagedMetadataVersion;
    prefix.kind = kMlaMetadataPaged;
    prefix.wordCount = outputWords;
    prefix.dtype = dtype;
    MlaPagedSchedule result;
    if (dtype != uint32_t(MlaMetadataDtype::Fp16) && dtype != uint32_t(MlaMetadataDtype::Bf16)) {
        prefix.status = uint32_t(MlaPagedMetadataStatus::UnsupportedDtype);
    } else if (!validLengths) {
        prefix.status = uint32_t(MlaPagedMetadataStatus::InvalidLengths);
    } else {
        if (fromQueryPrefix && shape.query.layout == MlaSequenceLayout::Tnd) {
            // No host readback. The scheduler still validates the initial
            // zero, monotonicity, per-segment bound, used and all KV lengths.
            shape.query.totalTokens = inputs.cuQuery.data[shape.batch];
        }
        const auto status = builder(shape, platform, inputs, result, context);
        prefix.scheduleStatus = uint32_t(status);
        if (status == MlaScheduleStatus::Success) {
            prefix.schedule = SerializeMlaSchedule(result.schedule);
            const uint64_t total = shape.query.totalTokens;
            prefix.query = {uint32_t(shape.query.layout),
                            {uint32_t(total), uint32_t(total >> 32)},
                            uint32_t(shape.query.maxLength)};
            prefix.usedFlags = inputs.usedQuery.data ? kMlaPagedUsedQueryPresent : 0;
            prefix.pageSize = result.pageSize;
            prefix.logicalKvCapacity = result.logicalKvCapacity;
        } else {
            prefix.status = uint32_t(MlaPagedMetadataStatus::ScheduleFailure);
        }
    }
    std::memset(output, 0, outputBytes);
    if (prefix.status == uint32_t(MlaPagedMetadataStatus::Success)) {
        uint32_t* span = output + kMlaPagedMetadataPrefixWords;
        // The scheduler owns these validated spans; never reread inputs.
        for (const auto& seq : result.lengths.sequences) {
            *span++ = uint32_t(seq.queryStart);
            *span++ = uint32_t(seq.queryStart >> 32);
            *span++ = seq.queryCapacity;
            *span++ = seq.queryLength;
            *span++ = seq.kvLength;
        }
    }
    std::memcpy(output, &prefix, sizeof(prefix));
    return true;
}

bool BuildPagedMetadata(const MlaPagedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype, uint32_t* output,
                        uint32_t outputWords, const MlaPagedInputs& inputs, bool fromQueryPrefix)
{
    return BuildPagedMetadataWithSchedule(shape, platform, dtype, output, outputWords, inputs, fromQueryPrefix,
                                          [](const auto& s, const auto& p, const auto& i, auto& r, const void*) {
                                              return BuildPagedSchedule(s, p, i, r);
                                          });
}

} // namespace flash_attn::mla

// Protocol adapter for the CANN-derived schedulers. Source/copyright/license
// are retained in mla_tiling.cpp, mla_tiling.cpp and LICENSE.cann.

namespace flash_attn::mla {
namespace {
bool MatchesFullSchedule(const MlaCausalFullSchedule* full, const MlaUsedShape& shape, const MlaUsedSchedule& result)
{
    const auto& s = result.schedule;
    if (full == nullptr || full->enabled != 1 || full->batch != shape.batch ||
        full->queryCapacity != shape.query.maxLength || full->kvCapacity != shape.kv.maxLength ||
        full->heads != shape.queryHeads || full->aicNum != s.aicNum || full->cvRatio != s.cvRatio ||
        full->mBaseSize != s.mBaseSize || full->s2BaseSize != s.s2BaseSize || full->usedCoreNum == 0 ||
        full->usedCoreNum > s.aicNum) {
        return false;
    }
    for (const auto& seq : result.lengths.sequences) {
        if (seq.queryLength != seq.queryCapacity || seq.kvLength != seq.kvCapacity) {
            return false;
        }
    }
    return true;
}

template <MlaSequenceLayout Layout, MlaSequenceLayout KvLayout = Layout>
MlaScheduleStatus BuildCausalAllocationSchedule(const MlaUsedShape& shape, const MlaPlatformInfo& platform,
                                                const MlaUsedInputs& inputs, MlaUsedSchedule& output,
                                                const void* context)
{
    if (shape.query.layout != Layout || shape.kv.layout != KvLayout) {
        return MlaScheduleStatus::UnsupportedShape;
    }
    try {
        MlaUsedSchedule result;
        auto status = PrepareUsedSchedule(shape, platform, inputs, result);
        if (status != MlaScheduleStatus::Success) {
            return status;
        }
        if (shape.batch != 0 && result.schedule.s1Size != 0 && result.schedule.s2Size != 0) {
            auto& s = result.schedule;
            const auto* full = static_cast<const MlaCausalFullSchedule*>(context);
            if (Layout == MlaSequenceLayout::Bsnd && MatchesFullSchedule(full, shape, result)) {
                s.usedCoreNum = full->usedCoreNum;
                s.bN2End = full->bN2End;
                s.gS1End = full->gS1End;
                s.s2End = full->s2End;
                s.maxCost = full->maxCost;
            } else {
                MlaCausalScheduleInput input{{}, shape.queryHeads, s.mBaseSize, s.s2BaseSize, s.aicNum, s.cvRatio};
                input.sequences.reserve(result.lengths.sequences.size());
                for (const auto& seq : result.lengths.sequences) {
                    input.sequences.push_back({seq.queryLength, seq.kvLength});
                }
                MlaCausalSchedule causal{};
                if (BuildCausalSchedule(input, causal) != MlaCausalStatus::Success) {
                    return MlaScheduleStatus::InvalidSplit;
                }
                s.usedCoreNum = causal.usedCoreNum;
                s.bN2End = causal.bN2End;
                s.gS1End = causal.gS1End;
                s.s2End = causal.s2End;
                s.maxCost = causal.maxCost;
            }
            s.blockDim = platform.blockDims[s.usedCoreNum - 1];
            const uint64_t slots = 2 * uint64_t(s.blockDim);
            auto& w = s.workspace;
            w.qkOffsetBytes = 0;
            w.pOffsetBytes = slots * w.mm1ResSize * 4;
            w.pvOffsetBytes = slots * w.mm1ResSize * 6;
            w.vec2OffsetBytes = slots * (w.mm1ResSize * 6 + w.mm2ResSize * 4);
            w.kernelBytes = slots * (w.mm1ResSize * 6 + w.mm2ResSize * 8);
            if (s.blockDim < s.usedCoreNum || s.blockDim > s.aicNum) {
                return MlaScheduleStatus::InvalidPlatform;
            }
            if (w.kernelBytes > w.userBytes) {
                return MlaScheduleStatus::WorkspaceOverflow;
            }
        }
        output = std::move(result);
        return MlaScheduleStatus::Success;
    } catch (const std::bad_alloc&) {
        return MlaScheduleStatus::AllocationFailure;
    }
}
} // namespace

bool BuildCausalBsndMetadata(const MlaUsedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype,
                             uint32_t* output, uint32_t outputWords, const MlaUsedInputs& inputs,
                             const MlaCausalFullSchedule* full)
{
    if (!BuildUsedMetadataWithSchedule(shape, platform, dtype, output, outputWords, inputs,
                                       BuildCausalAllocationSchedule<MlaSequenceLayout::Bsnd>, full)) {
        return false;
    }
    // Shared serialization has already zeroed failed payloads. The v7 identity
    // is the only wire change; consumers still validate all current lengths.
    output[1] = kMlaCausalBsndMetadataVersion;
    output[2] = kMlaMetadataCausalBsnd;
    return true;
}

bool BuildCausalTndMetadata(const MlaUsedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype,
                            uint32_t* output, uint32_t outputWords, const MlaUsedInputs& inputs)
{
    if (!BuildUsedMetadataWithSchedule(shape, platform, dtype, output, outputWords, inputs,
                                       BuildCausalAllocationSchedule<MlaSequenceLayout::Tnd>, nullptr)) {
        return false;
    }
    output[1] = kMlaCausalTndMetadataVersion;
    output[2] = kMlaMetadataCausalTnd;
    return true;
}

bool BuildCausalMixedMetadata(const MlaUsedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype,
                              uint32_t* output, uint32_t outputWords, const MlaUsedInputs& inputs,
                              bool queryTotalFromPrefix)
{
    if (!BuildUsedMetadataWithSchedule(shape, platform, dtype, output, outputWords, inputs,
                                       BuildCausalAllocationSchedule<MlaSequenceLayout::Tnd, MlaSequenceLayout::Bsnd>,
                                       nullptr, queryTotalFromPrefix)) {
        return false;
    }
    output[1] = kMlaCausalMixedMetadataVersion;
    output[2] = kMlaMetadataCausalMixed;
    return true;
}
} // namespace flash_attn::mla

// Protocol adapter for the pinned CANN/FA3 window scheduler in mla_tiling.cpp.

namespace flash_attn::mla {
namespace {
template <bool IsTnd, bool IsMixed = false>
MlaScheduleStatus BuildWindowAllocationSchedule(const MlaUsedShape& shape, const MlaPlatformInfo& platform,
                                                const MlaUsedInputs& inputs, MlaUsedSchedule& output,
                                                const void* context)
{
    const auto& config = *static_cast<const MlaWindowConfig*>(context);
    MlaWindow window{};
    constexpr auto layout = IsTnd ? MlaSequenceLayout::Tnd : MlaSequenceLayout::Bsnd;
    if (shape.query.layout != layout || shape.kv.layout != (IsMixed ? MlaSequenceLayout::Bsnd : layout) ||
        shape.query.maxLength != config.maxQueryLength || shape.kv.maxLength != config.maxKvLength ||
        NormalizeWindow(config, window) != MlaCausalStatus::Success) {
        return MlaScheduleStatus::UnsupportedShape;
    }
    try {
        MlaUsedSchedule result;
        auto status = PrepareUsedSchedule(shape, platform, inputs, result);
        if (status != MlaScheduleStatus::Success) {
            return status;
        }
        if (shape.batch != 0 && result.schedule.s1Size != 0 && result.schedule.s2Size != 0) {
            auto& s = result.schedule;
            MlaCausalScheduleInput input{{}, shape.queryHeads, s.mBaseSize, s.s2BaseSize, s.aicNum, s.cvRatio};
            input.sequences.reserve(result.lengths.sequences.size());
            for (const auto& seq : result.lengths.sequences) {
                input.sequences.push_back({seq.queryLength, seq.kvLength});
            }
            MlaCausalSchedule partition{};
            if (BuildWindowSchedule(input, config, partition) != MlaCausalStatus::Success) {
                return MlaScheduleStatus::InvalidSplit;
            }
            s.usedCoreNum = partition.usedCoreNum;
            s.bN2End = partition.bN2End;
            s.gS1End = partition.gS1End;
            s.s2End = partition.s2End;
            s.maxCost = partition.maxCost;
            s.blockDim = platform.blockDims[s.usedCoreNum - 1];
            const uint64_t slots = 2 * uint64_t(s.blockDim);
            auto& w = s.workspace;
            w.qkOffsetBytes = 0;
            w.pOffsetBytes = slots * w.mm1ResSize * 4;
            w.pvOffsetBytes = slots * w.mm1ResSize * 6;
            w.vec2OffsetBytes = slots * (w.mm1ResSize * 6 + w.mm2ResSize * 4);
            w.kernelBytes = slots * (w.mm1ResSize * 6 + w.mm2ResSize * 8);
            if (s.blockDim < s.usedCoreNum || s.blockDim > s.aicNum) {
                return MlaScheduleStatus::InvalidPlatform;
            }
            if (w.kernelBytes > w.userBytes) {
                return MlaScheduleStatus::WorkspaceOverflow;
            }
        }
        output = std::move(result);
        return MlaScheduleStatus::Success;
    } catch (const std::bad_alloc&) {
        return MlaScheduleStatus::AllocationFailure;
    }
}
} // namespace
template <bool IsTnd, bool IsMixed = false>
bool BuildWindowMetadata(const MlaUsedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype, uint32_t* output,
                         uint32_t outputWords, const MlaUsedInputs& inputs, const MlaWindowConfig& window,
                         bool queryTotalFromPrefix = false)
{
    if (!BuildUsedMetadataWithSchedule(shape, platform, dtype, output, outputWords, inputs,
                                       BuildWindowAllocationSchedule<IsTnd, IsMixed>, &window, queryTotalFromPrefix,
                                       kMlaWindowMetadataTailWords)) {
        return false;
    }
    output[1] = IsMixed ? kMlaWindowMixedMetadataVersion
                : IsTnd ? kMlaWindowTndMetadataVersion
                        : kMlaWindowBsndMetadataVersion;
    output[2] = IsMixed ? kMlaMetadataWindowMixed : IsTnd ? kMlaMetadataWindowTnd : kMlaMetadataWindowBsnd;
    if (output[5] == 0) {
        MlaWindow normalized{};
        NormalizeWindow(window, normalized); // Already checked by the partitioner, including empty allocations.
        output[outputWords - 3] = uint32_t(normalized.kind);
        output[outputWords - 2] = uint32_t(normalized.left);
        output[outputWords - 1] = uint32_t(normalized.right);
    }
    return true;
}
bool BuildWindowBsndMetadata(const MlaUsedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype,
                             uint32_t* output, uint32_t outputWords, const MlaUsedInputs& inputs,
                             const MlaWindowConfig& window)
{
    return BuildWindowMetadata<false>(shape, platform, dtype, output, outputWords, inputs, window);
}
bool BuildWindowTndMetadata(const MlaUsedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype,
                            uint32_t* output, uint32_t outputWords, const MlaUsedInputs& inputs,
                            const MlaWindowConfig& window)
{
    return BuildWindowMetadata<true>(shape, platform, dtype, output, outputWords, inputs, window);
}
bool BuildWindowMixedMetadata(const MlaUsedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype,
                              uint32_t* output, uint32_t outputWords, const MlaUsedInputs& inputs,
                              const MlaWindowConfig& window, bool queryTotalFromPrefix)
{
    return BuildWindowMetadata<true, true>(shape, platform, dtype, output, outputWords, inputs, window,
                                           queryTotalFromPrefix);
}
} // namespace flash_attn::mla

// BSND/TND PA causal adapter. CANN-derived algorithms and their license remain in
// mla_tiling.cpp, mla_tiling.cpp and LICENSE.cann; page Copy is unchanged.

namespace flash_attn::mla {
namespace {
template <MlaSequenceLayout Layout>
MlaScheduleStatus BuildCausalPagedSchedule(const MlaPagedShape& shape, const MlaPlatformInfo& platform,
                                           const MlaPagedInputs& inputs, MlaPagedSchedule& output, const void*)
{
    if (shape.query.layout != Layout) {
        return MlaScheduleStatus::UnsupportedShape;
    }
    try {
        MlaPagedSchedule result;
        const auto status = PreparePagedSchedule(shape, platform, inputs, result);
        if (status != MlaScheduleStatus::Success) {
            return status;
        }
        if (shape.batch != 0 && result.schedule.s1Size != 0 && result.schedule.s2Size != 0) {
            auto& s = result.schedule;
            MlaCausalScheduleInput input{{}, shape.queryHeads, s.mBaseSize, s.s2BaseSize, s.aicNum, s.cvRatio};
            input.sequences.reserve(result.lengths.sequences.size());
            for (const auto& seq : result.lengths.sequences) {
                input.sequences.push_back({seq.queryLength, seq.kvLength});
            }
            MlaCausalSchedule causal{};
            if (BuildCausalSchedule(input, causal) != MlaCausalStatus::Success) {
                return MlaScheduleStatus::InvalidSplit;
            }
            s.usedCoreNum = causal.usedCoreNum;
            s.bN2End = causal.bN2End;
            s.gS1End = causal.gS1End;
            s.s2End = causal.s2End;
            s.maxCost = causal.maxCost;
            s.blockDim = platform.blockDims[s.usedCoreNum - 1];
            const uint64_t slots = 2 * uint64_t(s.blockDim);
            auto& w = s.workspace;
            w.qkOffsetBytes = 0;
            w.pOffsetBytes = slots * w.mm1ResSize * 4;
            w.pvOffsetBytes = slots * w.mm1ResSize * 6;
            w.vec2OffsetBytes = slots * (w.mm1ResSize * 6 + w.mm2ResSize * 4);
            w.kernelBytes = slots * (w.mm1ResSize * 6 + w.mm2ResSize * 8);
            if (s.blockDim < s.usedCoreNum || s.blockDim > s.aicNum) {
                return MlaScheduleStatus::InvalidPlatform;
            }
            if (w.kernelBytes > w.userBytes) {
                return MlaScheduleStatus::WorkspaceOverflow;
            }
        }
        output = std::move(result);
        return MlaScheduleStatus::Success;
    } catch (const std::bad_alloc&) {
        return MlaScheduleStatus::AllocationFailure;
    }
}
} // namespace

bool BuildCausalPagedBsndMetadata(const MlaPagedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype,
                                  uint32_t* output, uint32_t outputWords, const MlaPagedInputs& inputs)
{
    if (!BuildPagedMetadataWithSchedule(shape, platform, dtype, output, outputWords, inputs, false,
                                        BuildCausalPagedSchedule<MlaSequenceLayout::Bsnd>)) {
        return false;
    }
    output[1] = kMlaCausalPagedBsndMetadataVersion;
    output[2] = kMlaMetadataCausalPagedBsnd;
    return true;
}
bool BuildCausalPagedTndMetadata(const MlaPagedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype,
                                 uint32_t* output, uint32_t outputWords, const MlaPagedInputs& inputs,
                                 bool fromQueryPrefix)
{
    if (!BuildPagedMetadataWithSchedule(shape, platform, dtype, output, outputWords, inputs, fromQueryPrefix,
                                        BuildCausalPagedSchedule<MlaSequenceLayout::Tnd>)) {
        return false;
    }
    output[1] = kMlaCausalPagedTndMetadataVersion;
    output[2] = kMlaMetadataCausalPagedTnd;
    return true;
}
} // namespace flash_attn::mla

// BSND/TND PA window adapters. Pinned CANN/FA3 algorithms and licenses remain in
// mla_tiling.cpp, mla_tiling.cpp, LICENSE.cann and LICENSE.fa3; page Copy is unchanged.

namespace flash_attn::mla {
namespace {
template <MlaSequenceLayout Layout>
MlaScheduleStatus BuildWindowPagedSchedule(const MlaPagedShape& shape, const MlaPlatformInfo& platform,
                                           const MlaPagedInputs& inputs, MlaPagedSchedule& output, const void* context)
{
    const auto& config = *static_cast<const MlaWindowConfig*>(context);
    MlaWindow normalized{};
    if (shape.query.layout != Layout || shape.query.maxLength != config.maxQueryLength ||
        shape.logicalKvCapacity != config.maxKvLength ||
        NormalizeWindow(config, normalized) != MlaCausalStatus::Success) {
        return MlaScheduleStatus::UnsupportedShape;
    }
    try {
        MlaPagedSchedule result;
        const auto status = PreparePagedSchedule(shape, platform, inputs, result);
        if (status != MlaScheduleStatus::Success) {
            return status;
        }
        if (shape.batch != 0 && result.schedule.s1Size != 0 && result.schedule.s2Size != 0) {
            auto& s = result.schedule;
            MlaCausalScheduleInput input{{}, shape.queryHeads, s.mBaseSize, s.s2BaseSize, s.aicNum, s.cvRatio};
            input.sequences.reserve(result.lengths.sequences.size());
            for (const auto& seq : result.lengths.sequences) {
                input.sequences.push_back({seq.queryLength, seq.kvLength});
            }
            MlaCausalSchedule causal{};
            if (BuildWindowSchedule(input, config, causal) != MlaCausalStatus::Success) {
                return MlaScheduleStatus::InvalidSplit;
            }
            s.usedCoreNum = causal.usedCoreNum;
            s.bN2End = causal.bN2End;
            s.gS1End = causal.gS1End;
            s.s2End = causal.s2End;
            s.maxCost = causal.maxCost;
            s.blockDim = platform.blockDims[s.usedCoreNum - 1];
            const uint64_t slots = 2 * uint64_t(s.blockDim);
            auto& w = s.workspace;
            w.qkOffsetBytes = 0;
            w.pOffsetBytes = slots * w.mm1ResSize * 4;
            w.pvOffsetBytes = slots * w.mm1ResSize * 6;
            w.vec2OffsetBytes = slots * (w.mm1ResSize * 6 + w.mm2ResSize * 4);
            w.kernelBytes = slots * (w.mm1ResSize * 6 + w.mm2ResSize * 8);
            if (s.blockDim < s.usedCoreNum || s.blockDim > s.aicNum) {
                return MlaScheduleStatus::InvalidPlatform;
            }
            if (w.kernelBytes > w.userBytes) {
                return MlaScheduleStatus::WorkspaceOverflow;
            }
        }
        output = std::move(result);
        return MlaScheduleStatus::Success;
    } catch (const std::bad_alloc&) {
        return MlaScheduleStatus::AllocationFailure;
    }
}
} // namespace

bool BuildWindowPagedBsndMetadata(const MlaPagedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype,
                                  uint32_t* output, uint32_t outputWords, const MlaPagedInputs& inputs,
                                  const MlaWindowConfig& window)
{
    if (!BuildPagedMetadataWithSchedule(shape, platform, dtype, output, outputWords, inputs, false,
                                        BuildWindowPagedSchedule<MlaSequenceLayout::Bsnd>, &window,
                                        kMlaWindowPagedMetadataTailWords)) {
        return false;
    }
    output[1] = kMlaWindowPagedBsndMetadataVersion;
    output[2] = kMlaMetadataWindowPagedBsnd;
    if (output[5] == 0) {
        MlaWindow normalized{};
        NormalizeWindow(window, normalized); // Checked even for empty allocations by the builder.
        output[outputWords - 3] = uint32_t(normalized.kind);
        output[outputWords - 2] = uint32_t(normalized.left);
        output[outputWords - 1] = uint32_t(normalized.right);
    }
    return true;
}
bool BuildWindowPagedTndMetadata(const MlaPagedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype,
                                 uint32_t* output, uint32_t outputWords, const MlaPagedInputs& inputs,
                                 const MlaWindowConfig& window, bool fromQueryPrefix)
{
    if (!BuildPagedMetadataWithSchedule(shape, platform, dtype, output, outputWords, inputs, fromQueryPrefix,
                                        BuildWindowPagedSchedule<MlaSequenceLayout::Tnd>, &window,
                                        kMlaWindowPagedMetadataTailWords)) {
        return false;
    }
    output[1] = kMlaWindowPagedTndMetadataVersion;
    output[2] = kMlaMetadataWindowPagedTnd;
    if (output[5] == 0) {
        MlaWindow normalized{};
        NormalizeWindow(window, normalized); // Checked even for empty allocations by the builder.
        output[outputWords - 3] = uint32_t(normalized.kind);
        output[outputWords - 2] = uint32_t(normalized.left);
        output[outputWords - 1] = uint32_t(normalized.right);
    }
    return true;
}
} // namespace flash_attn::mla

// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.

namespace flash_attn::mla {
namespace {
void WriteSplitTail(uint32_t* tail, const MlaWindowConfig& window, uint32_t splits, uint32_t batch,
                    const std::vector<uint64_t>& prefix)
{
    MlaWindow normalized{};
    NormalizeWindow(window, normalized);
    tail[0] = uint32_t(normalized.kind);
    tail[1] = uint32_t(normalized.left);
    tail[2] = uint32_t(normalized.right);
    tail[3] = uint32_t(splits);
    for (uint32_t b = 0; b <= batch; ++b) {
        const auto value = prefix[b];
        tail[4 + 2 * b] = uint32_t(value);
        tail[5 + 2 * b] = uint32_t(value >> 32);
    }
}
} // namespace

bool BuildSplitBsndMetadata(const MlaUsedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype,
                            uint32_t* output, uint32_t words, const MlaUsedInputs& inputs,
                            const MlaWindowConfig& window, int64_t splits, uint32_t hasSoftcap)
{
    uint32_t expected;
    if (!SplitBsndMetadataWords(shape.batch, expected) || words != expected)
        return false;
    struct Context {
        const MlaWindowConfig& window;
        int64_t splits;
        uint32_t hasSoftcap;
        MlaSplitKvPlan plan;
    } context{window, splits, hasSoftcap, {}};
    const auto builder = [](const MlaUsedShape& s, const MlaPlatformInfo& p, const MlaUsedInputs& in,
                            MlaUsedSchedule& schedule, const void* opaque) {
        auto& c = *const_cast<Context*>(static_cast<const Context*>(opaque));
        if (s.query.layout != MlaSequenceLayout::Bsnd || s.kv.layout != MlaSequenceLayout::Bsnd || c.hasSoftcap > 1 ||
            c.splits < (c.hasSoftcap ? 1 : 2) || c.splits > 256)
            return MlaScheduleStatus::InvalidSplit;
        auto status = BuildUsedSchedule(s, p, in, schedule);
        if (status != MlaScheduleStatus::Success)
            return status;
        const auto& t = schedule.schedule;
        std::vector<MlaCausalSequence> lengths;
        try {
            for (const auto& v : schedule.lengths.sequences)
                lengths.push_back({v.queryLength, v.kvLength});
        } catch (const std::bad_alloc&) {
            return MlaScheduleStatus::AllocationFailure;
        }
        const bool empty = !s.batch || !s.query.maxLength || !s.kv.maxLength;
        return BuildSplitKvPlan({s.batch, s.query.maxLength, s.kv.maxLength, s.queryHeads, empty ? 256 : t.mBaseSize,
                                 empty ? 512 : t.s2BaseSize, c.splits, p.aicNum, t.blockDim, t.cvRatio, 0},
                                c.window, lengths, c.plan);
    };
    const uint32_t base = kMlaUsedMetadataPrefixWords + kMlaUsedMetadataSequenceWords * uint32_t(shape.batch);
    if (!BuildUsedMetadataWithSchedule(shape, platform, dtype, output, words, inputs, builder, &context, false,
                                       words - base))
        return false;
    output[1] = output[2] = hasSoftcap ? kMlaSoftcapSplitBsndVersion : kMlaSplitBsndVersion;
    if (output[5] == 0) {
        WriteSplitTail(output + base, window, uint32_t(splits), uint32_t(shape.batch), context.plan.GroupPrefix());
    }
    return true;
}
bool BuildSplitTndMetadata(const MlaUsedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype, uint32_t* output,
                           uint32_t words, const MlaUsedInputs& inputs, const MlaWindowConfig& window, int64_t splits,
                           uint32_t hasSoftcap)
{
    uint32_t expected;
    if (!SplitTndMetadataWords(shape.batch, expected) || words != expected)
        return false;
    struct Context {
        const MlaWindowConfig& window;
        int64_t splits;
        uint32_t hasSoftcap;
        MlaSplitTndPlan plan;
    } context{window, splits, hasSoftcap, {}};
    const auto builder = [](const MlaUsedShape& s, const MlaPlatformInfo& p, const MlaUsedInputs& in,
                            MlaUsedSchedule& schedule, const void* opaque) {
        auto& c = *const_cast<Context*>(static_cast<const Context*>(opaque));
        if (s.query.layout != MlaSequenceLayout::Tnd || s.kv.layout != MlaSequenceLayout::Tnd || c.hasSoftcap > 1 ||
            c.splits < (c.hasSoftcap ? 1 : 2) || c.splits > 256)
            return MlaScheduleStatus::InvalidSplit;
        auto status = BuildUsedSchedule(s, p, in, schedule);
        if (status != MlaScheduleStatus::Success)
            return status;
        const auto& t = schedule.schedule;
        const bool empty = !s.query.totalTokens || !s.kv.totalTokens;
        return BuildSplitTndPlan(
            {{s.batch, s.query.totalTokens, s.kv.totalTokens, s.queryHeads, s.query.maxLength, s.kv.maxLength},
             empty ? 256 : t.mBaseSize,
             empty ? 512 : t.s2BaseSize,
             c.splits,
             p.aicNum,
             t.blockDim,
             t.cvRatio,
             0},
            c.window, in, c.plan);
    };
    const uint32_t base = kMlaUsedMetadataPrefixWords + kMlaUsedMetadataSequenceWords * uint32_t(shape.batch);
    if (!BuildUsedMetadataWithSchedule(shape, platform, dtype, output, words, inputs, builder, &context, false,
                                       words - base))
        return false;
    output[1] = output[2] = hasSoftcap ? kMlaSoftcapSplitTndVersion : kMlaSplitTndVersion;
    if (output[5] == 0) {
        WriteSplitTail(output + base, window, uint32_t(splits), uint32_t(shape.batch), context.plan.GroupPrefix());
    }
    return true;
}
bool BuildSplitMixedMetadata(const MlaUsedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype,
                             uint32_t* output, uint32_t words, const MlaUsedInputs& inputs,
                             const MlaWindowConfig& window, int64_t splits, bool queryTotalFromPrefix,
                             uint32_t hasSoftcap)
{
    uint32_t expected;
    if (!SplitMixedMetadataWords(shape.batch, expected) || words != expected)
        return false;
    struct Context {
        const MlaWindowConfig& window;
        int64_t splits;
        uint32_t hasSoftcap;
        MlaSplitMixedPlan plan;
    } context{window, splits, hasSoftcap, {}};
    const auto builder = [](const MlaUsedShape& s, const MlaPlatformInfo& p, const MlaUsedInputs& in,
                            MlaUsedSchedule& schedule, const void* opaque) {
        auto& c = *const_cast<Context*>(static_cast<const Context*>(opaque));
        if (s.query.layout != MlaSequenceLayout::Tnd || s.kv.layout != MlaSequenceLayout::Bsnd || c.hasSoftcap > 1 ||
            c.splits < (c.hasSoftcap ? 1 : 2) || c.splits > 256)
            return MlaScheduleStatus::InvalidSplit;
        auto status = BuildUsedSchedule(s, p, in, schedule);
        if (status != MlaScheduleStatus::Success)
            return status;
        const auto& t = schedule.schedule;
        const bool empty = !s.query.totalTokens || !s.kv.totalTokens;
        return BuildSplitMixedPlan({{s.batch, s.query.totalTokens, s.queryHeads, s.query.maxLength, s.kv.maxLength},
                                    empty ? 256 : t.mBaseSize,
                                    empty ? 512 : t.s2BaseSize,
                                    c.splits,
                                    p.aicNum,
                                    t.blockDim,
                                    t.cvRatio,
                                    0},
                                   c.window, in, c.plan);
    };
    const uint32_t base = kMlaUsedMetadataPrefixWords + kMlaUsedMetadataSequenceWords * uint32_t(shape.batch);
    if (!BuildUsedMetadataWithSchedule(shape, platform, dtype, output, words, inputs, builder, &context,
                                       queryTotalFromPrefix, words - base))
        return false;
    output[1] = output[2] = hasSoftcap ? kMlaSoftcapSplitMixedVersion : kMlaSplitMixedVersion;
    if (output[5] == 0) {
        WriteSplitTail(output + base, window, uint32_t(splits), uint32_t(shape.batch), context.plan.GroupPrefix());
    }
    return true;
}
bool BuildSplitPagedBsndMetadata(const MlaPagedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype,
                                 uint32_t* output, uint32_t words, const MlaPagedInputs& inputs,
                                 const MlaWindowConfig& window, int64_t splits, uint32_t hasSoftcap)
{
    uint32_t expected;
    if (!SplitPagedBsndMetadataWords(shape.batch, expected) || words != expected)
        return false;
    struct Context {
        const MlaWindowConfig& window;
        int64_t splits;
        uint32_t hasSoftcap;
        MlaSplitPagedBsndPlan plan;
    } context{window, splits, hasSoftcap, {}};
    const auto builder = [](const MlaPagedShape& s, const MlaPlatformInfo& p, const MlaPagedInputs& in,
                            MlaPagedSchedule& schedule, const void* opaque) {
        auto& c = *const_cast<Context*>(static_cast<const Context*>(opaque));
        if (c.hasSoftcap > 1 || c.splits < (c.hasSoftcap ? 1 : 2) || c.splits > 256)
            return MlaScheduleStatus::InvalidSplit;
        const auto status = BuildSplitPagedBsndPlan(s, p, in, c.window, c.splits, c.plan);
        if (status == MlaScheduleStatus::Success)
            schedule = std::move(c.plan.paged);
        return status;
    };
    const uint32_t base = kMlaPagedMetadataPrefixWords + kMlaPagedMetadataSequenceWords * uint32_t(shape.batch);
    if (!BuildPagedMetadataWithSchedule(shape, platform, dtype, output, words, inputs, false, builder, &context,
                                        words - base))
        return false;
    output[1] = output[2] = hasSoftcap ? kMlaSoftcapSplitPagedBsndVersion : kMlaSplitPagedBsndVersion;
    if (output[5] == 0) {
        WriteSplitTail(output + base, window, uint32_t(splits), uint32_t(shape.batch),
                       context.plan.split.GroupPrefix());
    }
    return true;
}
bool BuildSplitPagedTndMetadata(const MlaPagedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype,
                                uint32_t* output, uint32_t words, const MlaPagedInputs& inputs,
                                const MlaWindowConfig& window, int64_t splits, bool queryTotalFromPrefix,
                                uint32_t hasSoftcap)
{
    uint32_t expected;
    if (!SplitPagedTndMetadataWords(shape.batch, expected) || words != expected)
        return false;
    struct Context {
        const MlaWindowConfig& window;
        int64_t splits;
        bool fromPrefix;
        uint32_t hasSoftcap;
        MlaSplitPagedTndPlan plan;
    } context{window, splits, queryTotalFromPrefix, hasSoftcap, {}};
    const auto builder = [](const MlaPagedShape& s, const MlaPlatformInfo& p, const MlaPagedInputs& in,
                            MlaPagedSchedule& schedule, const void* opaque) {
        auto& c = *const_cast<Context*>(static_cast<const Context*>(opaque));
        if (c.hasSoftcap > 1 || c.splits < (c.hasSoftcap ? 1 : 2) || c.splits > 256)
            return MlaScheduleStatus::InvalidSplit;
        auto platform = p;
        if (c.fromPrefix)
            platform.blockDims.fill(s.batch && s.query.totalTokens && s.logicalKvCapacity ? p.aicNum : 1);
        const auto status = BuildSplitPagedTndPlan(s, platform, in, c.window, c.splits, c.plan);
        if (status == MlaScheduleStatus::Success)
            schedule = std::move(c.plan.paged);
        return status;
    };
    const uint32_t base = kMlaPagedMetadataPrefixWords + kMlaPagedMetadataSequenceWords * uint32_t(shape.batch);
    if (!BuildPagedMetadataWithSchedule(shape, platform, dtype, output, words, inputs, queryTotalFromPrefix, builder,
                                        &context, words - base))
        return false;
    output[1] = output[2] = hasSoftcap ? kMlaSoftcapSplitPagedTndVersion : kMlaSplitPagedTndVersion;
    if (output[5] == 0) {
        WriteSplitTail(output + base, window, uint32_t(splits), uint32_t(shape.batch), context.plan.GroupPrefix());
    }
    return true;
}
} // namespace flash_attn::mla

// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.

namespace flash_attn::mla {
namespace {
template <typename Plan>
void WriteAutoSplitTail(uint32_t* tail, const MlaWindowConfig& window, const Plan& plan)
{
    MlaWindow normalized{};
    NormalizeWindow(window, normalized);
    *tail++ = uint32_t(normalized.kind);
    *tail++ = uint32_t(normalized.left);
    *tail++ = uint32_t(normalized.right);
    const auto& n = plan.Native();
    *tail++ = n.usedCoreNum;
    const auto emit = [&](const auto& array) {
        for (auto x : array)
            *tail++ = x;
    };
    emit(n.bN2End);
    emit(n.gS1End);
    emit(n.s2End);
    const auto& f = n.fd;
    *tail++ = f.numOfFdHead;
    *tail++ = f.maxS2SplitNum;
    *tail++ = f.usedVecNumOfFd;
    *tail++ = f.gS1BaseSizeOfFd;
    emit(f.bN2IdxOfFdHead);
    emit(f.gS1IdxOfFdHead);
    emit(f.s2SplitNumOfFdHead);
    emit(f.s2SplitStartIdxOfCore);
    emit(f.gS1SplitNumOfFdHead);
    emit(f.gS1LastPartSizeOfFdHead);
    emit(f.gS1IdxEndOfFdHead);
    emit(f.gS1IdxEndOfFdHeadSplit);
    for (auto value : plan.GroupPrefix()) {
        *tail++ = uint32_t(value);
        *tail++ = uint32_t(value >> 32);
    }
}
} // namespace

bool BuildAutoSplitBsndMetadata(const MlaUsedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype,
                                uint32_t* output, uint32_t words, const MlaUsedInputs& inputs,
                                const MlaWindowConfig& window, uint32_t hasSoftcap)
{
    uint32_t expected;
    if (hasSoftcap > 1 || !AutoSplitBsndMetadataWords(shape.batch, expected) || words != expected)
        return false;
    struct Context {
        const MlaWindowConfig& window;
        MlaAutoSplitKvPlan plan;
    } context{window, {}};
    const auto builder = [](const MlaUsedShape& s, const MlaPlatformInfo& p, const MlaUsedInputs& in,
                            MlaUsedSchedule& schedule, const void* opaque) {
        auto& c = *const_cast<Context*>(static_cast<const Context*>(opaque));
        if (s.query.layout != MlaSequenceLayout::Bsnd || s.kv.layout != MlaSequenceLayout::Bsnd)
            return MlaScheduleStatus::UnsupportedShape;
        // The CPU base scheduler also models C1V1, but the reused device
        // schema/consumer is C1V2. Reject before publishing an unusable record.
        if (!p.aicNum || p.aicNum > kMlaMaxAic || p.aivNum != 2 * p.aicNum)
            return MlaScheduleStatus::InvalidPlatform;
        auto status = BuildUsedSchedule(s, p, in, schedule);
        if (status != MlaScheduleStatus::Success)
            return status;
        const auto& t = schedule.schedule;
        const bool empty = !s.batch || !s.query.maxLength || !s.kv.maxLength;
        // Auto scratch is indexed by physical AIC capacity, independent of the
        // number of native partitions. A producer must retain that launch ABI.
        if (!empty && t.blockDim != p.aicNum)
            return MlaScheduleStatus::InvalidPlatform;
        std::vector<MlaCausalSequence> lengths;
        try {
            lengths.reserve(schedule.lengths.sequences.size());
            for (const auto& v : schedule.lengths.sequences)
                lengths.push_back({v.queryLength, v.kvLength});
        } catch (const std::bad_alloc&) {
            return MlaScheduleStatus::AllocationFailure;
        }
        return BuildAutoSplitKvPlan({s.batch, s.query.maxLength, s.kv.maxLength, s.queryHeads,
                                     empty ? 256 : t.mBaseSize, empty ? 512 : t.s2BaseSize, p.aicNum, 2, 0},
                                    c.window, lengths, c.plan);
    };
    const uint32_t base = kMlaUsedMetadataPrefixWords + kMlaUsedMetadataSequenceWords * uint32_t(shape.batch);
    if (!BuildUsedMetadataWithSchedule(shape, platform, dtype, output, words, inputs, builder, &context, false,
                                       words - base))
        return false;
    output[1] = output[2] = hasSoftcap ? kMlaSoftcapAutoSplitBsndVersion : kMlaAutoSplitBsndVersion;
    if (output[5] != 0)
        return true;
    WriteAutoSplitTail(output + base, window, context.plan);
    return true;
}
bool BuildAutoSplitTndMetadata(const MlaUsedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype,
                               uint32_t* output, uint32_t words, const MlaUsedInputs& inputs,
                               const MlaWindowConfig& window, uint32_t hasSoftcap)
{
    uint32_t expected;
    if (!AutoSplitTndMetadataWords(shape.batch, expected) || words != expected)
        return false;
    struct Context {
        const MlaWindowConfig& window;
        uint32_t hasSoftcap;
        MlaAutoSplitTndPlan plan;
    } context{window, hasSoftcap, {}};
    const auto builder = [](const MlaUsedShape& s, const MlaPlatformInfo& p, const MlaUsedInputs& in,
                            MlaUsedSchedule& schedule, const void* opaque) {
        auto& c = *const_cast<Context*>(static_cast<const Context*>(opaque));
        if (s.query.layout != MlaSequenceLayout::Tnd || s.kv.layout != MlaSequenceLayout::Tnd || c.hasSoftcap > 1)
            return MlaScheduleStatus::UnsupportedShape;
        // The CPU base scheduler also models C1V1, but the reused device
        // schema/consumer is C1V2. Reject before publishing an unusable record.
        if (!p.aicNum || p.aicNum > kMlaMaxAic || p.aivNum != 2 * p.aicNum)
            return MlaScheduleStatus::InvalidPlatform;
        auto status = BuildUsedSchedule(s, p, in, schedule);
        if (status != MlaScheduleStatus::Success)
            return status;
        const auto& t = schedule.schedule;
        const bool empty = !s.query.totalTokens || !s.kv.totalTokens;
        // Auto scratch is indexed by physical AIC capacity, independent of the
        // number of native partitions. A producer must retain that launch ABI.
        if (!empty && t.blockDim != p.aicNum)
            return MlaScheduleStatus::InvalidPlatform;
        return BuildAutoSplitTndPlan(
            {{s.batch, s.query.totalTokens, s.kv.totalTokens, s.queryHeads, s.query.maxLength, s.kv.maxLength},
             empty ? 256 : t.mBaseSize,
             empty ? 512 : t.s2BaseSize,
             p.aicNum,
             2,
             0},
            c.window, in, c.plan);
    };
    const uint32_t base = kMlaUsedMetadataPrefixWords + kMlaUsedMetadataSequenceWords * uint32_t(shape.batch);
    if (!BuildUsedMetadataWithSchedule(shape, platform, dtype, output, words, inputs, builder, &context, false,
                                       words - base))
        return false;
    output[1] = output[2] = hasSoftcap ? kMlaSoftcapAutoSplitTndVersion : kMlaAutoSplitTndVersion;
    if (output[5] != 0)
        return true;
    WriteAutoSplitTail(output + base, window, context.plan);
    return true;
}
bool BuildAutoSplitMixedMetadata(const MlaUsedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype,
                                 uint32_t* output, uint32_t words, const MlaUsedInputs& inputs,
                                 const MlaWindowConfig& window, bool queryTotalFromPrefix, uint32_t hasSoftcap)
{
    uint32_t expected;
    if (!AutoSplitMixedMetadataWords(shape.batch, expected) || words != expected)
        return false;
    struct Context {
        const MlaWindowConfig& window;
        uint32_t hasSoftcap;
        MlaAutoSplitMixedPlan plan;
    } context{window, hasSoftcap, {}};
    const auto builder = [](const MlaUsedShape& s, const MlaPlatformInfo& p, const MlaUsedInputs& in,
                            MlaUsedSchedule& schedule, const void* opaque) {
        auto& c = *const_cast<Context*>(static_cast<const Context*>(opaque));
        if (s.query.layout != MlaSequenceLayout::Tnd || s.kv.layout != MlaSequenceLayout::Bsnd || c.hasSoftcap > 1)
            return MlaScheduleStatus::UnsupportedShape;
        // The CPU base scheduler also models C1V1, but the reused device
        // schema/consumer is C1V2. Reject before publishing an unusable record.
        if (!p.aicNum || p.aicNum > kMlaMaxAic || p.aivNum != 2 * p.aicNum)
            return MlaScheduleStatus::InvalidPlatform;
        auto status = BuildUsedSchedule(s, p, in, schedule);
        if (status != MlaScheduleStatus::Success)
            return status;
        const auto& t = schedule.schedule;
        const bool empty = !s.query.totalTokens || !s.kv.totalTokens;
        // Auto scratch is indexed by physical AIC capacity, independent of the
        // number of native partitions. A producer must retain that launch ABI.
        if (!empty && t.blockDim != p.aicNum)
            return MlaScheduleStatus::InvalidPlatform;
        return BuildAutoSplitMixedPlan({{s.batch, s.query.totalTokens, s.queryHeads, s.query.maxLength, s.kv.maxLength},
                                        empty ? 256 : t.mBaseSize,
                                        empty ? 512 : t.s2BaseSize,
                                        p.aicNum,
                                        2,
                                        0},
                                       c.window, in, c.plan);
    };
    const uint32_t base = kMlaUsedMetadataPrefixWords + kMlaUsedMetadataSequenceWords * uint32_t(shape.batch);
    if (!BuildUsedMetadataWithSchedule(shape, platform, dtype, output, words, inputs, builder, &context,
                                       queryTotalFromPrefix, words - base))
        return false;
    output[1] = output[2] = hasSoftcap ? kMlaSoftcapAutoSplitMixedVersion : kMlaAutoSplitMixedVersion;
    if (output[5] != 0)
        return true;
    WriteAutoSplitTail(output + base, window, context.plan);
    return true;
}
bool BuildAutoSplitPagedBsndMetadata(const MlaPagedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype,
                                     uint32_t* output, uint32_t words, const MlaPagedInputs& inputs,
                                     const MlaWindowConfig& window, uint32_t hasSoftcap)
{
    uint32_t expected;
    if (hasSoftcap > 1 || !AutoSplitPagedBsndMetadataWords(shape.batch, expected) || words != expected)
        return false;
    struct Context {
        const MlaWindowConfig& window;
        MlaAutoSplitPagedBsndPlan plan;
    } context{window, {}};
    const auto builder = [](const MlaPagedShape& s, const MlaPlatformInfo& p, const MlaPagedInputs& in,
                            MlaPagedSchedule& schedule, const void* opaque) {
        auto& c = *const_cast<Context*>(static_cast<const Context*>(opaque));
        const auto status = BuildAutoSplitPagedBsndPlan(s, p, in, c.window, c.plan);
        if (status == MlaScheduleStatus::Success)
            schedule = std::move(c.plan.paged);
        return status;
    };
    const uint32_t base = kMlaPagedMetadataPrefixWords + kMlaPagedMetadataSequenceWords * uint32_t(shape.batch);
    if (!BuildPagedMetadataWithSchedule(shape, platform, dtype, output, words, inputs, false, builder, &context,
                                        words - base))
        return false;
    output[1] = output[2] = hasSoftcap ? kMlaSoftcapAutoSplitPagedBsndVersion : kMlaAutoSplitPagedBsndVersion;
    if (output[5] != 0)
        return true;
    WriteAutoSplitTail(output + base, window, context.plan.split);
    return true;
}
bool BuildAutoSplitPagedTndMetadata(const MlaPagedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype,
                                    uint32_t* output, uint32_t words, const MlaPagedInputs& inputs,
                                    const MlaWindowConfig& window, bool queryTotalFromPrefix, uint32_t hasSoftcap)
{
    uint32_t expected;
    if (!AutoSplitPagedTndMetadataWords(shape.batch, expected) || words != expected)
        return false;
    struct Context {
        const MlaWindowConfig& window;
        bool fromPrefix;
        uint32_t hasSoftcap;
        MlaAutoSplitPagedTndPlan plan;
    } context{window, queryTotalFromPrefix, hasSoftcap, {}};
    const auto builder = [](const MlaPagedShape& s, const MlaPlatformInfo& p, const MlaPagedInputs& in,
                            MlaPagedSchedule& schedule, const void* opaque) {
        auto& c = *const_cast<Context*>(static_cast<const Context*>(opaque));
        if (c.hasSoftcap > 1)
            return MlaScheduleStatus::InvalidSplit;
        auto platform = p;
        if (c.fromPrefix)
            platform.blockDims.fill(s.batch && s.query.totalTokens && s.logicalKvCapacity ? p.aicNum : 1);
        MlaAutoSplitPagedTndBudget production{};
        auto status = BuildAutoSplitPagedTndProductionBudget(s, platform, production);
        if (status == MlaScheduleStatus::Success)
            status = BuildAutoSplitPagedTndPlan(s, platform, in, c.window, c.plan);
        if (status == MlaScheduleStatus::Success)
            schedule = std::move(c.plan.paged);
        return status;
    };
    const uint32_t base = kMlaPagedMetadataPrefixWords + kMlaPagedMetadataSequenceWords * uint32_t(shape.batch);
    if (!BuildPagedMetadataWithSchedule(shape, platform, dtype, output, words, inputs, queryTotalFromPrefix, builder,
                                        &context, words - base))
        return false;
    output[1] = output[2] = hasSoftcap ? kMlaSoftcapAutoSplitPagedTndVersion : kMlaAutoSplitPagedTndVersion;
    if (output[5] != 0)
        return true;
    WriteAutoSplitTail(output + base, window, context.plan.split);
    return true;
}
} // namespace flash_attn::mla

// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
// Native partition algorithm: ca18aee attention/common/op_host/split_core.cpp.

namespace flash_attn::mla {
bool BuildPrefillBsndMetadata(const MlaPrefillBsndMetadataArgs& a)
{
    auto c = a.context;
    MlaPrefillBsndLayout l{};
    if (!a.output || a.output % 4 || !PrefillBsndLayout(c, l) || a.words != l.words ||
        a.words > (UINT64_MAX - a.output) / 4)
        return false;
    // Same subtraction-based range checks as BuildUsedMetadataWithSchedule.
    // Neither read-only length array may overlap the writable record.
    for (uint32_t axis = 0; axis < 4; ++axis) {
        const uint64_t address = axis == 0 ? a.usedQuery : axis == 1 ? a.usedKv : axis == 2 ? a.cuQuery : a.cuKv;
        const uint64_t count = uint64_t(c.batch) + (axis >= 2);
        if (address && count && address < a.output + uint64_t(a.words) * 4 &&
            (address >= a.output || count > (a.output - address) / 4))
            return false;
    }
    auto* out = reinterpret_cast<uint32_t*>(a.output);
    std::fill(out, out + a.words, 0);
    out[0] = kMlaMetadataMagic;
    out[1] = out[2] =
        c.hasSoftcap ? (c.pageSize   ? (c.tnd ? kMlaSoftcapPrefillPagedTndVersion : kMlaSoftcapPrefillPagedBsndVersion)
                        : c.tnd == 2 ? kMlaSoftcapPrefillMixedVersion
                        : c.tnd == 1 ? kMlaSoftcapPrefillTndVersion
                                     : kMlaSoftcapPrefillBsndVersion)
        : c.pageSize ? (c.tnd ? kMlaPrefillPagedTndVersion : kMlaPrefillPagedBsndVersion)
        : c.tnd == 2 ? kMlaPrefillMixedVersion
        : c.tnd      ? kMlaPrefillTndVersion
                     : kMlaPrefillBsndVersion;
    out[3] = a.words;
    out[4] = c.dtype;
    out[5] = 1;
    if ((c.batch && (bool(c.usedFlags & 1) != bool(a.usedQuery) || bool(c.usedFlags & 2) != bool(a.usedKv))) ||
        a.usedQuery % 4 || a.usedKv % 4 || (a.usedQuery && c.batch > (UINT64_MAX - a.usedQuery) / 4) ||
        (a.usedKv && c.batch > (UINT64_MAX - a.usedKv) / 4) ||
        (c.dtype != uint32_t(MlaMetadataDtype::Fp16) && c.dtype != uint32_t(MlaMetadataDtype::Bf16)))
        return true;
    if (c.tnd && (!a.cuQuery || a.cuQuery % 4 || uint64_t(c.batch) + 1 > (UINT64_MAX - a.cuQuery) / 4 ||
                  (c.tnd == 1 && (!a.cuKv || a.cuKv % 4 || uint64_t(c.batch) + 1 > (UINT64_MAX - a.cuKv) / 4))))
        return true;
    if ((!c.tnd && a.cuQuery) || (c.tnd != 1 && a.cuKv) || a.inferTotalQ > 1 || (a.inferTotalQ && c.tnd != 2))
        return true;
    if (a.inferTotalQ) {
        c.totalQ = reinterpret_cast<const uint32_t*>(a.cuQuery)[c.batch];
        if (!PrefillBsndLayout(c, l))
            return true;
    }
    try {
        MlaUsedLengths allocation;
        if (c.tnd &&
            ParseUsedLengths({c.batch,
                              c.heads,
                              {MlaSequenceLayout::Tnd, c.totalQ, c.sq},
                              {c.tnd == 1 ? MlaSequenceLayout::Tnd : MlaSequenceLayout::Bsnd,
                               c.tnd == 1 ? int64_t(c.totalK) : int64_t(c.batch) * c.sk, c.sk}},
                             {{reinterpret_cast<const int32_t*>(a.cuQuery), uint64_t(c.batch) + 1},
                              {reinterpret_cast<const int32_t*>(a.cuKv), c.tnd == 1 ? uint64_t(c.batch) + 1 : 0},
                              {reinterpret_cast<const int32_t*>(a.usedQuery), a.usedQuery ? c.batch : 0},
                              {reinterpret_cast<const int32_t*>(a.usedKv), a.usedKv ? c.batch : 0}},
                             allocation) != MlaScheduleStatus::Success)
            return true;
        std::vector<uint32_t> result(a.words, 0);
        const uint32_t header[] = {kMlaMetadataMagic,
                                   out[1],
                                   out[2],
                                   a.words,
                                   c.dtype,
                                   0,
                                   c.batch,
                                   c.sq,
                                   c.sk,
                                   c.heads,
                                   c.kvHeads,
                                   c.splits,
                                   c.cores,
                                   l.m,
                                   l.n,
                                   c.usedFlags,
                                   c.window.kind,
                                   uint32_t(c.window.left),
                                   uint32_t(c.window.right),
                                   0};
        std::copy(header, header + 20, result.begin());
        if (c.tnd) {
            result[20] = c.totalQ;
            result[21] = c.totalK;
        }
        if (c.pageSize)
            result[c.tnd ? 21 : 20] = c.pageSize;
        detail::BaseInfo base{};
        base.bSize = c.batch;
        base.n2Size = c.kvHeads;
        base.gSize = c.heads / c.kvHeads;
        base.s1Size = c.sq;
        base.s2Size = c.sk;
        base.isS1G = true;
        base.actualLenQDims = base.actualLenKvDims = c.batch;
        base.attenMaskFlag = c.window.kind != 0;
        base.sparseMode = c.window.kind == 2 ? 4 : c.window.kind == 1 ? 3 : 0;
        base.preToken = c.window.kind == 1 ? INT32_MAX : c.window.left;
        base.nextToken = c.window.kind == 1 ? 0 : c.window.right;
        uint32_t* tail = result.data() + PrefillTailWord(c);
        uint32_t* prefix = tail + 346;
        uint64_t groups = 0, tiles = 0;
        for (uint32_t b = 0; b < c.batch; ++b) {
            const uint32_t q = c.tnd         ? allocation.sequences[b].queryLength
                               : a.usedQuery ? reinterpret_cast<const uint32_t*>(a.usedQuery)[b]
                                             : c.sq;
            const uint32_t k = c.tnd      ? allocation.sequences[b].kvLength
                               : a.usedKv ? reinterpret_cast<const uint32_t*>(a.usedKv)[b]
                                          : c.sk;
            if (q > c.sq || k > c.sk)
                return true;
            const uint32_t word = PrefillSequenceWord(c, b);
            if (c.tnd) {
                const auto& seq = allocation.sequences[b];
                result[word] = uint32_t(seq.queryStart);
                result[word + 2] = uint32_t(seq.kvStart);
                result[word + 3] = uint32_t(seq.kvStart >> 32);
                result[word + 4] = seq.queryCapacity;
                result[word + 5] = seq.kvCapacity;
            }
            result[PrefillLengthWord(c, b)] = q;
            result[PrefillLengthWord(c, b) + 1] = k;
            base.actualSeqS1Size.push_back(q);
            base.actualSeqS2Size.push_back(k);
            const uint64_t count = l.groups ? (uint64_t(q) * base.gSize + l.m - 1) / l.m : 0;
            for (uint32_t h = 0; h < c.kvHeads; ++h) {
                groups += count;
                prefix += 2;
                prefix[0] = uint32_t(groups);
                prefix[1] = uint32_t(groups >> 32);
            }
            tiles += count * c.kvHeads * ((uint64_t(k) + l.n - 1) / l.n);
            if (tiles > UINT32_MAX)
                return true;
        }
        detail::SplitParam param{};
        param.mBaseSize = l.m;
        param.s2BaseSize = l.n;
        param.gS1BaseSizeOfFd = 8;
        detail::SplitResult n{c.cores, 2};
        if (l.groups) {
            detail::SplitCore(c.cores, base, param, n);
        } else {
            n.usedCoreNum = 1;
            n.bN2End[0] = c.batch * c.kvHeads;
        }
        if (!n.usedCoreNum || n.usedCoreNum > c.cores || n.numOfFdHead > c.cores || n.maxS2SplitNum > c.cores)
            return true;
        tail[0] = c.window.kind;
        tail[1] = uint32_t(c.window.left);
        tail[2] = uint32_t(c.window.right);
        tail[3] = n.usedCoreNum;
        auto emit = [&](uint32_t offset, const auto& array) { std::copy(array.begin(), array.end(), tail + offset); };
        emit(4, n.bN2End);
        emit(30, n.gS1End);
        emit(56, n.s2End);
        tail[82] = n.numOfFdHead;
        tail[83] = n.maxS2SplitNum;
        tail[84] = n.usedVecNumOfFd;
        tail[85] = 8;
        emit(86, n.fdRes.bN2IdxOfFdHead);
        emit(112, n.fdRes.gS1IdxOfFdHead);
        emit(138, n.fdRes.s2SplitNumOfFdHead);
        emit(164, n.fdRes.s2SplitStartIdxOfCore);
        emit(190, n.fdRes.gS1SplitNumOfFdHead);
        emit(216, n.fdRes.gS1LastPartSizeOfFdHead);
        emit(242, n.fdRes.gS1IdxEndOfFdHead);
        emit(294, n.fdRes.gS1IdxEndOfFdHeadSplit);
        std::copy(result.begin(), result.end(), out);
        return true;
    } catch (const std::bad_alloc&) {
        return true;
    }
}
} // namespace flash_attn::mla

// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.

namespace flash_attn::mla {
bool BuildGqaMetadata(const MlaGqaMetadataArgs& a)
{
    auto c = a.context;
    MlaGqaLayout l{};
    if (!GqaLayout(c, l) || !a.output || a.output % 4 || a.words != l.words || a.words > (UINT64_MAX - a.output) / 4)
        return false;
    const uint64_t addresses[] = {a.cuQuery, a.cuKv, a.usedQuery, a.usedKv};
    for (uint32_t axis = 0; axis < 4; ++axis) {
        const uint64_t address = addresses[axis], count = uint64_t(c.batch) + (axis < 2);
        if (address && count && address < a.output + uint64_t(a.words) * 4 &&
            (address >= a.output || count > (a.output - address) / 4))
            return false;
    }
    auto* out = reinterpret_cast<uint32_t*>(a.output);
    std::fill(out, out + a.words, 0);
    for (uint32_t i = 0; i < 5; ++i)
        out[i] = GqaHeader(c, l, i);
    out[5] = 1;
    for (uint32_t axis = 0; axis < 4; ++axis) {
        const uint64_t count = uint64_t(c.batch) + (axis < 2), address = addresses[axis];
        const bool required = axis == 0 ? GqaTnd(c) : axis == 1 ? c.layout == 1 : c.usedFlags & (1U << (axis - 2));
        if ((required && count && !address) || (!required && address) || address % 4 ||
            (address && count > (UINT64_MAX - address) / 4))
            return true;
    }
    if ((c.dtype != uint32_t(MlaMetadataDtype::Fp16) && c.dtype != uint32_t(MlaMetadataDtype::Bf16)) ||
        a.inferTotalQ > 1 || (a.inferTotalQ && c.layout != 2 && c.layout != 4))
        return true;
    if (a.inferTotalQ) {
        c.totalQ = reinterpret_cast<const uint32_t*>(a.cuQuery)[c.batch];
        if (!GqaLayout(c, l))
            return true;
    }
    try {
        MlaPlatformInfo physical{};
        physical.aicNum = c.physicalCores;
        physical.aivNum = 2 * c.physicalCores;
        for (uint32_t i = 0; i < c.physicalCores; ++i)
            physical.blockDims[i] = i + 1;
        const MlaUsedShape shape{c.batch,
                                 c.heads,
                                 {GqaTnd(c) ? MlaSequenceLayout::Tnd : MlaSequenceLayout::Bsnd,
                                  GqaTnd(c) ? int64_t(c.totalQ) : int64_t(c.batch) * c.sq, c.sq},
                                 {c.layout == 1 ? MlaSequenceLayout::Tnd : MlaSequenceLayout::Bsnd,
                                  c.layout == 1 ? int64_t(c.totalK) : int64_t(c.batch) * c.sk, c.sk}};
        MlaUsedInputs inputs{};
        MlaLengthArray* arrays[] = {&inputs.cuQuery, &inputs.cuKv, &inputs.usedQuery, &inputs.usedKv};
        for (uint32_t i = 0; i < 4; ++i)
            *arrays[i] = {reinterpret_cast<const int32_t*>(addresses[i]),
                          addresses[i] ? uint64_t(c.batch) + (i < 2) : 0};
        MlaGqaSchedule schedule;
        // Positive split modes consume the same safe native tail, but use the
        // existing explicit fragment rule in the consumer, as prefill does.
        if (BuildGqaSchedule({shape, c.kvHeads, c.valueDim, l.m, l.n, 0, c.pageSize}, inputs,
                             {c.sq, c.sk, c.window.kind == 1 ? -1 : c.window.left,
                              c.window.kind == 1 ? -1 : c.window.right, c.window.kind == 1},
                             {c.packed, c.margin}, physical, schedule) != MlaScheduleStatus::Success)
            return true;
        std::vector<uint32_t> result(l.words, 0);
        for (uint32_t i = 0; i < 32; ++i)
            result[i] = GqaHeader(c, l, i);
        for (uint32_t b = 0; b < c.batch; ++b) {
            const auto& s = schedule.Lengths().sequences[b];
            const uint32_t values[] = {uint32_t(s.queryStart), uint32_t(s.queryStart >> 32),
                                       uint32_t(s.kvStart),    uint32_t(s.kvStart >> 32),
                                       s.queryCapacity,        s.kvCapacity,
                                       s.queryLength,          s.kvLength};
            std::copy(values, values + 8, result.begin() + 32 + 8 * b);
        }
        auto* tail = result.data() + GqaTail(c);
        const auto& n = schedule.Native();
        const auto& f = n.fd;
        tail[0] = c.window.kind;
        tail[1] = uint32_t(c.window.left);
        tail[2] = uint32_t(c.window.right);
        tail[3] = n.usedCoreNum;
        auto emit = [&](uint32_t offset, const auto& array) { std::copy(array.begin(), array.end(), tail + offset); };
        emit(4, n.bN2End);
        emit(30, n.gS1End);
        emit(56, n.s2End);
        tail[82] = f.numOfFdHead;
        tail[83] = f.maxS2SplitNum;
        tail[84] = f.usedVecNumOfFd;
        tail[85] = 8;
        emit(86, f.bN2IdxOfFdHead);
        emit(112, f.gS1IdxOfFdHead);
        emit(138, f.s2SplitNumOfFdHead);
        emit(164, f.s2SplitStartIdxOfCore);
        emit(190, f.gS1SplitNumOfFdHead);
        emit(216, f.gS1LastPartSizeOfFdHead);
        emit(242, f.gS1IdxEndOfFdHead);
        emit(294, f.gS1IdxEndOfFdHeadSplit);
        const auto& prefix = schedule.GroupPrefix();
        for (uint64_t i = 0; i < prefix.size(); ++i) {
            tail[346 + 2 * i] = uint32_t(prefix[i]);
            tail[347 + 2 * i] = uint32_t(prefix[i] >> 32);
        }
        std::copy(result.begin(), result.end(), out);
        return true;
    } catch (const std::bad_alloc&) {
        return true;
    }
}
} // namespace flash_attn::mla
