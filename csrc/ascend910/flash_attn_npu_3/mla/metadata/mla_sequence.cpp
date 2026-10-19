#include "mla_sequence.h"
#include "../tiling/mla_sequence.h"
#include <cstring>
#include <limits>

// Serialization only. Scheduling and its CANN source/license remain in
// tiling/mla_bsnd_schedule.cpp, tiling/mla_schedule.cpp and detail/cann_split_core.cpp.

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
// tiling/mla_sequence.cpp, tiling/mla_schedule.cpp and detail/cann_split_core.cpp.

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
