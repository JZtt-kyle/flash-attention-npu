// Serialization only. Scheduling and its CANN source/license remain in
// mla_tnd_schedule.cpp, mla_schedule.cpp and detail/cann_split_core.cpp.
#include "mla_tnd_bsnd_metadata.h"
#include "mla_tnd_bsnd_schedule.h"
#include <cstring>
#include <limits>

namespace flash_attn::mla {

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
