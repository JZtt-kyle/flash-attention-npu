// Serialization only. Scheduling and its CANN source/license remain in
// mla_tnd_schedule.cpp, mla_schedule.cpp and detail/cann_split_core.cpp.
#include "mla_tnd_metadata.h"
#include "mla_tnd_schedule.h"
#include <cstring>
#include <limits>

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

} // namespace flash_attn::mla
