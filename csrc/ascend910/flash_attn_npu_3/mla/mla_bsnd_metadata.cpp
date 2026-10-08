// Serialization only. Scheduling and its CANN source/license remain in
// mla_bsnd_schedule.cpp, mla_schedule.cpp and detail/cann_split_core.cpp.
#include "mla_bsnd_metadata.h"
#include "mla_bsnd_schedule.h"
#include <cstring>
#include <limits>

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
