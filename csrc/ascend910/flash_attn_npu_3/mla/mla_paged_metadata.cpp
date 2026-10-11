// Serialization only; reuse the CANN-derived PA scheduler and common encoding.
#include "mla_paged_metadata.h"
#include <cstring>
#include <limits>

namespace flash_attn::mla {

bool PagedMetadataWords(int64_t batch, uint32_t& words)
{
    if (batch < 0 || uint64_t(batch) > (UINT32_MAX - kMlaPagedMetadataPrefixWords) / kMlaPagedMetadataSequenceWords) {
        return false;
    }
    words = kMlaPagedMetadataPrefixWords + uint32_t(batch) * kMlaPagedMetadataSequenceWords;
    return true;
}

bool BuildPagedMetadata(const MlaPagedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype, uint32_t* output,
                        uint32_t outputWords, const MlaPagedInputs& inputs)
{
    uint32_t expectedWords;
    const uintptr_t outAddress = reinterpret_cast<uintptr_t>(output);
    constexpr auto maxAddress = std::numeric_limits<uintptr_t>::max();
    if (!PagedMetadataWords(shape.batch, expectedWords) || outputWords != expectedWords || output == nullptr ||
        outAddress % alignof(int32_t) != 0 || outputWords > (maxAddress - outAddress) / sizeof(int32_t)) {
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
        const auto status = BuildPagedSchedule(shape, platform, inputs, result);
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

} // namespace flash_attn::mla
