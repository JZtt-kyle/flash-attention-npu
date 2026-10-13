// Serialization only; reuse the CANN-derived allocation/used scheduler.
#include "mla_used_metadata.h"
#include <cstring>
#include <limits>

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
                                   MlaUsedScheduleBuilder builder, const void* context, bool queryTotalFromPrefix)
{
    auto shape = inputShape;
    auto platform = inputPlatform;
    uint32_t expectedWords;
    const uintptr_t outAddress = reinterpret_cast<uintptr_t>(output);
    constexpr auto maxAddress = std::numeric_limits<uintptr_t>::max();
    if (!UsedMetadataWords(shape.batch, expectedWords) || outputWords != expectedWords || output == nullptr ||
        outAddress % alignof(int32_t) != 0 || outputWords > (maxAddress - outAddress) / sizeof(int32_t)) {
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
