// Private host/AICPU launch ABI. Addresses are never serialized in v5 records.
#pragma once
#include "mla_used_schedule.h"
#include <type_traits>

namespace flash_attn::mla {
struct MlaUsedMetadataArray {
    uint64_t address, count;
};
struct MlaUsedMetadataArgs {
    uint64_t outputAddr;
    uint32_t outputWords, dtype;
    MlaUsedShape shape;
    MlaPlatformInfo platform;
    MlaUsedMetadataArray cuQuery, cuKv, usedQuery, usedKv;
};
static_assert(std::is_standard_layout_v<MlaUsedMetadataArgs> && std::is_trivially_copyable_v<MlaUsedMetadataArgs>);

inline MlaUsedInputs UsedMetadataInputs(const MlaUsedMetadataArgs& args)
{
    const auto array = [](const MlaUsedMetadataArray& input) {
        return MlaLengthArray{reinterpret_cast<const int32_t*>(input.address), input.count};
    };
    return {array(args.cuQuery), array(args.cuKv), array(args.usedQuery), array(args.usedKv)};
}
} // namespace flash_attn::mla
