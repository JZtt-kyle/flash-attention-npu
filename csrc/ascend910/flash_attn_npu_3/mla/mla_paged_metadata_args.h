// Private host/AICPU launch ABI. Addresses are never serialized in v6 records.
#pragma once
#include "mla_paged_schedule.h"
#include <type_traits>

namespace flash_attn::mla {
struct MlaPagedMetadataArray {
    uint64_t address, count;
};
struct MlaPagedMetadataArgs {
    uint64_t outputAddr;
    uint32_t outputWords, dtype;
    MlaPagedShape shape;
    MlaPlatformInfo platform;
    MlaPagedMetadataArray cuQuery, usedQuery, kvLengths;
};
static_assert(std::is_standard_layout_v<MlaPagedMetadataArgs> && std::is_trivially_copyable_v<MlaPagedMetadataArgs>);

inline MlaPagedInputs PagedMetadataInputs(const MlaPagedMetadataArgs& args)
{
    const auto array = [](const MlaPagedMetadataArray& input) {
        return MlaLengthArray{reinterpret_cast<const int32_t*>(input.address), input.count};
    };
    return {array(args.cuQuery), array(args.usedQuery), array(args.kvLengths)};
}
} // namespace flash_attn::mla
