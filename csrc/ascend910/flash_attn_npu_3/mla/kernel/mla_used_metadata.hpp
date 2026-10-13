// Independent v5 device consumer. Every Cube/Vector validates before its caller
// constructs TPipe/events, writes outputs/workspace or synchronizes.
#pragma once
#include "mla_metadata.hpp"
#include "../mla_used_metadata_consumer.h"

namespace flash_attn::mla::Kernel {
struct UsedMetadataReaders {
    MetadataReader data[4];
    __aicore__ inline uint32_t operator()(uint32_t axis, uint32_t index) const
    {
        return data[axis](index);
    }
};

template <typename InputT, bool IsCausal = false>
__aicore__ inline MlaMetadataReadStatus
ReadUsedMetadata(GM_ADDR address, uint32_t words, const MlaUsedMetadataContext& context,
                 const MlaUsedMetadataInputs& inputs, MlaUsedKernelSchedule& output)
{
    using Status = MlaMetadataReadStatus;
    static_assert(AscendC::IsSameType<InputT, half>::value || AscendC::IsSameType<InputT, bfloat16_t>::value);
    if (address == nullptr || !ValidUsedMetadataAddress(reinterpret_cast<uint64_t>(address), words)) {
        return Status::InvalidBuffer;
    }
    const uint32_t dtype =
        AscendC::IsSameType<InputT, half>::value ? uint32_t(MlaMetadataDtype::Fp16) : uint32_t(MlaMetadataDtype::Bf16);
    if (context.dtype != dtype) {
        return Status::DtypeMismatch;
    }
    if (context.blockDim != AscendC::GetBlockNum()) {
        return Status::PlatformMismatch;
    }
    MetadataReader read;
    read.data.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(address));
    UsedMetadataReaders arrays;
    const uint64_t addresses[4] = {inputs.cuQuery.address, inputs.cuKv.address, inputs.usedQuery.address,
                                   inputs.usedKv.address};
    for (uint32_t i = 0; i < 4; ++i) {
        if (addresses[i] != 0) {
            arrays.data[i].data.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(addresses[i]));
        }
    }
    return flash_attn::mla::ReadUsedMetadata<IsCausal>(read, words, context, inputs, arrays, output);
}

template <typename InputT, bool IsCausal = false>
__aicore__ inline bool LoadUsedMetadata(GM_ADDR address, uint32_t words, const MlaUsedMetadataContext& context,
                                        const MlaUsedMetadataInputs& inputs, MlaUsedKernelSchedule& output)
{
    if (ReadUsedMetadata<InputT, IsCausal>(address, words, context, inputs, output) != MlaMetadataReadStatus::Success) {
        AscendC::Trap();
        return false;
    }
    return true;
}
} // namespace flash_attn::mla::Kernel
