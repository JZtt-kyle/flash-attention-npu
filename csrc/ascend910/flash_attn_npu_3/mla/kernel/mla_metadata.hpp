// Device adapter for the private dense schedule protocol. Validate before the
// attention Kernel allocates events, initializes LSE or touches its workspace.
#pragma once
#include "kernel_operator.h"
#include "../mla_metadata_consumer.h"

namespace flash_attn::mla::Kernel {
struct MetadataReader {
    AscendC::GlobalTensor<uint32_t> data;
    __aicore__ inline uint32_t operator()(uint32_t word) const
    {
        return data.GetValue(word);
    }
};

template <typename InputT>
__aicore__ inline MlaMetadataReadStatus ReadDenseMetadata(GM_ADDR address, uint32_t words,
                                                          const MlaDenseMetadataContext& context,
                                                          MlaDenseKernelSchedule& output)
{
    static_assert(AscendC::IsSameType<InputT, half>::value || AscendC::IsSameType<InputT, bfloat16_t>::value);
    if (address == nullptr || reinterpret_cast<uint64_t>(address) % sizeof(uint32_t) != 0) {
        return MlaMetadataReadStatus::InvalidBuffer;
    }
    const uint32_t dtype =
        AscendC::IsSameType<InputT, half>::value ? uint32_t(MlaMetadataDtype::Fp16) : uint32_t(MlaMetadataDtype::Bf16);
    if (context.dtype != dtype) {
        return MlaMetadataReadStatus::DtypeMismatch;
    }
    if (context.blockDim != AscendC::GetBlockNum()) {
        return MlaMetadataReadStatus::PlatformMismatch;
    }
    MetadataReader read;
    read.data.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(address));
    return flash_attn::mla::ReadDenseMetadata(read, words, context, output);
}
} // namespace flash_attn::mla::Kernel
