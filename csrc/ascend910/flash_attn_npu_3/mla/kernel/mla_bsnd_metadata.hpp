// Device-only adapter for v2 BSND records. Every Cube/Vector validates before
// attention event allocation, barriers or output/workspace writes .
#pragma once
#include "mla_metadata.hpp"
#include "../metadata/mla_sequence_consumer.h"
namespace flash_attn::mla::Kernel {
template <typename InputT>
__aicore__ inline MlaMetadataReadStatus
ReadBsndKvLengthMetadata(GM_ADDR address, uint32_t words, const MlaDenseMetadataContext& context, GM_ADDR lengths,
                         uint64_t lengthCount, MlaDenseKernelSchedule& output)
{
    using Status = MlaMetadataReadStatus;
    static_assert(AscendC::IsSameType<InputT, half>::value || AscendC::IsSameType<InputT, bfloat16_t>::value);
    const uint64_t addr = reinterpret_cast<uint64_t>(address), lenAddr = reinterpret_cast<uint64_t>(lengths);
    if (address == nullptr || addr % sizeof(uint32_t) != 0 || words > (UINT64_MAX - addr) / sizeof(uint32_t)) {
        return Status::InvalidBuffer;
    }
    if (lenAddr % sizeof(int32_t) != 0 || lengthCount > (UINT64_MAX - lenAddr) / sizeof(int32_t)) {
        return Status::InvalidCacheLengths;
    }
    const uint32_t dtype =
        AscendC::IsSameType<InputT, half>::value ? uint32_t(MlaMetadataDtype::Fp16) : uint32_t(MlaMetadataDtype::Bf16);
    if (context.dtype != dtype) {
        return Status::DtypeMismatch;
    }
    if (context.blockDim != AscendC::GetBlockNum()) {
        return Status::PlatformMismatch;
    }
    MetadataReader read, current;
    read.data.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(address));
    if (lengths != nullptr) {
        current.data.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(lengths));
    }
    return flash_attn::mla::ReadBsndKvLengthMetadata(read, words, context, current, lengths != nullptr, lengthCount,
                                                     output);
}

template <typename InputT>
__aicore__ inline bool LoadBsndKvLengthMetadata(GM_ADDR address, uint32_t words, const MlaDenseMetadataContext& context,
                                                GM_ADDR lengths, uint64_t lengthCount, MlaDenseKernelSchedule& output)
{
    if (ReadBsndKvLengthMetadata<InputT>(address, words, context, lengths, lengthCount, output) !=
        MlaMetadataReadStatus::Success) {
        AscendC::Trap();
        return false;
    }
    return true;
}
} // namespace flash_attn::mla::Kernel
