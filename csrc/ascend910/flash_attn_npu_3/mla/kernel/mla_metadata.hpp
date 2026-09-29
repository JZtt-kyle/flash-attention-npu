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

// Reused metadata is a snapshot, not proof that the current lengths still
// match. Every Cube/Vector checks all B lengths before any attention barrier
// or write. The host has validated the optional tensor's size/device/stride.
__aicore__ inline bool CheckDenseCacheLengths(GM_ADDR address, uint32_t batch, uint32_t capacity)
{
    if (address == nullptr) {
        return true;
    }
    AscendC::GlobalTensor<int32_t> lengths;
    lengths.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(address));
    for (uint32_t b = 0; b < batch; ++b) {
        if (lengths.GetValue(b) != static_cast<int32_t>(capacity)) {
            AscendC::Trap();
            return false;
        }
    }
    return true;
}

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

// Runtime-facing adapter: an invalid record must fail the stream, not leave
// apparently successful, uninitialized O/LSE. Trap is unconditional on build
// flags (unlike debug assertions) and needs no status buffer or host readback.
// All Cube/Vector cores validate the same immutable record before the caller
// constructs TPipe, allocates events or enters any attention barrier.
template <typename InputT>
__aicore__ inline bool LoadDenseMetadata(GM_ADDR address, uint32_t words, const MlaDenseMetadataContext& context,
                                         MlaDenseKernelSchedule& output)
{
    if (ReadDenseMetadata<InputT>(address, words, context, output) != MlaMetadataReadStatus::Success) {
        AscendC::Trap();
        return false;
    }
    return true;
}
} // namespace flash_attn::mla::Kernel
