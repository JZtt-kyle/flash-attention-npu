// Independent v6 device validation. Every Cube/Vector must pass before its
// caller constructs TPipe/events, writes outputs/workspace or synchronizes.
#pragma once
#include "mla_metadata.hpp"
#include "../mla_paged_metadata_consumer.h"

namespace flash_attn::mla::Kernel {
struct PagedMetadataReaders {
    MetadataReader data[3];
    __aicore__ inline uint32_t operator()(uint32_t axis, uint32_t index) const
    {
        return data[axis](index);
    }
};

struct PageTableReader {
    AscendC::GlobalTensor<uint32_t> data;
    __aicore__ inline uint32_t operator()(uint64_t index) const
    {
        return data.GetValue(index);
    }
};

template <typename InputT>
__aicore__ inline MlaMetadataReadStatus
ReadPagedMetadata(GM_ADDR address, uint32_t words, const MlaPagedMetadataContext& context,
                  const MlaPagedMetadataInputs& inputs, const MlaPagedCacheInputs& cache,
                  MlaPagedKernelSchedule& output)
{
    using Status = MlaMetadataReadStatus;
    static_assert(AscendC::IsSameType<InputT, half>::value || AscendC::IsSameType<InputT, bfloat16_t>::value);
    if (address == nullptr || !ValidPagedMetadataAddress(reinterpret_cast<uint64_t>(address), words)) {
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
    PagedMetadataReaders arrays;
    const uint64_t addresses[3] = {inputs.cuQuery.address, inputs.usedQuery.address, inputs.kvLengths.address};
    for (uint32_t i = 0; i < 3; ++i) {
        if (addresses[i] != 0) {
            arrays.data[i].data.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(addresses[i]));
        }
    }
    PageTableReader pages;
    if (cache.table.address != 0) {
        pages.data.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(cache.table.address));
    }
    return flash_attn::mla::ReadPagedMetadata(read, words, context, inputs, arrays, cache, pages, output);
}

template <typename InputT>
__aicore__ inline bool LoadPagedMetadata(GM_ADDR address, uint32_t words, const MlaPagedMetadataContext& context,
                                         const MlaPagedMetadataInputs& inputs, const MlaPagedCacheInputs& cache,
                                         MlaPagedKernelSchedule& output)
{
    if (ReadPagedMetadata<InputT>(address, words, context, inputs, cache, output) != MlaMetadataReadStatus::Success) {
        AscendC::Trap();
        return false;
    }
    return true;
}
} // namespace flash_attn::mla::Kernel
