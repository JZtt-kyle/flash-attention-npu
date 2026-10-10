// Independent v4 mixed-layout device consumer. All Cube/Vector cores validate before the
// caller allocates TPipe/events, writes outputs/workspace or synchronizes.
#pragma once
#include "mla_metadata.hpp"
#include "../mla_tnd_bsnd_metadata_consumer.h"

namespace flash_attn::mla::Kernel {
template <typename InputT>
__aicore__ inline MlaMetadataReadStatus
ReadTndBsndMetadata(GM_ADDR address, uint32_t words, const MlaTndBsndMetadataContext& context, GM_ADDR cuQuery,
                    uint64_t queryCount, GM_ADDR kvLengths, uint64_t kvCount, MlaTndBsndKernelSchedule& output)
{
    using Status = MlaMetadataReadStatus;
    static_assert(AscendC::IsSameType<InputT, half>::value || AscendC::IsSameType<InputT, bfloat16_t>::value);
    const uint64_t addr = reinterpret_cast<uint64_t>(address);
    if (address == nullptr || addr % 4 != 0 || words > (UINT64_MAX - addr) / 4) {
        return Status::InvalidBuffer;
    }
    const uint64_t qAddr = reinterpret_cast<uint64_t>(cuQuery), kAddr = reinterpret_cast<uint64_t>(kvLengths);
    if (cuQuery == nullptr || qAddr % 4 != 0 || queryCount > (UINT64_MAX - qAddr) / 4) {
        return Status::InvalidPrefixes;
    }
    if ((kvLengths == nullptr && (context.batch != 0 || kvCount != 0)) || kAddr % 4 != 0 ||
        kvCount > (UINT64_MAX - kAddr) / 4) {
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
    MetadataReader read, query, kv;
    read.data.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(address));
    query.data.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(cuQuery));
    if (kvLengths != nullptr) {
        kv.data.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(kvLengths));
    }
    return flash_attn::mla::ReadTndBsndMetadata(read, words, context, query, queryCount, kv, kvCount, output);
}

template <typename InputT>
__aicore__ inline bool LoadTndBsndMetadata(GM_ADDR address, uint32_t words, const MlaTndBsndMetadataContext& context,
                                           GM_ADDR cuQuery, uint64_t queryCount, GM_ADDR kvLengths, uint64_t kvCount,
                                           MlaTndBsndKernelSchedule& output)
{
    if (ReadTndBsndMetadata<InputT>(address, words, context, cuQuery, queryCount, kvLengths, kvCount, output) !=
        MlaMetadataReadStatus::Success) {
        AscendC::Trap();
        return false;
    }
    return true;
}
} // namespace flash_attn::mla::Kernel
