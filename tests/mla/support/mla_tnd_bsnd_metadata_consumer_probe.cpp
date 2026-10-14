// Validation-only mixed-core probe, never linked into the public extension.
// Successful validation exports the decoded schedule and touches caller-owned
// sentinel buffers; rejection must leave all three buffers unchanged.
#include "kernel_operator.h"
#include "acl/acl.h"
#include "mla/mla_metadata.h"
#include "mla/mla_tnd_bsnd_metadata_consumer.h"
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
#include "mla/kernel/mla_tnd_bsnd_metadata.hpp"
#endif

using namespace AscendC;
using namespace flash_attn::mla;
#ifdef MLA_TND_BSND_BF16
using InputT = bfloat16_t;
#else
using InputT = half;
#endif

__global__ __aicore__ __mix__(1, 2) void mixed_metadata_consumer_probe(
    GM_ADDR metadata, uint32_t words, MlaTndBsndMetadataContext context, GM_ADDR cuQuery, uint64_t queryCount,
    GM_ADDR kvLengths, uint64_t kvCount, GM_ADDR status, GM_ADDR output, GM_ADDR lse, GM_ADDR workspace, bool checked)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
    MlaTndBsndKernelSchedule schedule{};
    if (checked) {
        if (!Kernel::LoadTndBsndMetadata<InputT>(metadata, words, context, cuQuery, queryCount, kvLengths, kvCount,
                                                 schedule))
            return;
    }
    const auto result = checked ? MlaMetadataReadStatus::Success
                                : Kernel::ReadTndBsndMetadata<InputT>(metadata, words, context, cuQuery, queryCount,
                                                                      kvLengths, kvCount, schedule);
    uint32_t index = GetBlockIdx();
    if ASCEND_IS_AIV {
        index += GetBlockNum();
    }
    GlobalTensor<uint32_t> state;
    state.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(status) + index * 16);
    state.SetValue(0, uint32_t(result));
    DataCacheCleanAndInvalid<uint32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(state);
    if (result != MlaMetadataReadStatus::Success)
        return;
    GlobalTensor<uint32_t> out;
    out.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(output) + index * 96);
    static_assert(sizeof(schedule) == 92 * sizeof(uint32_t));
    const auto* decoded = reinterpret_cast<const uint32_t*>(&schedule);
    for (uint32_t i = 0; i < 92; ++i)
        out.SetValue(i, decoded[i]);
    for (uint32_t i = 0; i < 96; i += 16)
        DataCacheCleanAndInvalid<uint32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(out[i]);
    for (uint32_t i = 0; i < 2; ++i) {
        GlobalTensor<uint32_t> marker;
        marker.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(i == 0 ? lse : workspace) + index * 16);
        marker.SetValue(0, 1);
        DataCacheCleanAndInvalid<uint32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(marker);
    }
#endif
}

extern "C" int launch_mixed_consumer(void* stream, void* metadata, uint32_t words,
                                     const MlaTndBsndMetadataContext* context, void* cuQuery, uint64_t queryCount,
                                     void* kvLengths, uint64_t kvCount, void* status, void* output, void* lse,
                                     void* workspace, uint32_t blocks, bool checked)
{
    if (!context || !status || !output || !lse || !workspace || blocks == 0 || blocks > 26)
        return -1;
    auto error = aclrtSetOpExecuteTimeOut(15);
    if (error != ACL_SUCCESS)
        return error;
    mixed_metadata_consumer_probe<<<blocks, nullptr, stream>>>(
        (uint8_t*)metadata, words, *context, (uint8_t*)cuQuery, queryCount, (uint8_t*)kvLengths, kvCount,
        (uint8_t*)status, (uint8_t*)output, (uint8_t*)lse, (uint8_t*)workspace, checked);
    return 0;
}
