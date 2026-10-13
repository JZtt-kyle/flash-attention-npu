// Test-only host bridge to the production AICPU entry. Never linked into FA3.
#include "mla/mla_metadata_args.h"
#include "acl/acl.h"
#include "tiling/platform/platform_ascendc.h"
#include <cstring>

extern __global__ __aicpu__ uint32_t ComputeMlaDenseMetadata(void* args);
#ifdef MLA_BSND_METADATA
#include "mla/mla_bsnd_metadata.h"
extern __global__ __aicpu__ uint32_t ComputeMlaBsndKvLengthMetadata(void* args);
#endif
#ifdef MLA_TND_METADATA
#include "mla/mla_tnd_metadata.h"
#include "mla/mla_tnd_metadata_args.h"
extern __global__ __aicpu__ uint32_t ComputeMlaTndMetadata(void* args);
#endif
#ifdef MLA_TND_BSND_METADATA
#include "mla/mla_tnd_bsnd_metadata.h"
#include "mla/mla_tnd_bsnd_metadata_args.h"
extern __global__ __aicpu__ uint32_t ComputeMlaTndBsndMetadata(void* args);
#endif
#ifdef MLA_USED_METADATA
#include "mla/mla_used_metadata.h"
#include "mla/mla_used_metadata_args.h"
extern __global__ __aicpu__ uint32_t ComputeMlaUsedMetadata(void* args);
#endif
#ifdef MLA_PAGED_METADATA
#include "mla/mla_paged_metadata.h"
#include "mla_paged_metadata_probe.h"
extern __global__ __aicpu__ uint32_t ComputeMlaPagedMetadata(void* args);
extern __global__ __aicpu__ uint32_t ProbeMlaPagedMetadata(void* args);
#endif
using namespace flash_attn::mla;

extern "C" uint32_t metadata_args_size()
{
    return sizeof(MlaMetadataArgs);
}

#ifdef MLA_PAGED_METADATA
extern "C" uint32_t paged_metadata_args_size()
{
    return sizeof(MlaPagedMetadataArgs);
}

extern "C" uint32_t paged_metadata_probe_args_size()
{
    return sizeof(MlaPagedMetadataProbeArgs);
}

extern "C" uint32_t paged_metadata_host(const MlaPagedMetadataArgs* args, uint32_t* output)
{
    // Explicit CPU total, independently compared with AICPU's prefix-derived total.
    return BuildPagedMetadata(args->shape, args->platform, args->dtype, output, args->outputWords,
                              PagedMetadataInputs(*args))
               ? 0
               : 1;
}

extern "C" uint32_t causal_paged_metadata_host(const MlaPagedMetadataArgs* args, uint32_t* output)
{
    const bool ok = args->shape.query.layout == MlaSequenceLayout::Tnd
                        ? BuildCausalPagedTndMetadata(args->shape, args->platform, args->dtype, output,
                                                      args->outputWords, PagedMetadataInputs(*args))
                        : BuildCausalPagedBsndMetadata(args->shape, args->platform, args->dtype, output,
                                                       args->outputWords, PagedMetadataInputs(*args));
    return ok ? 0 : 1;
}

extern "C" int launch_paged_metadata(void* current, void* worker, void* ready, void* done,
                                     const MlaPagedMetadataArgs* args, uint64_t diagnosticStatus, bool nullArgs)
{
    auto status = aclrtRecordEvent(ready, current);
    if (status != ACL_SUCCESS)
        return status;
    status = aclrtStreamWaitEvent(worker, ready);
    if (status != ACL_SUCCESS)
        return status;
    if (diagnosticStatus != 0) {
        MlaPagedMetadataProbeArgs launchArgs{*args, diagnosticStatus, uint32_t(nullArgs)};
        ProbeMlaPagedMetadata<<<1, nullptr, worker>>>(&launchArgs, sizeof(launchArgs));
    } else {
        MlaPagedMetadataArgs launchArgs = *args;
        ComputeMlaPagedMetadata<<<1, nullptr, worker>>>(&launchArgs, sizeof(launchArgs));
    }
    status = aclrtRecordEvent(done, worker);
    if (status != ACL_SUCCESS)
        return status;
    return aclrtStreamWaitEvent(current, done);
}
#endif

#ifdef MLA_USED_METADATA
extern "C" uint32_t used_metadata_args_size()
{
    return sizeof(MlaUsedMetadataArgs);
}

extern "C" uint32_t used_metadata_host(const MlaUsedMetadataArgs* args, uint32_t* output)
{
    return BuildUsedMetadata(args->shape, args->platform, args->dtype, output, args->outputWords,
                             UsedMetadataInputs(*args))
               ? 0
               : 1;
}

extern "C" int launch_used_metadata(void* current, void* worker, void* ready, void* done,
                                    const MlaUsedMetadataArgs* args)
{
    auto status = aclrtRecordEvent(ready, current);
    if (status != ACL_SUCCESS)
        return status;
    status = aclrtStreamWaitEvent(worker, ready);
    if (status != ACL_SUCCESS)
        return status;
    MlaUsedMetadataArgs launchArgs = *args;
    ComputeMlaUsedMetadata<<<1, nullptr, worker>>>(&launchArgs, sizeof(launchArgs));
    status = aclrtRecordEvent(done, worker);
    if (status != ACL_SUCCESS)
        return status;
    return aclrtStreamWaitEvent(current, done);
}
#endif
extern "C" uint32_t metadata_words()
{
    return kMlaDenseMetadataWords;
}

extern "C" int metadata_platform(MlaPlatformInfo* p)
{
    auto* platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (platform == nullptr)
        return -1;
    *p = {};
    p->aicNum = platform->GetCoreNumAic();
    p->aivNum = platform->GetCoreNumAiv();
    if (p->aicNum == 0 || p->aicNum > kMlaMaxAic || p->aivNum != 2 * p->aicNum)
        return -2;
    using Mem = platform_ascendc::CoreMemType;
    platform->GetCoreMemSize(Mem::L1, p->localMemory.l1Bytes);
    platform->GetCoreMemSize(Mem::UB, p->localMemory.ubBytes);
    platform->GetCoreMemSize(Mem::L0_A, p->localMemory.l0aBytes);
    platform->GetCoreMemSize(Mem::L0_B, p->localMemory.l0bBytes);
    platform->GetCoreMemSize(Mem::L0_C, p->localMemory.l0cBytes);
    p->libApiWorkspaceBytes = platform->GetLibApiWorkSpaceSize();
    for (uint32_t i = 1; i <= p->aicNum; ++i) {
        p->blockDims[i - 1] = platform->CalcTschBlockDim(i * 2, i, i * 2);
    }
    return 0;
}

extern "C" void metadata_host(const MlaMetadataArgs* args, void* output)
{
    MlaDenseMetadata metadata;
    BuildDenseMetadata(args->shape, args->platform, args->dtype, metadata);
    std::memcpy(output, &metadata, sizeof(metadata));
}

template <bool Bsnd>
int LaunchMetadata(void* current, void* worker, void* ready, void* done, const MlaMetadataArgs* args)
{
    // Mirror the existing FA3 metadata stream handshake. Events are owned by
    // the test fixture; it synchronizes before releasing them or any storage.
    auto status = aclrtRecordEvent(ready, current);
    if (status != ACL_SUCCESS)
        return status;
    status = aclrtStreamWaitEvent(worker, ready);
    if (status != ACL_SUCCESS)
        return status;
    MlaMetadataArgs launchArgs = *args;
#ifdef MLA_BSND_METADATA
    if constexpr (Bsnd) {
        ComputeMlaBsndKvLengthMetadata<<<1, nullptr, worker>>>(&launchArgs, sizeof(launchArgs));
    } else
#endif

    {
        ComputeMlaDenseMetadata<<<1, nullptr, worker>>>(&launchArgs, sizeof(launchArgs));
    }
    status = aclrtRecordEvent(done, worker);
    if (status != ACL_SUCCESS)
        return status;
    return aclrtStreamWaitEvent(current, done);
}

extern "C" int launch_metadata(void* current, void* worker, void* ready, void* done, const MlaMetadataArgs* args)
{
    return LaunchMetadata<false>(current, worker, ready, done, args);
}

#ifdef MLA_BSND_METADATA
extern "C" int launch_bsnd_metadata(void* current, void* worker, void* ready, void* done, const MlaMetadataArgs* args)
{
    return LaunchMetadata<true>(current, worker, ready, done, args);
}

extern "C" uint32_t bsnd_metadata_host(const MlaMetadataArgs* args, uint32_t* output)
{
    return BuildBsndKvLengthMetadata(args->shape, args->platform, args->dtype, output, args->outputWords,
                                     reinterpret_cast<const int32_t*>(args->cacheSeqlensAddr), args->cacheSeqlensCount)
               ? 0
               : 1;
}
#endif

#ifdef MLA_TND_METADATA
extern "C" uint32_t tnd_metadata_args_size()
{
    return sizeof(MlaTndMetadataArgs);
}

extern "C" uint32_t tnd_metadata_host(const MlaTndMetadataArgs* args, uint32_t* output)
{
    return BuildTndMetadata(args->shape, args->platform, args->dtype, output, args->outputWords,
                            reinterpret_cast<const int32_t*>(args->cuQueryAddr), args->queryCount,
                            reinterpret_cast<const int32_t*>(args->cuKvAddr), args->kvCount)
               ? 0
               : 1;
}

extern "C" int launch_tnd_metadata(void* current, void* worker, void* ready, void* done, const MlaTndMetadataArgs* args)
{
    auto status = aclrtRecordEvent(ready, current);
    if (status != ACL_SUCCESS)
        return status;
    status = aclrtStreamWaitEvent(worker, ready);
    if (status != ACL_SUCCESS)
        return status;
    MlaTndMetadataArgs launchArgs = *args;
    ComputeMlaTndMetadata<<<1, nullptr, worker>>>(&launchArgs, sizeof(launchArgs));
    status = aclrtRecordEvent(done, worker);
    if (status != ACL_SUCCESS)
        return status;
    return aclrtStreamWaitEvent(current, done);
}
#endif

#ifdef MLA_TND_BSND_METADATA
extern "C" uint32_t mixed_metadata_args_size()
{
    return sizeof(MlaTndBsndMetadataArgs);
}

extern "C" uint32_t mixed_metadata_host(const MlaTndBsndMetadataArgs* args, uint32_t* output)
{
    return BuildTndBsndMetadata(args->shape, args->platform, args->dtype, output, args->outputWords,
                                reinterpret_cast<const int32_t*>(args->cuQueryAddr), args->queryCount,
                                reinterpret_cast<const int32_t*>(args->kvLengthsAddr), args->kvCount)
               ? 0
               : 1;
}

extern "C" int launch_mixed_metadata(void* current, void* worker, void* ready, void* done,
                                     const MlaTndBsndMetadataArgs* args)
{
    auto status = aclrtRecordEvent(ready, current);
    if (status != ACL_SUCCESS)
        return status;
    status = aclrtStreamWaitEvent(worker, ready);
    if (status != ACL_SUCCESS)
        return status;
    MlaTndBsndMetadataArgs launchArgs = *args;
    ComputeMlaTndBsndMetadata<<<1, nullptr, worker>>>(&launchArgs, sizeof(launchArgs));
    status = aclrtRecordEvent(done, worker);
    if (status != ACL_SUCCESS)
        return status;
    return aclrtStreamWaitEvent(current, done);
}
#endif
