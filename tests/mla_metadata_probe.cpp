// Test-only host bridge to the production AICPU entry. Never linked into FA3.
#include "mla/mla_metadata_args.h"
#include "acl/acl.h"
#include "tiling/platform/platform_ascendc.h"
#include <cstring>

extern __global__ __aicpu__ uint32_t ComputeMlaDenseMetadata(void* args);
using namespace flash_attn::mla;

extern "C" uint32_t metadata_args_size()
{
    return sizeof(MlaMetadataArgs);
}
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

extern "C" int launch_metadata(void* current, void* worker, void* ready, void* done, const MlaMetadataArgs* args)
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
    ComputeMlaDenseMetadata<<<1, nullptr, worker>>>(&launchArgs, sizeof(launchArgs));
    status = aclrtRecordEvent(done, worker);
    if (status != ACL_SUCCESS)
        return status;
    return aclrtStreamWaitEvent(current, done);
}
