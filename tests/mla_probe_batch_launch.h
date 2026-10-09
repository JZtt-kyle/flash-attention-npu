// Shared standalone-probe BATCH_MODE launcher; not linked into production.
#pragma once
#if !defined(__NPU_ARCH__)
// Bisheng's standalone <<<>>> launcher defaults to NORMAL_MODE. Only this
// isolated probe DSO wraps the runtime call to preserve source BATCH_MODE.
// The counter below makes a compiler/runtime path that bypasses it fail.
static uint64_t batchLaunchCount = 0;
static aclError batchLaunchError = ACL_SUCCESS;
extern "C" aclError __real_aclrtLaunchKernelWithHostArgs(aclrtFuncHandle, uint32_t, aclrtStream, aclrtLaunchKernelCfg*,
                                                         void*, size_t, aclrtPlaceHolderInfo*, size_t);
extern "C" aclError __wrap_aclrtLaunchKernelWithHostArgs(aclrtFuncHandle function, uint32_t blocks, aclrtStream stream,
                                                         aclrtLaunchKernelCfg* config, void* args, size_t argsSize,
                                                         aclrtPlaceHolderInfo* placeholders, size_t placeholderCount)
{
    ++batchLaunchCount;
    aclrtLaunchKernelAttr attrs[9]{};
    size_t count = 0;
    if (config) {
        if (config->numAttrs > 8 || (config->numAttrs && !config->attrs)) {
            batchLaunchError = ACL_ERROR_INVALID_PARAM;
            return batchLaunchError;
        }
        for (size_t i = 0; i < config->numAttrs; ++i) {
            if (config->attrs[i].id != ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE) {
                attrs[count++] = config->attrs[i];
            }
        }
    }
    attrs[count].id = ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE;
    attrs[count++].value.schemMode = 1;
    aclrtLaunchKernelCfg batchConfig{attrs, count};
    batchLaunchError = __real_aclrtLaunchKernelWithHostArgs(function, blocks, stream, &batchConfig, args, argsSize,
                                                            placeholders, placeholderCount);
    return batchLaunchError;
}
#endif
