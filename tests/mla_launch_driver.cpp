// Host-only test: substitute the ACL runtime, never touch an NPU or FA kernel.
#include "mla/mla_launch.h"
#include <cassert>
#include <thread>
#include <vector>

using namespace flash_attn::mla;
static thread_local std::vector<aclrtLaunchKernelAttr> observed;
static thread_local aclrtLaunchKernelCfg* observedConfig;
static thread_local aclError runtimeError = ACL_SUCCESS;
static thread_local uint32_t runtimeCalls = 0;
extern "C" aclError __wrap_aclrtLaunchKernelWithHostArgs(aclrtFuncHandle, uint32_t, aclrtStream, aclrtLaunchKernelCfg*,
                                                         void*, size_t, aclrtPlaceHolderInfo*, size_t);
extern "C" aclError __real_aclrtLaunchKernelWithHostArgs(aclrtFuncHandle function, uint32_t blocks, aclrtStream stream,
                                                         aclrtLaunchKernelCfg* config, void* args, size_t size,
                                                         aclrtPlaceHolderInfo* placeholders, size_t count)
{
    assert(function == reinterpret_cast<void*>(1) && blocks == 7 && stream == reinterpret_cast<void*>(2));
    assert(args == reinterpret_cast<void*>(3) && size == 99);
    assert(placeholders == reinterpret_cast<aclrtPlaceHolderInfo*>(4) && count == 2);
    observedConfig = config;
    observed.clear();
    if (config) {
        observed.assign(config->attrs, config->attrs + config->numAttrs);
    }
    ++runtimeCalls;
    return runtimeError;
}

aclError Launch(aclrtLaunchKernelCfg* config)
{
    return __wrap_aclrtLaunchKernelWithHostArgs(reinterpret_cast<void*>(1), 7, reinterpret_cast<void*>(2), config,
                                                reinterpret_cast<void*>(3), 99,
                                                reinterpret_cast<aclrtPlaceHolderInfo*>(4), 2);
}

int main()
{
    aclrtLaunchKernelAttr attrs[2]{};
    attrs[0].id = ACL_RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE;
    attrs[0].value.dynUBufSize = 8192;
    attrs[1].id = ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE;
    attrs[1].value.schemMode = 2;
    aclrtLaunchKernelCfg config{attrs, 2};
    assert(Launch(&config) == ACL_SUCCESS && observedConfig == &config);
    assert(observed[1].value.schemMode == 2);
    assert(Launch(nullptr) == ACL_SUCCESS && observedConfig == nullptr);
    {
        MlaBatchLaunchScope scope;
        assert(scope.Result() != ACL_SUCCESS); // a compiler path that bypasses ACL must fail
        assert(Launch(&config) == ACL_SUCCESS && scope.Result() == ACL_SUCCESS);
        assert(observed.size() == 2 && observed[0].value.dynUBufSize == 8192);
        assert(observed[1].id == ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE && observed[1].value.schemMode == 1);
        assert(attrs[1].value.schemMode == 2); // caller config was not mutated
        std::thread other([&] {
            assert(activeMlaLaunch == nullptr);
            assert(Launch(&config) == ACL_SUCCESS && observed[1].value.schemMode == 2);
        });
        other.join();
        {
            MlaBatchLaunchScope nested;
            runtimeError = ACL_ERROR_INVALID_PARAM;
            assert(Launch(nullptr) == runtimeError && nested.Result() == runtimeError);
            runtimeError = ACL_SUCCESS;
        }
        assert(scope.Result() == ACL_SUCCESS);
        assert(Launch(nullptr) == ACL_SUCCESS);
        assert(scope.Result() != ACL_SUCCESS); // exactly one launch per scope
    }
    assert(activeMlaLaunch == nullptr);
    assert(Launch(&config) == ACL_SUCCESS && observedConfig == &config);
    for (auto bad : {aclrtLaunchKernelCfg{nullptr, 1}, aclrtLaunchKernelCfg{attrs, 9}}) {
        MlaBatchLaunchScope scope;
        auto before = runtimeCalls;
        assert(Launch(&bad) == ACL_ERROR_INVALID_PARAM && scope.Result() == ACL_ERROR_INVALID_PARAM);
        assert(runtimeCalls == before);
    }
    return 0;
}
