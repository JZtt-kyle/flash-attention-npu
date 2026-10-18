// Exercise the exact production host query using CANN's installed B/C configs.
#include "mla_schedule.h"
#include "tiling/platform/platform_ascendc.h"
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace flash_attn::mla;
static const char* runtimeSoc;
const char* aclrtGetSocName()
{
    return runtimeSoc;
}
#define TORCH_CHECK(condition, message)                                                                                \
    do {                                                                                                               \
        if (!(condition))                                                                                              \
            throw std::runtime_error(message);                                                                         \
    } while (false)
#include "platform_query.h"

int main(int argc, char** argv)
{
    if (argc != 5)
        return 2;
    runtimeSoc = std::strcmp(argv[1], "null") == 0 ? nullptr : argv[1];
    try {
        TORCH_CHECK(platform_ascendc::PlatformAscendCManager::GetInstance(argv[2]), "config missing");
        auto p = DensePlatform(std::stoi(argv[3]) != 0);
        const std::string mutation(argv[4]);
        if (mutation == "zero-cores")
            p.aicNum = 0;
        if (mutation == "too-many-cores")
            p.aicNum = kMlaMaxAic + 1;
        if (mutation == "invalid-vectors")
            p.aivNum = 3 * p.aicNum;
        if (mutation == "l1")
            p.localMemory.l1Bytes = 0;
        if (mutation == "ub")
            p.localMemory.ubBytes = 0;
        if (mutation == "l0a")
            p.localMemory.l0aBytes = 0;
        if (mutation == "l0b")
            p.localMemory.l0bBytes = 0;
        if (mutation == "l0c")
            p.localMemory.l0cBytes = 0;
        if (mutation == "block-dim")
            p.blockDims.fill(0);
        MlaSchedule schedule{};
        const auto status = BuildDenseSchedule({1, 17, 8, 65}, p, schedule);
        TORCH_CHECK(status == MlaScheduleStatus::Success, "schedule rejected resources");
        std::cout << "{\"aic\":" << p.aicNum << ",\"aiv\":" << p.aivNum << ",\"memory\":[" << p.localMemory.l1Bytes
                  << ',' << p.localMemory.ubBytes << ',' << p.localMemory.l0aBytes << ',' << p.localMemory.l0bBytes
                  << ',' << p.localMemory.l0cBytes << "],\"lib_api\":" << p.libApiWorkspaceBytes << ",\"blocks\":[";
        for (uint32_t i = 0; i < kMlaMaxAic; ++i) {
            if (i)
                std::cout << ',';
            std::cout << p.blockDims[i];
        }
        std::cout << "]}\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
