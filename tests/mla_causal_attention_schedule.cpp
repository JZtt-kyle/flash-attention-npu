// Test-only bridge: use the existing resource-checked tile selection and the
// independently accepted M08a whole-M causal partition. No production dispatch.
#include <algorithm>
#include "mla/mla_causal.h"
#include "mla/mla_dense_params.h"
using namespace flash_attn::mla;
extern "C" int prepare_dense(int64_t, int64_t, int64_t, int64_t, MlaDenseKernelSchedule*, uint64_t*);

extern "C" int prepare_causal(int64_t batch, int64_t sq, int64_t heads, int64_t sk, const int64_t* lengths,
                              int64_t coreLimit, int64_t m, int64_t n, MlaDenseKernelSchedule* result,
                              uint64_t* details)
{
    if (!result || !details || batch < 0 || sq < 0 || sk < 0 || sk > 16384 || (batch && !lengths) || coreLimit < 0 ||
        coreLimit > 26 || m < 0 || m > 256 || n < 0 || n > 512) {
        return -1;
    }
    MlaDenseKernelSchedule params{};
    uint64_t resources[16]{};
    int code = prepare_dense(std::max(batch, int64_t(1)), std::max(sq, int64_t(1)), heads, std::max(sk, int64_t(1)),
                             &params, resources);
    if (code) {
        return code;
    }
    if (coreLimit > int64_t(resources[2])) {
        return -1;
    }
    params.batch = batch;
    params.sq = sq;
    params.sk = sk;
    if (m) {
        params.mBaseSize = m;
    }
    if (n) {
        params.s2BaseSize = n;
    }
    MlaCausalScheduleInput input{
        {}, heads, params.mBaseSize, params.s2BaseSize, coreLimit ? coreLimit : int64_t(resources[2]), 2};
    for (int64_t b = 0; b < batch; ++b) {
        if (lengths[b] < 0 || lengths[b] > sk) {
            return -1;
        }
        input.sequences.push_back({sq, lengths[b]});
    }
    MlaCausalSchedule schedule{};
    if (BuildCausalSchedule(input, schedule) != MlaCausalStatus::Success) {
        return -1;
    }
    params.usedCoreNum = schedule.usedCoreNum;
    params.blockDim = schedule.usedCoreNum; // C1V2 on this standalone 910B probe.
    std::copy(schedule.bN2End.begin(), schedule.bN2End.end(), params.bN2End);
    std::copy(schedule.gS1End.begin(), schedule.gS1End.end(), params.gS1End);
    std::copy(schedule.s2End.begin(), schedule.s2End.end(), params.s2End);
    const uint32_t alignedM = (params.mBaseSize + 15) / 16 * 16;
    params.mm1ResSize = alignedM * ((params.s2BaseSize + 31) / 32 * 32);
    params.mm2ResSize = alignedM * 512;
    const uint64_t slots = 2 * params.blockDim;
    resources[10] = 0;
    resources[11] = slots * params.mm1ResSize * 4;
    resources[12] = resources[11] + slots * params.mm1ResSize * 2;
    resources[13] = resources[12] + slots * params.mm2ResSize * 4;
    resources[1] = resources[13] + slots * params.mm2ResSize * 4;
    resources[0] = resources[1] + resources[9];
    resources[14] = 1;
    resources[15] = 0;
    *result = params;
    std::copy(std::begin(resources), std::end(resources), details);
    return 0;
}
