// Independent M10b1 bridge. Reuse validated resource selection and M10a
// explicit fragment/workspace builders; no production metadata or dispatcher.
#include <algorithm>
#include "mla/mla_split_kv.h"
#include "mla/mla_split_kv_params.h"
#include "mla/mla_dense_params.h"
using namespace flash_attn::mla;
extern "C" int prepare_window(int64_t, int64_t, int64_t, int64_t, const int64_t*, int64_t, int64_t, int64_t,
                              MlaDenseKernelSchedule*, uint64_t*, int64_t, int64_t, int64_t, int64_t, int64_t,
                              MlaWindowKernelParams*);
extern "C" int prepare_partial(int64_t batch, int64_t sq, int64_t heads, int64_t sk, const int64_t* lengths,
                               int64_t cores, int64_t m, int64_t n, int64_t splits, uint64_t first, int64_t left,
                               int64_t right, int64_t causal, MlaDenseKernelSchedule* schedule,
                               MlaSplitKvKernelParams* partial, uint64_t* details)
{
    if (!schedule || !partial || !details || batch < 1 || sq < 1 || splits < 1 || splits > 256) {
        return -1;
    }
    MlaDenseKernelSchedule s{};
    MlaSplitKvKernelParams p{};
    uint64_t d[16]{};
    int error =
        prepare_window(batch, sq, heads, sk, lengths, cores, m, n, &s, d, sq, sk, left, right, causal, &p.window);
    if (error) {
        return error;
    }
    const uint64_t mCount = (uint64_t(sq) * heads + s.mBaseSize - 1) / s.mBaseSize;
    p.totalTasks = uint64_t(batch) * mCount * splits;
    if (first >= p.totalTasks) {
        return -1;
    }
    s.usedCoreNum = s.blockDim = std::min(p.totalTasks - first, cores ? uint64_t(cores) : d[2]);
    MlaSplitKvWorkspace w{};
    if (BuildSplitKvWorkspace({s.mBaseSize, s.s2BaseSize, sq * heads, int64_t(d[2]), s.blockDim, true, d[9]}, w) !=
        MlaScheduleStatus::Success) {
        return -1;
    }
    s.mm1ResSize = w.attention.mm1ResSize;
    s.mm2ResSize = w.attention.mm2ResSize;
    p.outOffsetBytes = w.fdAccumOffsetBytes;
    p.sumOffsetBytes = w.fdSumOffsetBytes;
    p.maxOffsetBytes = w.fdMaxOffsetBytes;
    d[0] = w.attention.userBytes;
    d[1] = w.attention.kernelBytes;
    d[10] = 0;
    d[11] = w.attention.pOffsetBytes;
    d[12] = w.attention.pvOffsetBytes;
    d[13] = w.attention.vec2OffsetBytes;
    for (uint64_t i = 0; i < s.usedCoreNum; ++i) {
        const uint64_t index = first + i;
        const uint32_t b = index / (mCount * splits), block = (index / splits) % mCount, split = index % splits;
        MlaSplitKvBlock fragment{};
        if (BuildSplitKvBlock({sq, lengths[b], heads, s.mBaseSize, s.s2BaseSize, block},
                              {sq, sk, left, right, causal != 0}, splits, split,
                              fragment) != MlaCausalStatus::Success) {
            return -1;
        }
        p.tasks[i] = {b, block, split, fragment.s2Begin, fragment.s2End};
    }
    *schedule = s;
    *partial = p;
    std::copy(std::begin(d), std::end(d), details);
    return 0;
}
