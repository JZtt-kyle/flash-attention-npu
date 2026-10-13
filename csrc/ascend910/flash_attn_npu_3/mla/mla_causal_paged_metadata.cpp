// BSND PA causal adapter. CANN-derived algorithms and their license remain in
// mla_causal.cpp, mla_tnd_schedule.cpp and LICENSE.cann; page Copy is unchanged.
#include "mla_paged_metadata.h"
#include "mla_causal.h"
#include <new>
#include <utility>

namespace flash_attn::mla {
namespace {
MlaScheduleStatus BuildCausalPagedSchedule(const MlaPagedShape& shape, const MlaPlatformInfo& platform,
                                           const MlaPagedInputs& inputs, MlaPagedSchedule& output)
{
    if (shape.query.layout != MlaSequenceLayout::Bsnd) {
        return MlaScheduleStatus::UnsupportedShape;
    }
    try {
        MlaPagedSchedule result;
        const auto status = PreparePagedSchedule(shape, platform, inputs, result);
        if (status != MlaScheduleStatus::Success) {
            return status;
        }
        if (shape.batch != 0 && result.schedule.s1Size != 0 && result.schedule.s2Size != 0) {
            auto& s = result.schedule;
            MlaCausalScheduleInput input{{}, shape.queryHeads, s.mBaseSize, s.s2BaseSize, s.aicNum, s.cvRatio};
            input.sequences.reserve(result.lengths.sequences.size());
            for (const auto& seq : result.lengths.sequences) {
                input.sequences.push_back({seq.queryLength, seq.kvLength});
            }
            MlaCausalSchedule causal{};
            if (BuildCausalSchedule(input, causal) != MlaCausalStatus::Success) {
                return MlaScheduleStatus::InvalidSplit;
            }
            s.usedCoreNum = causal.usedCoreNum;
            s.bN2End = causal.bN2End;
            s.gS1End = causal.gS1End;
            s.s2End = causal.s2End;
            s.maxCost = causal.maxCost;
            s.blockDim = platform.blockDims[s.usedCoreNum - 1];
            const uint64_t slots = 2 * uint64_t(s.blockDim);
            auto& w = s.workspace;
            w.qkOffsetBytes = 0;
            w.pOffsetBytes = slots * w.mm1ResSize * 4;
            w.pvOffsetBytes = slots * w.mm1ResSize * 6;
            w.vec2OffsetBytes = slots * (w.mm1ResSize * 6 + w.mm2ResSize * 4);
            w.kernelBytes = slots * (w.mm1ResSize * 6 + w.mm2ResSize * 8);
            if (s.blockDim < s.usedCoreNum || s.blockDim > s.aicNum) {
                return MlaScheduleStatus::InvalidPlatform;
            }
            if (w.kernelBytes > w.userBytes) {
                return MlaScheduleStatus::WorkspaceOverflow;
            }
        }
        output = std::move(result);
        return MlaScheduleStatus::Success;
    } catch (const std::bad_alloc&) {
        return MlaScheduleStatus::AllocationFailure;
    }
}
} // namespace

bool BuildCausalPagedBsndMetadata(const MlaPagedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype,
                                  uint32_t* output, uint32_t outputWords, const MlaPagedInputs& inputs)
{
    if (!BuildPagedMetadataWithSchedule(shape, platform, dtype, output, outputWords, inputs, false,
                                        BuildCausalPagedSchedule)) {
        return false;
    }
    output[1] = kMlaCausalPagedBsndMetadataVersion;
    output[2] = kMlaMetadataCausalPagedBsnd;
    return true;
}
} // namespace flash_attn::mla
