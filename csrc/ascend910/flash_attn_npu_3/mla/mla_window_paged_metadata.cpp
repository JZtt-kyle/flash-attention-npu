// BSND/TND PA window adapters. Pinned CANN/FA3 algorithms and licenses remain in
// mla_window.cpp, mla_tnd_schedule.cpp, LICENSE.cann and LICENSE.fa3; page Copy is unchanged.
#include "mla_window_paged_metadata.h"
#include <new>
#include <utility>

namespace flash_attn::mla {
namespace {
template <MlaSequenceLayout Layout>
MlaScheduleStatus BuildWindowPagedSchedule(const MlaPagedShape& shape, const MlaPlatformInfo& platform,
                                           const MlaPagedInputs& inputs, MlaPagedSchedule& output, const void* context)
{
    const auto& config = *static_cast<const MlaWindowConfig*>(context);
    MlaWindow normalized{};
    if (shape.query.layout != Layout || shape.query.maxLength != config.maxQueryLength ||
        shape.logicalKvCapacity != config.maxKvLength ||
        NormalizeWindow(config, normalized) != MlaCausalStatus::Success) {
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
            if (BuildWindowSchedule(input, config, causal) != MlaCausalStatus::Success) {
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

bool BuildWindowPagedBsndMetadata(const MlaPagedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype,
                                  uint32_t* output, uint32_t outputWords, const MlaPagedInputs& inputs,
                                  const MlaWindowConfig& window)
{
    if (!BuildPagedMetadataWithSchedule(shape, platform, dtype, output, outputWords, inputs, false,
                                        BuildWindowPagedSchedule<MlaSequenceLayout::Bsnd>, &window,
                                        kMlaWindowPagedMetadataTailWords)) {
        return false;
    }
    output[1] = kMlaWindowPagedBsndMetadataVersion;
    output[2] = kMlaMetadataWindowPagedBsnd;
    if (output[5] == 0) {
        MlaWindow normalized{};
        NormalizeWindow(window, normalized); // Checked even for empty allocations by the builder.
        output[outputWords - 3] = uint32_t(normalized.kind);
        output[outputWords - 2] = uint32_t(normalized.left);
        output[outputWords - 1] = uint32_t(normalized.right);
    }
    return true;
}
bool BuildWindowPagedTndMetadata(const MlaPagedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype,
                                 uint32_t* output, uint32_t outputWords, const MlaPagedInputs& inputs,
                                 const MlaWindowConfig& window, bool fromQueryPrefix)
{
    if (!BuildPagedMetadataWithSchedule(shape, platform, dtype, output, outputWords, inputs, fromQueryPrefix,
                                        BuildWindowPagedSchedule<MlaSequenceLayout::Tnd>, &window,
                                        kMlaWindowPagedMetadataTailWords)) {
        return false;
    }
    output[1] = kMlaWindowPagedTndMetadataVersion;
    output[2] = kMlaMetadataWindowPagedTnd;
    if (output[5] == 0) {
        MlaWindow normalized{};
        NormalizeWindow(window, normalized); // Checked even for empty allocations by the builder.
        output[outputWords - 3] = uint32_t(normalized.kind);
        output[outputWords - 2] = uint32_t(normalized.left);
        output[outputWords - 1] = uint32_t(normalized.right);
    }
    return true;
}
} // namespace flash_attn::mla
