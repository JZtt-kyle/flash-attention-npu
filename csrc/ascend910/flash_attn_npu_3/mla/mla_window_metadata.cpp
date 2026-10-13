// Protocol adapter for the pinned CANN/FA3 window scheduler in mla_window.cpp.
#include "mla_window_metadata.h"
#include <new>
#include <utility>

namespace flash_attn::mla {
namespace {
MlaScheduleStatus BuildWindowAllocationSchedule(const MlaUsedShape& shape, const MlaPlatformInfo& platform,
                                                const MlaUsedInputs& inputs, MlaUsedSchedule& output,
                                                const void* context)
{
    const auto& config = *static_cast<const MlaWindowConfig*>(context);
    MlaWindow window{};
    if (shape.query.layout != MlaSequenceLayout::Bsnd || shape.kv.layout != MlaSequenceLayout::Bsnd ||
        shape.query.maxLength != config.maxQueryLength || shape.kv.maxLength != config.maxKvLength ||
        NormalizeWindow(config, window) != MlaCausalStatus::Success) {
        return MlaScheduleStatus::UnsupportedShape;
    }
    try {
        MlaUsedSchedule result;
        auto status = PrepareUsedSchedule(shape, platform, inputs, result);
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
            MlaCausalSchedule partition{};
            if (BuildWindowSchedule(input, config, partition) != MlaCausalStatus::Success) {
                return MlaScheduleStatus::InvalidSplit;
            }
            s.usedCoreNum = partition.usedCoreNum;
            s.bN2End = partition.bN2End;
            s.gS1End = partition.gS1End;
            s.s2End = partition.s2End;
            s.maxCost = partition.maxCost;
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
bool BuildWindowBsndMetadata(const MlaUsedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype,
                             uint32_t* output, uint32_t outputWords, const MlaUsedInputs& inputs,
                             const MlaWindowConfig& window)
{
    if (!BuildUsedMetadataWithSchedule(shape, platform, dtype, output, outputWords, inputs,
                                       BuildWindowAllocationSchedule, &window, false, kMlaWindowMetadataTailWords)) {
        return false;
    }
    output[1] = kMlaWindowBsndMetadataVersion;
    output[2] = kMlaMetadataWindowBsnd;
    if (output[5] == 0) {
        MlaWindow normalized{};
        NormalizeWindow(window, normalized); // Already checked by the partitioner, including empty allocations.
        output[outputWords - 3] = uint32_t(normalized.kind);
        output[outputWords - 2] = uint32_t(normalized.left);
        output[outputWords - 1] = uint32_t(normalized.right);
    }
    return true;
}
} // namespace flash_attn::mla
