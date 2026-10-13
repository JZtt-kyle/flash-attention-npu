// Protocol adapter for the CANN-derived schedulers. Source/copyright/license
// are retained in mla_causal.cpp, mla_tnd_schedule.cpp and LICENSE.cann.
#include "mla_causal_metadata.h"
#include "mla_causal.h"
#include <new>
#include <utility>

namespace flash_attn::mla {
namespace {
bool MatchesFullSchedule(const MlaCausalFullSchedule* full, const MlaUsedShape& shape, const MlaUsedSchedule& result)
{
    const auto& s = result.schedule;
    if (full == nullptr || full->enabled != 1 || full->batch != shape.batch ||
        full->queryCapacity != shape.query.maxLength || full->kvCapacity != shape.kv.maxLength ||
        full->heads != shape.queryHeads || full->aicNum != s.aicNum || full->cvRatio != s.cvRatio ||
        full->mBaseSize != s.mBaseSize || full->s2BaseSize != s.s2BaseSize || full->usedCoreNum == 0 ||
        full->usedCoreNum > s.aicNum) {
        return false;
    }
    for (const auto& seq : result.lengths.sequences) {
        if (seq.queryLength != seq.queryCapacity || seq.kvLength != seq.kvCapacity) {
            return false;
        }
    }
    return true;
}

MlaScheduleStatus BuildCausalBsndSchedule(const MlaUsedShape& shape, const MlaPlatformInfo& platform,
                                          const MlaUsedInputs& inputs, MlaUsedSchedule& output, const void* context)
{
    if (shape.query.layout != MlaSequenceLayout::Bsnd || shape.kv.layout != MlaSequenceLayout::Bsnd) {
        return MlaScheduleStatus::UnsupportedShape;
    }
    try {
        MlaUsedSchedule result;
        auto status = PrepareUsedSchedule(shape, platform, inputs, result);
        if (status != MlaScheduleStatus::Success) {
            return status;
        }
        if (shape.batch != 0 && shape.query.maxLength != 0 && shape.kv.maxLength != 0) {
            auto& s = result.schedule;
            const auto* full = static_cast<const MlaCausalFullSchedule*>(context);
            if (MatchesFullSchedule(full, shape, result)) {
                s.usedCoreNum = full->usedCoreNum;
                s.bN2End = full->bN2End;
                s.gS1End = full->gS1End;
                s.s2End = full->s2End;
                s.maxCost = full->maxCost;
            } else {
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
            }
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

bool BuildCausalBsndMetadata(const MlaUsedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype,
                             uint32_t* output, uint32_t outputWords, const MlaUsedInputs& inputs,
                             const MlaCausalFullSchedule* full)
{
    if (!BuildUsedMetadataWithSchedule(shape, platform, dtype, output, outputWords, inputs, BuildCausalBsndSchedule,
                                       full)) {
        return false;
    }
    // Shared serialization has already zeroed failed payloads. The v7 identity
    // is the only wire change; consumers still validate all current lengths.
    output[1] = kMlaCausalBsndMetadataVersion;
    output[2] = kMlaMetadataCausalBsnd;
    return true;
}
} // namespace flash_attn::mla
