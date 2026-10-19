#include "mla_mask.h"
#include "../tiling/mla_causal.h"
#include <new>
#include <utility>
#include "mla_paged.h"

// Protocol adapter for the CANN-derived schedulers. Source/copyright/license
// are retained in tiling/mla_causal.cpp, tiling/mla_sequence.cpp and LICENSE.cann.

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

template <MlaSequenceLayout Layout, MlaSequenceLayout KvLayout = Layout>
MlaScheduleStatus BuildCausalAllocationSchedule(const MlaUsedShape& shape, const MlaPlatformInfo& platform,
                                                const MlaUsedInputs& inputs, MlaUsedSchedule& output,
                                                const void* context)
{
    if (shape.query.layout != Layout || shape.kv.layout != KvLayout) {
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
            const auto* full = static_cast<const MlaCausalFullSchedule*>(context);
            if (Layout == MlaSequenceLayout::Bsnd && MatchesFullSchedule(full, shape, result)) {
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
    if (!BuildUsedMetadataWithSchedule(shape, platform, dtype, output, outputWords, inputs,
                                       BuildCausalAllocationSchedule<MlaSequenceLayout::Bsnd>, full)) {
        return false;
    }
    // Shared serialization has already zeroed failed payloads. The v7 identity
    // is the only wire change; consumers still validate all current lengths.
    output[1] = kMlaCausalBsndMetadataVersion;
    output[2] = kMlaMetadataCausalBsnd;
    return true;
}

bool BuildCausalTndMetadata(const MlaUsedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype,
                            uint32_t* output, uint32_t outputWords, const MlaUsedInputs& inputs)
{
    if (!BuildUsedMetadataWithSchedule(shape, platform, dtype, output, outputWords, inputs,
                                       BuildCausalAllocationSchedule<MlaSequenceLayout::Tnd>, nullptr)) {
        return false;
    }
    output[1] = kMlaCausalTndMetadataVersion;
    output[2] = kMlaMetadataCausalTnd;
    return true;
}

bool BuildCausalMixedMetadata(const MlaUsedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype,
                              uint32_t* output, uint32_t outputWords, const MlaUsedInputs& inputs,
                              bool queryTotalFromPrefix)
{
    if (!BuildUsedMetadataWithSchedule(shape, platform, dtype, output, outputWords, inputs,
                                       BuildCausalAllocationSchedule<MlaSequenceLayout::Tnd, MlaSequenceLayout::Bsnd>,
                                       nullptr, queryTotalFromPrefix)) {
        return false;
    }
    output[1] = kMlaCausalMixedMetadataVersion;
    output[2] = kMlaMetadataCausalMixed;
    return true;
}
} // namespace flash_attn::mla

// Protocol adapter for the pinned CANN/FA3 window scheduler in tiling/mla_window.cpp.

namespace flash_attn::mla {
namespace {
template <bool IsTnd, bool IsMixed = false>
MlaScheduleStatus BuildWindowAllocationSchedule(const MlaUsedShape& shape, const MlaPlatformInfo& platform,
                                                const MlaUsedInputs& inputs, MlaUsedSchedule& output,
                                                const void* context)
{
    const auto& config = *static_cast<const MlaWindowConfig*>(context);
    MlaWindow window{};
    constexpr auto layout = IsTnd ? MlaSequenceLayout::Tnd : MlaSequenceLayout::Bsnd;
    if (shape.query.layout != layout || shape.kv.layout != (IsMixed ? MlaSequenceLayout::Bsnd : layout) ||
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
template <bool IsTnd, bool IsMixed = false>
bool BuildWindowMetadata(const MlaUsedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype, uint32_t* output,
                         uint32_t outputWords, const MlaUsedInputs& inputs, const MlaWindowConfig& window,
                         bool queryTotalFromPrefix = false)
{
    if (!BuildUsedMetadataWithSchedule(shape, platform, dtype, output, outputWords, inputs,
                                       BuildWindowAllocationSchedule<IsTnd, IsMixed>, &window, queryTotalFromPrefix,
                                       kMlaWindowMetadataTailWords)) {
        return false;
    }
    output[1] = IsMixed ? kMlaWindowMixedMetadataVersion
                : IsTnd ? kMlaWindowTndMetadataVersion
                        : kMlaWindowBsndMetadataVersion;
    output[2] = IsMixed ? kMlaMetadataWindowMixed : IsTnd ? kMlaMetadataWindowTnd : kMlaMetadataWindowBsnd;
    if (output[5] == 0) {
        MlaWindow normalized{};
        NormalizeWindow(window, normalized); // Already checked by the partitioner, including empty allocations.
        output[outputWords - 3] = uint32_t(normalized.kind);
        output[outputWords - 2] = uint32_t(normalized.left);
        output[outputWords - 1] = uint32_t(normalized.right);
    }
    return true;
}
bool BuildWindowBsndMetadata(const MlaUsedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype,
                             uint32_t* output, uint32_t outputWords, const MlaUsedInputs& inputs,
                             const MlaWindowConfig& window)
{
    return BuildWindowMetadata<false>(shape, platform, dtype, output, outputWords, inputs, window);
}
bool BuildWindowTndMetadata(const MlaUsedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype,
                            uint32_t* output, uint32_t outputWords, const MlaUsedInputs& inputs,
                            const MlaWindowConfig& window)
{
    return BuildWindowMetadata<true>(shape, platform, dtype, output, outputWords, inputs, window);
}
bool BuildWindowMixedMetadata(const MlaUsedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype,
                              uint32_t* output, uint32_t outputWords, const MlaUsedInputs& inputs,
                              const MlaWindowConfig& window, bool queryTotalFromPrefix)
{
    return BuildWindowMetadata<true, true>(shape, platform, dtype, output, outputWords, inputs, window,
                                           queryTotalFromPrefix);
}
} // namespace flash_attn::mla

// BSND/TND PA causal adapter. CANN-derived algorithms and their license remain in
// tiling/mla_causal.cpp, tiling/mla_sequence.cpp and LICENSE.cann; page Copy is unchanged.

namespace flash_attn::mla {
namespace {
template <MlaSequenceLayout Layout>
MlaScheduleStatus BuildCausalPagedSchedule(const MlaPagedShape& shape, const MlaPlatformInfo& platform,
                                           const MlaPagedInputs& inputs, MlaPagedSchedule& output, const void*)
{
    if (shape.query.layout != Layout) {
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
                                        BuildCausalPagedSchedule<MlaSequenceLayout::Bsnd>)) {
        return false;
    }
    output[1] = kMlaCausalPagedBsndMetadataVersion;
    output[2] = kMlaMetadataCausalPagedBsnd;
    return true;
}
bool BuildCausalPagedTndMetadata(const MlaPagedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype,
                                 uint32_t* output, uint32_t outputWords, const MlaPagedInputs& inputs,
                                 bool fromQueryPrefix)
{
    if (!BuildPagedMetadataWithSchedule(shape, platform, dtype, output, outputWords, inputs, fromQueryPrefix,
                                        BuildCausalPagedSchedule<MlaSequenceLayout::Tnd>)) {
        return false;
    }
    output[1] = kMlaCausalPagedTndMetadataVersion;
    output[2] = kMlaMetadataCausalPagedTnd;
    return true;
}
} // namespace flash_attn::mla

// BSND/TND PA window adapters. Pinned CANN/FA3 algorithms and licenses remain in
// tiling/mla_window.cpp, tiling/mla_sequence.cpp, LICENSE.cann and LICENSE.fa3; page Copy is unchanged.

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
