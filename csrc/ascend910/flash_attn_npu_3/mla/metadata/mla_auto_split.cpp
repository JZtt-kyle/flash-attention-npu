// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
#include "mla_auto_split.h"
#include "../tiling/mla_auto_split_plan.h"
#include <new>
#include <utility>

namespace flash_attn::mla {
namespace {
template <typename Plan>
void WriteAutoSplitTail(uint32_t* tail, const MlaWindowConfig& window, const Plan& plan)
{
    MlaWindow normalized{};
    NormalizeWindow(window, normalized);
    *tail++ = uint32_t(normalized.kind);
    *tail++ = uint32_t(normalized.left);
    *tail++ = uint32_t(normalized.right);
    const auto& n = plan.Native();
    *tail++ = n.usedCoreNum;
    const auto emit = [&](const auto& array) {
        for (auto x : array)
            *tail++ = x;
    };
    emit(n.bN2End);
    emit(n.gS1End);
    emit(n.s2End);
    const auto& f = n.fd;
    *tail++ = f.numOfFdHead;
    *tail++ = f.maxS2SplitNum;
    *tail++ = f.usedVecNumOfFd;
    *tail++ = f.gS1BaseSizeOfFd;
    emit(f.bN2IdxOfFdHead);
    emit(f.gS1IdxOfFdHead);
    emit(f.s2SplitNumOfFdHead);
    emit(f.s2SplitStartIdxOfCore);
    emit(f.gS1SplitNumOfFdHead);
    emit(f.gS1LastPartSizeOfFdHead);
    emit(f.gS1IdxEndOfFdHead);
    emit(f.gS1IdxEndOfFdHeadSplit);
    for (auto value : plan.GroupPrefix()) {
        *tail++ = uint32_t(value);
        *tail++ = uint32_t(value >> 32);
    }
}
} // namespace

bool BuildAutoSplitBsndMetadata(const MlaUsedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype,
                                uint32_t* output, uint32_t words, const MlaUsedInputs& inputs,
                                const MlaWindowConfig& window, uint32_t hasSoftcap)
{
    uint32_t expected;
    if (hasSoftcap > 1 || !AutoSplitBsndMetadataWords(shape.batch, expected) || words != expected)
        return false;
    struct Context {
        const MlaWindowConfig& window;
        MlaAutoSplitKvPlan plan;
    } context{window, {}};
    const auto builder = [](const MlaUsedShape& s, const MlaPlatformInfo& p, const MlaUsedInputs& in,
                            MlaUsedSchedule& schedule, const void* opaque) {
        auto& c = *const_cast<Context*>(static_cast<const Context*>(opaque));
        if (s.query.layout != MlaSequenceLayout::Bsnd || s.kv.layout != MlaSequenceLayout::Bsnd)
            return MlaScheduleStatus::UnsupportedShape;
        // The CPU base scheduler also models C1V1, but the reused device
        // schema/consumer is C1V2. Reject before publishing an unusable record.
        if (!p.aicNum || p.aicNum > kMlaMaxAic || p.aivNum != 2 * p.aicNum)
            return MlaScheduleStatus::InvalidPlatform;
        auto status = BuildUsedSchedule(s, p, in, schedule);
        if (status != MlaScheduleStatus::Success)
            return status;
        const auto& t = schedule.schedule;
        const bool empty = !s.batch || !s.query.maxLength || !s.kv.maxLength;
        // Auto scratch is indexed by physical AIC capacity, independent of the
        // number of native partitions. A producer must retain that launch ABI.
        if (!empty && t.blockDim != p.aicNum)
            return MlaScheduleStatus::InvalidPlatform;
        std::vector<MlaCausalSequence> lengths;
        try {
            lengths.reserve(schedule.lengths.sequences.size());
            for (const auto& v : schedule.lengths.sequences)
                lengths.push_back({v.queryLength, v.kvLength});
        } catch (const std::bad_alloc&) {
            return MlaScheduleStatus::AllocationFailure;
        }
        return BuildAutoSplitKvPlan({s.batch, s.query.maxLength, s.kv.maxLength, s.queryHeads,
                                     empty ? 256 : t.mBaseSize, empty ? 512 : t.s2BaseSize, p.aicNum, 2, 0},
                                    c.window, lengths, c.plan);
    };
    const uint32_t base = kMlaUsedMetadataPrefixWords + kMlaUsedMetadataSequenceWords * uint32_t(shape.batch);
    if (!BuildUsedMetadataWithSchedule(shape, platform, dtype, output, words, inputs, builder, &context, false,
                                       words - base))
        return false;
    output[1] = output[2] = hasSoftcap ? kMlaSoftcapAutoSplitBsndVersion : kMlaAutoSplitBsndVersion;
    if (output[5] != 0)
        return true;
    WriteAutoSplitTail(output + base, window, context.plan);
    return true;
}
bool BuildAutoSplitTndMetadata(const MlaUsedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype,
                               uint32_t* output, uint32_t words, const MlaUsedInputs& inputs,
                               const MlaWindowConfig& window, uint32_t hasSoftcap)
{
    uint32_t expected;
    if (!AutoSplitTndMetadataWords(shape.batch, expected) || words != expected)
        return false;
    struct Context {
        const MlaWindowConfig& window;
        uint32_t hasSoftcap;
        MlaAutoSplitTndPlan plan;
    } context{window, hasSoftcap, {}};
    const auto builder = [](const MlaUsedShape& s, const MlaPlatformInfo& p, const MlaUsedInputs& in,
                            MlaUsedSchedule& schedule, const void* opaque) {
        auto& c = *const_cast<Context*>(static_cast<const Context*>(opaque));
        if (s.query.layout != MlaSequenceLayout::Tnd || s.kv.layout != MlaSequenceLayout::Tnd || c.hasSoftcap > 1)
            return MlaScheduleStatus::UnsupportedShape;
        // The CPU base scheduler also models C1V1, but the reused device
        // schema/consumer is C1V2. Reject before publishing an unusable record.
        if (!p.aicNum || p.aicNum > kMlaMaxAic || p.aivNum != 2 * p.aicNum)
            return MlaScheduleStatus::InvalidPlatform;
        auto status = BuildUsedSchedule(s, p, in, schedule);
        if (status != MlaScheduleStatus::Success)
            return status;
        const auto& t = schedule.schedule;
        const bool empty = !s.query.totalTokens || !s.kv.totalTokens;
        // Auto scratch is indexed by physical AIC capacity, independent of the
        // number of native partitions. A producer must retain that launch ABI.
        if (!empty && t.blockDim != p.aicNum)
            return MlaScheduleStatus::InvalidPlatform;
        return BuildAutoSplitTndPlan(
            {{s.batch, s.query.totalTokens, s.kv.totalTokens, s.queryHeads, s.query.maxLength, s.kv.maxLength},
             empty ? 256 : t.mBaseSize,
             empty ? 512 : t.s2BaseSize,
             p.aicNum,
             2,
             0},
            c.window, in, c.plan);
    };
    const uint32_t base = kMlaUsedMetadataPrefixWords + kMlaUsedMetadataSequenceWords * uint32_t(shape.batch);
    if (!BuildUsedMetadataWithSchedule(shape, platform, dtype, output, words, inputs, builder, &context, false,
                                       words - base))
        return false;
    output[1] = output[2] = hasSoftcap ? kMlaSoftcapAutoSplitTndVersion : kMlaAutoSplitTndVersion;
    if (output[5] != 0)
        return true;
    WriteAutoSplitTail(output + base, window, context.plan);
    return true;
}
bool BuildAutoSplitMixedMetadata(const MlaUsedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype,
                                 uint32_t* output, uint32_t words, const MlaUsedInputs& inputs,
                                 const MlaWindowConfig& window, bool queryTotalFromPrefix, uint32_t hasSoftcap)
{
    uint32_t expected;
    if (!AutoSplitMixedMetadataWords(shape.batch, expected) || words != expected)
        return false;
    struct Context {
        const MlaWindowConfig& window;
        uint32_t hasSoftcap;
        MlaAutoSplitMixedPlan plan;
    } context{window, hasSoftcap, {}};
    const auto builder = [](const MlaUsedShape& s, const MlaPlatformInfo& p, const MlaUsedInputs& in,
                            MlaUsedSchedule& schedule, const void* opaque) {
        auto& c = *const_cast<Context*>(static_cast<const Context*>(opaque));
        if (s.query.layout != MlaSequenceLayout::Tnd || s.kv.layout != MlaSequenceLayout::Bsnd || c.hasSoftcap > 1)
            return MlaScheduleStatus::UnsupportedShape;
        // The CPU base scheduler also models C1V1, but the reused device
        // schema/consumer is C1V2. Reject before publishing an unusable record.
        if (!p.aicNum || p.aicNum > kMlaMaxAic || p.aivNum != 2 * p.aicNum)
            return MlaScheduleStatus::InvalidPlatform;
        auto status = BuildUsedSchedule(s, p, in, schedule);
        if (status != MlaScheduleStatus::Success)
            return status;
        const auto& t = schedule.schedule;
        const bool empty = !s.query.totalTokens || !s.kv.totalTokens;
        // Auto scratch is indexed by physical AIC capacity, independent of the
        // number of native partitions. A producer must retain that launch ABI.
        if (!empty && t.blockDim != p.aicNum)
            return MlaScheduleStatus::InvalidPlatform;
        return BuildAutoSplitMixedPlan({{s.batch, s.query.totalTokens, s.queryHeads, s.query.maxLength, s.kv.maxLength},
                                        empty ? 256 : t.mBaseSize,
                                        empty ? 512 : t.s2BaseSize,
                                        p.aicNum,
                                        2,
                                        0},
                                       c.window, in, c.plan);
    };
    const uint32_t base = kMlaUsedMetadataPrefixWords + kMlaUsedMetadataSequenceWords * uint32_t(shape.batch);
    if (!BuildUsedMetadataWithSchedule(shape, platform, dtype, output, words, inputs, builder, &context,
                                       queryTotalFromPrefix, words - base))
        return false;
    output[1] = output[2] = hasSoftcap ? kMlaSoftcapAutoSplitMixedVersion : kMlaAutoSplitMixedVersion;
    if (output[5] != 0)
        return true;
    WriteAutoSplitTail(output + base, window, context.plan);
    return true;
}
bool BuildAutoSplitPagedBsndMetadata(const MlaPagedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype,
                                     uint32_t* output, uint32_t words, const MlaPagedInputs& inputs,
                                     const MlaWindowConfig& window, uint32_t hasSoftcap)
{
    uint32_t expected;
    if (hasSoftcap > 1 || !AutoSplitPagedBsndMetadataWords(shape.batch, expected) || words != expected)
        return false;
    struct Context {
        const MlaWindowConfig& window;
        MlaAutoSplitPagedBsndPlan plan;
    } context{window, {}};
    const auto builder = [](const MlaPagedShape& s, const MlaPlatformInfo& p, const MlaPagedInputs& in,
                            MlaPagedSchedule& schedule, const void* opaque) {
        auto& c = *const_cast<Context*>(static_cast<const Context*>(opaque));
        const auto status = BuildAutoSplitPagedBsndPlan(s, p, in, c.window, c.plan);
        if (status == MlaScheduleStatus::Success)
            schedule = std::move(c.plan.paged);
        return status;
    };
    const uint32_t base = kMlaPagedMetadataPrefixWords + kMlaPagedMetadataSequenceWords * uint32_t(shape.batch);
    if (!BuildPagedMetadataWithSchedule(shape, platform, dtype, output, words, inputs, false, builder, &context,
                                        words - base))
        return false;
    output[1] = output[2] = hasSoftcap ? kMlaSoftcapAutoSplitPagedBsndVersion : kMlaAutoSplitPagedBsndVersion;
    if (output[5] != 0)
        return true;
    WriteAutoSplitTail(output + base, window, context.plan.split);
    return true;
}
bool BuildAutoSplitPagedTndMetadata(const MlaPagedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype,
                                    uint32_t* output, uint32_t words, const MlaPagedInputs& inputs,
                                    const MlaWindowConfig& window, bool queryTotalFromPrefix, uint32_t hasSoftcap)
{
    uint32_t expected;
    if (!AutoSplitPagedTndMetadataWords(shape.batch, expected) || words != expected)
        return false;
    struct Context {
        const MlaWindowConfig& window;
        bool fromPrefix;
        uint32_t hasSoftcap;
        MlaAutoSplitPagedTndPlan plan;
    } context{window, queryTotalFromPrefix, hasSoftcap, {}};
    const auto builder = [](const MlaPagedShape& s, const MlaPlatformInfo& p, const MlaPagedInputs& in,
                            MlaPagedSchedule& schedule, const void* opaque) {
        auto& c = *const_cast<Context*>(static_cast<const Context*>(opaque));
        if (c.hasSoftcap > 1)
            return MlaScheduleStatus::InvalidSplit;
        auto platform = p;
        if (c.fromPrefix)
            platform.blockDims.fill(s.batch && s.query.totalTokens && s.logicalKvCapacity ? p.aicNum : 1);
        MlaAutoSplitPagedTndBudget production{};
        auto status = BuildAutoSplitPagedTndProductionBudget(s, platform, production);
        if (status == MlaScheduleStatus::Success)
            status = BuildAutoSplitPagedTndPlan(s, platform, in, c.window, c.plan);
        if (status == MlaScheduleStatus::Success)
            schedule = std::move(c.plan.paged);
        return status;
    };
    const uint32_t base = kMlaPagedMetadataPrefixWords + kMlaPagedMetadataSequenceWords * uint32_t(shape.batch);
    if (!BuildPagedMetadataWithSchedule(shape, platform, dtype, output, words, inputs, queryTotalFromPrefix, builder,
                                        &context, words - base))
        return false;
    output[1] = output[2] = hasSoftcap ? kMlaSoftcapAutoSplitPagedTndVersion : kMlaAutoSplitPagedTndVersion;
    if (output[5] != 0)
        return true;
    WriteAutoSplitTail(output + base, window, context.plan.split);
    return true;
}
} // namespace flash_attn::mla
