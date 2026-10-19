// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
#include "mla_split.h"
#include "../tiling/mla_split_plan.h"
#include <new>
#include <cstring>
#include <utility>

namespace flash_attn::mla {
namespace {
void WriteSplitTail(uint32_t* tail, const MlaWindowConfig& window, uint32_t splits, uint32_t batch,
                    const std::vector<uint64_t>& prefix)
{
    MlaWindow normalized{};
    NormalizeWindow(window, normalized);
    tail[0] = uint32_t(normalized.kind);
    tail[1] = uint32_t(normalized.left);
    tail[2] = uint32_t(normalized.right);
    tail[3] = uint32_t(splits);
    for (uint32_t b = 0; b <= batch; ++b) {
        const auto value = prefix[b];
        tail[4 + 2 * b] = uint32_t(value);
        tail[5 + 2 * b] = uint32_t(value >> 32);
    }
}
} // namespace

bool BuildSplitBsndMetadata(const MlaUsedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype,
                            uint32_t* output, uint32_t words, const MlaUsedInputs& inputs,
                            const MlaWindowConfig& window, int64_t splits, uint32_t hasSoftcap)
{
    uint32_t expected;
    if (!SplitBsndMetadataWords(shape.batch, expected) || words != expected)
        return false;
    struct Context {
        const MlaWindowConfig& window;
        int64_t splits;
        uint32_t hasSoftcap;
        MlaSplitKvPlan plan;
    } context{window, splits, hasSoftcap, {}};
    const auto builder = [](const MlaUsedShape& s, const MlaPlatformInfo& p, const MlaUsedInputs& in,
                            MlaUsedSchedule& schedule, const void* opaque) {
        auto& c = *const_cast<Context*>(static_cast<const Context*>(opaque));
        if (s.query.layout != MlaSequenceLayout::Bsnd || s.kv.layout != MlaSequenceLayout::Bsnd || c.hasSoftcap > 1 ||
            c.splits < (c.hasSoftcap ? 1 : 2) || c.splits > 256)
            return MlaScheduleStatus::InvalidSplit;
        auto status = BuildUsedSchedule(s, p, in, schedule);
        if (status != MlaScheduleStatus::Success)
            return status;
        const auto& t = schedule.schedule;
        std::vector<MlaCausalSequence> lengths;
        try {
            for (const auto& v : schedule.lengths.sequences)
                lengths.push_back({v.queryLength, v.kvLength});
        } catch (const std::bad_alloc&) {
            return MlaScheduleStatus::AllocationFailure;
        }
        const bool empty = !s.batch || !s.query.maxLength || !s.kv.maxLength;
        return BuildSplitKvPlan({s.batch, s.query.maxLength, s.kv.maxLength, s.queryHeads, empty ? 256 : t.mBaseSize,
                                 empty ? 512 : t.s2BaseSize, c.splits, p.aicNum, t.blockDim, t.cvRatio, 0},
                                c.window, lengths, c.plan);
    };
    const uint32_t base = kMlaUsedMetadataPrefixWords + kMlaUsedMetadataSequenceWords * uint32_t(shape.batch);
    if (!BuildUsedMetadataWithSchedule(shape, platform, dtype, output, words, inputs, builder, &context, false,
                                       words - base))
        return false;
    output[1] = output[2] = hasSoftcap ? kMlaSoftcapSplitBsndVersion : kMlaSplitBsndVersion;
    if (output[5] == 0) {
        WriteSplitTail(output + base, window, uint32_t(splits), uint32_t(shape.batch), context.plan.GroupPrefix());
    }
    return true;
}
bool BuildSplitTndMetadata(const MlaUsedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype, uint32_t* output,
                           uint32_t words, const MlaUsedInputs& inputs, const MlaWindowConfig& window, int64_t splits,
                           uint32_t hasSoftcap)
{
    uint32_t expected;
    if (!SplitTndMetadataWords(shape.batch, expected) || words != expected)
        return false;
    struct Context {
        const MlaWindowConfig& window;
        int64_t splits;
        uint32_t hasSoftcap;
        MlaSplitTndPlan plan;
    } context{window, splits, hasSoftcap, {}};
    const auto builder = [](const MlaUsedShape& s, const MlaPlatformInfo& p, const MlaUsedInputs& in,
                            MlaUsedSchedule& schedule, const void* opaque) {
        auto& c = *const_cast<Context*>(static_cast<const Context*>(opaque));
        if (s.query.layout != MlaSequenceLayout::Tnd || s.kv.layout != MlaSequenceLayout::Tnd || c.hasSoftcap > 1 ||
            c.splits < (c.hasSoftcap ? 1 : 2) || c.splits > 256)
            return MlaScheduleStatus::InvalidSplit;
        auto status = BuildUsedSchedule(s, p, in, schedule);
        if (status != MlaScheduleStatus::Success)
            return status;
        const auto& t = schedule.schedule;
        const bool empty = !s.query.totalTokens || !s.kv.totalTokens;
        return BuildSplitTndPlan(
            {{s.batch, s.query.totalTokens, s.kv.totalTokens, s.queryHeads, s.query.maxLength, s.kv.maxLength},
             empty ? 256 : t.mBaseSize,
             empty ? 512 : t.s2BaseSize,
             c.splits,
             p.aicNum,
             t.blockDim,
             t.cvRatio,
             0},
            c.window, in, c.plan);
    };
    const uint32_t base = kMlaUsedMetadataPrefixWords + kMlaUsedMetadataSequenceWords * uint32_t(shape.batch);
    if (!BuildUsedMetadataWithSchedule(shape, platform, dtype, output, words, inputs, builder, &context, false,
                                       words - base))
        return false;
    output[1] = output[2] = hasSoftcap ? kMlaSoftcapSplitTndVersion : kMlaSplitTndVersion;
    if (output[5] == 0) {
        WriteSplitTail(output + base, window, uint32_t(splits), uint32_t(shape.batch), context.plan.GroupPrefix());
    }
    return true;
}
bool BuildSplitMixedMetadata(const MlaUsedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype,
                             uint32_t* output, uint32_t words, const MlaUsedInputs& inputs,
                             const MlaWindowConfig& window, int64_t splits, bool queryTotalFromPrefix,
                             uint32_t hasSoftcap)
{
    uint32_t expected;
    if (!SplitMixedMetadataWords(shape.batch, expected) || words != expected)
        return false;
    struct Context {
        const MlaWindowConfig& window;
        int64_t splits;
        uint32_t hasSoftcap;
        MlaSplitMixedPlan plan;
    } context{window, splits, hasSoftcap, {}};
    const auto builder = [](const MlaUsedShape& s, const MlaPlatformInfo& p, const MlaUsedInputs& in,
                            MlaUsedSchedule& schedule, const void* opaque) {
        auto& c = *const_cast<Context*>(static_cast<const Context*>(opaque));
        if (s.query.layout != MlaSequenceLayout::Tnd || s.kv.layout != MlaSequenceLayout::Bsnd || c.hasSoftcap > 1 ||
            c.splits < (c.hasSoftcap ? 1 : 2) || c.splits > 256)
            return MlaScheduleStatus::InvalidSplit;
        auto status = BuildUsedSchedule(s, p, in, schedule);
        if (status != MlaScheduleStatus::Success)
            return status;
        const auto& t = schedule.schedule;
        const bool empty = !s.query.totalTokens || !s.kv.totalTokens;
        return BuildSplitMixedPlan({{s.batch, s.query.totalTokens, s.queryHeads, s.query.maxLength, s.kv.maxLength},
                                    empty ? 256 : t.mBaseSize,
                                    empty ? 512 : t.s2BaseSize,
                                    c.splits,
                                    p.aicNum,
                                    t.blockDim,
                                    t.cvRatio,
                                    0},
                                   c.window, in, c.plan);
    };
    const uint32_t base = kMlaUsedMetadataPrefixWords + kMlaUsedMetadataSequenceWords * uint32_t(shape.batch);
    if (!BuildUsedMetadataWithSchedule(shape, platform, dtype, output, words, inputs, builder, &context,
                                       queryTotalFromPrefix, words - base))
        return false;
    output[1] = output[2] = hasSoftcap ? kMlaSoftcapSplitMixedVersion : kMlaSplitMixedVersion;
    if (output[5] == 0) {
        WriteSplitTail(output + base, window, uint32_t(splits), uint32_t(shape.batch), context.plan.GroupPrefix());
    }
    return true;
}
bool BuildSplitPagedBsndMetadata(const MlaPagedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype,
                                 uint32_t* output, uint32_t words, const MlaPagedInputs& inputs,
                                 const MlaWindowConfig& window, int64_t splits, uint32_t hasSoftcap)
{
    uint32_t expected;
    if (!SplitPagedBsndMetadataWords(shape.batch, expected) || words != expected)
        return false;
    struct Context {
        const MlaWindowConfig& window;
        int64_t splits;
        uint32_t hasSoftcap;
        MlaSplitPagedBsndPlan plan;
    } context{window, splits, hasSoftcap, {}};
    const auto builder = [](const MlaPagedShape& s, const MlaPlatformInfo& p, const MlaPagedInputs& in,
                            MlaPagedSchedule& schedule, const void* opaque) {
        auto& c = *const_cast<Context*>(static_cast<const Context*>(opaque));
        if (c.hasSoftcap > 1 || c.splits < (c.hasSoftcap ? 1 : 2) || c.splits > 256)
            return MlaScheduleStatus::InvalidSplit;
        const auto status = BuildSplitPagedBsndPlan(s, p, in, c.window, c.splits, c.plan);
        if (status == MlaScheduleStatus::Success)
            schedule = std::move(c.plan.paged);
        return status;
    };
    const uint32_t base = kMlaPagedMetadataPrefixWords + kMlaPagedMetadataSequenceWords * uint32_t(shape.batch);
    if (!BuildPagedMetadataWithSchedule(shape, platform, dtype, output, words, inputs, false, builder, &context,
                                        words - base))
        return false;
    output[1] = output[2] = hasSoftcap ? kMlaSoftcapSplitPagedBsndVersion : kMlaSplitPagedBsndVersion;
    if (output[5] == 0) {
        WriteSplitTail(output + base, window, uint32_t(splits), uint32_t(shape.batch),
                       context.plan.split.GroupPrefix());
    }
    return true;
}
bool BuildSplitPagedTndMetadata(const MlaPagedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype,
                                uint32_t* output, uint32_t words, const MlaPagedInputs& inputs,
                                const MlaWindowConfig& window, int64_t splits, bool queryTotalFromPrefix,
                                uint32_t hasSoftcap)
{
    uint32_t expected;
    if (!SplitPagedTndMetadataWords(shape.batch, expected) || words != expected)
        return false;
    struct Context {
        const MlaWindowConfig& window;
        int64_t splits;
        bool fromPrefix;
        uint32_t hasSoftcap;
        MlaSplitPagedTndPlan plan;
    } context{window, splits, queryTotalFromPrefix, hasSoftcap, {}};
    const auto builder = [](const MlaPagedShape& s, const MlaPlatformInfo& p, const MlaPagedInputs& in,
                            MlaPagedSchedule& schedule, const void* opaque) {
        auto& c = *const_cast<Context*>(static_cast<const Context*>(opaque));
        if (c.hasSoftcap > 1 || c.splits < (c.hasSoftcap ? 1 : 2) || c.splits > 256)
            return MlaScheduleStatus::InvalidSplit;
        auto platform = p;
        if (c.fromPrefix)
            platform.blockDims.fill(s.batch && s.query.totalTokens && s.logicalKvCapacity ? p.aicNum : 1);
        const auto status = BuildSplitPagedTndPlan(s, platform, in, c.window, c.splits, c.plan);
        if (status == MlaScheduleStatus::Success)
            schedule = std::move(c.plan.paged);
        return status;
    };
    const uint32_t base = kMlaPagedMetadataPrefixWords + kMlaPagedMetadataSequenceWords * uint32_t(shape.batch);
    if (!BuildPagedMetadataWithSchedule(shape, platform, dtype, output, words, inputs, queryTotalFromPrefix, builder,
                                        &context, words - base))
        return false;
    output[1] = output[2] = hasSoftcap ? kMlaSoftcapSplitPagedTndVersion : kMlaSplitPagedTndVersion;
    if (output[5] == 0) {
        WriteSplitTail(output + base, window, uint32_t(splits), uint32_t(shape.batch), context.plan.GroupPrefix());
    }
    return true;
}
} // namespace flash_attn::mla
