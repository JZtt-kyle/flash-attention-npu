// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
#include "mla_split_mixed_metadata.h"
#include "mla_split_mixed_plan.h"
#include <new>
#include <cstring>

namespace flash_attn::mla {
bool BuildSplitMixedMetadata(const MlaUsedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype,
                             uint32_t* output, uint32_t words, const MlaUsedInputs& inputs,
                             const MlaWindowConfig& window, int64_t splits, bool queryTotalFromPrefix)
{
    uint32_t expected;
    if (!SplitMixedMetadataWords(shape.batch, expected) || words != expected)
        return false;
    struct Context {
        const MlaWindowConfig& window;
        int64_t splits;
        MlaSplitMixedPlan plan;
    } context{window, splits, {}};
    const auto builder = [](const MlaUsedShape& s, const MlaPlatformInfo& p, const MlaUsedInputs& in,
                            MlaUsedSchedule& schedule, const void* opaque) {
        auto& c = *const_cast<Context*>(static_cast<const Context*>(opaque));
        if (s.query.layout != MlaSequenceLayout::Tnd || s.kv.layout != MlaSequenceLayout::Bsnd || c.splits < 2 ||
            c.splits > 256)
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
    output[1] = output[2] = kMlaSplitMixedVersion;
    if (output[5] == 0) {
        MlaWindow normalized{};
        NormalizeWindow(window, normalized);
        output[base] = uint32_t(normalized.kind);
        output[base + 1] = uint32_t(normalized.left);
        output[base + 2] = uint32_t(normalized.right);
        output[base + 3] = uint32_t(splits);
        for (uint32_t b = 0; b <= uint32_t(shape.batch); ++b) {
            const auto value = context.plan.GroupPrefix()[b];
            output[base + 4 + 2 * b] = uint32_t(value);
            output[base + 5 + 2 * b] = uint32_t(value >> 32);
        }
    }
    return true;
}
} // namespace flash_attn::mla
