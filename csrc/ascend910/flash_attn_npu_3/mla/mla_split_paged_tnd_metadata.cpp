// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
#include "mla_split_paged_tnd_metadata.h"
#include "mla_split_paged_tnd_plan.h"
#include <utility>

namespace flash_attn::mla {
bool BuildSplitPagedTndMetadata(const MlaPagedShape& shape, const MlaPlatformInfo& platform, uint32_t dtype,
                                uint32_t* output, uint32_t words, const MlaPagedInputs& inputs,
                                const MlaWindowConfig& window, int64_t splits)
{
    uint32_t expected;
    if (!SplitPagedTndMetadataWords(shape.batch, expected) || words != expected)
        return false;
    struct Context {
        const MlaWindowConfig& window;
        int64_t splits;
        MlaSplitPagedTndPlan plan;
    } context{window, splits, {}};
    const auto builder = [](const MlaPagedShape& s, const MlaPlatformInfo& p, const MlaPagedInputs& in,
                            MlaPagedSchedule& schedule, const void* opaque) {
        auto& c = *const_cast<Context*>(static_cast<const Context*>(opaque));
        const auto status = BuildSplitPagedTndPlan(s, p, in, c.window, c.splits, c.plan);
        if (status == MlaScheduleStatus::Success)
            schedule = std::move(c.plan.paged);
        return status;
    };
    const uint32_t base = kMlaPagedMetadataPrefixWords + kMlaPagedMetadataSequenceWords * uint32_t(shape.batch);
    if (!BuildPagedMetadataWithSchedule(shape, platform, dtype, output, words, inputs, false, builder, &context,
                                        words - base))
        return false;
    output[1] = output[2] = kMlaSplitPagedTndVersion;
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
