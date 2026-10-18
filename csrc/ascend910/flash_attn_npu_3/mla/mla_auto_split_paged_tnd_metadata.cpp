// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
#include "mla_auto_split_paged_tnd_metadata.h"
#include "mla_auto_split_paged_tnd_plan.h"
#include <utility>

namespace flash_attn::mla {
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
    MlaWindow normalized{};
    NormalizeWindow(window, normalized);
    uint32_t* tail = output + base;
    *tail++ = uint32_t(normalized.kind);
    *tail++ = uint32_t(normalized.left);
    *tail++ = uint32_t(normalized.right);
    const auto& n = context.plan.split.Native();
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
    for (auto value : context.plan.split.GroupPrefix()) {
        *tail++ = uint32_t(value);
        *tail++ = uint32_t(value >> 32);
    }
    return true;
}
} // namespace flash_attn::mla
