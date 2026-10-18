// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
#include "mla_auto_split_mixed_metadata.h"
#include "mla_auto_split_mixed_plan.h"

namespace flash_attn::mla {
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
    MlaWindow normalized{};
    NormalizeWindow(window, normalized);
    uint32_t* tail = output + base;
    *tail++ = uint32_t(normalized.kind);
    *tail++ = uint32_t(normalized.left);
    *tail++ = uint32_t(normalized.right);
    const auto& n = context.plan.Native();
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
    for (auto value : context.plan.GroupPrefix()) {
        *tail++ = uint32_t(value);
        *tail++ = uint32_t(value >> 32);
    }
    return true;
}
} // namespace flash_attn::mla
