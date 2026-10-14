// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
// Fixed CANN/FA3 provenance and licenses: mla_auto_split_mixed_plan.h.
#include "mla_auto_split_mixed_plan.h"
namespace flash_attn::mla {
namespace {
MlaScheduleStatus CompactConfig(const MlaAutoSplitMixedPlanConfig& c, MlaAutoSplitTndPlanConfig& output)
{
    const auto& s = c.shape;
    // Bound both factors before multiplying; the rectangular KV total may
    // exceed int32/uint32 and is never narrowed to a TND prefix terminal.
    if (s.batch < 0 || s.kvCapacity < 0)
        return MlaScheduleStatus::UnsupportedShape;
    if (uint64_t(s.batch) > UINT32_MAX || s.kvCapacity > kMlaMaxKvLength)
        return MlaScheduleStatus::FieldOverflow;
    output = {{s.batch, s.totalQuery, s.batch * s.kvCapacity, s.queryHeads, s.maxQueryLength, s.kvCapacity},
              c.mBaseSize,
              c.s2BaseSize,
              c.aicNum,
              c.cvRatio,
              c.libApiBytes};
    return MlaScheduleStatus::Success;
}
} // namespace
MlaScheduleStatus BuildAutoSplitMixedPlanLayout(const MlaAutoSplitMixedPlanConfig& c, MlaAutoSplitKvLayout& output)
{
    MlaAutoSplitTndPlanConfig compact{};
    const auto status = CompactConfig(c, compact);
    if (status != MlaScheduleStatus::Success)
        return status;
    return BuildAutoSplitCompactPlanLayout(compact, MlaSequenceLayout::Bsnd, output);
}
MlaScheduleStatus BuildAutoSplitMixedPlan(const MlaAutoSplitMixedPlanConfig& c, const MlaWindowConfig& window,
                                          const MlaUsedInputs& inputs, MlaAutoSplitMixedPlan& output)
{
    MlaAutoSplitTndPlanConfig compact{};
    const auto status = CompactConfig(c, compact);
    if (status != MlaScheduleStatus::Success)
        return status;
    return BuildAutoSplitCompactPlan(compact, MlaSequenceLayout::Bsnd, window, inputs, output);
}
} // namespace flash_attn::mla
