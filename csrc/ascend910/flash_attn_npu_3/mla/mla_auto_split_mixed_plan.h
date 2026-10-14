// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
// CANN ca18aee610729ca5270a44e2d84c751bc687fbf0 native AIC/FD partition;
// FA3 d15f1531a460ba456f41b01a774f33ab2db8febf accumulator convention.
// FA3 portions are BSD 3-Clause; see LICENSE.fa3.
#pragma once
#include "mla_auto_split_tnd_plan.h"
#include "mla_tnd_bsnd_schedule.h"

namespace flash_attn::mla {
struct MlaAutoSplitMixedPlanConfig {
    MlaTndBsndShape shape;
    int64_t mBaseSize, s2BaseSize, aicNum, cvRatio;
    uint64_t libApiBytes;
};
// The owned compact-Q plan and queries are layout independent after parsing.
// Mixed builders parse BSND KV origins as uint64(batch)*capacity, never as a
// TND int32 prefix. Only Q has an allocation prefix; optional used values do
// not move either input's physical origin. Each input applies its own stride.
using MlaAutoSplitMixedPlan = MlaAutoSplitTndPlan;
using MlaAutoSplitMixedTask = MlaAutoSplitTndTask;
// Static budget uses compact totalQ, at most 2*AIC FD slots and
// [AIC,H,totalQ,512] / [AIC,H,totalQ] FP32 accumulator. Empty capacity validates
// all descriptors and has no tasks/workspace. All failures are atomic.
MlaScheduleStatus BuildAutoSplitMixedPlanLayout(const MlaAutoSplitMixedPlanConfig&, MlaAutoSplitKvLayout&);
MlaScheduleStatus BuildAutoSplitMixedPlan(const MlaAutoSplitMixedPlanConfig&, const MlaWindowConfig&,
                                          const MlaUsedInputs&, MlaAutoSplitMixedPlan&);
} // namespace flash_attn::mla
