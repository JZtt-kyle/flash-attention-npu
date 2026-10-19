// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
// Source scheduling: CANN ca18aee610729ca5270a44e2d84c751bc687fbf0 SplitCore.
// FA3 pack/resource semantics: d15f1531a460ba456f41b01a774f33ab2db8febf
// hopper/flash_api.cpp, BSD 3-Clause; see LICENSE.fa3.
#pragma once
#include "mla_split_kv.h"
#include "mla_sequence.h"
namespace flash_attn::mla {
struct MlaGqaOptions {
    int64_t packGqa = -1; // -1 = backend default, 0 = separate query heads, 1 = packed GQA.
    int64_t smMargin = 0; // Reserve this many AICs and twice as many AIVs.
};
struct MlaGqaResources {
    MlaPlatformInfo platform;
    uint32_t headGroups, headsPerGroup;
    bool packed;
};
// The NPU default preserves the existing CANN S1G path. G=1 normalizes to
// unpacked. Validate signed options before narrowing/subtraction. Keep the
// physical platform's blockDim mapping for the remaining C1V2 resources.
MlaScheduleStatus ResolveGqaResources(int64_t heads, int64_t kvHeads, const MlaGqaOptions& options,
                                      const MlaPlatformInfo& physical, MlaGqaResources& output);

struct MlaGqaConfig {
    MlaUsedShape shape;
    int64_t kvHeads, valueDim, mBase, nBase, splits;
    int64_t pageSize = 0; // PA uses logical rectangular KV spans; page lookup remains separate.
};
struct MlaGqaTask {
    uint32_t batch, headGroup, kvHead, m, split, splitCount, owner;
    uint32_t rowBegin, rowEnd, s2Begin, s2End;
};
struct MlaGqaRow {
    uint32_t batch, head, kvHead, query;
    uint64_t queryToken, kvStart;
    // Byte offsets into separate FP16/BF16 O, FP32 LSE and public accumulators.
    // UINT64_MAX denotes absent accumulators (splits=1). No workspace address
    // is inferred from these public offsets, or from a virtual batch index.
    uint64_t out, lse, accumulatorOut, accumulatorLse;
};

// Owned CPU scheduling/coordinate protocol, not serialized device metadata.
// S1G within a KV head (packed) or S1 within a query head (unpacked) is purely
// logical: tensors keep their original allocation and independent strides.
// Production v47 maps these coordinates to existing partial/merge numerics.
class MlaGqaSchedule {
  public:
    const MlaGqaResources& Resources() const
    {
        return resources_;
    }
    const MlaUsedLengths& Lengths() const
    {
        return lengths_;
    }
    const std::vector<uint64_t>& GroupPrefix() const
    {
        return prefix_;
    }
    const MlaNativeSplitKvSchedule& Native() const
    {
        return native_;
    }
    uint64_t Groups() const
    {
        return prefix_.empty() ? 0 : prefix_.back();
    }
    uint32_t AccumulatorCapacity() const
    {
        return capacity_;
    }
    // Auto retains native endpoint intersections/owners. Positive splits use
    // the existing mask-dependent FA3 fragment rule and cyclic AIC ownership.
    // Empty KV/masked groups retain explicit initialization tasks; zero Q has none.
    MlaScheduleStatus Task(uint64_t group, uint32_t split, MlaGqaTask& output) const;
    MlaScheduleStatus Row(uint64_t group, uint32_t split, uint32_t localRow, MlaGqaRow& output) const;

  private:
    friend MlaScheduleStatus BuildGqaSchedule(const MlaGqaConfig&, const MlaUsedInputs&, const MlaWindowConfig&,
                                              const MlaGqaOptions&, const MlaPlatformInfo&, MlaGqaSchedule&);
    MlaGqaConfig config_{};
    MlaGqaResources resources_{};
    MlaUsedLengths lengths_;
    MlaWindow window_{};
    MlaNativeSplitKvSchedule native_{}; // Populated only for splits=0.
    std::vector<uint64_t> prefix_;      // One prefix per (batch, headGroup), not per task.
    uint32_t capacity_ = 0;
};
// Reuses allocation/used parsing and source SplitCore; all failures are atomic.
// M/N are caller-selected logical tiles (M=32/64/128/256/512, N=1..8192).
// This does not certify local memory, physical pages or a numerical kernel.
// Two representations: valueDim=512 requires Hkv=1; valueDim=128 permits GQA.
MlaScheduleStatus BuildGqaSchedule(const MlaGqaConfig& config, const MlaUsedInputs& inputs,
                                   const MlaWindowConfig& window, const MlaGqaOptions& options,
                                   const MlaPlatformInfo& physical, MlaGqaSchedule& output);
} // namespace flash_attn::mla
