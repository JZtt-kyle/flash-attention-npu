// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
// Independent v22 mixed consumer; no producer or CPU plan calls.
#pragma once
#include "mla_auto_split_mixed_metadata.h"
#include "mla_auto_split_tnd_consumer.h"
#if defined(__NPU_ARCH__)
#define MLA_AUTO_MIXED_INLINE __aicore__ inline
#else
#define MLA_AUTO_MIXED_INLINE inline
#endif
namespace flash_attn::mla {
using MlaAutoSplitMixedKernelSchedule = MlaAutoSplitTndKernelSchedule;
using MlaAutoSplitMixedKernelTask = MlaAutoSplitTndKernelTask;
using MlaAutoSplitMixedKernelMerge = MlaAutoSplitTndKernelMerge;
using MlaAutoSplitMixedKernelRow = MlaAutoSplitTndKernelRow;
template <typename Reader, typename Arrays>
MLA_AUTO_MIXED_INLINE MlaMetadataReadStatus ReadAutoSplitMixedMetadata(const Reader& read, uint32_t words,
                                                                       const MlaAutoSplitMixedContext& context,
                                                                       const MlaUsedMetadataInputs& inputs,
                                                                       const Arrays& arrays,
                                                                       MlaAutoSplitMixedKernelSchedule& output)
{
    return ReadAutoSplitMetadata<true, kMlaAutoSplitMixedVersion, false, kMlaSoftcapAutoSplitMixedVersion>(
        read, words, context, inputs, arrays, output);
}
// Queries require successful validation and unchanged record/context. Shared
// compact-Q queries load both 64-bit origins verbatim; the KV stride is supplied
// by the caller. Mask lengths and native endpoints stay per sample and per M.
template <typename Reader>
MLA_AUTO_MIXED_INLINE bool AutoSplitMixedTask(const Reader& read, const MlaAutoSplitMixedKernelSchedule& s,
                                              const MlaWindowKernelParams& w, uint64_t group, uint32_t split,
                                              MlaAutoSplitMixedKernelTask& output)
{
    return AutoSplitTndTask(read, s, w, group, split, output);
}
template <typename Reader>
MLA_AUTO_MIXED_INLINE bool AutoSplitMixedMerge(const Reader& read, const MlaAutoSplitMixedKernelSchedule& s,
                                               uint32_t head, uint32_t part, MlaAutoSplitMixedKernelMerge& output)
{
    return AutoSplitTndMerge(read, s, head, part, output);
}
MLA_AUTO_MIXED_INLINE bool AutoSplitMixedRow(const MlaAutoSplitMixedKernelSchedule& s,
                                             const MlaAutoSplitMixedKernelTask& task, uint32_t row,
                                             MlaAutoSplitMixedKernelRow& output)
{
    return AutoSplitTndRow(s, task, row, output);
}
} // namespace flash_attn::mla
#undef MLA_AUTO_MIXED_INLINE
