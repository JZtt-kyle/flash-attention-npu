// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
// Independent CPU/AICore validation of v21. The producer's plan is never trusted.
#pragma once
#include "mla_split_mixed_metadata.h"
#include "mla_split_bsnd_consumer.h"
#if defined(__NPU_ARCH__)
#define MLA_SPLIT_INLINE __aicore__ inline
#else
#define MLA_SPLIT_INLINE inline
#endif
namespace flash_attn::mla {
struct MlaSplitMixedSchedule {
    MlaUsedKernelSchedule used;
    uint64_t groups, tasks, outOffset, sumOffset, maxOffset, bytes;
};
template <typename Reader, typename Arrays>
MLA_SPLIT_INLINE MlaMetadataReadStatus ReadSplitMixedMetadata(const Reader& read, uint32_t words,
                                                              const MlaSplitMixedContext& context,
                                                              const MlaUsedMetadataInputs& inputs, const Arrays& arrays,
                                                              MlaSplitMixedSchedule& output)
{
    using Status = MlaMetadataReadStatus;
    const auto& c = context.used;
    if (c.batch > (UINT32_MAX - 416) / 10)
        return Status::InvalidBuffer;
    if (context.hasSoftcap > 1)
        return Status::ShapeMismatch;
    const uint32_t version = context.hasSoftcap ? kMlaSoftcapSplitMixedVersion : kMlaSplitMixedVersion;
    auto status = ReadMetadataHeader(read, words, 416 + 10 * c.batch, version, version, c.dtype);
    if (status != Status::Success)
        return status;
    if (c.query.layout != 1 || c.kv.layout != 0 || context.splits < (context.hasSoftcap ? 1 : 2) ||
        context.splits > 256)
        return Status::ShapeMismatch;
    const uint32_t base = kMlaUsedMetadataPrefixWords + kMlaUsedMetadataSequenceWords * c.batch;
    const auto& w = context.window;
    if (w.kind > 2 || w.left < -1 || w.right < -1 || w.left > int64_t(c.kv.maxLength) - 1 ||
        w.right > (c.query.maxLength == 0 ? 0 : int64_t(c.query.maxLength) - 1) || read(base) != w.kind ||
        int32_t(read(base + 1)) != w.left || int32_t(read(base + 2)) != w.right || read(base + 3) != context.splits)
        return Status::ShapeMismatch;
    MlaSplitMixedSchedule result{};
    status = ReadUsedMetadata(SplitUsedView<Reader>{read, base}, base, c, inputs, arrays, result.used);
    if (status != Status::Success)
        return status;
    const auto& s = result.used.tiling;
    const uint64_t totalQ = uint64_t(c.query.totalTokens.low) | (uint64_t(c.query.totalTokens.high) << 32);
    // Check the FP32 accumulator even when Q/KV is empty or all used lengths
    // are zero. TND Q output/accumulator strides depend on allocation totalQ.
    const uint64_t strideQ = totalQ ? totalQ : 1;
    if (strideQ * c.heads > uint64_t(INT64_MAX) / (uint64_t(context.splits) * 512 * sizeof(float)))
        return Status::InvalidWorkspace;
    const bool empty = !totalQ || !c.batch || !c.kv.maxLength;
    uint64_t prefix = 0;
    for (uint32_t b = 0; b <= c.batch; ++b) {
        if (ReadMetadataWide(read, base + 4 + 2 * b) != prefix)
            return Status::InvalidEndpoints;
        if (b != c.batch && !empty)
            prefix +=
                (uint64_t(read(kMlaUsedMetadataPrefixWords + 8 * b + 6)) * c.heads + s.mBaseSize - 1) / s.mBaseSize;
    }
    result.groups = prefix;
    result.tasks = prefix * context.splits;
    if (!empty) {
        // Independently bound every allocation partition, including one tail
        // per nonempty segment. Never substitute B*maxQ for compact totalQ.
        const uint64_t active = c.batch < totalQ ? c.batch : totalQ;
        const uint64_t rows = uint64_t(c.query.maxLength) * c.heads;
        const uint64_t perSample = active * ((rows + s.mBaseSize - 1) / s.mBaseSize);
        const uint64_t padded = (totalQ * c.heads + active * (s.mBaseSize - 1)) / s.mBaseSize;
        const uint64_t bound = perSample < padded ? perSample : padded;
        if (prefix > bound)
            return Status::InvalidEndpoints;
        const uint64_t partialRows = bound * context.splits * s.mBaseSize;
        const uint64_t reserve =
            2 * uint64_t(c.aicNum) * (6 * uint64_t(s.mm1ResSize) + 8 * uint64_t(s.mm2ResSize) + 8 * s.mBaseSize);
        result.outOffset = (reserve + 31) / 32 * 32;
        if (partialRows > (uint64_t(INT64_MAX) - result.outOffset) / (528 * sizeof(float)))
            return Status::InvalidWorkspace;
        result.sumOffset = result.outOffset + partialRows * 512 * sizeof(float);
        result.maxOffset = result.sumOffset + partialRows * 8 * sizeof(float);
        result.bytes = result.maxOffset + partialRows * 8 * sizeof(float);
        if (result.bytes > c.workspaceBytes)
            return Status::InvalidWorkspace;
    }
    output = result;
    return Status::Success;
}
// Fragment geometry is layout independent; both paths use original used
// lengths and the same independently checked FA3 ceil-div rule.
} // namespace flash_attn::mla
#undef MLA_SPLIT_INLINE
