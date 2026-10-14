// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
// Native AIC/Vector endpoints: mla_auto_split_kv_plan.h. Numerical provenance
// remains in the reused partial and merge Kernel/Block/Tile components.
#pragma once
#include "mla_auto_split_bsnd_launch.h"
#include "mla_auto_split_tnd_launch.h"
#include "kernel_operator.h"
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
#include "kernel/mla_used_metadata.hpp"
#include "mla_auto_split_tnd_consumer.h"
#include "kernel/mla_split_kv.hpp"
#include "block/mla_split_kv_merge.hpp"
#endif
namespace flash_attn::mla {
namespace {
#if MLA_AUTO_TND
using AutoContext = MlaAutoSplitTndContext;
using AutoLaunchArgs = MlaSplitTndLaunchArgs;
#else
using AutoContext = MlaAutoSplitBsndContext;
using AutoLaunchArgs = MlaSplitBsndLaunchArgs;
#endif
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
__aicore__ inline bool LoadAutoSplit(GM_ADDR metadata, const AutoContext& c, const MlaUsedMetadataInputs& inputs,
                                     MlaAutoSplitBsndSchedule& s, Kernel::MetadataReader& read)
{
    const uint32_t dtype = AscendC::IsSameType<MLA_AUTO_TYPE, half>::value ? uint32_t(MlaMetadataDtype::Fp16)
                                                                           : uint32_t(MlaMetadataDtype::Bf16);
    if (!metadata || c.used.batch > (UINT32_MAX - kMlaAutoSplitFixedWords) / 10 || c.used.dtype != dtype ||
        c.used.blockDim != AscendC::GetBlockNum() ||
        !ValidUsedMetadataAddress(reinterpret_cast<uint64_t>(metadata), kMlaAutoSplitFixedWords + 10 * c.used.batch)) {
        AscendC::Trap();
        return false;
    }
    read.data.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(metadata));
    Kernel::UsedMetadataReaders arrays;
    const uint64_t addresses[4] = {inputs.cuQuery.address, inputs.cuKv.address, inputs.usedQuery.address,
                                   inputs.usedKv.address};
    for (uint32_t i = 0; i < 4; ++i)
        if (addresses[i])
            arrays.data[i].data.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(addresses[i]));
    if (ReadAutoSplitMetadata<MLA_AUTO_TND, MLA_AUTO_TND ? kMlaAutoSplitTndVersion : kMlaAutoSplitBsndVersion>(
            read, kMlaAutoSplitFixedWords + 10 * c.used.batch, c, inputs, arrays, s) !=
        MlaMetadataReadStatus::Success) {
        AscendC::Trap();
        return false;
    }
    // Direct M blocks reuse one extra raw slot per physical AIC. This region
    // follows all persistent FD slots, so later direct work cannot overwrite FD.
    if (s.groups && s.bytes + uint64_t(s.aicNum) * s.used.tiling.mBaseSize * 528 * 4 > c.used.workspaceBytes) {
        AscendC::Trap();
        return false;
    }
    return true;
}
template <typename T>
__aicore__ inline void AutoInitialize(GM_ADDR address, uint64_t elements, T value)
{
    const uint64_t cores = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
    const uint64_t chunk = (elements + cores * 64 - 1) / (cores * 64) * 64;
    const uint64_t start = uint64_t(AscendC::GetBlockIdx()) * chunk;
    if (start < elements) {
        AscendC::GlobalTensor<T> tensor;
        tensor.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(address));
        matmul::InitOutput<T>(tensor[start], chunk < elements - start ? chunk : elements - start, value);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    }
}
__aicore__ inline MlaSplitKvMergeParams AutoMergeParams(const Kernel::MetadataReader& read,
                                                        const MlaAutoSplitBsndSchedule& s,
                                                        const MlaWindowKernelParams& w, const MlaAutoSplitBsndTask& t)
{
    MlaSplitKvMergeParams p{};
    const auto& a = s.used.tiling;
    p.batch = a.batch;
    p.sq = a.sq;
    p.heads = a.heads;
    p.batchIndex = t.batch;
    p.rowStart = t.rowBegin;
    p.rows = t.rowEnd - t.rowBegin;
    p.mBase = a.mBaseSize;
    p.splits = t.splitCount;
    p.adapt = 1;
    p.queryLength = read(kMlaUsedMetadataPrefixWords + 8 * t.batch + 6);
    p.kvLength = read(kMlaUsedMetadataPrefixWords + 8 * t.batch + 7);
    if constexpr (MLA_AUTO_TND) {
        p.batch = 1;
        p.sq = s.used.query.totalTokens.low;
        p.batchIndex = read(kMlaUsedMetadataPrefixWords + 8 * t.batch);
    }
    p.maskKind = w.kind;
    p.left = w.left;
    p.right = w.right;
    for (uint32_t split = 0; split < p.splits; ++split) {
        MlaAutoSplitBsndTask fragment;
        AutoSplitBsndTask(read, s, w, t.group, split, fragment);
        p.begin[split] = fragment.kvBegin;
        p.end[split] = fragment.kvEnd;
    }
    return p;
}
#endif
} // namespace
// Stream ordering supplies the initialization and partial->FD barriers. Every
// stage checks its layout-specific record before any output access, including globally empty inputs.
__global__ __aicore__ __mix__(1, 2) void MLA_AUTO_INIT(GM_ADDR metadata, AutoContext c, MlaUsedMetadataInputs inputs,
                                                       GM_ADDR out, GM_ADDR lse, GM_ADDR accum, GM_ADDR accumLse)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
    MlaAutoSplitBsndSchedule s{};
    Kernel::MetadataReader read;
    if (!LoadAutoSplit(metadata, c, inputs, s, read))
        return;
    if ASCEND_IS_AIV {
        AscendC::TPipe pipe;
        const uint64_t rows = uint64_t(c.used.heads) * UsedMetadataTotal(c.used.query);
        AutoInitialize<MLA_AUTO_TYPE>(out, rows * 512, MLA_AUTO_TYPE(0));
        AutoInitialize<float>(lse, rows, UsedMetadataTotal(c.used.kv) ? -float(1.0 / 0.0) : float(1.0 / 0.0));
        AutoInitialize<float>(accum, rows * kMlaAutoSplitCapacity * 512, 0.0f);
        AutoInitialize<float>(accumLse, rows * kMlaAutoSplitCapacity, -float(1.0 / 0.0));
    }
#endif
}
__global__ __aicore__ __mix__(1, 2) void MLA_AUTO_PARTIAL(uint64_t ffts, GM_ADDR qc, GM_ADDR qr, GM_ADDR kv, GM_ADDR kr,
                                                          GM_ADDR workspace, GM_ADDR metadata, AutoContext c,
                                                          MlaUsedMetadataInputs inputs, float scale,
                                                          MlaBsndInputLayouts layouts, GM_ADDR out, GM_ADDR lse,
                                                          GM_ADDR accum, GM_ADDR accumLse)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
    MlaAutoSplitBsndSchedule s{};
    Kernel::MetadataReader read;
    if (!LoadAutoSplit(metadata, c, inputs, s, read) || !s.groups)
        return;
    AscendC::SetSyncBaseAddr(ffts);
    uint32_t core = AscendC::GetBlockIdx();
    if ASCEND_IS_AIV {
        core /= AscendC::GetSubBlockNum();
    }
    if (core >= s.usedCoreNum)
        return;
    auto tiling = s.used.tiling;
    tiling.usedCoreNum = tiling.blockDim;
    const uint32_t prefix = auto_split_detail::Base(c.used.batch) + kMlaAutoSplitTailPrefix;
    const uint32_t b = core ? s.bEnd[core - 1] : 0;
    const uint64_t first = ReadMetadataWide(read, prefix + 2 * b) + (core ? s.mEnd[core - 1] : 0);
    uint64_t last = ReadMetadataWide(read, prefix + 2 * s.bEnd[core]) + s.mEnd[core] + (s.nEnd[core] ? 1 : 0);
    if (last > s.groups)
        last = s.groups;
    for (uint64_t group = first; group < last; ++group) {
        MlaAutoSplitBsndTask t{};
        if (!AutoSplitBsndTask(read, s, c.window, group, 0, t) || t.kind == 2)
            continue;
        uint32_t split = 0;
        while (t.owner != core && ++split < t.splitCount)
            AutoSplitBsndTask(read, s, c.window, group, split, t);
        if (t.owner != core)
            continue;
        MlaSplitKvKernelParams p{};
        p.window = c.window;
        p.totalTasks = tiling.blockDim;
        p.tasks[core] = {t.batch, t.m, t.split, t.s2Begin, t.s2End};
        const uint64_t coreRow = uint64_t(core) * tiling.mBaseSize;
        if (t.kind == 1) {
            p.outOffsetBytes = t.outOffset;
            p.sumOffsetBytes = t.sumOffset;
            p.maxOffsetBytes = t.maxOffset;
        } else {
            p.outOffsetBytes = s.bytes;
            p.sumOffsetBytes = p.outOffsetBytes + uint64_t(s.aicNum) * tiling.mBaseSize * 512 * 4;
            p.maxOffsetBytes = p.sumOffsetBytes + uint64_t(s.aicNum) * tiling.mBaseSize * 8 * 4;
            p.outOffsetBytes += coreRow * 512 * 4;
            p.sumOffsetBytes += coreRow * 8 * 4;
            p.maxOffsetBytes += coreRow * 8 * 4;
        }
        {
            AscendC::TPipe pipe;
            if constexpr (MLA_AUTO_TND) {
                Kernel::MlaSplitKvUsedTnd<MLA_AUTO_TYPE> op;
                op(tiling, p, scale, qc, qr, kv, kr, workspace, &pipe, metadata + kMlaUsedMetadataPrefixWords * 4,
                   layouts, c.used.query.totalTokens.low);
            } else {
                Kernel::MlaSplitKvUsedBsnd<MLA_AUTO_TYPE, true> op;
                op(tiling, p, scale, qc, qr, kv, kr, workspace, &pipe, metadata + kMlaUsedMetadataPrefixWords * 4,
                   layouts);
            }
        }
        if ASCEND_IS_AIV {
            if (t.kind == 0) {
                auto merge = AutoMergeParams(read, s, c.window, t);
                // Match the source's 16-row aligned C1V2 ownership. Each vector
                // reads only its completed raw rows; no cross-vector handoff.
                const uint32_t firstRows = merge.rows <= 16 ? merge.rows : ((merge.rows + 15) / 16 + 1) / 2 * 16;
                const uint32_t begin = AscendC::GetBlockIdx() % 2 ? firstRows : 0;
                const uint32_t end = AscendC::GetBlockIdx() % 2 ? merge.rows : firstRows;
                AscendC::TPipe pipe;
                Block::MlaSplitKvMerge<MLA_AUTO_TYPE, true, MLA_AUTO_TND> block;
                block.Init(merge, workspace + p.outOffsetBytes, workspace + p.sumOffsetBytes,
                           workspace + p.maxOffsetBytes, out, lse, accum, accumLse, &pipe);
                for (uint32_t row = begin; row < end; row += 8)
                    block.Run(row, end - row < 8 ? end - row : 8);
            }
        }
    }
#endif
}
__global__ __aicore__ __mix__(1, 2) void MLA_AUTO_MERGE(GM_ADDR workspace, GM_ADDR metadata, AutoContext c,
                                                        MlaUsedMetadataInputs inputs, GM_ADDR out, GM_ADDR lse,
                                                        GM_ADDR accum, GM_ADDR accumLse)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
    MlaAutoSplitBsndSchedule s{};
    Kernel::MetadataReader read;
    if (!LoadAutoSplit(metadata, c, inputs, s, read))
        return;
    if ASCEND_IS_AIV {
        const uint32_t owner = AscendC::GetBlockIdx();
        if (owner >= s.vectors)
            return;
        for (uint32_t h = 0; h < s.fdHeads; ++h) {
            const uint32_t parts = read(auto_split_detail::Base(c.used.batch) + 190 + h);
            for (uint32_t part = 0; part < parts; ++part) {
                MlaAutoSplitBsndMerge r{};
                AutoSplitBsndMerge(read, s, h, part, r);
                if (r.owner != owner)
                    continue;
                MlaAutoSplitBsndTask t{};
                AutoSplitBsndTask(read, s, c.window, r.group, 0, t);
                auto p = AutoMergeParams(read, s, c.window, t);
                p.rowStart = r.rowBegin;
                p.rows = r.rowEnd - r.rowBegin;
                AscendC::TPipe pipe;
                Block::MlaSplitKvMerge<MLA_AUTO_TYPE, false, MLA_AUTO_TND> block;
                block.Init(p, workspace + r.outOffset, workspace + r.sumOffset, workspace + r.maxOffset, out, lse,
                           accum, accumLse, &pipe);
                const uint32_t tile = block.TileRows();
                for (uint32_t row = 0; row < p.rows; row += tile)
                    block.Run(row, p.rows - row < tile ? p.rows - row : tile);
            }
        }
    }
#endif
}
aclError MLA_AUTO_LAUNCH(const AutoLaunchArgs& args)
{
    const auto& a = args.used;
    AutoContext c{a.context, a.window};
    {
        MlaBatchLaunchScope launch;
        MLA_AUTO_INIT<<<a.context.blockDim, nullptr, a.stream>>>(a.metadata, c, a.inputs, a.output, a.lse, args.accum,
                                                                 args.accumLse);
        const auto error = launch.Result();
        if (error != ACL_SUCCESS)
            return error;
    }
    if ((!a.context.query.totalTokens.low && !a.context.query.totalTokens.high) ||
        (!a.context.kv.totalTokens.low && !a.context.kv.totalTokens.high))
        return ACL_SUCCESS;
    {
        MlaBatchLaunchScope launch;
        MLA_AUTO_PARTIAL<<<a.context.blockDim, nullptr, a.stream>>>(a.ffts, a.qc, a.qr, a.kv, a.kr, a.workspace,
                                                                    a.metadata, c, a.inputs, a.scale, a.layouts,
                                                                    a.output, a.lse, args.accum, args.accumLse);
        const auto error = launch.Result();
        if (error != ACL_SUCCESS)
            return error;
    }
    MlaBatchLaunchScope launch;
    MLA_AUTO_MERGE<<<a.context.blockDim, nullptr, a.stream>>>(a.workspace, a.metadata, c, a.inputs, a.output, a.lse,
                                                              args.accum, args.accumLse);
    return launch.Result();
}
} // namespace flash_attn::mla
