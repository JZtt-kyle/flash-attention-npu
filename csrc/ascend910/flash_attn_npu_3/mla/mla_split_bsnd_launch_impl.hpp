// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
// Explicit ownership/persistent storage: mla_split_kv_plan.h. Numerical source
// provenance remains in the partial and merge Kernel/Block/Tile components.
#pragma once
#include "mla_split_bsnd_launch.h"
#include "kernel_operator.h"
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
#include "kernel/mla_used_metadata.hpp"
#include "mla_split_bsnd_consumer.h"
#include "kernel/mla_split_kv.hpp"
#include "block/mla_split_kv_merge.hpp"
#endif
namespace flash_attn::mla {
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
__aicore__ inline bool LoadSplitBsnd(GM_ADDR metadata, const MlaSplitBsndContext& c,
                                     const MlaUsedMetadataInputs& inputs, MlaSplitBsndSchedule& s,
                                     Kernel::MetadataReader& read)
{
    const uint32_t dtype = AscendC::IsSameType<MLA_SPLIT_TYPE, half>::value ? uint32_t(MlaMetadataDtype::Fp16)
                                                                            : uint32_t(MlaMetadataDtype::Bf16);
    if (!metadata || c.used.batch > (UINT32_MAX - 416) / 10 || c.used.dtype != dtype ||
        c.used.blockDim != AscendC::GetBlockNum() ||
        !ValidUsedMetadataAddress(reinterpret_cast<uint64_t>(metadata), 416 + 10 * c.used.batch)) {
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
    if (ReadSplitBsndMetadata(read, 416 + 10 * c.used.batch, c, inputs, arrays, s) != MlaMetadataReadStatus::Success) {
        AscendC::Trap();
        return false;
    }
    return true;
}
__aicore__ inline uint32_t SplitBatch(const Kernel::MetadataReader& read, uint32_t batch, uint64_t group, uint32_t& m)
{
    const uint32_t base = 414 + 8 * batch;
    uint32_t lo = 0, hi = batch;
    while (lo < hi) {
        const uint32_t mid = lo + (hi - lo) / 2;
        if (ReadMetadataWide(read, base + 2 * (mid + 1)) <= group)
            lo = mid + 1;
        else
            hi = mid;
    }
    m = uint32_t(group - ReadMetadataWide(read, base + 2 * lo));
    return lo;
}
#endif
// Separate stream-ordered kernels provide the global partial->merge barrier.
// Each core has its own scratch and each task has a persistent raw slot.
__global__ __aicore__ __mix__(1, 2) void MLA_SPLIT_PARTIAL(uint64_t ffts, GM_ADDR qc, GM_ADDR qr, GM_ADDR kv,
                                                           GM_ADDR kr, GM_ADDR workspace, GM_ADDR metadata,
                                                           MlaSplitBsndContext c, MlaUsedMetadataInputs inputs,
                                                           float scale, MlaBsndInputLayouts layouts, float softcap)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
    MlaSplitBsndSchedule s{};
    Kernel::MetadataReader read;
    if (!LoadSplitBsnd(metadata, c, inputs, s, read))
        return;
    AscendC::SetSyncBaseAddr(ffts);
    uint32_t core = AscendC::GetBlockIdx();
    if ASCEND_IS_AIV {
        core /= AscendC::GetSubBlockNum();
    }
    auto tiling = s.used.tiling;
    tiling.usedCoreNum = tiling.blockDim;
    for (uint64_t task = core; task < s.tasks; task += tiling.blockDim) {
        MlaSplitKvKernelParams partial{};
        partial.window = c.window;
        partial.totalTasks = tiling.blockDim;
        auto& fragment = partial.tasks[core];
        fragment.batch = SplitBatch(read, c.used.batch, task / c.splits, fragment.m);
        fragment.split = task % c.splits;
        const uint32_t span = kMlaUsedMetadataPrefixWords + 8 * fragment.batch;
        SplitBsndFragment(read(span + 6), read(span + 7), tiling.heads, fragment.m, tiling.mBaseSize, tiling.s2BaseSize,
                          c.window, c.splits, fragment.split, fragment.s2Begin, fragment.s2End);
        const uint64_t row = (task - core) * tiling.mBaseSize;
        partial.outOffsetBytes = s.outOffset + row * 512 * 4;
        partial.sumOffsetBytes = s.sumOffset + row * 8 * 4;
        partial.maxOffsetBytes = s.maxOffset + row * 8 * 4;
        // The ring drains all events before destruction/reuse of local storage.
        AscendC::TPipe pipe;
        Kernel::MlaSplitKvUsedBsnd<MLA_SPLIT_TYPE> op;
        op(tiling, partial, scale, qc, qr, kv, kr, workspace, &pipe, metadata + kMlaUsedMetadataPrefixWords * 4,
           layouts, softcap);
    }
#endif
}
__global__ __aicore__ __mix__(1, 2) void MLA_SPLIT_MERGE(uint64_t ffts, GM_ADDR workspace, GM_ADDR metadata,
                                                         MlaSplitBsndContext c, MlaUsedMetadataInputs inputs,
                                                         GM_ADDR out, GM_ADDR lse, GM_ADDR accum, GM_ADDR accumLse,
                                                         float softcap)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
    MlaSplitBsndSchedule s{};
    Kernel::MetadataReader read;
    if (!LoadSplitBsnd(metadata, c, inputs, s, read))
        return;
    AscendC::SetSyncBaseAddr(ffts);
    if ASCEND_IS_AIV {
        const auto& t = s.used.tiling;
        for (uint64_t group = AscendC::GetBlockIdx(); group < s.groups;
             group += t.blockDim * AscendC::GetSubBlockNum()) {
            MlaSplitKvMergeParams p{};
            uint32_t m;
            p.batchIndex = SplitBatch(read, c.used.batch, group, m);
            const uint32_t span = kMlaUsedMetadataPrefixWords + 8 * p.batchIndex;
            p.batch = t.batch;
            p.sq = t.sq;
            p.heads = t.heads;
            p.queryLength = read(span + 6);
            p.kvLength = read(span + 7);
            p.rowStart = m * t.mBaseSize;
            p.rows = (t.mBaseSize < p.queryLength * t.heads - p.rowStart ? t.mBaseSize
                                                                         : p.queryLength * t.heads - p.rowStart);
            p.mBase = t.mBaseSize;
            p.splits = c.splits;
            p.adapt = 1;
            p.maskKind = c.window.kind;
            p.left = c.window.left;
            p.right = c.window.right;
            for (uint32_t split = 0; split < p.splits; ++split) {
                uint32_t begin, end;
                SplitBsndFragment(p.queryLength, p.kvLength, p.heads, m, p.mBase, t.s2BaseSize, c.window, p.splits,
                                  split, begin, end);
                p.begin[split] = (begin * t.s2BaseSize < p.kvLength ? begin * t.s2BaseSize : p.kvLength);
                p.end[split] = (end * t.s2BaseSize < p.kvLength ? end * t.s2BaseSize : p.kvLength);
            }
            const uint64_t row = group * p.splits * p.mBase;
            AscendC::TPipe pipe;
            Block::MlaSplitKvMerge<MLA_SPLIT_TYPE> block;
            block.Init(p, workspace + s.outOffset + row * 512 * 4, workspace + s.sumOffset + row * 8 * 4,
                       workspace + s.maxOffset + row * 8 * 4, out, lse, accum, accumLse, &pipe, softcap);
            const uint32_t tile = block.TileRows();
            for (uint32_t start = 0; start < p.rows; start += tile)
                block.Run(start, (tile < p.rows - start ? tile : p.rows - start));
        }
    }
#endif
}
aclError MLA_SPLIT_LAUNCH(const MlaSplitBsndLaunchArgs& args)
{
    const auto& a = args.used;
    MlaSplitBsndContext c{a.context, a.window, args.splits, uint32_t(a.softcap > 0)};
    {
        MlaBatchLaunchScope launch;
        MLA_SPLIT_PARTIAL<<<a.context.blockDim, nullptr, a.stream>>>(
            a.ffts, a.qc, a.qr, a.kv, a.kr, a.workspace, a.metadata, c, a.inputs, a.scale, a.layouts, a.softcap);
        const auto error = launch.Result();
        if (error != ACL_SUCCESS)
            return error;
    }
    MlaBatchLaunchScope launch;
    MLA_SPLIT_MERGE<<<a.context.blockDim, nullptr, a.stream>>>(a.ffts, a.workspace, a.metadata, c, a.inputs, a.output,
                                                               a.lse, args.accum, args.accumLse, a.softcap);
    return launch.Result();
}
} // namespace flash_attn::mla
