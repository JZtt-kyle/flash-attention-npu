// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
// v23 ownership from metadata/mla_split_consumer.h; numerical provenance
// stays in the reused Kernel/Block/Tile components.
#pragma once
#include "mla_launch.h"
#include "kernel_operator.h"
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
#include "../kernel/mla_paged_metadata.hpp"
#include "../metadata/mla_split_consumer.h"
#include "../kernel/mla_split_kv.hpp"
#include "../block/mla_split_kv_merge.hpp"
#endif
namespace flash_attn::mla {
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
__aicore__ inline bool LoadSplitPagedBsnd(GM_ADDR metadata, const MlaSplitPagedBsndContext& c,
                                          const MlaPagedMetadataInputs& inputs, const MlaPagedCacheInputs& cache,
                                          MlaSplitPagedBsndSchedule& s, Kernel::MetadataReader& read)
{
    const uint32_t dtype = AscendC::IsSameType<MLA_SPLIT_TYPE, half>::value ? 1 : 2;
    if (!metadata || c.paged.batch > (UINT32_MAX - 414) / 7 || c.paged.dtype != dtype ||
        c.paged.blockDim != AscendC::GetBlockNum() ||
        !ValidPagedMetadataAddress(reinterpret_cast<uint64_t>(metadata), 414 + 7 * c.paged.batch)) {
        AscendC::Trap();
        return false;
    }
    read.data.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(metadata));
    Kernel::PagedMetadataReaders arrays;
    const uint64_t addresses[3] = {inputs.cuQuery.address, inputs.usedQuery.address, inputs.kvLengths.address};
    for (uint32_t i = 0; i < 3; ++i)
        if (addresses[i])
            arrays.data[i].data.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(addresses[i]));
    Kernel::PageTableReader pages;
    if (cache.table.address)
        pages.data.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(cache.table.address));
    if (ReadSplitPagedBsndMetadata(read, 414 + 7 * c.paged.batch, c, inputs, arrays, cache, pages, s) !=
        MlaMetadataReadStatus::Success) {
        AscendC::Trap();
        return false;
    }
    return true;
}
template <typename T>
__aicore__ inline void InitSplitPagedEmpty(GM_ADDR addr, uint64_t elements, T value)
{
    const uint64_t lanes = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
    const uint64_t chunk = (elements + lanes - 1) / lanes;
    const uint64_t begin = uint64_t(AscendC::GetBlockIdx()) * chunk;
    if (begin >= elements)
        return;
    AscendC::GlobalTensor<T> gm;
    gm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(addr));
    matmul::InitOutput<T>(gm[begin], elements - begin < chunk ? elements - begin : chunk, value);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
}
#endif
// Stream order is the global partial -> merge barrier. Every stage rechecks
// every effective page before writes/events, including static empty capacity.
__global__ __aicore__ __mix__(1, 2) void MLA_SPLIT_PARTIAL(uint64_t ffts, GM_ADDR qc, GM_ADDR qr, GM_ADDR workspace,
                                                           GM_ADDR metadata, MlaSplitPagedBsndContext c,
                                                           MlaPagedMetadataInputs inputs, MlaPagedCacheInputs cache,
                                                           MlaBsndInputLayouts layouts, float scale, float softcap)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
    MlaSplitPagedBsndSchedule s{};
    Kernel::MetadataReader read;
    if (!LoadSplitPagedBsnd(metadata, c, inputs, cache, s, read))
        return;
    AscendC::SetSyncBaseAddr(ffts);
    uint32_t core = AscendC::GetBlockIdx();
    if ASCEND_IS_AIV {
        core /= AscendC::GetSubBlockNum();
    }
    auto tiling = s.paged.tiling;
    tiling.usedCoreNum = tiling.blockDim;
    tiling.mBaseSize = s.mBase;
    tiling.s2BaseSize = s.nBase;
    for (uint64_t index = core; index < s.tasks; index += tiling.blockDim) {
        MlaSplitPagedBsndTask task{};
        SplitPagedBsndTask(read, c, s, index, task);
        MlaSplitKvKernelParams partial{};
        partial.window = c.window;
        partial.totalTasks = tiling.blockDim;
        partial.tasks[core] = {task.batch, task.m, task.split, task.s2Begin, task.s2End};
        partial.outOffsetBytes = task.outOffset;
        partial.sumOffsetBytes = task.sumOffset;
        partial.maxOffsetBytes = task.maxOffset;
        AscendC::TPipe pipe;
        Kernel::MlaSplitKvPagedBsnd<MLA_SPLIT_TYPE> op;
        op(tiling, partial, scale, qc, qr, cache, workspace, &pipe, metadata + kMlaPagedMetadataPrefixWords * 4,
           layouts, 0, softcap);
    }
#endif
}
__global__ __aicore__ __mix__(1, 2) void MLA_SPLIT_MERGE(uint64_t ffts, GM_ADDR workspace, GM_ADDR metadata,
                                                         MlaSplitPagedBsndContext c, MlaPagedMetadataInputs inputs,
                                                         MlaPagedCacheInputs cache, GM_ADDR out, GM_ADDR lse,
                                                         GM_ADDR accum, GM_ADDR accumLse, float softcap)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
    MlaSplitPagedBsndSchedule s{};
    Kernel::MetadataReader read;
    if (!LoadSplitPagedBsnd(metadata, c, inputs, cache, s, read))
        return;
    AscendC::SetSyncBaseAddr(ffts);
    if ASCEND_IS_AIV {
        if (c.paged.logicalKvCapacity == 0) {
            const uint64_t rows = uint64_t(c.paged.batch) * c.paged.query.maxLength * c.paged.heads;
            AscendC::TPipe pipe;
            InitSplitPagedEmpty<MLA_SPLIT_TYPE>(out, rows * 512, 0);
            InitSplitPagedEmpty<float>(lse, rows, -float(1.0 / 0.0));
            InitSplitPagedEmpty<float>(accum, rows * c.splits * 512, 0);
            InitSplitPagedEmpty<float>(accumLse, rows * c.splits, -float(1.0 / 0.0));
            return;
        }
        for (uint64_t group = AscendC::GetBlockIdx(); group < s.groups;
             group += c.paged.blockDim * AscendC::GetSubBlockNum()) {
            MlaSplitPagedBsndTask task{};
            SplitPagedBsndTask(read, c, s, group * c.splits, task);
            const uint32_t span = kMlaPagedMetadataPrefixWords + 5 * task.batch;
            MlaSplitKvMergeParams p{};
            p.batchIndex = task.batch;
            p.batch = c.paged.batch;
            p.sq = c.paged.query.maxLength;
            p.heads = c.paged.heads;
            p.queryLength = read(span + 3);
            p.kvLength = read(span + 4);
            p.rowStart = task.rowBegin;
            p.rows = task.rowEnd - task.rowBegin;
            p.mBase = s.mBase;
            p.splits = c.splits;
            p.adapt = 1;
            p.maskKind = c.window.kind;
            p.left = c.window.left;
            p.right = c.window.right;
            for (uint32_t split = 0; split < p.splits; ++split) {
                uint32_t begin, end;
                SplitBsndFragment(p.queryLength, p.kvLength, p.heads, task.m, p.mBase, s.nBase, c.window, p.splits,
                                  split, begin, end);
                p.begin[split] = begin * s.nBase < p.kvLength ? begin * s.nBase : p.kvLength;
                p.end[split] = end * s.nBase < p.kvLength ? end * s.nBase : p.kvLength;
            }
            AscendC::TPipe pipe;
            Block::MlaSplitKvMerge<MLA_SPLIT_TYPE> block;
            block.Init(p, workspace + task.outOffset, workspace + task.sumOffset, workspace + task.maxOffset, out, lse,
                       accum, accumLse, &pipe, softcap);
            const uint32_t tile = block.TileRows();
            for (uint32_t start = 0; start < p.rows; start += tile)
                block.Run(start, tile < p.rows - start ? tile : p.rows - start);
        }
    }
#endif
}
aclError MLA_SPLIT_LAUNCH(const MlaSplitPagedBsndLaunchArgs& args)
{
    const auto& a = args.paged;
    const MlaSplitPagedBsndContext c{a.context, a.window, args.splits, uint32_t(args.softcap > 0)};
    {
        MlaBatchLaunchScope launch;
        MLA_SPLIT_PARTIAL<<<a.context.blockDim, nullptr, a.stream>>>(
            a.ffts, a.qc, a.qr, a.workspace, a.metadata, c, a.inputs, a.cache, a.layouts, a.scale, args.softcap);
        const auto error = launch.Result();
        if (error != ACL_SUCCESS)
            return error;
    }
    MlaBatchLaunchScope launch;
    MLA_SPLIT_MERGE<<<a.context.blockDim, nullptr, a.stream>>>(a.ffts, a.workspace, a.metadata, c, a.inputs, a.cache,
                                                               a.output, a.lse, args.accum, args.accumLse,
                                                               args.softcap);
    return launch.Result();
}
} // namespace flash_attn::mla
