// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
// Layout variants share the dispatch implementation, as in fwd_dispatch_impl.hpp.
// autogen/ selects one dtype/layout per compiler unit; numerical layers are reused.
#pragma once
#include "kernel_operator.h"
#include "mla_dispatch.h"
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
#include "mla_kernel.hpp"
#include "mla_split_metadata_consumer.h"
#include "mla_split_kv_kernel.hpp"
#include "mla_split_kv_merge_block.hpp"
#endif

#include <type_traits>
namespace flash_attn::mla {
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
template <typename DType>
__aicore__ inline bool LoadSplitBsnd(GM_ADDR metadata, const MlaSplitBsndContext& c,
                                     const MlaUsedMetadataInputs& inputs, MlaSplitBsndSchedule& s,
                                     Kernel::MetadataReader& read)
{
    const uint32_t dtype =
        AscendC::IsSameType<DType, half>::value ? uint32_t(MlaMetadataDtype::Fp16) : uint32_t(MlaMetadataDtype::Bf16);
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
template <typename DType>
__global__ __aicore__ __mix__(1, 2) void SplitBsndPartial(uint64_t ffts, GM_ADDR qc, GM_ADDR qr, GM_ADDR kv, GM_ADDR kr,
                                                          GM_ADDR workspace, GM_ADDR metadata, MlaSplitBsndContext c,
                                                          MlaUsedMetadataInputs inputs, float scale,
                                                          MlaBsndInputLayouts layouts, float softcap)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
    MlaSplitBsndSchedule s{};
    Kernel::MetadataReader read;
    if (!LoadSplitBsnd<DType>(metadata, c, inputs, s, read))
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
        Kernel::MlaSplitKvUsedBsnd<DType> op;
        op(tiling, partial, scale, qc, qr, kv, kr, workspace, &pipe, metadata + kMlaUsedMetadataPrefixWords * 4,
           layouts, softcap);
    }
#endif
}
template <typename DType>
__global__ __aicore__ __mix__(1, 2) void SplitBsndMerge(uint64_t ffts, GM_ADDR workspace, GM_ADDR metadata,
                                                        MlaSplitBsndContext c, MlaUsedMetadataInputs inputs,
                                                        GM_ADDR out, GM_ADDR lse, GM_ADDR accum, GM_ADDR accumLse,
                                                        float softcap)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
    MlaSplitBsndSchedule s{};
    Kernel::MetadataReader read;
    if (!LoadSplitBsnd<DType>(metadata, c, inputs, s, read))
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
            Block::MlaSplitKvMerge<DType> block;
            block.Init(p, workspace + s.outOffset + row * 512 * 4, workspace + s.sumOffset + row * 8 * 4,
                       workspace + s.maxOffset + row * 8 * 4, out, lse, accum, accumLse, &pipe, softcap);
            const uint32_t tile = block.TileRows();
            for (uint32_t start = 0; start < p.rows; start += tile)
                block.Run(start, (tile < p.rows - start ? tile : p.rows - start));
        }
    }
#endif
}
template <typename DType>
aclError LaunchSplitBsnd(const MlaSplitBsndLaunchArgs& args)
{
    const auto& a = args.used;
    MlaSplitBsndContext c{a.context, a.window, args.splits, uint32_t(a.softcap > 0)};
    {
        MlaBatchLaunchScope launch;
        SplitBsndPartial<DType><<<a.context.blockDim, nullptr, a.stream>>>(
            a.ffts, a.qc, a.qr, a.kv, a.kr, a.workspace, a.metadata, c, a.inputs, a.scale, a.layouts, a.softcap);
        const auto error = launch.Result();
        if (error != ACL_SUCCESS)
            return error;
    }
    MlaBatchLaunchScope launch;
    SplitBsndMerge<DType><<<a.context.blockDim, nullptr, a.stream>>>(
        a.ffts, a.workspace, a.metadata, c, a.inputs, a.output, a.lse, args.accum, args.accumLse, a.softcap);
    return launch.Result();
}
} // namespace flash_attn::mla

namespace flash_attn::mla {

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
template <typename DType, bool IsMixed>
__aicore__ inline bool
LoadSplitTnd(GM_ADDR metadata, const std::conditional_t<IsMixed, MlaSplitMixedContext, MlaSplitTndContext>& c,
             const MlaUsedMetadataInputs& inputs,
             std::conditional_t<IsMixed, MlaSplitMixedSchedule, MlaSplitTndSchedule>& s, Kernel::MetadataReader& read)
{
    const uint32_t dtype =
        AscendC::IsSameType<DType, half>::value ? uint32_t(MlaMetadataDtype::Fp16) : uint32_t(MlaMetadataDtype::Bf16);
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
    MlaMetadataReadStatus status;
    if constexpr (IsMixed)
        status = ReadSplitMixedMetadata(read, 416 + 10 * c.used.batch, c, inputs, arrays, s);
    else
        status = ReadSplitTndMetadata(read, 416 + 10 * c.used.batch, c, inputs, arrays, s);
    if (status != MlaMetadataReadStatus::Success) {
        AscendC::Trap();
        return false;
    }
    return true;
}
__aicore__ inline uint32_t SplitTndBatch(const Kernel::MetadataReader& read, uint32_t batch, uint64_t group,
                                         uint32_t& m)
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
template <typename DType, bool IsMixed>
__global__ __aicore__ __mix__(1, 2) void SplitTndPartial(
    uint64_t ffts, GM_ADDR qc, GM_ADDR qr, GM_ADDR kv, GM_ADDR kr, GM_ADDR workspace, GM_ADDR metadata,
    std::conditional_t<IsMixed, MlaSplitMixedContext, MlaSplitTndContext> c, MlaUsedMetadataInputs inputs, float scale,
    MlaBsndInputLayouts layouts, float softcap)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
    std::conditional_t<IsMixed, MlaSplitMixedSchedule, MlaSplitTndSchedule> s{};
    Kernel::MetadataReader read;
    if (!LoadSplitTnd<DType, IsMixed>(metadata, c, inputs, s, read))
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
        fragment.batch = SplitTndBatch(read, c.used.batch, task / c.splits, fragment.m);
        fragment.split = task % c.splits;
        const uint32_t span = kMlaUsedMetadataPrefixWords + 8 * fragment.batch;
        SplitBsndFragment(read(span + 6), read(span + 7), tiling.heads, fragment.m, tiling.mBaseSize, tiling.s2BaseSize,
                          c.window, c.splits, fragment.split, fragment.s2Begin, fragment.s2End);
        const uint64_t row = task * tiling.mBaseSize;
        partial.outOffsetBytes = s.outOffset + row * 512 * 4;
        partial.sumOffsetBytes = s.sumOffset + row * 8 * 4;
        partial.maxOffsetBytes = s.maxOffset + row * 8 * 4;
        // The ring drains all events before destruction/reuse of local storage.
        AscendC::TPipe pipe;
        Kernel::MlaSplitKvUsedTnd<DType> op;
        op(tiling, partial, scale, qc, qr, kv, kr, workspace, &pipe, metadata + kMlaUsedMetadataPrefixWords * 4,
           layouts, c.used.query.totalTokens.low, c.used.kv.layout == uint32_t(MlaSequenceLayout::Tnd), softcap);
    }
#endif
}
template <typename DType, bool IsMixed>
__global__
    __aicore__ __mix__(1, 2) void SplitTndMerge(uint64_t ffts, GM_ADDR workspace, GM_ADDR metadata,
                                                std::conditional_t<IsMixed, MlaSplitMixedContext, MlaSplitTndContext> c,
                                                MlaUsedMetadataInputs inputs, GM_ADDR out, GM_ADDR lse, GM_ADDR accum,
                                                GM_ADDR accumLse, float softcap)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
    std::conditional_t<IsMixed, MlaSplitMixedSchedule, MlaSplitTndSchedule> s{};
    Kernel::MetadataReader read;
    if (!LoadSplitTnd<DType, IsMixed>(metadata, c, inputs, s, read))
        return;
    AscendC::SetSyncBaseAddr(ffts);
    if ASCEND_IS_AIV {
        const auto& t = s.used.tiling;
        for (uint64_t group = AscendC::GetBlockIdx(); group < s.groups;
             group += t.blockDim * AscendC::GetSubBlockNum()) {
            MlaSplitKvMergeParams p{};
            uint32_t m;
            p.batchIndex = SplitTndBatch(read, c.used.batch, group, m);
            const uint32_t span = kMlaUsedMetadataPrefixWords + 8 * p.batchIndex;
            // Merge's TND specialization interprets batchIndex as original
            // allocation token and sq as totalQ, never the sample max/used.
            p.batchIndex = read(span);
            p.batch = 1;
            p.sq = c.used.query.totalTokens.low;
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
            Block::MlaSplitKvMerge<DType, false, true> block;
            block.Init(p, workspace + s.outOffset + row * 512 * 4, workspace + s.sumOffset + row * 8 * 4,
                       workspace + s.maxOffset + row * 8 * 4, out, lse, accum, accumLse, &pipe, softcap);
            const uint32_t tile = block.TileRows();
            for (uint32_t start = 0; start < p.rows; start += tile)
                block.Run(start, (tile < p.rows - start ? tile : p.rows - start));
        }
    }
#endif
}
template <typename DType, bool IsMixed>
aclError LaunchSplitTnd(const std::conditional_t<IsMixed, MlaSplitMixedLaunchArgs, MlaSplitTndLaunchArgs>& args)
{
    const auto& a = args.used;
    std::conditional_t<IsMixed, MlaSplitMixedContext, MlaSplitTndContext> c{a.context, a.window, args.splits};
    c.hasSoftcap = a.softcap > 0;
    {
        MlaBatchLaunchScope launch;
        SplitTndPartial<DType, IsMixed><<<a.context.blockDim, nullptr, a.stream>>>(
            a.ffts, a.qc, a.qr, a.kv, a.kr, a.workspace, a.metadata, c, a.inputs, a.scale, a.layouts, a.softcap);
        const auto error = launch.Result();
        if (error != ACL_SUCCESS)
            return error;
    }
    MlaBatchLaunchScope launch;
    SplitTndMerge<DType, IsMixed><<<a.context.blockDim, nullptr, a.stream>>>(
        a.ffts, a.workspace, a.metadata, c, a.inputs, a.output, a.lse, args.accum, args.accumLse, a.softcap);
    return launch.Result();
}
} // namespace flash_attn::mla

namespace flash_attn::mla {
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
template <typename DType>
__aicore__ inline bool LoadSplitPagedBsnd(GM_ADDR metadata, const MlaSplitPagedBsndContext& c,
                                          const MlaPagedMetadataInputs& inputs, const MlaPagedCacheInputs& cache,
                                          MlaSplitPagedBsndSchedule& s, Kernel::MetadataReader& read)
{
    const uint32_t dtype = AscendC::IsSameType<DType, half>::value ? 1 : 2;
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
    if (!addr)
        return;
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
template <typename DType>
__global__ __aicore__ __mix__(1, 2) void SplitPagedBsndPartial(uint64_t ffts, GM_ADDR qc, GM_ADDR qr, GM_ADDR workspace,
                                                               GM_ADDR metadata, MlaSplitPagedBsndContext c,
                                                               MlaPagedMetadataInputs inputs, MlaPagedCacheInputs cache,
                                                               MlaBsndInputLayouts layouts, float scale, float softcap)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
    MlaSplitPagedBsndSchedule s{};
    Kernel::MetadataReader read;
    if (!LoadSplitPagedBsnd<DType>(metadata, c, inputs, cache, s, read))
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
        Kernel::MlaSplitKvPagedBsnd<DType> op;
        op(tiling, partial, scale, qc, qr, cache, workspace, &pipe, metadata + kMlaPagedMetadataPrefixWords * 4,
           layouts, 0, softcap);
    }
#endif
}
template <typename DType>
__global__ __aicore__ __mix__(1, 2) void SplitPagedBsndMerge(uint64_t ffts, GM_ADDR workspace, GM_ADDR metadata,
                                                             MlaSplitPagedBsndContext c, MlaPagedMetadataInputs inputs,
                                                             MlaPagedCacheInputs cache, GM_ADDR out, GM_ADDR lse,
                                                             GM_ADDR accum, GM_ADDR accumLse, float softcap)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
    MlaSplitPagedBsndSchedule s{};
    Kernel::MetadataReader read;
    if (!LoadSplitPagedBsnd<DType>(metadata, c, inputs, cache, s, read))
        return;
    AscendC::SetSyncBaseAddr(ffts);
    if ASCEND_IS_AIV {
        if (c.paged.logicalKvCapacity == 0) {
            const uint64_t rows = uint64_t(c.paged.batch) * c.paged.query.maxLength * c.paged.heads;
            AscendC::TPipe pipe;
            InitSplitPagedEmpty<DType>(out, rows * 512, 0);
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
            Block::MlaSplitKvMerge<DType> block;
            block.Init(p, workspace + task.outOffset, workspace + task.sumOffset, workspace + task.maxOffset, out, lse,
                       accum, accumLse, &pipe, softcap);
            const uint32_t tile = block.TileRows();
            for (uint32_t start = 0; start < p.rows; start += tile)
                block.Run(start, tile < p.rows - start ? tile : p.rows - start);
        }
    }
#endif
}
template <typename DType>
aclError LaunchSplitPagedBsnd(const MlaSplitPagedBsndLaunchArgs& args)
{
    const auto& a = args.paged;
    const MlaSplitPagedBsndContext c{a.context, a.window, args.splits, uint32_t(args.softcap > 0)};
    {
        MlaBatchLaunchScope launch;
        SplitPagedBsndPartial<DType><<<a.context.blockDim, nullptr, a.stream>>>(
            a.ffts, a.qc, a.qr, a.workspace, a.metadata, c, a.inputs, a.cache, a.layouts, a.scale, args.softcap);
        const auto error = launch.Result();
        if (error != ACL_SUCCESS)
            return error;
    }
    MlaBatchLaunchScope launch;
    SplitPagedBsndMerge<DType><<<a.context.blockDim, nullptr, a.stream>>>(a.ffts, a.workspace, a.metadata, c, a.inputs,
                                                                          a.cache, a.output, a.lse, args.accum,
                                                                          args.accumLse, args.softcap);
    return launch.Result();
}
} // namespace flash_attn::mla

namespace flash_attn::mla {
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
template <typename DType>
__aicore__ inline bool LoadSplitPagedTnd(GM_ADDR metadata, const MlaSplitPagedTndContext& c,
                                         const MlaPagedMetadataInputs& inputs, const MlaPagedCacheInputs& cache,
                                         MlaSplitPagedTndSchedule& s, Kernel::MetadataReader& read)
{
    const uint32_t dtype = AscendC::IsSameType<DType, half>::value ? 1 : 2;
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
    if (ReadSplitPagedTndMetadata(read, 414 + 7 * c.paged.batch, c, inputs, arrays, cache, pages, s) !=
        MlaMetadataReadStatus::Success) {
        AscendC::Trap();
        return false;
    }
    return true;
}
template <typename T>
__aicore__ inline void InitSplitPagedTndEmpty(GM_ADDR addr, uint64_t elements, T value)
{
    if (!addr)
        return;
    const uint64_t lanes = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
    const uint64_t chunk = ((elements + lanes - 1) / lanes + 31) / 32 * 32;
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
// Initialization owns every allocation token, including Q-used gaps and
// empty samples; stream order prevents races with partial/merge writeback.
template <typename DType>
__global__ __aicore__ __mix__(1, 2) void SplitPagedTndInit(GM_ADDR metadata, MlaSplitPagedTndContext c,
                                                           MlaPagedMetadataInputs inputs, MlaPagedCacheInputs cache,
                                                           GM_ADDR out, GM_ADDR lse, GM_ADDR accum, GM_ADDR accumLse)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
    MlaSplitPagedTndSchedule s{};
    Kernel::MetadataReader read;
    if (!LoadSplitPagedTnd<DType>(metadata, c, inputs, cache, s, read))
        return;
    if ASCEND_IS_AIV {
        const uint64_t rows = uint64_t(c.paged.query.totalTokens.low) * c.paged.heads;
        AscendC::TPipe pipe;
        InitSplitPagedTndEmpty<DType>(out, rows * 512, 0);
        InitSplitPagedTndEmpty<float>(lse, rows, -float(1.0 / 0.0));
        InitSplitPagedTndEmpty<float>(accum, rows * c.splits * 512, 0);
        InitSplitPagedTndEmpty<float>(accumLse, rows * c.splits, -float(1.0 / 0.0));
    }
#endif
}
// Stream order is the global partial -> merge barrier. Every stage rechecks
// every effective page before writes/events, including static empty capacity.
template <typename DType>
__global__ __aicore__ __mix__(1, 2) void SplitPagedTndPartial(uint64_t ffts, GM_ADDR qc, GM_ADDR qr, GM_ADDR workspace,
                                                              GM_ADDR metadata, MlaSplitPagedTndContext c,
                                                              MlaPagedMetadataInputs inputs, MlaPagedCacheInputs cache,
                                                              MlaBsndInputLayouts layouts, float scale, float softcap)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
    MlaSplitPagedTndSchedule s{};
    Kernel::MetadataReader read;
    if (!LoadSplitPagedTnd<DType>(metadata, c, inputs, cache, s, read))
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
        MlaSplitPagedTndTask task{};
        SplitPagedTndTask(read, c, s, index, task);
        MlaSplitKvKernelParams partial{};
        partial.window = c.window;
        partial.totalTasks = tiling.blockDim;
        partial.tasks[core] = {task.batch, task.m, task.split, task.s2Begin, task.s2End};
        partial.outOffsetBytes = task.outOffset;
        partial.sumOffsetBytes = task.sumOffset;
        partial.maxOffsetBytes = task.maxOffset;
        AscendC::TPipe pipe;
        Kernel::MlaSplitKvPaged<DType, true> op;
        op(tiling, partial, scale, qc, qr, cache, workspace, &pipe, metadata + kMlaPagedMetadataPrefixWords * 4,
           layouts, c.paged.query.totalTokens.low, softcap);
    }
#endif
}
template <typename DType>
__global__ __aicore__ __mix__(1, 2) void SplitPagedTndMerge(uint64_t ffts, GM_ADDR workspace, GM_ADDR metadata,
                                                            MlaSplitPagedTndContext c, MlaPagedMetadataInputs inputs,
                                                            MlaPagedCacheInputs cache, GM_ADDR out, GM_ADDR lse,
                                                            GM_ADDR accum, GM_ADDR accumLse, float softcap)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
    MlaSplitPagedTndSchedule s{};
    Kernel::MetadataReader read;
    if (!LoadSplitPagedTnd<DType>(metadata, c, inputs, cache, s, read))
        return;
    AscendC::SetSyncBaseAddr(ffts);
    if ASCEND_IS_AIV {
        for (uint64_t group = AscendC::GetBlockIdx(); group < s.groups;
             group += c.paged.blockDim * AscendC::GetSubBlockNum()) {
            MlaSplitPagedTndTask task{};
            SplitPagedTndTask(read, c, s, group * c.splits, task);
            const uint32_t span = kMlaPagedMetadataPrefixWords + 5 * task.batch;
            MlaSplitKvMergeParams p{};
            p.batchIndex = task.queryStart;
            p.batch = 1;
            p.sq = c.paged.query.totalTokens.low;
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
            Block::MlaSplitKvMerge<DType, false, true> block;
            block.Init(p, workspace + task.outOffset, workspace + task.sumOffset, workspace + task.maxOffset, out, lse,
                       accum, accumLse, &pipe, softcap);
            const uint32_t tile = block.TileRows();
            for (uint32_t start = 0; start < p.rows; start += tile)
                block.Run(start, tile < p.rows - start ? tile : p.rows - start);
        }
    }
#endif
}
template <typename DType>
aclError LaunchSplitPagedTnd(const MlaSplitPagedTndLaunchArgs& args)
{
    const auto& a = args.paged;
    const MlaSplitPagedTndContext c{a.context, a.window, args.splits, uint32_t(args.softcap > 0)};
    {
        MlaBatchLaunchScope launch;
        SplitPagedTndInit<DType><<<a.context.blockDim, nullptr, a.stream>>>(a.metadata, c, a.inputs, a.cache, a.output,
                                                                            a.lse, args.accum, args.accumLse);
        const auto error = launch.Result();
        if (error != ACL_SUCCESS)
            return error;
    }
    if (!a.context.query.totalTokens.low || !a.context.logicalKvCapacity)
        return ACL_SUCCESS;
    {
        MlaBatchLaunchScope launch;
        SplitPagedTndPartial<DType><<<a.context.blockDim, nullptr, a.stream>>>(
            a.ffts, a.qc, a.qr, a.workspace, a.metadata, c, a.inputs, a.cache, a.layouts, a.scale, args.softcap);
        const auto error = launch.Result();
        if (error != ACL_SUCCESS)
            return error;
    }
    MlaBatchLaunchScope launch;
    SplitPagedTndMerge<DType><<<a.context.blockDim, nullptr, a.stream>>>(a.ffts, a.workspace, a.metadata, c, a.inputs,
                                                                         a.cache, a.output, a.lse, args.accum,
                                                                         args.accumLse, args.softcap);
    return launch.Result();
}
} // namespace flash_attn::mla
