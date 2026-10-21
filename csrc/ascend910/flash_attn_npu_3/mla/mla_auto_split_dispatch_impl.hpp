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
namespace {

#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
template <typename DType, bool IsTnd, bool IsMixed>
__aicore__ inline bool
LoadAutoSplit(GM_ADDR metadata,
              const std::conditional_t<IsMixed, MlaAutoSplitMixedContext,
                                       std::conditional_t<IsTnd, MlaAutoSplitTndContext, MlaAutoSplitBsndContext>>& c,
              const MlaUsedMetadataInputs& inputs, MlaAutoSplitBsndSchedule& s, Kernel::MetadataReader& read)
{
    const uint32_t dtype =
        AscendC::IsSameType<DType, half>::value ? uint32_t(MlaMetadataDtype::Fp16) : uint32_t(MlaMetadataDtype::Bf16);
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
    constexpr uint32_t version = IsMixed ? kMlaAutoSplitMixedVersion
                                 : IsTnd ? kMlaAutoSplitTndVersion
                                         : kMlaAutoSplitBsndVersion;
    constexpr bool kvIsTnd = IsTnd && !IsMixed;
    if (ReadAutoSplitMetadata<IsTnd, version, kvIsTnd,
                              IsMixed   ? kMlaSoftcapAutoSplitMixedVersion
                              : kvIsTnd ? kMlaSoftcapAutoSplitTndVersion
                                        : kMlaSoftcapAutoSplitBsndVersion>(
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
    if (!address)
        return;
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
template <bool IsTnd>
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
    if constexpr (IsTnd) {
        p.batch = 1;
        p.sq = s.used.query.totalTokens.low;
        p.batchIndex = read(kMlaUsedMetadataPrefixWords + 8 * t.batch);
    }
    p.maskKind = w.kind;
    p.left = w.left;
    p.right = w.right;
    auto geometry = t;
    auto_split_detail::Geometry<false>(read, s, w, geometry);
    auto_split_detail::Parts(s, geometry, UINT32_MAX, nullptr, p.begin, p.end);
    for (uint32_t split = 0; split < p.splits; ++split) {
        const uint32_t begin = p.begin[split] * a.s2BaseSize, end = p.end[split] * a.s2BaseSize;
        p.begin[split] = geometry.kvBegin > begin ? geometry.kvBegin : begin;
        p.end[split] = geometry.kvEnd < end ? geometry.kvEnd : end;
    }
    return p;
}
#endif
} // namespace
// Validate once, then reuse the checked schedule for initialization,
// attention and FD. The numerical stages retain their existing ownership.
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
namespace {
template <typename DType, bool IsTnd, bool IsMixed>
__aicore__ inline void
AutoInit(const std::conditional_t<IsMixed, MlaAutoSplitMixedContext,
                                  std::conditional_t<IsTnd, MlaAutoSplitTndContext, MlaAutoSplitBsndContext>>& c,
         GM_ADDR out, GM_ADDR lse, GM_ADDR accum, GM_ADDR accumLse)
{
    if ASCEND_IS_AIV {
        AscendC::TPipe pipe;
        const uint64_t rows = uint64_t(c.used.heads) * UsedMetadataTotal(c.used.query);
        AutoInitialize<DType>(out, rows * 512, DType(0));
        AutoInitialize<float>(lse, rows, UsedMetadataTotal(c.used.kv) ? -float(1.0 / 0.0) : float(1.0 / 0.0));
        AutoInitialize<float>(accum, rows * kMlaAutoSplitCapacity * 512, 0.0f);
        AutoInitialize<float>(accumLse, rows * kMlaAutoSplitCapacity, -float(1.0 / 0.0));
    }
}
template <typename DType, bool IsTnd, bool IsMixed>
__aicore__ inline void
AutoPartial(GM_ADDR qc, GM_ADDR qr, GM_ADDR kv, GM_ADDR kr, GM_ADDR workspace, GM_ADDR metadata,
            const std::conditional_t<IsMixed, MlaAutoSplitMixedContext,
                                     std::conditional_t<IsTnd, MlaAutoSplitTndContext, MlaAutoSplitBsndContext>>& c,
            float scale, MlaBsndInputLayouts layouts, GM_ADDR out, GM_ADDR lse, GM_ADDR accum, GM_ADDR accumLse,
            float softcap, const MlaAutoSplitBsndSchedule& s, const Kernel::MetadataReader& read)
{
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
        if (t.owner != core) {
            auto geometry = t;
            auto_split_detail::Geometry<false>(read, s, c.window, geometry);
            uint32_t split = 0;
            auto_split_detail::Parts(s, geometry, core, &split);
            AutoSplitBsndTask(read, s, c.window, group, split, t);
        }
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
            if constexpr (IsTnd) {
                Kernel::MlaSplitKvUsedTnd<DType> op;
                op(tiling, p, scale, qc, qr, kv, kr, workspace, &pipe, metadata + kMlaUsedMetadataPrefixWords * 4,
                   layouts, c.used.query.totalTokens.low, !IsMixed, softcap);
            } else {
                Kernel::MlaSplitKvUsedBsnd<DType, true> op;
                op(tiling, p, scale, qc, qr, kv, kr, workspace, &pipe, metadata + kMlaUsedMetadataPrefixWords * 4,
                   layouts, softcap);
            }
        }
        if ASCEND_IS_AIV {
            if (t.kind == 0) {
                auto merge = AutoMergeParams<IsTnd>(read, s, c.window, t);
                // Match the source's 16-row aligned C1V2 ownership. Each vector
                // reads only its completed raw rows; no cross-vector handoff.
                const uint32_t firstRows = merge.rows <= 16 ? merge.rows : ((merge.rows + 15) / 16 + 1) / 2 * 16;
                const uint32_t begin = AscendC::GetBlockIdx() % 2 ? firstRows : 0;
                const uint32_t end = AscendC::GetBlockIdx() % 2 ? merge.rows : firstRows;
                AscendC::TPipe pipe;
                Block::MlaSplitKvMerge<DType, true, IsTnd> block;
                block.Init(merge, workspace + p.outOffsetBytes, workspace + p.sumOffsetBytes,
                           workspace + p.maxOffsetBytes, out, lse, accum, accumLse, &pipe, softcap);
                for (uint32_t row = begin; row < end; row += 8)
                    block.Run(row, end - row < 8 ? end - row : 8);
            }
        }
    }
}
template <typename DType, bool IsTnd, bool IsMixed>
__aicore__ inline void
AutoMerge(GM_ADDR workspace,
          const std::conditional_t<IsMixed, MlaAutoSplitMixedContext,
                                   std::conditional_t<IsTnd, MlaAutoSplitTndContext, MlaAutoSplitBsndContext>>& c,
          GM_ADDR out, GM_ADDR lse, GM_ADDR accum, GM_ADDR accumLse, float softcap, const MlaAutoSplitBsndSchedule& s,
          const Kernel::MetadataReader& read)
{
    if ASCEND_IS_AIV {
        const uint32_t owner = AscendC::GetBlockIdx();
        if (owner >= s.vectors)
            return;
        const uint32_t base = auto_split_detail::Base(c.used.batch);
        const uint32_t firstHead = owner ? read(base + 242 + owner - 1) : 0;
        const uint32_t firstPart = owner ? read(base + 294 + owner - 1) + 1 : 0;
        const uint32_t lastHead = read(base + 242 + owner);
        const uint32_t lastPart = read(base + 294 + owner);
        // The validated vector endpoints already give this lane's contiguous
        // FD interval. Do not rediscover ownership for every other lane's row.
        for (uint32_t h = firstHead; h <= lastHead; ++h) {
            const uint32_t parts = h == lastHead ? lastPart + 1 : read(base + 190 + h);
            for (uint32_t part = h == firstHead ? firstPart : 0; part < parts; ++part) {
                MlaAutoSplitBsndMerge r{};
                AutoSplitBsndMerge(read, s, h, part, r);
                if (r.owner != owner)
                    continue;
                MlaAutoSplitBsndTask t{};
                AutoSplitBsndTask(read, s, c.window, r.group, 0, t);
                auto p = AutoMergeParams<IsTnd>(read, s, c.window, t);
                p.rowStart = r.rowBegin;
                p.rows = r.rowEnd - r.rowBegin;
                AscendC::TPipe pipe;
                Block::MlaSplitKvMerge<DType, false, IsTnd> block;
                block.Init(p, workspace + r.outOffset, workspace + r.sumOffset, workspace + r.maxOffset, out, lse,
                           accum, accumLse, &pipe, softcap);
                const uint32_t tile = block.TileRows();
                for (uint32_t row = 0; row < p.rows; row += tile)
                    block.Run(row, p.rows - row < tile ? p.rows - row : tile);
            }
        }
    }
}
} // namespace
#endif
template <typename DType, bool IsTnd, bool IsMixed>
__global__ __aicore__ __mix__(1, 2) void AutoPartialKernel(
    uint64_t ffts, GM_ADDR qc, GM_ADDR qr, GM_ADDR kv, GM_ADDR kr, GM_ADDR workspace, GM_ADDR metadata,
    std::conditional_t<IsMixed, MlaAutoSplitMixedContext,
                       std::conditional_t<IsTnd, MlaAutoSplitTndContext, MlaAutoSplitBsndContext>>
        c,
    MlaUsedMetadataInputs inputs, float scale, MlaBsndInputLayouts layouts, GM_ADDR out, GM_ADDR lse, GM_ADDR accum,
    GM_ADDR accumLse, float softcap)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
    MlaAutoSplitBsndSchedule s{};
    Kernel::MetadataReader read;
    if (!LoadAutoSplit<DType, IsTnd, IsMixed>(metadata, c, inputs, s, read))
        return;
    AscendC::SetSyncBaseAddr(ffts);
    AutoInit<DType, IsTnd, IsMixed>(c, out, lse, accum, accumLse);
    if (!s.groups)
        return;
    // Direct fragments write final output, so all output initialization must
    // finish first. Idle vector cores participate in both barriers.
    if ASCEND_IS_AIV {
        AscendC::SyncAll();
    }
    AutoPartial<DType, IsTnd, IsMixed>(qc, qr, kv, kr, workspace, metadata, c, scale, layouts, out, lse, accum,
                                       accumLse, softcap, s, read);
    if ASCEND_IS_AIV {
        AscendC::SyncAll();
        AutoMerge<DType, IsTnd, IsMixed>(workspace, c, out, lse, accum, accumLse, softcap, s, read);
    }
#endif
}
template <typename DType, bool IsTnd, bool IsMixed>
aclError LaunchAutoSplit(
    const std::conditional_t<IsMixed, MlaSplitMixedLaunchArgs,
                             std::conditional_t<IsTnd, MlaSplitTndLaunchArgs, MlaSplitBsndLaunchArgs>>& args)
{
    const auto& a = args.used;
    std::conditional_t<IsMixed, MlaAutoSplitMixedContext,
                       std::conditional_t<IsTnd, MlaAutoSplitTndContext, MlaAutoSplitBsndContext>>
        c{a.context, a.window};
    c.hasSoftcap = a.softcap > 0;
    MlaBatchLaunchScope launch;
    AutoPartialKernel<DType, IsTnd, IsMixed><<<a.context.blockDim, nullptr, a.stream>>>(
        a.ffts, a.qc, a.qr, a.kv, a.kr, a.workspace, a.metadata, c, a.inputs, a.scale, a.layouts, a.output, a.lse,
        args.accum, args.accumLse, a.softcap);
    return launch.Result();
}
} // namespace flash_attn::mla

namespace flash_attn::mla {
namespace {
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
template <typename DType, bool IsTnd>
__aicore__ inline bool
LoadAutoSplitPaged(GM_ADDR metadata,
                   const std::conditional_t<IsTnd, MlaAutoSplitPagedTndContext, MlaAutoSplitPagedBsndContext>& c,
                   const MlaPagedMetadataInputs& inputs, const MlaPagedCacheInputs& cache, MlaAutoSplitBsndSchedule& s,
                   Kernel::MetadataReader& read)
{
    const uint32_t dtype = AscendC::IsSameType<DType, half>::value ? 1 : 2;
    if (!metadata ||
        c.paged.batch >
            (UINT32_MAX - (IsTnd ? kMlaAutoSplitPagedTndFixedWords : kMlaAutoSplitPagedBsndFixedWords)) / 7 ||
        c.paged.dtype != dtype || c.paged.blockDim != AscendC::GetBlockNum() ||
        !ValidPagedMetadataAddress(reinterpret_cast<uint64_t>(metadata),
                                   (IsTnd ? kMlaAutoSplitPagedTndFixedWords : kMlaAutoSplitPagedBsndFixedWords) +
                                       7 * c.paged.batch)) {
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
    std::conditional_t<IsTnd, MlaAutoSplitPagedTndSchedule, MlaAutoSplitPagedBsndSchedule> paged{};
    MlaMetadataReadStatus status;
    if constexpr (IsTnd)
        status = ReadAutoSplitPagedTndMetadata(read, kMlaAutoSplitPagedTndFixedWords + 7 * c.paged.batch, c, inputs,
                                               arrays, cache, pages, paged);
    else
        status = ReadAutoSplitPagedBsndMetadata(read, kMlaAutoSplitPagedBsndFixedWords + 7 * c.paged.batch, c, inputs,
                                                arrays, cache, pages, paged);
    if (status != MlaMetadataReadStatus::Success) {
        AscendC::Trap();
        return false;
    }
    s = paged.native;
    // Direct slots follow all persistent FD slots; every stage verifies the
    // additional production scratch before writes or synchronization.
    if (s.groups && s.bytes + uint64_t(s.aicNum) * s.used.tiling.mBaseSize * 528 * 4 > c.paged.workspaceBytes) {
        AscendC::Trap();
        return false;
    }
    return true;
}
template <typename T>
__aicore__ inline void AutoInitializePaged(GM_ADDR address, uint64_t elements, T value)
{
    if (!address)
        return;
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
template <bool IsTnd>
__aicore__ inline MlaSplitKvMergeParams
AutoMergeParamsPaged(const Kernel::MetadataReader& read, const MlaAutoSplitBsndSchedule& s,
                     const MlaWindowKernelParams& w, const MlaAutoSplitBsndTask& t)
{
    MlaSplitKvMergeParams p{};
    const auto& a = s.used.tiling;
    if constexpr (IsTnd) {
        p.batch = 1;
        p.sq = UsedMetadataTotal(s.used.query);
    } else {
        p.batch = a.batch;
        p.sq = a.sq;
    }
    p.heads = a.heads;
    if constexpr (IsTnd) {
        p.batchIndex = ReadMetadataWide(read, kMlaPagedMetadataPrefixWords + 5 * t.batch);
    } else {
        p.batchIndex = t.batch;
    }
    p.rowStart = t.rowBegin;
    p.rows = t.rowEnd - t.rowBegin;
    p.mBase = a.mBaseSize;
    p.splits = t.splitCount;
    p.adapt = 1;
    p.queryLength = auto_split_detail::QueryLength<true>(read, t.batch);
    p.kvLength = auto_split_detail::KvLength<true>(read, t.batch);
    p.maskKind = w.kind;
    p.left = w.left;
    p.right = w.right;
    auto geometry = t;
    auto_split_detail::Geometry<true>(read, s, w, geometry);
    auto_split_detail::Parts(s, geometry, UINT32_MAX, nullptr, p.begin, p.end);
    for (uint32_t split = 0; split < p.splits; ++split) {
        const uint32_t begin = p.begin[split] * a.s2BaseSize, end = p.end[split] * a.s2BaseSize;
        p.begin[split] = geometry.kvBegin > begin ? geometry.kvBegin : begin;
        p.end[split] = geometry.kvEnd < end ? geometry.kvEnd : end;
    }
    return p;
}
#endif
} // namespace
// Validate once, then reuse the checked schedule for initialization,
// attention and FD. The numerical stages retain their existing ownership.
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
namespace {
template <typename DType, bool IsTnd>
__aicore__ inline void
AutoInit(const std::conditional_t<IsTnd, MlaAutoSplitPagedTndContext, MlaAutoSplitPagedBsndContext>& c, GM_ADDR out,
         GM_ADDR lse, GM_ADDR accum, GM_ADDR accumLse)
{
    if ASCEND_IS_AIV {
        AscendC::TPipe pipe;
        const uint64_t rows = uint64_t(c.paged.heads) * (uint64_t(c.paged.query.totalTokens.low) |
                                                         (uint64_t(c.paged.query.totalTokens.high) << 32));
        AutoInitializePaged<DType>(out, rows * 512, DType(0));
        AutoInitializePaged<float>(lse, rows, -float(1.0 / 0.0));
        AutoInitializePaged<float>(accum, rows * kMlaAutoSplitCapacity * 512, 0.0f);
        AutoInitializePaged<float>(accumLse, rows * kMlaAutoSplitCapacity, -float(1.0 / 0.0));
    }
}
template <typename DType, bool IsTnd>
__aicore__ inline void
AutoPartial(GM_ADDR qc, GM_ADDR qr, GM_ADDR workspace, GM_ADDR metadata,
            const std::conditional_t<IsTnd, MlaAutoSplitPagedTndContext, MlaAutoSplitPagedBsndContext>& c,
            const MlaPagedCacheInputs& cache, float scale, MlaBsndInputLayouts layouts, GM_ADDR out, GM_ADDR lse,
            GM_ADDR accum, GM_ADDR accumLse, float softcap, const MlaAutoSplitBsndSchedule& s,
            const Kernel::MetadataReader& read)
{
    uint32_t core = AscendC::GetBlockIdx();
    if ASCEND_IS_AIV {
        core /= AscendC::GetSubBlockNum();
    }
    if (core >= s.usedCoreNum)
        return;
    auto tiling = s.used.tiling;
    tiling.usedCoreNum = tiling.blockDim;
    const uint32_t prefix = auto_split_detail::Base<true>(c.paged.batch) + kMlaAutoSplitTailPrefix;
    const uint32_t b = core ? s.bEnd[core - 1] : 0;
    const uint64_t first = ReadMetadataWide(read, prefix + 2 * b) + (core ? s.mEnd[core - 1] : 0);
    uint64_t last = ReadMetadataWide(read, prefix + 2 * s.bEnd[core]) + s.mEnd[core] + (s.nEnd[core] ? 1 : 0);
    if (last > s.groups)
        last = s.groups;
    for (uint64_t group = first; group < last; ++group) {
        MlaAutoSplitBsndTask t{};
        if (!AutoSplitBsndTask<true>(read, s, c.window, group, 0, t) || t.kind == 2)
            continue;
        if (t.owner != core) {
            auto geometry = t;
            auto_split_detail::Geometry<true>(read, s, c.window, geometry);
            uint32_t split = 0;
            auto_split_detail::Parts(s, geometry, core, &split);
            AutoSplitBsndTask<true>(read, s, c.window, group, split, t);
        }
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
            Kernel::MlaSplitKvPaged<DType, IsTnd> op;
            op(tiling, p, scale, qc, qr, cache, workspace, &pipe, metadata + kMlaPagedMetadataPrefixWords * 4, layouts,
               IsTnd ? c.paged.query.totalTokens.low : 0, softcap);
        }
        if ASCEND_IS_AIV {
            if (t.kind == 0) {
                auto merge = AutoMergeParamsPaged<IsTnd>(read, s, c.window, t);
                // Match the source's 16-row aligned C1V2 ownership. Each vector
                // reads only its completed raw rows; no cross-vector handoff.
                const uint32_t firstRows = merge.rows <= 16 ? merge.rows : ((merge.rows + 15) / 16 + 1) / 2 * 16;
                const uint32_t begin = AscendC::GetBlockIdx() % 2 ? firstRows : 0;
                const uint32_t end = AscendC::GetBlockIdx() % 2 ? merge.rows : firstRows;
                AscendC::TPipe pipe;
                Block::MlaSplitKvMerge<DType, true, IsTnd> block;
                block.Init(merge, workspace + p.outOffsetBytes, workspace + p.sumOffsetBytes,
                           workspace + p.maxOffsetBytes, out, lse, accum, accumLse, &pipe, softcap);
                for (uint32_t row = begin; row < end; row += 8)
                    block.Run(row, end - row < 8 ? end - row : 8);
            }
        }
    }
}
template <typename DType, bool IsTnd>
__aicore__ inline void
AutoMerge(GM_ADDR workspace,
          const std::conditional_t<IsTnd, MlaAutoSplitPagedTndContext, MlaAutoSplitPagedBsndContext>& c, GM_ADDR out,
          GM_ADDR lse, GM_ADDR accum, GM_ADDR accumLse, float softcap, const MlaAutoSplitBsndSchedule& s,
          const Kernel::MetadataReader& read)
{
    if ASCEND_IS_AIV {
        const uint32_t owner = AscendC::GetBlockIdx();
        if (owner >= s.vectors)
            return;
        const uint32_t base = auto_split_detail::Base<true>(c.paged.batch);
        const uint32_t firstHead = owner ? read(base + 242 + owner - 1) : 0;
        const uint32_t firstPart = owner ? read(base + 294 + owner - 1) + 1 : 0;
        const uint32_t lastHead = read(base + 242 + owner);
        const uint32_t lastPart = read(base + 294 + owner);
        // The validated vector endpoints already give this lane's contiguous
        // FD interval. Do not rediscover ownership for every other lane's row.
        for (uint32_t h = firstHead; h <= lastHead; ++h) {
            const uint32_t parts = h == lastHead ? lastPart + 1 : read(base + 190 + h);
            for (uint32_t part = h == firstHead ? firstPart : 0; part < parts; ++part) {
                MlaAutoSplitBsndMerge r{};
                AutoSplitBsndMerge<true>(read, s, h, part, r);
                if (r.owner != owner)
                    continue;
                MlaAutoSplitBsndTask t{};
                AutoSplitBsndTask<true>(read, s, c.window, r.group, 0, t);
                auto p = AutoMergeParamsPaged<IsTnd>(read, s, c.window, t);
                p.rowStart = r.rowBegin;
                p.rows = r.rowEnd - r.rowBegin;
                AscendC::TPipe pipe;
                Block::MlaSplitKvMerge<DType, false, IsTnd> block;
                block.Init(p, workspace + r.outOffset, workspace + r.sumOffset, workspace + r.maxOffset, out, lse,
                           accum, accumLse, &pipe, softcap);
                const uint32_t tile = block.TileRows();
                for (uint32_t row = 0; row < p.rows; row += tile)
                    block.Run(row, p.rows - row < tile ? p.rows - row : tile);
            }
        }
    }
}
} // namespace
#endif
template <typename DType, bool IsTnd>
__global__ __aicore__ __mix__(1, 2) void AutoPagedPartialKernel(
    uint64_t ffts, GM_ADDR qc, GM_ADDR qr, GM_ADDR workspace, GM_ADDR metadata,
    std::conditional_t<IsTnd, MlaAutoSplitPagedTndContext, MlaAutoSplitPagedBsndContext> c,
    MlaPagedMetadataInputs inputs, MlaPagedCacheInputs cache, float scale, MlaBsndInputLayouts layouts, GM_ADDR out,
    GM_ADDR lse, GM_ADDR accum, GM_ADDR accumLse, float softcap)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
    MlaAutoSplitBsndSchedule s{};
    Kernel::MetadataReader read;
    if (!LoadAutoSplitPaged<DType, IsTnd>(metadata, c, inputs, cache, s, read))
        return;
    AscendC::SetSyncBaseAddr(ffts);
    AutoInit<DType, IsTnd>(c, out, lse, accum, accumLse);
    if (!s.groups)
        return;
    // Direct fragments write final output, so all output initialization must
    // finish first. Idle vector cores participate in both barriers.
    if ASCEND_IS_AIV {
        AscendC::SyncAll();
    }
    AutoPartial<DType, IsTnd>(qc, qr, workspace, metadata, c, cache, scale, layouts, out, lse, accum, accumLse, softcap,
                              s, read);
    if ASCEND_IS_AIV {
        AscendC::SyncAll();
        AutoMerge<DType, IsTnd>(workspace, c, out, lse, accum, accumLse, softcap, s, read);
    }
#endif
}
template <typename DType, bool IsTnd>
aclError LaunchAutoSplitPaged(const MlaSplitPagedBsndLaunchArgs& args)
{
    const auto& a = args.paged;
    std::conditional_t<IsTnd, MlaAutoSplitPagedTndContext, MlaAutoSplitPagedBsndContext> c{a.context, a.window,
                                                                                           uint32_t(args.softcap > 0)};
    MlaBatchLaunchScope launch;
    AutoPagedPartialKernel<DType, IsTnd><<<a.context.blockDim, nullptr, a.stream>>>(
        a.ffts, a.qc, a.qr, a.workspace, a.metadata, c, a.inputs, a.cache, a.scale, a.layouts, a.output, a.lse,
        args.accum, args.accumLse, args.softcap);
    return launch.Result();
}
} // namespace flash_attn::mla
