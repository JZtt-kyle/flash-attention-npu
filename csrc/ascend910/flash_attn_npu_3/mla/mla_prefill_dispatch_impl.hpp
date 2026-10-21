// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
#pragma once
#include "kernel_operator.h"
#include "mla_dispatch.h"
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
#include "lib/matmul_intf.h"
#include "mla_prefill_kernel.hpp"
#include "mla_kernel.hpp"
#include "mla_split_metadata_consumer.h"
#include "mla_split_kv_merge_block.hpp"
#endif
namespace flash_attn::mla {
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
template <typename DType>
__aicore__ inline bool LoadPrefill(const MlaPrefillBsndLaunchArgs& a, Kernel::MetadataReader& read,
                                   MlaAutoSplitBsndSchedule& s, MlaPrefillBsndLayout& layout)
{
    const uint32_t dtype =
        AscendC::IsSameType<DType, half>::value ? uint32_t(MlaMetadataDtype::Fp16) : uint32_t(MlaMetadataDtype::Bf16);
    if (!PrefillBsndLayout(a.context, layout)) {
        AscendC::Trap();
        return false;
    }
    if (!LoadCheckedAutoSplitSchedule(a.checked, dtype, s))
        return false;
    read.data.SetGlobalBuffer((__gm__ uint32_t*)a.metadata);
    return true;
}
template <typename DType, bool Brc, bool Partial, bool Paged>
__aicore__ inline void RunPrefillImpl(const MlaPrefillBsndLaunchArgs& a, const Kernel::MetadataReader& read,
                                      const MlaAutoSplitBsndSchedule& s, const MlaPrefillBsndLayout& l, uint32_t core,
                                      AscendC::TPipe* pipe)
{
    const auto& c = a.context;
    const bool autoSplit = c.splits == 0;
    uint64_t first = core, last = s.groups * l.capacity;
    if (autoSplit) {
        if (core >= s.usedCoreNum)
            return;
        const PrefillPartitionView<Kernel::MetadataReader> view{read, c};
        const uint32_t prefix = auto_split_detail::Base(s.used.tiling.batch) + kMlaAutoSplitTailPrefix;
        const uint32_t b = core ? s.bEnd[core - 1] : 0;
        first = ReadMetadataWide(view, prefix + 2 * b) + (core ? s.mEnd[core - 1] : 0);
        last = ReadMetadataWide(view, prefix + 2 * s.bEnd[core]) + s.mEnd[core] + (s.nEnd[core] ? 1 : 0);
        if (last > s.groups)
            last = s.groups;
    }
    if (first >= last)
        return;
    Kernel::MlaPrefill<DType, Brc, Partial, Paged> kernel;
    auto params = a.params;
    if (c.tnd) {
        params.batch = 1;
        params.sq = c.totalQ;
    }
    kernel.Init(params, core, (GM_ADDR)a.q, (GM_ADDR)a.k, (GM_ADDR)a.v, (GM_ADDR)a.out, (GM_ADDR)a.lse,
                (GM_ADDR)a.workspace, (GM_ADDR)a.workspace + l.outOffset, (GM_ADDR)a.workspace + l.sumOffset,
                (GM_ADDR)a.workspace + l.maxOffset, pipe, &a.cache, Partial ? (GM_ADDR)a.accum : nullptr,
                Partial ? (GM_ADDR)a.accumLse : nullptr);
    for (uint64_t index = first; index < last; index += autoSplit ? 1 : c.cores) {
        const uint64_t group = autoSplit ? index : index / l.capacity;
        const uint32_t split = autoSplit ? 0 : index % l.capacity;
        MlaAutoSplitBsndTask t{};
        if (!PrefillBsndTask(read, c, s, group, split, t, autoSplit ? core : UINT32_MAX) || (autoSplit && t.kind == 2))
            continue;
        const uint32_t b = t.batch / c.kvHeads;
        Block::PrefillTaskContext context{};
        context.task = {c.tnd ? 0U : b,
                        t.batch % c.kvHeads,
                        t.rowBegin,
                        t.rowEnd - t.rowBegin,
                        read(PrefillLengthWord(c, b)),
                        read(PrefillLengthWord(c, b) + 1),
                        t.s2Begin,
                        t.s2End};
        context.cacheBatch = b;
        context.partial = t.splitCount > 1;
        context.partialRow = (group * l.capacity + t.split) * l.m;
        if (c.tnd) {
            const uint64_t qStart = read(PrefillSequenceWord(c, b));
            const uint64_t kStart = read(PrefillSequenceWord(c, b) + 2);
            context.qOffset = qStart * a.params.q.rowStride;
            if constexpr (!Paged) {
                context.kOffset = c.tnd == 1 ? kStart * a.params.k.rowStride : uint64_t(b) * a.params.k.batchStride;
                context.vOffset = c.tnd == 1 ? kStart * a.params.v.rowStride : uint64_t(b) * a.params.v.batchStride;
            }
            context.outputOffset = qStart * c.heads;
            context.lseOffset = qStart;
        }
        kernel.RunTask(context);
    }
    kernel.Finish();
}
template <typename DType, bool IsTnd>
__aicore__ inline void PrefillMerge(const MlaPrefillBsndLaunchArgs& a, const MlaSplitKvMergeParams& p,
                                    const MlaPrefillBsndLayout& l, uint64_t row, uint32_t begin, uint32_t end,
                                    Block::MlaSplitKvMerge<DType, true, IsTnd, 128>& merge)
{
    merge.Bind(p, (GM_ADDR)a.workspace + l.outOffset + row * 128 * 4, (GM_ADDR)a.workspace + l.sumOffset + row * 8 * 4,
               (GM_ADDR)a.workspace + l.maxOffset + row * 8 * 4, (GM_ADDR)a.out, (GM_ADDR)a.lse, (GM_ADDR)a.accum,
               (GM_ADDR)a.accumLse, a.params.softcap);
    for (uint32_t start = begin; start < end; start += merge.TileRows())
        merge.Run(start, merge.TileRows() < end - start ? merge.TileRows() : end - start);
}
// Same disjoint, 64-element aligned initialization as AutoInitialize in
// mla_auto_split_dispatch_impl.hpp. All vector cores join before output writeback.
template <typename T>
__aicore__ inline void PrefillInitialize(uint64_t address, uint64_t elements, T value)
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

template <typename DType>
__aicore__ inline void InitializePrefillOutputs(const MlaPrefillBsndLaunchArgs& a)
{
    if ASCEND_IS_AIV {
        const auto& c = a.context;
        const uint64_t rows = (c.tnd ? uint64_t(c.totalQ) : uint64_t(c.batch) * c.sq) * c.heads;
        PrefillInitialize<DType>(a.out, rows * 128, DType(0));
        PrefillInitialize<float>(a.lse, rows,
                                 (c.pageSize || (c.tnd == 1 ? c.totalK : c.sk)) ? -float(1.0 / 0.0) : float(1.0 / 0.0));
        if (c.splits != 1) {
            const uint64_t capacity = c.splits ? c.splits : 26;
            PrefillInitialize<float>(a.accum, rows * capacity * 128, 0.0f);
            PrefillInitialize<float>(a.accumLse, rows * capacity, -float(1.0 / 0.0));
        }
    }
}
template <typename DType, bool IsTnd>
__aicore__ inline void MergePrefillRows(const MlaPrefillBsndLaunchArgs& a, const Kernel::MetadataReader& read,
                                        const MlaAutoSplitBsndSchedule& s, const MlaPrefillBsndLayout& l,
                                        uint64_t group, uint32_t begin, uint32_t end,
                                        Block::MlaSplitKvMerge<DType, true, IsTnd, 128>& merge)
{
    const auto& c = a.context;
    MlaAutoSplitBsndTask t{};
    if (!PrefillBsndTask(read, c, s, group, 0, t)) {
        AscendC::Trap();
        return;
    }
    if (begin >= t.rowEnd - t.rowBegin)
        return;
    MlaSplitKvMergeParams p{};
    p.batch = c.batch;
    p.sq = c.sq;
    p.heads = c.heads / c.kvHeads;
    p.batchIndex = t.batch / c.kvHeads;
    p.kvHeads = c.kvHeads;
    p.kvHead = t.batch % c.kvHeads;
    p.rowStart = t.rowBegin;
    p.rows = t.rowEnd - t.rowBegin;
    p.mBase = l.m;
    p.splits = t.splitCount;
    p.queryLength = read(PrefillLengthWord(c, p.batchIndex));
    p.kvLength = read(PrefillLengthWord(c, p.batchIndex) + 1);
    p.adapt = 1;
    p.maskKind = c.window.kind;
    p.left = c.window.left;
    p.right = c.window.right;
    if (c.splits == 0) {
        const PrefillPartitionView<Kernel::MetadataReader> view{read, c};
        auto geometry = t;
        auto_split_detail::Geometry<false>(view, s, c.window, geometry);
        if (auto_split_detail::Parts(s, geometry, UINT32_MAX, nullptr, p.begin, p.end) != p.splits) {
            AscendC::Trap();
            return;
        }
        for (uint32_t split = 0; split < p.splits; ++split) {
            const uint32_t first = p.begin[split] * l.n, last = p.end[split] * l.n;
            p.begin[split] = p.kvLength < first ? p.kvLength : first;
            p.end[split] = p.kvLength < last ? p.kvLength : last;
        }
    } else {
        for (uint32_t split = 0; split < p.splits; ++split) {
            if (!PrefillBsndTask(read, c, s, group, split, t)) {
                AscendC::Trap();
                return;
            }
            p.begin[split] = p.kvLength < t.s2Begin * l.n ? p.kvLength : t.s2Begin * l.n;
            p.end[split] = p.kvLength < t.s2End * l.n ? p.kvLength : t.s2End * l.n;
        }
    }
    const uint64_t row = group * l.capacity * l.m;
    end = end < p.rows ? end : p.rows;
    if constexpr (IsTnd) {
        p.batch = 1;
        p.sq = c.totalQ;
        p.batchIndex = read(PrefillSequenceWord(c, t.batch / c.kvHeads));
    }
    PrefillMerge<DType, IsTnd>(a, p, l, row, begin, end, merge);
}
template <typename DType, bool IsTnd>
__aicore__ inline void MergePrefillOutputsImpl(const MlaPrefillBsndLaunchArgs& a, const Kernel::MetadataReader& read,
                                               const MlaAutoSplitBsndSchedule& s, const MlaPrefillBsndLayout& l,
                                               AscendC::TPipe* pipe)
{
    const auto& c = a.context;
    const uint32_t owner = AscendC::GetBlockIdx();
    if (c.splits == 0 && owner >= s.vectors) {
        AscendC::SyncAll();
        return;
    }
    pipe->Reset();
    Block::MlaSplitKvMerge<DType, true, IsTnd, 128> merge;
    merge.InitBuffers(pipe);
    AscendC::ICachePreLoad(2);
    AscendC::SyncAll();
    if (c.splits == 0) {
        // The native tail assigns only FD groups to vectors. Direct S1G
        // blocks have already written O/LSE and public accumulator slot 0.
        const PrefillPartitionView<Kernel::MetadataReader> view{read, c};
        const uint32_t base = auto_split_detail::Base(s.used.tiling.batch);
        const uint32_t firstHead = owner ? view(base + 242 + owner - 1) : 0;
        const uint32_t firstPart = owner ? view(base + 294 + owner - 1) + 1 : 0;
        const uint32_t lastHead = view(base + 242 + owner);
        const uint32_t lastPart = view(base + 294 + owner);
        for (uint32_t head = firstHead; head <= lastHead; ++head) {
            const uint32_t parts = head == lastHead ? lastPart + 1 : view(base + 190 + head);
            const uint32_t part = head == firstHead ? firstPart : 0;
            if (part >= parts)
                continue;
            MlaAutoSplitBsndMerge r{};
            if (!AutoSplitBsndMerge(view, s, head, part, r)) {
                merge.Finish();
                AscendC::Trap();
                return;
            }
            MergePrefillRows<DType, IsTnd>(a, read, s, l, r.group, part * 8, parts * 8, merge);
        }
        merge.Finish();
        return;
    }
    // TND and PA can use M=512 for short sequences. Distribute independent row
    // tiles across vectors instead of serializing an entire M group on one
    // vector. The existing merge's 192 split-row bound remains unchanged.
    const uint32_t capacityRows = 192U / l.capacity;
    const uint32_t tileRows = capacityRows > 8 ? 8 : capacityRows ? capacityRows : 1;
    const bool parallelRows = c.tnd || c.pageSize;
    const uint32_t tiles = parallelRows ? (l.m + tileRows - 1) / tileRows : 1;
    for (uint64_t index = AscendC::GetBlockIdx(); index < s.groups * tiles; index += c.cores * 2) {
        const uint64_t group = index / tiles;
        const uint32_t begin = parallelRows ? (index % tiles) * tileRows : 0;
        MergePrefillRows<DType, IsTnd>(a, read, s, l, group, begin, parallelRows ? begin + tileRows : l.m, merge);
    }
    merge.Finish();
}
#endif
template <typename DType, bool Brc, bool Partial, bool Paged, bool IsTnd>
__global__ __aicore__ __mix__(1, 2) void PrefillAttention(MlaPrefillBsndLaunchArgs a)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
    Kernel::MetadataReader read;
    MlaAutoSplitBsndSchedule s;
    MlaPrefillBsndLayout l{};
    if (!LoadPrefill<DType>(a, read, s, l))
        return;
    AscendC::SetSyncBaseAddr(a.ffts);
    AscendC::TPipe pipe;
    InitializePrefillOutputs<DType>(a);
    if (!s.groups)
        return;
    if ASCEND_IS_AIV {
        // Direct output must wait for all vector cores, including idle ones.
        if (a.context.splits <= 1)
            AscendC::SyncAll();
    }
    uint32_t core = AscendC::GetBlockIdx();
    if ASCEND_IS_AIV {
        core /= AscendC::GetSubBlockNum();
    }
    RunPrefillImpl<DType, Brc, Partial, Paged>(a, read, s, l, core, &pipe);
    if ASCEND_IS_AIV {
        if constexpr (Partial) {
            MergePrefillOutputsImpl<DType, IsTnd>(a, read, s, l, &pipe);
        }
    }
#endif
}
// Resolve device specialization on host, as in fwd_dispatch_impl.hpp.
#define MLA_PREFILL_BOOL_SWITCH(COND, CONST_NAME, ...)                                                                 \
    do {                                                                                                               \
        if (COND) {                                                                                                    \
            constexpr bool CONST_NAME = true;                                                                          \
            __VA_ARGS__                                                                                                \
        } else {                                                                                                       \
            constexpr bool CONST_NAME = false;                                                                         \
            __VA_ARGS__                                                                                                \
        }                                                                                                              \
    } while (0)

template <typename DType>
aclError LaunchPrefillBsnd(const MlaPrefillBsndLaunchArgs& a)
{
    MlaBatchLaunchScope launch;
    MLA_PREFILL_BOOL_SWITCH(a.params.mBase <= 128, Brc, {
        MLA_PREFILL_BOOL_SWITCH(a.context.splits != 1, Partial, {
            MLA_PREFILL_BOOL_SWITCH(a.context.pageSize != 0, Paged, {
                MLA_PREFILL_BOOL_SWITCH(a.context.tnd != 0, IsTnd, {
                    PrefillAttention<DType, Brc, Partial, Paged, IsTnd><<<a.context.cores, nullptr, a.stream>>>(a);
                });
            });
        });
    });
    return launch.Result();
}
#undef MLA_PREFILL_BOOL_SWITCH
} // namespace flash_attn::mla
