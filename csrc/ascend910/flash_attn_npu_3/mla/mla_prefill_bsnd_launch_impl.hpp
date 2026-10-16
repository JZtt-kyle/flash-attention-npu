// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
#pragma once
#include "kernel_operator.h"
#include "mla_prefill_bsnd_launch.h"
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
#include "lib/matmul_intf.h"
#include "kernel/mla_prefill.hpp"
#include "kernel/mla_used_metadata.hpp"
#include "kernel/mla_paged_metadata.hpp"
#include "mla_prefill_bsnd_consumer.h"
#include "block/mla_split_kv_merge.hpp"
#endif
namespace flash_attn::mla {
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
__aicore__ inline bool LoadPrefill(const MlaPrefillBsndLaunchArgs& a, Kernel::MetadataReader& read,
                                   MlaAutoSplitBsndSchedule& s, MlaPrefillBsndLayout& layout)
{
    const auto& c = a.context;
    if (!PrefillBsndLayout(c, layout) || !ValidUsedMetadataAddress(a.metadata, layout.words) ||
        (a.usedQuery && !ValidUsedMetadataAddress(a.usedQuery, c.batch)) ||
        (a.usedKv && !ValidUsedMetadataAddress(a.usedKv, c.batch)) ||
        (c.tnd && (!ValidUsedMetadataAddress(a.cuQuery, uint64_t(c.batch) + 1) ||
                   (c.tnd == 1 && !ValidUsedMetadataAddress(a.cuKv, uint64_t(c.batch) + 1)))) ||
        AscendC::GetBlockNum() != c.cores ||
        c.dtype != (AscendC::IsSameType<MLA_PREFILL_TYPE, half>::value ? uint32_t(MlaMetadataDtype::Fp16)
                                                                       : uint32_t(MlaMetadataDtype::Bf16)) ||
        (c.batch && (bool(c.usedFlags & 1) != bool(a.usedQuery) || bool(c.usedFlags & 2) != bool(a.usedKv)))) {
        AscendC::Trap();
        return false;
    }
    read.data.SetGlobalBuffer((__gm__ uint32_t*)a.metadata);
    Kernel::UsedMetadataReaders arrays;
    const uint32_t usedIndex = c.tnd ? 2 : 0;
    if (c.tnd) {
        arrays.data[0].data.SetGlobalBuffer((__gm__ uint32_t*)a.cuQuery);
        arrays.data[1].data.SetGlobalBuffer((__gm__ uint32_t*)a.cuKv);
    }
    if (a.usedQuery)
        arrays.data[usedIndex].data.SetGlobalBuffer((__gm__ uint32_t*)a.usedQuery);
    if (a.usedKv)
        arrays.data[usedIndex + 1].data.SetGlobalBuffer((__gm__ uint32_t*)a.usedKv);
    if (ReadPrefillBsndMetadata(read, layout.words, c, arrays, s) != MlaMetadataReadStatus::Success) {
        AscendC::Trap();
        return false;
    }
    if (c.pageSize) {
        Kernel::PageTableReader pages;
        pages.data.SetGlobalBuffer((__gm__ uint32_t*)a.cache.table.address);
        if (ReadPrefillPagedCache(read, c, a.cache, pages) != MlaMetadataReadStatus::Success) {
            AscendC::Trap();
            return false;
        }
    }
    return true;
}
template <bool Brc, bool Partial, bool Paged>
__aicore__ inline void RunPrefillImpl(const MlaPrefillBsndLaunchArgs& a, const MlaPrefillTask& task, uint32_t core,
                                      uint64_t row, const MlaPrefillBsndLayout& l)
{
    AscendC::TPipe pipe;
    Kernel::MlaPrefill<MLA_PREFILL_TYPE, Brc, Partial, Paged> kernel;
    kernel.Run(a.params, task, core, (GM_ADDR)a.q, (GM_ADDR)a.k, (GM_ADDR)a.v, (GM_ADDR)a.out, (GM_ADDR)a.lse,
               (GM_ADDR)a.workspace, (GM_ADDR)a.workspace + l.outOffset + row * 128 * 4,
               (GM_ADDR)a.workspace + l.sumOffset + row * 8 * 4, (GM_ADDR)a.workspace + l.maxOffset + row * 8 * 4,
               &pipe, &a.cache);
}
template <bool Brc, bool Partial>
__aicore__ inline void RunPrefill(const MlaPrefillBsndLaunchArgs& a, const MlaPrefillTask& task, uint32_t core,
                                  uint64_t row, const MlaPrefillBsndLayout& l)
{
    if (a.context.pageSize)
        RunPrefillImpl<Brc, Partial, true>(a, task, core, row, l);
    else
        RunPrefillImpl<Brc, Partial, false>(a, task, core, row, l);
}
template <bool IsTnd>
__aicore__ inline void PrefillMerge(const MlaPrefillBsndLaunchArgs& a, const MlaSplitKvMergeParams& p,
                                    const MlaPrefillBsndLayout& l, uint64_t row, uint32_t begin, uint32_t end)
{
    AscendC::TPipe pipe;
    Block::MlaSplitKvMerge<MLA_PREFILL_TYPE, true, IsTnd, 128> merge;
    merge.Init(p, (GM_ADDR)a.workspace + l.outOffset + row * 128 * 4, (GM_ADDR)a.workspace + l.sumOffset + row * 8 * 4,
               (GM_ADDR)a.workspace + l.maxOffset + row * 8 * 4, (GM_ADDR)a.out, (GM_ADDR)a.lse, (GM_ADDR)a.accum,
               (GM_ADDR)a.accumLse, &pipe, a.params.softcap);
    for (uint32_t start = begin; start < end; start += merge.TileRows())
        merge.Run(start, merge.TileRows() < end - start ? merge.TileRows() : end - start);
}
// Same disjoint, 64-element aligned initialization as AutoInitialize in
// mla_auto_split_launch_impl.hpp. Stream ordering separates it from attention.
template <typename T>
__aicore__ inline void PrefillInitialize(uint64_t address, uint64_t elements, T value)
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

#endif
__global__ __aicore__ __mix__(1, 2) void MLA_PREFILL_INIT(MlaPrefillBsndLaunchArgs a)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
    Kernel::MetadataReader read;
    MlaAutoSplitBsndSchedule s{};
    MlaPrefillBsndLayout l{};
    if (!LoadPrefill(a, read, s, l))
        return;
    if ASCEND_IS_AIV {
        AscendC::TPipe pipe;
        const auto& c = a.context;
        const uint64_t rows = (c.tnd ? uint64_t(c.totalQ) : uint64_t(c.batch) * c.sq) * c.heads;
        PrefillInitialize<MLA_PREFILL_TYPE>(a.out, rows * 128, MLA_PREFILL_TYPE(0));
        PrefillInitialize<float>(a.lse, rows,
                                 (c.pageSize || (c.tnd == 1 ? c.totalK : c.sk)) ? -float(1.0 / 0.0) : float(1.0 / 0.0));
        if (c.splits != 1) {
            const uint64_t capacity = c.splits ? c.splits : 26;
            PrefillInitialize<float>(a.accum, rows * capacity * 128, 0.0f);
            PrefillInitialize<float>(a.accumLse, rows * capacity, -float(1.0 / 0.0));
        }
    }
#endif
}
__global__ __aicore__ __mix__(1, 2) void MLA_PREFILL_ATTENTION(MlaPrefillBsndLaunchArgs a)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
    Kernel::MetadataReader read;
    MlaAutoSplitBsndSchedule s{};
    MlaPrefillBsndLayout l{};
    if (!LoadPrefill(a, read, s, l))
        return;
    AscendC::SetSyncBaseAddr(a.ffts);
    uint32_t core = AscendC::GetBlockIdx();
    if ASCEND_IS_AIV {
        core /= AscendC::GetSubBlockNum();
    }
    const auto& c = a.context;
    for (uint64_t index = core; index < s.groups * l.capacity; index += c.cores) {
        MlaAutoSplitBsndTask t{};
        if (!PrefillBsndTask(read, c, s, index / l.capacity, index % l.capacity, t))
            continue;
        const uint32_t b = t.batch / c.kvHeads;
        MlaPrefillTask task{a.context.tnd ? 0U : b,
                            t.batch % c.kvHeads,
                            t.rowBegin,
                            t.rowEnd - t.rowBegin,
                            read(PrefillLengthWord(c, b)),
                            read(PrefillLengthWord(c, b) + 1),
                            t.s2Begin,
                            t.s2End};
        auto current = a;
        if (c.tnd) {
            const uint64_t qStart = read(PrefillSequenceWord(c, b));
            const uint64_t kStart = read(PrefillSequenceWord(c, b) + 2);
            current.q += qStart * a.params.q.rowStride * 2;
            if (c.pageSize) {
                // The task's tensor batch is zero after rebasing Q. PA keeps
                // its physical pools and selects only the logical table row.
                current.cache.table.address += uint64_t(b) * a.cache.layout.tableRowStride * 4;
            } else {
                current.k += (c.tnd == 1 ? kStart * a.params.k.rowStride : uint64_t(b) * a.params.k.batchStride) * 2;
                current.v += (c.tnd == 1 ? kStart * a.params.v.rowStride : uint64_t(b) * a.params.v.batchStride) * 2;
            }
            current.out += qStart * c.heads * 128 * 2;
            current.lse += qStart * 4;
            current.params.batch = 1;
            current.params.sq = c.totalQ;
        }
        if (c.splits == 1) {
            if (l.m <= 128)
                RunPrefill<true, false>(current, task, core, index * l.m, l);
            else
                RunPrefill<false, false>(current, task, core, index * l.m, l);
        } else {
            if (l.m <= 128)
                RunPrefill<true, true>(current, task, core, index * l.m, l);
            else
                RunPrefill<false, true>(current, task, core, index * l.m, l);
        }
    }
#endif
}
__global__ __aicore__ __mix__(1, 2) void MLA_PREFILL_MERGE(MlaPrefillBsndLaunchArgs a)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
    Kernel::MetadataReader read;
    MlaAutoSplitBsndSchedule s{};
    MlaPrefillBsndLayout l{};
    if (!LoadPrefill(a, read, s, l))
        return;
    if ASCEND_IS_AIV {
        const auto& c = a.context;
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
            MlaAutoSplitBsndTask t{};
            if (!PrefillBsndTask(read, c, s, group, 0, t)) {
                AscendC::Trap();
                return;
            }
            if (begin >= t.rowEnd - t.rowBegin)
                continue;
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
            for (uint32_t split = 0; split < p.splits; ++split) {
                if (!PrefillBsndTask(read, c, s, group, split, t)) {
                    AscendC::Trap();
                    return;
                }
                p.begin[split] = p.kvLength < t.s2Begin * l.n ? p.kvLength : t.s2Begin * l.n;
                p.end[split] = p.kvLength < t.s2End * l.n ? p.kvLength : t.s2End * l.n;
            }
            const uint64_t row = group * l.capacity * l.m;
            if (c.tnd) {
                p.batch = 1;
                p.sq = c.totalQ;
                p.batchIndex = read(PrefillSequenceWord(c, t.batch / c.kvHeads));
                PrefillMerge<true>(a, p, l, row, begin, begin + tileRows < p.rows ? begin + tileRows : p.rows);
            } else
                PrefillMerge<false>(a, p, l, row, begin,
                                    parallelRows && begin + tileRows < p.rows ? begin + tileRows : p.rows);
        }
    }
#endif
}
aclError MLA_PREFILL_LAUNCH(const MlaPrefillBsndLaunchArgs& a)
{
    if (a.context.tnd || a.context.pageSize) {
        MlaBatchLaunchScope launch;
        MLA_PREFILL_INIT<<<a.context.cores, nullptr, a.stream>>>(a);
        const auto error = launch.Result();
        if (error != ACL_SUCCESS)
            return error;
    }
    {
        MlaBatchLaunchScope launch;
        MLA_PREFILL_ATTENTION<<<a.context.cores, nullptr, a.stream>>>(a);
        const auto error = launch.Result();
        if (error != ACL_SUCCESS || a.context.splits == 1)
            return error;
    }
    MlaBatchLaunchScope launch;
    MLA_PREFILL_MERGE<<<a.context.cores, nullptr, a.stream>>>(a);
    return launch.Result();
}
} // namespace flash_attn::mla
