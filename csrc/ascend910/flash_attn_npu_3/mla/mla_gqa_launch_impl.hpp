// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
// Reuse the validated partial/FD numerics with M15g1 scheduling coordinates.
#pragma once
#include "kernel_operator.h"
#include "mla_gqa_launch.h"
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
#include "mla_gqa_consumer.h"
#include "kernel/mla_used_metadata.hpp"
#include "kernel/mla_paged_metadata.hpp"
#include "kernel/mla_split_kv.hpp"
#include "kernel/mla_prefill.hpp"
#include "block/mla_split_kv_merge.hpp"
namespace flash_attn::mla {
__aicore__ inline bool LoadGqa(const MlaGqaLaunchArgs& a, Kernel::MetadataReader& read, MlaGqaLayout& l,
                               MlaAutoSplitBsndSchedule& s)
{
    const auto& c = a.context;
    if (!GqaLayout(c, l) || !ValidUsedMetadataAddress(a.metadata, l.words) || AscendC::GetBlockNum() != l.cores ||
        c.dtype != (AscendC::IsSameType<MLA_GQA_TYPE, half>::value ? uint32_t(MlaMetadataDtype::Fp16)
                                                                   : uint32_t(MlaMetadataDtype::Bf16))) {
        AscendC::Trap();
        return false;
    }
    Kernel::UsedMetadataReaders arrays;
    const uint64_t addresses[] = {a.cuQuery, a.cuKv, a.usedQuery, a.usedKv};
    for (uint32_t i = 0; i < 4; ++i) {
        const uint64_t count = uint64_t(c.batch) + (i < 2);
        const bool present = i == 0 ? GqaTnd(c) : i == 1 ? c.layout == 1 : c.usedFlags & (1U << (i - 2));
        if ((present && !ValidUsedMetadataAddress(addresses[i], count)) || (!present && addresses[i])) {
            AscendC::Trap();
            return false;
        }
        if (addresses[i])
            arrays.data[i].data.SetGlobalBuffer((__gm__ uint32_t*)addresses[i]);
    }
    read.data.SetGlobalBuffer((__gm__ uint32_t*)a.metadata);
    if (ReadGqaMetadata(read, l.words, c, arrays, s) != MlaMetadataReadStatus::Success) {
        AscendC::Trap();
        return false;
    }
    if (c.pageSize) {
        MlaPagedMetadataContext p{};
        p.batch = c.batch;
        p.pageSize = c.pageSize;
        p.logicalKvCapacity = c.sk;
        if (CheckPagedCacheDescriptor(p, a.cache, c.kvHeads, c.valueDim == 512 ? 64 : 192, c.valueDim) !=
            MlaMetadataReadStatus::Success) {
            AscendC::Trap();
            return false;
        }
        Kernel::PageTableReader pages;
        pages.data.SetGlobalBuffer((__gm__ uint32_t*)a.cache.table.address);
        for (uint32_t b = 0; b < c.batch; ++b)
            for (uint32_t i = 0; i < (read(32 + 8 * b + 7) + c.pageSize - 1) / c.pageSize; ++i) {
                const int32_t id = int32_t(pages(uint64_t(b) * a.cache.layout.tableRowStride + i));
                if (id < 0 || uint64_t(id) >= a.cache.layout.physicalPages) {
                    AscendC::Trap();
                    return false;
                }
            }
    }
    return true;
}
template <typename T>
__aicore__ inline void GqaInitialize(uint64_t address, uint64_t count, T value)
{
    const uint64_t cores = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
    const uint64_t chunk = (count + cores * 64 - 1) / (cores * 64) * 64;
    const uint64_t start = uint64_t(AscendC::GetBlockIdx()) * chunk;
    if (start < count) {
        AscendC::GlobalTensor<T> tensor;
        tensor.SetGlobalBuffer((__gm__ T*)address);
        matmul::InitOutput<T>(tensor[start], chunk < count - start ? chunk : count - start, value);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    }
}
template <bool IsTnd, bool Paged>
__aicore__ inline void GqaCompressed(const MlaGqaLaunchArgs& a, const MlaGqaLayout& l, const MlaAutoSplitBsndTask& t,
                                     const Kernel::MetadataReader& read, uint32_t core, uint64_t row)
{
    const auto& c = a.context;
    const uint32_t b = t.batch / l.headGroups, head = (t.batch % l.headGroups) * l.g;
    MlaDenseKernelSchedule schedule{};
    schedule.batch = c.batch;
    schedule.sq = c.sq;
    schedule.sk = c.sk;
    schedule.heads = l.g;
    schedule.mBaseSize = l.m;
    schedule.s2BaseSize = l.n;
    schedule.blockDim = schedule.usedCoreNum = l.cores;
    schedule.cvRatio = 2;
    schedule.mm1ResSize = l.m * ((l.n + 31) / 32 * 32);
    schedule.mm2ResSize = l.m * 512;
    MlaSplitKvKernelParams p{};
    p.window = c.window;
    p.tasks[core] = {b, t.m, t.split, t.s2Begin, t.s2End};
    p.totalTasks = l.cores;
    p.outOffsetBytes = l.outOffset + row * 512 * 4;
    p.sumOffsetBytes = l.sumOffset + row * 8 * 4;
    p.maxOffsetBytes = l.maxOffset + row * 8 * 4;
    MlaTaskLengths lengths{read(32 + 8 * b), read(32 + 8 * b + 6), read(32 + 8 * b + 7),
                           ReadMetadataWide(read, 32 + 8 * b + 2)};
    AscendC::TPipe pipe;
    Kernel::MlaGqaPartial<MLA_GQA_TYPE, IsTnd, Paged> op;
    op.RunFragment(schedule, p, lengths, a.scale, (GM_ADDR)a.qv + uint64_t(head) * a.inputs.qv.headStride * 2,
                   (GM_ADDR)a.q + uint64_t(head) * a.inputs.q.headStride * 2, (GM_ADDR)a.v, (GM_ADDR)a.k,
                   (GM_ADDR)a.workspace, &pipe, a.inputs, c.layout == 1, a.cache, a.softcap);
}
template <bool Brc, bool Paged>
__aicore__ inline void GqaPrefill(const MlaGqaLaunchArgs& a, const MlaGqaLayout& l, const MlaAutoSplitBsndTask& t,
                                  const Kernel::MetadataReader& read, uint32_t core, uint64_t row)
{
    const auto& c = a.context;
    const uint32_t b = t.batch / l.headGroups, head = (t.batch % l.headGroups) * l.g;
    const uint32_t kvHead = head / (c.heads / c.kvHeads);
    const auto& in = a.inputs;
    const uint64_t qStart = GqaTnd(c) ? uint64_t(read(32 + 8 * b)) * in.q.rowStride : uint64_t(b) * in.q.batchStride;
    const uint64_t kStart = c.layout == 1 ? uint64_t(read(32 + 8 * b + 2)) * in.k.rowStride
                            : Paged       ? 0
                                          : uint64_t(b) * in.k.batchStride;
    const uint64_t vStart = c.layout == 1 ? uint64_t(read(32 + 8 * b + 2)) * in.v.rowStride
                            : Paged       ? 0
                                          : uint64_t(b) * in.v.batchStride;
    MlaPrefillParams p{1, c.sq, c.sk, l.g, 1, l.m, l.n, l.cores, a.scale, c.window, in.q, in.k, in.v, a.softcap};
    MlaPrefillTask task{
        0, 0, t.rowBegin, t.rowEnd - t.rowBegin, read(32 + 8 * b + 6), read(32 + 8 * b + 7), t.s2Begin, t.s2End};
    auto cache = a.cache;
    if constexpr (Paged)
        cache.table.address += uint64_t(b) * cache.layout.tableRowStride * 4;
    AscendC::TPipe pipe;
    Kernel::MlaPrefill<MLA_GQA_TYPE, Brc, true, Paged> op;
    op.Run(p, task, core, (GM_ADDR)a.q + (qStart + uint64_t(head) * in.q.headStride) * 2,
           (GM_ADDR)a.k + (kStart + uint64_t(kvHead) * in.k.headStride) * 2,
           (GM_ADDR)a.v + (vStart + uint64_t(kvHead) * in.v.headStride) * 2, (GM_ADDR)a.out, (GM_ADDR)a.lse,
           (GM_ADDR)a.workspace, (GM_ADDR)a.workspace + l.outOffset + row * 128 * 4,
           (GM_ADDR)a.workspace + l.sumOffset + row * 8 * 4, (GM_ADDR)a.workspace + l.maxOffset + row * 8 * 4, &pipe,
           &cache);
}
template <uint32_t Dim, bool IsTnd, bool Accumulator>
__aicore__ inline void GqaMerge(const MlaGqaLaunchArgs& a, const MlaGqaLayout& l, const MlaSplitKvMergeParams& p,
                                uint64_t row, uint32_t begin, uint32_t count)
{
    AscendC::TPipe pipe;
    Block::MlaSplitKvMerge<MLA_GQA_TYPE, Accumulator, IsTnd, Dim, true> op;
    op.Init(p, (GM_ADDR)a.workspace + l.outOffset + row * Dim * 4, (GM_ADDR)a.workspace + l.sumOffset + row * 8 * 4,
            (GM_ADDR)a.workspace + l.maxOffset + row * 8 * 4, (GM_ADDR)a.out, (GM_ADDR)a.lse, (GM_ADDR)a.accum,
            (GM_ADDR)a.accumLse, &pipe, a.softcap);
    op.Run(begin, count);
}
} // namespace flash_attn::mla
#endif
namespace flash_attn::mla {
__global__ __aicore__ __mix__(1, 2) void MLA_GQA_INIT(MlaGqaLaunchArgs a)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
    Kernel::MetadataReader read;
    MlaGqaLayout l{};
    MlaAutoSplitBsndSchedule s{};
    if (!LoadGqa(a, read, l, s))
        return;
    if ASCEND_IS_AIV {
        AscendC::TPipe pipe;
        const auto& c = a.context;
        const uint64_t rows = (GqaTnd(c) ? uint64_t(c.totalQ) : uint64_t(c.batch) * c.sq) * c.heads;
        GqaInitialize<MLA_GQA_TYPE>(a.out, rows * c.valueDim, MLA_GQA_TYPE(0));
        GqaInitialize<float>(a.lse, rows, !c.pageSize && GqaEmptyKv(c) ? float(1.0 / 0.0) : -float(1.0 / 0.0));
        if (c.splits != 1) {
            const uint64_t capacity = c.splits ? c.splits : 26;
            GqaInitialize<float>(a.accum, rows * capacity * c.valueDim, 0.0f);
            GqaInitialize<float>(a.accumLse, rows * capacity, -float(1.0 / 0.0));
        }
    }
#endif
}
__global__ __aicore__ __mix__(1, 2) void MLA_GQA_ATTENTION(MlaGqaLaunchArgs a)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
    Kernel::MetadataReader read;
    MlaGqaLayout l{};
    MlaAutoSplitBsndSchedule s{};
    if (!LoadGqa(a, read, l, s) || GqaEmptyKv(a.context))
        return;
    AscendC::SetSyncBaseAddr(a.ffts);
    uint32_t core = AscendC::GetBlockIdx();
    if ASCEND_IS_AIV
        core /= AscendC::GetSubBlockNum();
    const auto& c = a.context;
    for (uint64_t index = core; index < s.groups * l.capacity; index += l.cores) {
        MlaAutoSplitBsndTask t{};
        if (!GqaTask(read, c, l, s, index / l.capacity, index % l.capacity, t))
            continue;
        const uint64_t row = index * l.m;
        if (c.valueDim == 512) {
            if (GqaTnd(c)) {
                if (c.pageSize)
                    GqaCompressed<true, true>(a, l, t, read, core, row);
                else
                    GqaCompressed<true, false>(a, l, t, read, core, row);
            } else if (c.pageSize)
                GqaCompressed<false, true>(a, l, t, read, core, row);
            else
                GqaCompressed<false, false>(a, l, t, read, core, row);
        } else if (l.m <= 128) {
            if (c.pageSize)
                GqaPrefill<true, true>(a, l, t, read, core, row);
            else
                GqaPrefill<true, false>(a, l, t, read, core, row);
        } else if (c.pageSize)
            GqaPrefill<false, true>(a, l, t, read, core, row);
        else
            GqaPrefill<false, false>(a, l, t, read, core, row);
    }
#endif
}
__global__ __aicore__ __mix__(1, 2) void MLA_GQA_MERGE(MlaGqaLaunchArgs a)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
    Kernel::MetadataReader read;
    MlaGqaLayout l{};
    MlaAutoSplitBsndSchedule s{};
    if (!LoadGqa(a, read, l, s) || GqaEmptyKv(a.context))
        return;
    if ASCEND_IS_AIV {
        const auto& c = a.context;
        const uint32_t tile = 192 / l.capacity > 8 ? 8 : 192 / l.capacity ? 192 / l.capacity : 1;
        const uint32_t tiles = (l.m + tile - 1) / tile;
        for (uint64_t index = AscendC::GetBlockIdx(); index < s.groups * tiles; index += l.cores * 2) {
            const uint64_t group = index / tiles;
            const uint32_t begin = (index % tiles) * tile;
            MlaAutoSplitBsndTask t{};
            if (!GqaTask(read, c, l, s, group, 0, t)) {
                AscendC::Trap();
                return;
            }
            const uint32_t b = t.batch / l.headGroups;
            MlaSplitKvMergeParams p{};
            p.batch = GqaTnd(c) ? 1 : c.batch;
            p.sq = GqaTnd(c) ? c.totalQ : c.sq;
            p.batchIndex = GqaTnd(c) ? read(32 + 8 * b) : b;
            p.heads = l.g;
            p.kvHeads = l.headGroups;
            p.kvHead = t.batch % l.headGroups;
            p.rowStart = t.rowBegin;
            p.rows = t.rowEnd - t.rowBegin;
            if (begin >= p.rows)
                continue;
            p.mBase = l.m;
            p.splits = t.splitCount;
            p.queryLength = read(32 + 8 * b + 6);
            p.kvLength = read(32 + 8 * b + 7);
            p.adapt = 1;
            p.maskKind = c.window.kind;
            p.left = c.window.left;
            p.right = c.window.right;
            for (uint32_t split = 0; split < p.splits; ++split) {
                if (!GqaTask(read, c, l, s, group, split, t)) {
                    AscendC::Trap();
                    return;
                }
                p.begin[split] = p.kvLength < t.s2Begin * l.n ? p.kvLength : t.s2Begin * l.n;
                p.end[split] = p.kvLength < t.s2End * l.n ? p.kvLength : t.s2End * l.n;
            }
            const uint64_t row = group * l.capacity * l.m;
            const uint32_t count = tile < p.rows - begin ? tile : p.rows - begin;
#define MLA_GQA_MERGE_DIM(D, T)                                                                                        \
    if (c.splits == 1)                                                                                                 \
        GqaMerge<D, T, false>(a, l, p, row, begin, count);                                                             \
    else                                                                                                               \
        GqaMerge<D, T, true>(a, l, p, row, begin, count)
            if (c.valueDim == 512) {
                if (GqaTnd(c)) {
                    MLA_GQA_MERGE_DIM(512, true);
                } else {
                    MLA_GQA_MERGE_DIM(512, false);
                }
            } else if (GqaTnd(c)) {
                MLA_GQA_MERGE_DIM(128, true);
            } else {
                MLA_GQA_MERGE_DIM(128, false);
            }
#undef MLA_GQA_MERGE_DIM
        }
    }
#endif
}
aclError MLA_GQA_LAUNCH(const MlaGqaLaunchArgs& a)
{
    const uint32_t cores = a.context.physicalCores - a.context.margin;
    {
        MlaBatchLaunchScope launch;
        MLA_GQA_INIT<<<cores, nullptr, a.stream>>>(a);
        const auto error = launch.Result();
        if (error != ACL_SUCCESS)
            return error;
    }
    {
        MlaBatchLaunchScope launch;
        MLA_GQA_ATTENTION<<<cores, nullptr, a.stream>>>(a);
        const auto error = launch.Result();
        if (error != ACL_SUCCESS)
            return error;
    }
    MlaBatchLaunchScope launch;
    MLA_GQA_MERGE<<<cores, nullptr, a.stream>>>(a);
    return launch.Result();
}
} // namespace flash_attn::mla
