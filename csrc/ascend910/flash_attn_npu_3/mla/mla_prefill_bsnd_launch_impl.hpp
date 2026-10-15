// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
#pragma once
#include "kernel_operator.h"
#include "mla_prefill_bsnd_launch.h"
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
#include "kernel/mla_prefill.hpp"
#include "kernel/mla_used_metadata.hpp"
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
        (a.usedKv && !ValidUsedMetadataAddress(a.usedKv, c.batch)) || AscendC::GetBlockNum() != c.cores ||
        c.dtype != (AscendC::IsSameType<MLA_PREFILL_TYPE, half>::value ? uint32_t(MlaMetadataDtype::Fp16)
                                                                       : uint32_t(MlaMetadataDtype::Bf16)) ||
        (c.batch && (bool(c.usedFlags & 1) != bool(a.usedQuery) || bool(c.usedFlags & 2) != bool(a.usedKv)))) {
        AscendC::Trap();
        return false;
    }
    read.data.SetGlobalBuffer((__gm__ uint32_t*)a.metadata);
    Kernel::UsedMetadataReaders arrays;
    if (a.usedQuery)
        arrays.data[0].data.SetGlobalBuffer((__gm__ uint32_t*)a.usedQuery);
    if (a.usedKv)
        arrays.data[1].data.SetGlobalBuffer((__gm__ uint32_t*)a.usedKv);
    if (ReadPrefillBsndMetadata(read, layout.words, c, arrays, s) != MlaMetadataReadStatus::Success) {
        AscendC::Trap();
        return false;
    }
    return true;
}
template <bool Brc, bool Partial>
__aicore__ inline void RunPrefill(const MlaPrefillBsndLaunchArgs& a, const MlaPrefillTask& task, uint32_t core,
                                  uint64_t row, const MlaPrefillBsndLayout& l)
{
    AscendC::TPipe pipe;
    Kernel::MlaPrefill<MLA_PREFILL_TYPE, Brc, Partial> kernel;
    kernel.Run(a.params, task, core, (GM_ADDR)a.q, (GM_ADDR)a.k, (GM_ADDR)a.v, (GM_ADDR)a.out, (GM_ADDR)a.lse,
               (GM_ADDR)a.workspace, (GM_ADDR)a.workspace + l.outOffset + row * 128 * 4,
               (GM_ADDR)a.workspace + l.sumOffset + row * 8 * 4, (GM_ADDR)a.workspace + l.maxOffset + row * 8 * 4,
               &pipe);
}
#endif
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
        MlaPrefillTask task{b,
                            t.batch % c.kvHeads,
                            t.rowBegin,
                            t.rowEnd - t.rowBegin,
                            read(20 + 2 * b),
                            read(21 + 2 * b),
                            t.s2Begin,
                            t.s2End};
        if (c.splits == 1) {
            if (l.m <= 128)
                RunPrefill<true, false>(a, task, core, index * l.m, l);
            else
                RunPrefill<false, false>(a, task, core, index * l.m, l);
        } else {
            if (l.m <= 128)
                RunPrefill<true, true>(a, task, core, index * l.m, l);
            else
                RunPrefill<false, true>(a, task, core, index * l.m, l);
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
        for (uint64_t group = AscendC::GetBlockIdx(); group < s.groups; group += c.cores * 2) {
            MlaAutoSplitBsndTask t{};
            if (!PrefillBsndTask(read, c, s, group, 0, t)) {
                AscendC::Trap();
                return;
            }
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
            p.queryLength = read(20 + 2 * p.batchIndex);
            p.kvLength = read(21 + 2 * p.batchIndex);
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
            AscendC::TPipe pipe;
            Block::MlaSplitKvMerge<MLA_PREFILL_TYPE, true, false, 128> merge;
            merge.Init(p, (GM_ADDR)a.workspace + l.outOffset + row * 128 * 4,
                       (GM_ADDR)a.workspace + l.sumOffset + row * 8 * 4,
                       (GM_ADDR)a.workspace + l.maxOffset + row * 8 * 4, (GM_ADDR)a.out, (GM_ADDR)a.lse,
                       (GM_ADDR)a.accum, (GM_ADDR)a.accumLse, &pipe);
            for (uint32_t start = 0; start < p.rows; start += merge.TileRows())
                merge.Run(start, merge.TileRows() < p.rows - start ? merge.TileRows() : p.rows - start);
        }
    }
#endif
}
aclError MLA_PREFILL_LAUNCH(const MlaPrefillBsndLaunchArgs& a)
{
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
