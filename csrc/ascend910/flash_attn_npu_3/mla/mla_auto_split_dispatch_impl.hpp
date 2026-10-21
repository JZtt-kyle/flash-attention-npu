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
template <typename DType>
__aicore__ inline bool LoadAutoSplit(GM_ADDR metadata, uint64_t checked, MlaAutoSplitBsndSchedule& s,
                                     Kernel::MetadataReader& read)
{
    const uint32_t dtype =
        AscendC::IsSameType<DType, half>::value ? uint32_t(MlaMetadataDtype::Fp16) : uint32_t(MlaMetadataDtype::Bf16);
    if (!LoadCheckedAutoSplitSchedule(checked, dtype, s))
        return false;
    read.data.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(metadata));
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
__aicore__ inline void AutoInit(const MlaAutoSplitContext& c, GM_ADDR out, GM_ADDR lse, GM_ADDR accum, GM_ADDR accumLse)
{
    if ASCEND_IS_AIV {
        const uint64_t rows = uint64_t(c.used.heads) * UsedMetadataTotal(c.used.query);
        AutoInitialize<DType>(out, rows * 512, DType(0));
        AutoInitialize<float>(lse, rows, UsedMetadataTotal(c.used.kv) ? -float(1.0 / 0.0) : float(1.0 / 0.0));
        AutoInitialize<float>(accum, rows * kMlaAutoSplitCapacity * 512, 0.0f);
        AutoInitialize<float>(accumLse, rows * kMlaAutoSplitCapacity, -float(1.0 / 0.0));
    }
}
template <typename DType, bool IsTnd, bool IsMixed>
__aicore__ inline void AutoPartial(GM_ADDR qc, GM_ADDR qr, GM_ADDR kv, GM_ADDR kr, GM_ADDR workspace, GM_ADDR metadata,
                                   const MlaAutoSplitContext& c, float scale, MlaBsndInputLayouts layouts, GM_ADDR out,
                                   GM_ADDR lse, GM_ADDR accum, GM_ADDR accumLse, float softcap,
                                   const MlaAutoSplitBsndSchedule& s, const Kernel::MetadataReader& read,
                                   AscendC::TPipe* pipe)
{
    Kernel::MlaAutoSplit<DType, IsTnd, false> op;
    op(s, read, c.window, scale, qc, qr, kv, kr, out, lse, workspace, pipe, metadata + kMlaUsedMetadataPrefixWords * 4,
       c.used.query.totalTokens.low, layouts, IsTnd && !IsMixed, accum, accumLse, softcap);
}
template <typename DType, bool IsTnd, bool IsMixed>
__aicore__ inline void AutoMerge(GM_ADDR workspace, const MlaAutoSplitContext& c, GM_ADDR out, GM_ADDR lse,
                                 GM_ADDR accum, GM_ADDR accumLse, float softcap, const MlaAutoSplitBsndSchedule& s,
                                 const Kernel::MetadataReader& read, AscendC::TPipe* pipe)
{
    if ASCEND_IS_AIV {
        const uint32_t owner = AscendC::GetBlockIdx();
        if (owner >= s.vectors) {
            AscendC::SyncAll();
            return;
        }
        const uint32_t base = auto_split_detail::Base(c.used.batch);
        const uint32_t firstHead = owner ? read(base + 242 + owner - 1) : 0;
        const uint32_t firstPart = owner ? read(base + 294 + owner - 1) + 1 : 0;
        const uint32_t lastHead = read(base + 242 + owner);
        const uint32_t lastPart = read(base + 294 + owner);
        pipe->Reset();
        Block::MlaSplitKvMerge<DType, false, IsTnd> block;
        block.InitBuffers(pipe);
        AscendC::ICachePreLoad(2);
        AscendC::SyncAll();
        // The validated vector endpoints already give this lane's contiguous
        // FD interval. Do not rediscover ownership for every other lane's row.
        for (uint32_t h = firstHead; h <= lastHead; ++h) {
            const uint32_t parts = h == lastHead ? lastPart + 1 : read(base + 190 + h);
            const uint32_t first = h == firstHead ? firstPart : 0;
            if (first >= parts)
                continue;
            MlaAutoSplitBsndMerge r{};
            AutoSplitBsndMerge(read, s, h, 0, r);
            MlaAutoSplitBsndTask t{};
            AutoSplitBsndTask(read, s, c.window, r.group, 0, t);
            auto p = AutoMergeParams<IsTnd>(read, s, c.window, t);
            block.Bind(p, workspace + r.outOffset, workspace + r.sumOffset, workspace + r.maxOffset, out, lse, accum,
                       accumLse, softcap);
            const uint32_t tile = block.TileRows();
            for (uint32_t part = first; part < parts; ++part) {
                const uint32_t end = p.rows < (part + 1) * 8 ? p.rows : (part + 1) * 8;
                for (uint32_t row = part * 8; row < end; row += tile)
                    block.Run(row, end - row < tile ? end - row : tile);
            }
        }
        block.Finish();
    }
}
} // namespace
#endif
template <typename DType, bool IsTnd, bool IsMixed>
__global__ __aicore__ __mix__(1, 2) void AutoPartialKernel(uint64_t ffts, GM_ADDR qc, GM_ADDR qr, GM_ADDR kv,
                                                           GM_ADDR kr, GM_ADDR workspace, GM_ADDR metadata,
                                                           MlaAutoSplitContext c, uint64_t checked, float scale,
                                                           MlaBsndInputLayouts layouts, GM_ADDR out, GM_ADDR lse,
                                                           GM_ADDR accum, GM_ADDR accumLse, float softcap)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
    MlaAutoSplitBsndSchedule s;
    Kernel::MetadataReader read;
    if (!LoadAutoSplit<DType>(metadata, checked, s, read))
        return;
    AscendC::SetSyncBaseAddr(ffts);
    AscendC::TPipe pipe;
    AutoInit<DType, IsTnd, IsMixed>(c, out, lse, accum, accumLse);
    if (!s.groups)
        return;
    // Direct fragments write final output, so all output initialization must
    // finish first. Idle vector cores participate in both barriers.
    if ASCEND_IS_AIV {
        AscendC::SyncAll();
    }
    AutoPartial<DType, IsTnd, IsMixed>(qc, qr, kv, kr, workspace, metadata, c, scale, layouts, out, lse, accum,
                                       accumLse, softcap, s, read, &pipe);
    if ASCEND_IS_AIV {
        AutoMerge<DType, IsTnd, IsMixed>(workspace, c, out, lse, accum, accumLse, softcap, s, read, &pipe);
    }
#endif
}
template <typename DType, bool IsTnd, bool IsMixed>
aclError LaunchAutoSplit(const MlaSplitLaunchArgs& args)
{
    const auto& a = args.used;
    MlaAutoSplitContext c{a.context, a.window};
    c.hasSoftcap = a.softcap > 0;
    MlaBatchLaunchScope launch;
    AutoPartialKernel<DType, IsTnd, IsMixed><<<a.context.blockDim, nullptr, a.stream>>>(
        a.ffts, a.qc, a.qr, a.kv, a.kr, a.workspace, a.metadata, c, a.checked, a.scale, a.layouts, a.output, a.lse,
        args.accum, args.accumLse, a.softcap);
    return launch.Result();
}
} // namespace flash_attn::mla

namespace flash_attn::mla {
namespace {
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
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
__aicore__ inline void AutoInit(const MlaAutoSplitPagedContext& c, GM_ADDR out, GM_ADDR lse, GM_ADDR accum,
                                GM_ADDR accumLse)
{
    if ASCEND_IS_AIV {
        const uint64_t rows = uint64_t(c.paged.heads) * (uint64_t(c.paged.query.totalTokens.low) |
                                                         (uint64_t(c.paged.query.totalTokens.high) << 32));
        AutoInitializePaged<DType>(out, rows * 512, DType(0));
        AutoInitializePaged<float>(lse, rows, -float(1.0 / 0.0));
        AutoInitializePaged<float>(accum, rows * kMlaAutoSplitCapacity * 512, 0.0f);
        AutoInitializePaged<float>(accumLse, rows * kMlaAutoSplitCapacity, -float(1.0 / 0.0));
    }
}
template <typename DType, bool IsTnd>
__aicore__ inline void AutoPartial(GM_ADDR qc, GM_ADDR qr, GM_ADDR workspace, GM_ADDR metadata,
                                   const MlaAutoSplitPagedContext& c, const MlaPagedCacheInputs& cache, float scale,
                                   MlaBsndInputLayouts layouts, GM_ADDR out, GM_ADDR lse, GM_ADDR accum,
                                   GM_ADDR accumLse, float softcap, const MlaAutoSplitBsndSchedule& s,
                                   const Kernel::MetadataReader& read, AscendC::TPipe* pipe)
{
    Kernel::MlaAutoSplit<DType, IsTnd, true> op;
    op(s, read, c.window, scale, qc, qr, reinterpret_cast<GM_ADDR>(cache.v.address),
       reinterpret_cast<GM_ADDR>(cache.k.address), out, lse, workspace, pipe,
       metadata + kMlaPagedMetadataPrefixWords * 4, c.paged.query.totalTokens.low, layouts, false, accum, accumLse,
       softcap, reinterpret_cast<GM_ADDR>(cache.table.address), &cache.layout);
}
template <typename DType, bool IsTnd>
__aicore__ inline void AutoMerge(GM_ADDR workspace, const MlaAutoSplitPagedContext& c, GM_ADDR out, GM_ADDR lse,
                                 GM_ADDR accum, GM_ADDR accumLse, float softcap, const MlaAutoSplitBsndSchedule& s,
                                 const Kernel::MetadataReader& read, AscendC::TPipe* pipe)
{
    if ASCEND_IS_AIV {
        const uint32_t owner = AscendC::GetBlockIdx();
        if (owner >= s.vectors) {
            AscendC::SyncAll();
            return;
        }
        const uint32_t base = auto_split_detail::Base<true>(c.paged.batch);
        const uint32_t firstHead = owner ? read(base + 242 + owner - 1) : 0;
        const uint32_t firstPart = owner ? read(base + 294 + owner - 1) + 1 : 0;
        const uint32_t lastHead = read(base + 242 + owner);
        const uint32_t lastPart = read(base + 294 + owner);
        pipe->Reset();
        Block::MlaSplitKvMerge<DType, false, IsTnd> block;
        block.InitBuffers(pipe);
        AscendC::ICachePreLoad(2);
        AscendC::SyncAll();
        // The validated vector endpoints already give this lane's contiguous
        // FD interval. Do not rediscover ownership for every other lane's row.
        for (uint32_t h = firstHead; h <= lastHead; ++h) {
            const uint32_t parts = h == lastHead ? lastPart + 1 : read(base + 190 + h);
            const uint32_t first = h == firstHead ? firstPart : 0;
            if (first >= parts)
                continue;
            MlaAutoSplitBsndMerge r{};
            AutoSplitBsndMerge<true>(read, s, h, 0, r);
            MlaAutoSplitBsndTask t{};
            AutoSplitBsndTask<true>(read, s, c.window, r.group, 0, t);
            auto p = AutoMergeParamsPaged<IsTnd>(read, s, c.window, t);
            block.Bind(p, workspace + r.outOffset, workspace + r.sumOffset, workspace + r.maxOffset, out, lse, accum,
                       accumLse, softcap);
            const uint32_t tile = block.TileRows();
            for (uint32_t part = first; part < parts; ++part) {
                const uint32_t end = p.rows < (part + 1) * 8 ? p.rows : (part + 1) * 8;
                for (uint32_t row = part * 8; row < end; row += tile)
                    block.Run(row, end - row < tile ? end - row : tile);
            }
        }
        block.Finish();
    }
}
} // namespace
#endif
template <typename DType, bool IsTnd>
__global__ __aicore__ __mix__(1, 2) void AutoPagedPartialKernel(uint64_t ffts, GM_ADDR qc, GM_ADDR qr,
                                                                GM_ADDR workspace, GM_ADDR metadata,
                                                                MlaAutoSplitPagedContext c, uint64_t checked,
                                                                MlaPagedCacheInputs cache, float scale,
                                                                MlaBsndInputLayouts layouts, GM_ADDR out, GM_ADDR lse,
                                                                GM_ADDR accum, GM_ADDR accumLse, float softcap)
{
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 2201
    MlaAutoSplitBsndSchedule s;
    Kernel::MetadataReader read;
    if (!LoadAutoSplit<DType>(metadata, checked, s, read))
        return;
    AscendC::SetSyncBaseAddr(ffts);
    AscendC::TPipe pipe;
    AutoInit<DType, IsTnd>(c, out, lse, accum, accumLse);
    if (!s.groups)
        return;
    // Direct fragments write final output, so all output initialization must
    // finish first. Idle vector cores participate in both barriers.
    if ASCEND_IS_AIV {
        AscendC::SyncAll();
    }
    AutoPartial<DType, IsTnd>(qc, qr, workspace, metadata, c, cache, scale, layouts, out, lse, accum, accumLse, softcap,
                              s, read, &pipe);
    if ASCEND_IS_AIV {
        AutoMerge<DType, IsTnd>(workspace, c, out, lse, accum, accumLse, softcap, s, read, &pipe);
    }
#endif
}
template <typename DType, bool IsTnd>
aclError LaunchAutoSplitPaged(const MlaSplitPagedLaunchArgs& args)
{
    const auto& a = args.paged;
    MlaAutoSplitPagedContext c{a.context, a.window, uint32_t(args.softcap > 0)};
    MlaBatchLaunchScope launch;
    AutoPagedPartialKernel<DType, IsTnd><<<a.context.blockDim, nullptr, a.stream>>>(
        a.ffts, a.qc, a.qr, a.workspace, a.metadata, c, a.checked, a.cache, a.scale, a.layouts, a.output, a.lse,
        args.accum, args.accumLse, args.softcap);
    return launch.Result();
}
} // namespace flash_attn::mla
