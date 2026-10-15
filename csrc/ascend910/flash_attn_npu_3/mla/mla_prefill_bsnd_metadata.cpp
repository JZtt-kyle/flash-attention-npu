// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
// Native partition algorithm: ca18aee attention/common/op_host/split_core.cpp.
#include "mla_prefill_bsnd_metadata.h"
#include "detail/cann_split_core.h"
#include <algorithm>
#include <new>
#include <vector>
namespace flash_attn::mla {
bool BuildPrefillBsndMetadata(const MlaPrefillBsndMetadataArgs& a)
{
    const auto& c = a.context;
    MlaPrefillBsndLayout l{};
    if (!a.output || a.output % 4 || !PrefillBsndLayout(c, l) || a.words != l.words ||
        a.words > (UINT64_MAX - a.output) / 4)
        return false;
    // Same subtraction-based range checks as BuildUsedMetadataWithSchedule.
    // Neither read-only length array may overlap the writable record.
    for (const auto address : {a.usedQuery, a.usedKv}) {
        if (address && c.batch && address < a.output + uint64_t(a.words) * 4 &&
            (address >= a.output || c.batch > (a.output - address) / 4))
            return false;
    }
    auto* out = reinterpret_cast<uint32_t*>(a.output);
    std::fill(out, out + a.words, 0);
    out[0] = kMlaMetadataMagic;
    out[1] = out[2] = kMlaPrefillBsndVersion;
    out[3] = a.words;
    out[4] = c.dtype;
    out[5] = 1;
    if ((c.batch && (bool(c.usedFlags & 1) != bool(a.usedQuery) || bool(c.usedFlags & 2) != bool(a.usedKv))) ||
        a.usedQuery % 4 || a.usedKv % 4 || (a.usedQuery && c.batch > (UINT64_MAX - a.usedQuery) / 4) ||
        (a.usedKv && c.batch > (UINT64_MAX - a.usedKv) / 4) ||
        (c.dtype != uint32_t(MlaMetadataDtype::Fp16) && c.dtype != uint32_t(MlaMetadataDtype::Bf16)))
        return true;
    try {
        std::vector<uint32_t> result(a.words, 0);
        const uint32_t header[] = {kMlaMetadataMagic,
                                   27,
                                   27,
                                   a.words,
                                   c.dtype,
                                   0,
                                   c.batch,
                                   c.sq,
                                   c.sk,
                                   c.heads,
                                   c.kvHeads,
                                   c.splits,
                                   c.cores,
                                   l.m,
                                   l.n,
                                   c.usedFlags,
                                   c.window.kind,
                                   uint32_t(c.window.left),
                                   uint32_t(c.window.right),
                                   0};
        std::copy(header, header + 20, result.begin());
        detail::BaseInfo base{};
        base.bSize = c.batch;
        base.n2Size = c.kvHeads;
        base.gSize = c.heads / c.kvHeads;
        base.s1Size = c.sq;
        base.s2Size = c.sk;
        base.isS1G = true;
        base.actualLenQDims = base.actualLenKvDims = c.batch;
        base.attenMaskFlag = c.window.kind != 0;
        base.sparseMode = c.window.kind == 2 ? 4 : c.window.kind == 1 ? 3 : 0;
        base.preToken = c.window.kind == 1 ? INT32_MAX : c.window.left;
        base.nextToken = c.window.kind == 1 ? 0 : c.window.right;
        uint32_t* tail = result.data() + 20 + 2 * c.batch;
        uint32_t* prefix = tail + 346;
        uint64_t groups = 0, tiles = 0;
        for (uint32_t b = 0; b < c.batch; ++b) {
            const uint32_t q = a.usedQuery ? reinterpret_cast<const uint32_t*>(a.usedQuery)[b] : c.sq;
            const uint32_t k = a.usedKv ? reinterpret_cast<const uint32_t*>(a.usedKv)[b] : c.sk;
            if (q > c.sq || k > c.sk)
                return true;
            result[20 + 2 * b] = q;
            result[21 + 2 * b] = k;
            base.actualSeqS1Size.push_back(q);
            base.actualSeqS2Size.push_back(k);
            const uint64_t count = l.groups ? (uint64_t(q) * base.gSize + l.m - 1) / l.m : 0;
            for (uint32_t h = 0; h < c.kvHeads; ++h) {
                groups += count;
                prefix += 2;
                prefix[0] = uint32_t(groups);
                prefix[1] = uint32_t(groups >> 32);
            }
            tiles += count * c.kvHeads * ((uint64_t(k) + l.n - 1) / l.n);
            if (tiles > UINT32_MAX)
                return true;
        }
        detail::SplitParam param{};
        param.mBaseSize = l.m;
        param.s2BaseSize = l.n;
        param.gS1BaseSizeOfFd = 8;
        detail::SplitResult n{c.cores, 2};
        if (l.groups) {
            detail::SplitCore(c.cores, base, param, n);
        } else {
            n.usedCoreNum = 1;
            n.bN2End[0] = c.batch * c.kvHeads;
        }
        if (!n.usedCoreNum || n.usedCoreNum > c.cores || n.numOfFdHead > c.cores || n.maxS2SplitNum > c.cores)
            return true;
        tail[0] = c.window.kind;
        tail[1] = uint32_t(c.window.left);
        tail[2] = uint32_t(c.window.right);
        tail[3] = n.usedCoreNum;
        auto emit = [&](uint32_t offset, const auto& array) { std::copy(array.begin(), array.end(), tail + offset); };
        emit(4, n.bN2End);
        emit(30, n.gS1End);
        emit(56, n.s2End);
        tail[82] = n.numOfFdHead;
        tail[83] = n.maxS2SplitNum;
        tail[84] = n.usedVecNumOfFd;
        tail[85] = 8;
        emit(86, n.fdRes.bN2IdxOfFdHead);
        emit(112, n.fdRes.gS1IdxOfFdHead);
        emit(138, n.fdRes.s2SplitNumOfFdHead);
        emit(164, n.fdRes.s2SplitStartIdxOfCore);
        emit(190, n.fdRes.gS1SplitNumOfFdHead);
        emit(216, n.fdRes.gS1LastPartSizeOfFdHead);
        emit(242, n.fdRes.gS1IdxEndOfFdHead);
        emit(294, n.fdRes.gS1IdxEndOfFdHeadSplit);
        std::copy(result.begin(), result.end(), out);
        return true;
    } catch (const std::bad_alloc&) {
        return true;
    }
}
} // namespace flash_attn::mla
