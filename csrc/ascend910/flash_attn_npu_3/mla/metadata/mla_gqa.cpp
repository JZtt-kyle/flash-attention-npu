// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
#include "mla_gqa.h"
#include "../tiling/mla_gqa_schedule.h"
#include <algorithm>
#include <new>

namespace flash_attn::mla {
bool BuildGqaMetadata(const MlaGqaMetadataArgs& a)
{
    auto c = a.context;
    MlaGqaLayout l{};
    if (!GqaLayout(c, l) || !a.output || a.output % 4 || a.words != l.words || a.words > (UINT64_MAX - a.output) / 4)
        return false;
    const uint64_t addresses[] = {a.cuQuery, a.cuKv, a.usedQuery, a.usedKv};
    for (uint32_t axis = 0; axis < 4; ++axis) {
        const uint64_t address = addresses[axis], count = uint64_t(c.batch) + (axis < 2);
        if (address && count && address < a.output + uint64_t(a.words) * 4 &&
            (address >= a.output || count > (a.output - address) / 4))
            return false;
    }
    auto* out = reinterpret_cast<uint32_t*>(a.output);
    std::fill(out, out + a.words, 0);
    for (uint32_t i = 0; i < 5; ++i)
        out[i] = GqaHeader(c, l, i);
    out[5] = 1;
    for (uint32_t axis = 0; axis < 4; ++axis) {
        const uint64_t count = uint64_t(c.batch) + (axis < 2), address = addresses[axis];
        const bool required = axis == 0 ? GqaTnd(c) : axis == 1 ? c.layout == 1 : c.usedFlags & (1U << (axis - 2));
        if ((required && count && !address) || (!required && address) || address % 4 ||
            (address && count > (UINT64_MAX - address) / 4))
            return true;
    }
    if ((c.dtype != uint32_t(MlaMetadataDtype::Fp16) && c.dtype != uint32_t(MlaMetadataDtype::Bf16)) ||
        a.inferTotalQ > 1 || (a.inferTotalQ && c.layout != 2 && c.layout != 4))
        return true;
    if (a.inferTotalQ) {
        c.totalQ = reinterpret_cast<const uint32_t*>(a.cuQuery)[c.batch];
        if (!GqaLayout(c, l))
            return true;
    }
    try {
        MlaPlatformInfo physical{};
        physical.aicNum = c.physicalCores;
        physical.aivNum = 2 * c.physicalCores;
        for (uint32_t i = 0; i < c.physicalCores; ++i)
            physical.blockDims[i] = i + 1;
        const MlaUsedShape shape{c.batch,
                                 c.heads,
                                 {GqaTnd(c) ? MlaSequenceLayout::Tnd : MlaSequenceLayout::Bsnd,
                                  GqaTnd(c) ? int64_t(c.totalQ) : int64_t(c.batch) * c.sq, c.sq},
                                 {c.layout == 1 ? MlaSequenceLayout::Tnd : MlaSequenceLayout::Bsnd,
                                  c.layout == 1 ? int64_t(c.totalK) : int64_t(c.batch) * c.sk, c.sk}};
        MlaUsedInputs inputs{};
        MlaLengthArray* arrays[] = {&inputs.cuQuery, &inputs.cuKv, &inputs.usedQuery, &inputs.usedKv};
        for (uint32_t i = 0; i < 4; ++i)
            *arrays[i] = {reinterpret_cast<const int32_t*>(addresses[i]),
                          addresses[i] ? uint64_t(c.batch) + (i < 2) : 0};
        MlaGqaSchedule schedule;
        // Positive split modes consume the same safe native tail, but use the
        // existing explicit fragment rule in the consumer, as prefill does.
        if (BuildGqaSchedule({shape, c.kvHeads, c.valueDim, l.m, l.n, 0, c.pageSize}, inputs,
                             {c.sq, c.sk, c.window.kind == 1 ? -1 : c.window.left,
                              c.window.kind == 1 ? -1 : c.window.right, c.window.kind == 1},
                             {c.packed, c.margin}, physical, schedule) != MlaScheduleStatus::Success)
            return true;
        std::vector<uint32_t> result(l.words, 0);
        for (uint32_t i = 0; i < 32; ++i)
            result[i] = GqaHeader(c, l, i);
        for (uint32_t b = 0; b < c.batch; ++b) {
            const auto& s = schedule.Lengths().sequences[b];
            const uint32_t values[] = {uint32_t(s.queryStart), uint32_t(s.queryStart >> 32),
                                       uint32_t(s.kvStart),    uint32_t(s.kvStart >> 32),
                                       s.queryCapacity,        s.kvCapacity,
                                       s.queryLength,          s.kvLength};
            std::copy(values, values + 8, result.begin() + 32 + 8 * b);
        }
        auto* tail = result.data() + GqaTail(c);
        const auto& n = schedule.Native();
        const auto& f = n.fd;
        tail[0] = c.window.kind;
        tail[1] = uint32_t(c.window.left);
        tail[2] = uint32_t(c.window.right);
        tail[3] = n.usedCoreNum;
        auto emit = [&](uint32_t offset, const auto& array) { std::copy(array.begin(), array.end(), tail + offset); };
        emit(4, n.bN2End);
        emit(30, n.gS1End);
        emit(56, n.s2End);
        tail[82] = f.numOfFdHead;
        tail[83] = f.maxS2SplitNum;
        tail[84] = f.usedVecNumOfFd;
        tail[85] = 8;
        emit(86, f.bN2IdxOfFdHead);
        emit(112, f.gS1IdxOfFdHead);
        emit(138, f.s2SplitNumOfFdHead);
        emit(164, f.s2SplitStartIdxOfCore);
        emit(190, f.gS1SplitNumOfFdHead);
        emit(216, f.gS1LastPartSizeOfFdHead);
        emit(242, f.gS1IdxEndOfFdHead);
        emit(294, f.gS1IdxEndOfFdHeadSplit);
        const auto& prefix = schedule.GroupPrefix();
        for (uint64_t i = 0; i < prefix.size(); ++i) {
            tail[346 + 2 * i] = uint32_t(prefix[i]);
            tail[347 + 2 * i] = uint32_t(prefix[i] >> 32);
        }
        std::copy(result.begin(), result.end(), out);
        return true;
    } catch (const std::bad_alloc&) {
        return true;
    }
}
} // namespace flash_attn::mla
