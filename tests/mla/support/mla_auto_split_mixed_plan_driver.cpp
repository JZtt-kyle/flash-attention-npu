// Test-only CPU bridge. No torch, device runtime or production dispatch.
#include "mla_auto_split_mixed_plan.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <sstream>

using namespace flash_attn::mla;

template <typename T>
void Array(std::ostream& out, const T& values)
{
    out << '[';
    bool first = true;
    for (auto v : values) {
        out << (first ? "" : ",") << v;
        first = false;
    }
    out << ']';
}
template <typename... T>
void Values(std::ostream& out, T... values)
{
    Array(out, std::array<uint64_t, sizeof...(T)>{uint64_t(values)...});
}
void Tensor(std::ostream& out, const MlaAccumulatorTensor& t)
{
    out << "{\"rank\":" << t.rank << ",\"shape\":";
    Array(out, t.shape);
    out << ",\"strides\":";
    Array(out, t.strides);
    out << ",\"elements\":" << t.elements << ",\"bytes\":" << t.bytes << '}';
}
void Layout(std::ostream& out, const MlaAutoSplitKvLayout& l)
{
    out << "{\"geometry\":";
    Values(out, l.maxGroups, l.maxPartialSlots, l.partialRows, l.emptyCapacity);
    out << ",\"buffers\":";
    Values(out, l.outOffsetBytes, l.sumOffsetBytes, l.maxOffsetBytes, l.userBytes, l.totalBytes);
    out << ",\"scratch\":";
    const auto& w = l.scratch.attention;
    Values(out, w.mm1ResSize, w.mm2ResSize, w.fdAccumOutSize, w.fdLogSumExpSize, w.qkOffsetBytes, w.pOffsetBytes,
           w.pvOffsetBytes, w.vec2OffsetBytes, w.kernelBytes, w.userBytes, w.libApiBytes, w.totalBytes);
    out << ",\"accum\":";
    Tensor(out, l.accumulator.out);
    out << ",\"lse_accum\":";
    Tensor(out, l.accumulator.lse);
    out << '}';
}
std::string Summary(const MlaAutoSplitMixedPlan& plan)
{
    std::ostringstream out;
    out << "{\"groups\":" << plan.Groups() << ",\"slots\":" << plan.PartialSlots() << ",\"prefix\":";
    Array(out, plan.GroupPrefix());
    out << ",\"partial_prefix\":";
    Array(out, plan.PartialPrefix());
    const auto& n = plan.Native();
    out << ",\"native\":";
    Values(out, n.usedCoreNum, n.maxCost, n.fd.numOfFdHead, n.fd.maxS2SplitNum, n.fd.usedVecNumOfFd, n.totalCost,
           n.totalS2Blocks, n.totalMBlocks);
    out << ",\"arrays\":[";
    Array(out, n.bN2End);
    out << ',';
    Array(out, n.gS1End);
    out << ',';
    Array(out, n.s2End);
    const auto& f = n.fd;
    out << ',';
    Array(out, f.bN2IdxOfFdHead);
    out << ',';
    Array(out, f.gS1IdxOfFdHead);
    out << ',';
    Array(out, f.s2SplitNumOfFdHead);
    out << ',';
    Array(out, f.s2SplitStartIdxOfCore);
    out << ',';
    Array(out, f.gS1SplitNumOfFdHead);
    out << ',';
    Array(out, f.gS1LastPartSizeOfFdHead);
    out << ',';
    Array(out, f.gS1IdxEndOfFdHead);
    out << ',';
    Array(out, f.gS1IdxEndOfFdHeadSplit);
    out << ']' << ",\"layout\":";
    Layout(out, plan.Layout());
    out << '}';
    return out.str();
}
MlaAutoSplitMixedPlanConfig Config()
{
    MlaAutoSplitMixedPlanConfig c{};
    std::cin >> c.shape.batch >> c.shape.maxQueryLength >> c.shape.kvCapacity >> c.shape.queryHeads >> c.mBaseSize >>
        c.s2BaseSize >> c.aicNum >> c.cvRatio >> c.libApiBytes >> c.shape.totalQuery;
    return c;
}
template <typename T, typename F>
void Checked(T& output, F function)
{
    unsigned char before[sizeof(T)];
    std::memcpy(before, &output, sizeof(T));
    auto status = function();
    std::cout << "{\"status\":" << int(status)
              << ",\"unchanged\":" << (std::memcmp(before, &output, sizeof(T)) == 0 ? "true" : "false");
}
int main()
{
    MlaAutoSplitMixedPlan plan;
    int mode;
    while (std::cin >> mode) {
        if (mode == 0) {
            const auto c = Config();
            MlaAutoSplitKvLayout l{};
            l.totalBytes = 987;
            Checked(l, [&] { return BuildAutoSplitMixedPlanLayout(c, l); });
            std::cout << ",\"layout\":";
            Layout(std::cout, l);
        } else if (mode == 1) {
            auto c = Config();
            MlaWindowConfig window{};
            int causal;
            std::cin >> window.maxQueryLength >> window.maxKvLength >> window.windowLeft >> window.windowRight >>
                causal;
            window.causal = causal;
            std::array<std::vector<int32_t>, 4> buffers;
            std::array<MlaLengthArray, 4> arrays{};
            for (size_t i = 0; i < arrays.size(); ++i) {
                int present;
                uint32_t count;
                std::cin >> present >> count;
                buffers[i].resize(std::max(1U, count));
                for (uint32_t j = 0; j < count; ++j)
                    std::cin >> buffers[i][j];
                arrays[i] = {present ? buffers[i].data() : nullptr, count};
            }
            const auto before = Summary(plan);
            const auto status = BuildAutoSplitMixedPlan(c, window, {arrays[0], arrays[1], arrays[2], arrays[3]}, plan);
            // Destroy/mutate every caller-owned object before querying the plan.
            c = {};
            window = {};
            for (auto& buffer : buffers)
                std::fill(buffer.begin(), buffer.end(), -1);
            std::cout << "{\"status\":" << int(status)
                      << ",\"unchanged\":" << (before == Summary(plan) ? "true" : "false")
                      << ",\"plan\":" << Summary(plan);
        } else if (mode == 2) {
            uint64_t index;
            uint32_t split;
            std::cin >> index >> split;
            MlaAutoSplitMixedTask task{};
            auto& t = task.native;
            t.group = 987;
            Checked(task, [&] { return plan.Task(index, split, task); });
            const auto& seq = task.sequence;
            std::cout << ",\"sequence\":";
            Values(std::cout, seq.queryStart, seq.kvStart, seq.queryCapacity, seq.kvCapacity, seq.queryLength,
                   seq.kvLength);
            std::cout << ",\"task\":";
            Values(std::cout, t.group, t.slot, t.batch, t.m, t.split, t.splitCount, t.owner, t.fdHead, t.kind);
            const auto& f = t.fragment;
            const auto& w = f.whole;
            std::cout << ",\"fragment\":";
            Values(std::cout, f.s2Begin, f.s2End, f.kvBegin, f.kvEnd, f.cost);
            std::cout << ",\"whole\":";
            Values(std::cout, w.rowBegin, w.rowEnd, w.queryBegin, w.queryEnd, w.kvBegin, w.kvEnd, w.s2Begin, w.s2End,
                   w.leadingEmptyRows, w.cost, w.normalS2Cost, w.lastS2Cost);
            std::cout << ",\"offsets\":";
            Values(std::cout, t.outOffsetBytes, t.sumOffsetBytes, t.maxOffsetBytes);
        } else if (mode == 3) {
            uint32_t index, part;
            std::cin >> index >> part;
            MlaAutoSplitKvMerge m{};
            m.group = 987;
            Checked(m, [&] { return plan.Merge(index, part, m); });
            std::cout << ",\"merge\":";
            Values(std::cout, m.group, m.firstSlot, m.batch, m.m, m.head, m.part, m.owner, m.splitCount, m.rowBegin,
                   m.rowEnd);
            std::cout << ",\"offsets\":";
            Values(std::cout, m.outOffsetBytes, m.sumOffsetBytes, m.maxOffsetBytes);
        } else if (mode == 4) {
            uint64_t index;
            uint32_t split, row;
            std::cin >> index >> split >> row;
            MlaAutoSplitKvRow r{};
            r.out = 987;
            Checked(r, [&] { return plan.Row(index, split, row, r); });
            std::cout << ",\"row\":";
            Values(std::cout, r.out, r.lse, r.accumulatorOut, r.accumulatorLse, r.partialOut, r.partialSum,
                   r.partialMax);
        } else {
            return 2;
        }
        std::cout << "}\n" << std::flush;
        if (!std::cin) {
            return 2;
        }
    }
}
