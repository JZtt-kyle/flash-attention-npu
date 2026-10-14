// Test-only CPU bridge. No torch, device runtime or production dispatch.
#include "mla_split_kv_plan.h"
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
void Layout(std::ostream& out, const MlaSplitKvPlanLayout& l)
{
    out << "{\"geometry\":";
    Values(out, l.maxGroups, l.maxTasks, l.partialRows, l.emptyCapacity);
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
std::string Summary(const MlaSplitKvPlan& plan)
{
    std::ostringstream out;
    out << "{\"groups\":" << plan.Groups() << ",\"tasks\":" << plan.Tasks() << ",\"prefix\":";
    Array(out, plan.GroupPrefix());
    out << ",\"layout\":";
    Layout(out, plan.Layout());
    out << '}';
    return out.str();
}
MlaSplitKvPlanConfig Config()
{
    MlaSplitKvPlanConfig c{};
    std::cin >> c.batch >> c.queryCapacity >> c.kvCapacity >> c.queryHeads >> c.mBaseSize >> c.s2BaseSize >>
        c.numSplits >> c.aicNum >> c.blockDim >> c.cvRatio >> c.libApiBytes;
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
    MlaSplitKvPlan plan;
    int mode;
    while (std::cin >> mode) {
        if (mode == 0) {
            const auto c = Config();
            MlaSplitKvPlanLayout l{};
            l.totalBytes = 987;
            Checked(l, [&] { return BuildSplitKvPlanLayout(c, l); });
            std::cout << ",\"layout\":";
            Layout(std::cout, l);
        } else if (mode == 1) {
            auto c = Config();
            MlaWindowConfig window{};
            int causal;
            uint32_t count;
            std::cin >> window.maxQueryLength >> window.maxKvLength >> window.windowLeft >> window.windowRight >>
                causal >> count;
            window.causal = causal;
            std::vector<MlaCausalSequence> sequences(count);
            for (auto& s : sequences) {
                std::cin >> s.queryLength >> s.kvLength;
            }
            const auto before = Summary(plan);
            const auto status = BuildSplitKvPlan(c, window, sequences, plan);
            // Lifetime/ownership: every caller-owned input is modified and
            // destroyed before subsequent task/address queries.
            c = {};
            window = {};
            for (auto& s : sequences) {
                s = {-1, -1};
            }
            std::cout << "{\"status\":" << int(status)
                      << ",\"unchanged\":" << (before == Summary(plan) ? "true" : "false")
                      << ",\"plan\":" << Summary(plan);
        } else if (mode == 2) {
            uint64_t index;
            std::cin >> index;
            MlaSplitKvPlannedTask t{};
            t.task = 987;
            Checked(t, [&] { return plan.Task(index, t); });
            std::cout << ",\"task\":";
            Values(std::cout, t.task, t.group, t.iteration, t.batch, t.m, t.split, t.owner);
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
            uint64_t index;
            std::cin >> index;
            MlaSplitKvPlannedMerge m{};
            m.group = 987;
            Checked(m, [&] { return plan.Merge(index, m); });
            std::cout << ",\"merge\":";
            Values(std::cout, m.group, m.iteration, m.batch, m.m, m.owner, m.rowBegin, m.rowEnd);
            std::cout << ",\"offsets\":";
            Values(std::cout, m.outOffsetBytes, m.sumOffsetBytes, m.maxOffsetBytes);
        } else if (mode == 4) {
            uint64_t index;
            uint32_t row;
            std::cin >> index >> row;
            MlaSplitKvPlannedRow r{};
            r.out = 987;
            Checked(r, [&] { return plan.Row(index, row, r); });
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
