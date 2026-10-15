// Test-only CPU bridge. No torch, device runtime or production dispatch.
#include "mla_split_paged_tnd_plan.h"
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
std::string Summary(const MlaSplitPagedTndPlan& plan)
{
    std::ostringstream out;
    out << "{\"groups\":" << plan.Groups() << ",\"tasks\":" << plan.Tasks() << ",\"prefix\":";
    Array(out, plan.GroupPrefix());
    out << ",\"layout\":";
    Layout(out, plan.Layout());
    out << '}';
    return out.str();
}
struct Request {
    MlaPagedShape shape{};
    MlaPlatformInfo platform{};
    int64_t splits;
};
Request Config()
{
    Request r{};
    int64_t total;
    int layout, block, memory;
    std::cin >> r.shape.batch >> r.shape.query.maxLength >> r.shape.logicalKvCapacity >> r.shape.pageSize >>
        r.shape.queryHeads >> r.splits >> r.platform.aicNum >> r.platform.libApiWorkspaceBytes >> total >> layout >>
        block >> memory;
    r.shape.query.layout = MlaSequenceLayout(layout);
    r.shape.query.totalTokens = total;
    r.platform.aivNum = 2 * r.platform.aicNum;
    const bool active = r.shape.batch && total && r.shape.logicalKvCapacity;
    r.platform.blockDims.fill(block >= 0 ? uint32_t(block) : active ? r.platform.aicNum : 1);
    r.platform.localMemory = {524288, uint64_t(memory ? 1 : 196608), 65536, 65536, 131072};
    return r;
}
struct Lengths {
    int32_t empty = 0;
    std::array<std::vector<int32_t>, 3> values;
    std::array<MlaLengthArray, 3> arrays;
    void Read()
    {
        for (uint32_t i = 0; i < 3; ++i) {
            int present;
            uint64_t count, size;
            std::cin >> present >> count >> size;
            if (size > 300000)
                std::abort();
            values[i].resize(size);
            for (auto& v : values[i])
                std::cin >> v;
            // Only matching descriptors may be read by the implementation.
            arrays[i] = {present ? (size ? values[i].data() : &empty) : nullptr, count};
        }
    }
    MlaPagedInputs Inputs() const
    {
        return {arrays[0], arrays[1], arrays[2]};
    }
};
void Budget(std::ostream& out, const MlaSplitPagedTndBudget& b)
{
    out << "{\"bounds\":";
    Values(out, b.maxGroups, b.maxTasks, b.partialRows, b.scratchBytes, b.userBytes, b.totalBytes);
    out << ",\"accum\":";
    Tensor(out, b.accumulator.out);
    out << ",\"lse_accum\":";
    Tensor(out, b.accumulator.lse);
    out << '}';
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
int PlanMain()
{
    MlaSplitPagedTndPlan owned;
    auto& plan = owned;
    int mode;
    while (std::cin >> mode) {
        if (mode == 0) {
            const auto c = Config();
            MlaSplitPagedTndBudget l{};
            l.totalBytes = 987;
            Checked(l, [&] { return BuildSplitPagedTndBudget(c.shape, c.platform, c.splits, l); });
            std::cout << ",\"layout\":";
            Budget(std::cout, l);
        } else if (mode == 1) {
            auto c = Config();
            MlaWindowConfig window{};
            int causal;
            std::cin >> window.maxQueryLength >> window.maxKvLength >> window.windowLeft >> window.windowRight >>
                causal;
            window.causal = causal;
            Lengths lengths;
            lengths.Read();
            const auto before = Summary(plan);
            const auto status = BuildSplitPagedTndPlan(c.shape, c.platform, lengths.Inputs(), window, c.splits, owned);
            for (auto& v : lengths.values)
                std::fill(v.begin(), v.end(), -1);
            c = {};
            window = {};
            std::cout << "{\"status\":" << int(status)
                      << ",\"unchanged\":" << (before == Summary(plan) ? "true" : "false")
                      << ",\"plan\":" << Summary(plan) << ",\"budget\":";
            Budget(std::cout, owned.budget);
            std::cout << ",\"tiling\":";
            Values(std::cout, owned.paged.schedule.mBaseSize, owned.paged.schedule.s2BaseSize,
                   owned.paged.schedule.s2Size, owned.mBase, owned.nBase);
        } else if (mode == 2) {
            uint64_t index;
            std::cin >> index;
            MlaSplitPagedTndPlannedTask task{};
            auto& t = task.partial;
            t.task = 987;
            Checked(task, [&] { return plan.Task(index, task); });
            std::cout << ",\"task\":";
            Values(std::cout, t.task, t.group, t.iteration, t.batch, t.m, t.split, t.owner);
            std::cout << ",\"sequence\":";
            Values(std::cout, task.queryStart, task.queryCapacity, task.queryLength, task.kvLength);
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
    return 0;
}
