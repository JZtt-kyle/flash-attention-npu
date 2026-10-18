// CPU-only GQA scheduling harness. Native oracle is a separate executable.
#include "mla_gqa_schedule.h"
#include <algorithm>
#include <iostream>
#include <vector>

using namespace flash_attn::mla;
template <typename T>
void Emit(const T& values, uint32_t count)
{
    for (uint32_t i = 0; i < count; ++i)
        std::cout << ',' << values[i];
}
MlaPlatformInfo Platform(uint32_t cores, uint32_t vectors, int missing)
{
    MlaPlatformInfo p{};
    p.aicNum = cores;
    p.aivNum = vectors;
    p.localMemory = {524288, 196608, 65536, 65536, 131072};
    p.libApiWorkspaceBytes = 16777216;
    for (uint32_t i = 0; i < kMlaMaxAic; ++i)
        p.blockDims[i] = int(i) == missing ? 0 : i + 1;
    return p;
}
int main()
{
    int mode;
    MlaGqaSchedule plan; // Failed builds must preserve the previous valid plan.
    while (std::cin >> mode) {
        if (mode == 0) {
            int64_t h, hk;
            MlaGqaOptions o{};
            uint32_t cores, vectors;
            int missing;
            std::cin >> h >> hk >> o.packGqa >> o.smMargin >> cores >> vectors >> missing;
            MlaGqaResources r{};
            r.headGroups = 123;
            const auto status = ResolveGqaResources(h, hk, o, Platform(cores, vectors, missing), r);
            std::cout << '[' << int(status) << ',' << r.headGroups << ',' << r.headsPerGroup << ',' << r.packed << ','
                      << r.platform.aicNum << ',' << r.platform.aivNum;
            Emit(r.platform.blockDims, kMlaMaxAic);
            std::cout << ',' << r.platform.libApiWorkspaceBytes << ',' << r.platform.localMemory.ubBytes << "]\n";
        } else if (mode == 1) {
            MlaGqaConfig c{};
            MlaGqaOptions o{};
            MlaWindowConfig w{};
            int qLayout, kLayout, mutate, flags;
            uint32_t cores;
            std::cin >> c.shape.batch >> c.shape.queryHeads >> c.kvHeads >> c.valueDim >> qLayout >> kLayout >>
                c.shape.query.totalTokens >> c.shape.kv.totalTokens >> c.shape.query.maxLength >>
                c.shape.kv.maxLength >> c.mBase >> c.nBase >> c.splits >> c.pageSize >> o.packGqa >> o.smMargin >>
                cores >> w.windowLeft >> w.windowRight >> w.causal >> mutate >> flags;
            c.shape.query.layout = MlaSequenceLayout(qLayout);
            c.shape.kv.layout = MlaSequenceLayout(kLayout);
            w.maxQueryLength = c.shape.query.maxLength;
            w.maxKvLength = c.shape.kv.maxLength;
            std::vector<int32_t> data[4];
            MlaUsedInputs inputs{};
            MlaLengthArray* arrays[] = {&inputs.cuQuery, &inputs.cuKv, &inputs.usedQuery, &inputs.usedKv};
            for (uint32_t axis = 0; axis < 4; ++axis) {
                uint32_t count;
                std::cin >> count;
                data[axis].resize(count);
                for (auto& value : data[axis])
                    std::cin >> value;
                *arrays[axis] = {flags & (1 << axis) ? data[axis].data() : nullptr, count};
            }
            const auto status = BuildGqaSchedule(c, inputs, w, o, Platform(cores, cores * 2, -1), plan);
            // Poison original caller buffers immediately after construction.
            if (mutate)
                for (auto& values : data)
                    std::fill(values.begin(), values.end(), -99);
            const auto& r = plan.Resources();
            std::cout << "{\"status\":" << int(status) << ",\"groups\":" << plan.Groups()
                      << ",\"capacity\":" << plan.AccumulatorCapacity() << ",\"geometry\":[" << r.packed << ','
                      << r.headGroups << ',' << r.headsPerGroup << ',' << r.platform.aicNum << "],\"prefix\":[0";
            for (size_t i = 1; i < plan.GroupPrefix().size(); ++i)
                std::cout << ',' << plan.GroupPrefix()[i];
            std::cout << "],\"native\":[";
            const auto& n = plan.Native();
            const auto& f = n.fd;
            std::cout << n.usedCoreNum << ',' << n.maxCost << ',' << f.numOfFdHead << ',' << f.maxS2SplitNum << ','
                      << f.usedVecNumOfFd << ',' << n.totalCost << ',' << n.totalS2Blocks;
            const uint32_t count = r.platform.aicNum;
            Emit(n.bN2End, count);
            Emit(n.gS1End, count);
            Emit(n.s2End, count);
            Emit(f.bN2IdxOfFdHead, count);
            Emit(f.gS1IdxOfFdHead, count);
            Emit(f.s2SplitNumOfFdHead, count);
            Emit(f.s2SplitStartIdxOfCore, count);
            Emit(f.gS1SplitNumOfFdHead, count);
            Emit(f.gS1LastPartSizeOfFdHead, count);
            Emit(f.gS1IdxEndOfFdHead, 2 * count);
            Emit(f.gS1IdxEndOfFdHeadSplit, 2 * count);
            std::cout << "]}\n";
        } else if (mode == 2) {
            uint64_t group;
            uint32_t split, localRow;
            std::cin >> group >> split >> localRow;
            MlaGqaTask t{};
            t.batch = 123;
            MlaGqaRow r{};
            r.out = 456;
            const auto ts = plan.Task(group, split, t);
            const auto rs = plan.Row(group, split, localRow, r);
            std::cout << "{\"task_status\":" << int(ts) << ",\"row_status\":" << int(rs) << ",\"task\":[" << t.batch
                      << ',' << t.headGroup << ',' << t.kvHead << ',' << t.m << ',' << t.split << ',' << t.splitCount
                      << ',' << t.owner << ',' << t.rowBegin << ',' << t.rowEnd << ',' << t.s2Begin << ',' << t.s2End
                      << "],\"row\":[" << r.batch << ',' << r.head << ',' << r.kvHead << ',' << r.query << ','
                      << r.queryToken << ',' << r.kvStart << ',' << r.out << ',' << r.lse << ',' << r.accumulatorOut
                      << ',' << r.accumulatorLse << "]}\n";
        } else {
            return 2;
        }
        if (!std::cin)
            return 2;
        std::cout << std::flush;
    }
}
