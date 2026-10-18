// CPU production-codec harness; fixed CANN oracle is a separate executable.
#include "mla_gqa_metadata.h"
#include "mla_gqa_consumer.h"
#include <cstring>
#include <iostream>
#include <vector>
using namespace flash_attn::mla;
struct Reader {
    const std::vector<uint32_t>& data;
    uint32_t operator()(uint32_t i) const
    {
        return data.at(i);
    }
};
struct Arrays {
    const std::vector<int32_t>* data;
    uint32_t operator()(uint32_t a, uint32_t i) const
    {
        return data[a].at(i);
    }
};
int main()
{
    int mode;
    MlaGqaContext c{};
    MlaGqaLayout l{};
    MlaAutoSplitBsndSchedule schedule{};
    std::vector<uint32_t> record;
    while (std::cin >> mode) {
        if (mode == 1) {
            int fault, value, causal;
            int64_t left, right;
            std::cin >> c.batch >> c.sq >> c.sk >> c.heads >> c.kvHeads >> c.splits >> c.dtype >> c.physicalCores >>
                c.margin >> c.packed >> c.layout >> c.totalQ >> c.totalK >> c.pageSize >> c.usedFlags >> c.valueDim >>
                c.hasSoftcap >> left >> right >> causal >> fault >> value;
            std::vector<int32_t> data[4];
            for (auto& d : data) {
                uint32_t count;
                std::cin >> count;
                d.resize(count);
                for (auto& x : d)
                    std::cin >> x;
            }
            MlaWindow w{};
            const auto ws = NormalizeWindow({c.sq, c.sk, left, right, bool(causal)}, w);
            c.window = {uint32_t(w.kind), w.left, w.right};
            if (ws != MlaCausalStatus::Success || !GqaLayout(c, l)) {
                std::cout << "{\"layout\":false}\n" << std::flush;
                continue;
            }
            c.workspaceBytes = l.bytes;
            record.assign(l.words, 0xA5A5A5A5);
            MlaGqaMetadataArgs a{c,
                                 data[0].empty() ? 0 : uint64_t(data[0].data()),
                                 data[1].empty() ? 0 : uint64_t(data[1].data()),
                                 data[2].empty() ? 0 : uint64_t(data[2].data()),
                                 data[3].empty() ? 0 : uint64_t(data[3].data()),
                                 uint64_t(record.data()),
                                 l.words};
            if (fault == -2)
                a.output = 0;
            if (fault == -3)
                ++a.output;
            if (fault == -4)
                a.output = uint64_t(data[2].data());
            if (fault == -5)
                a.output = UINT64_MAX - 3;
            if (fault == -6)
                ++a.usedQuery;
            if (fault == -7)
                a.usedQuery = UINT64_MAX - 3;
            if (fault == -8)
                --a.words;
            if (fault == -9)
                a.cuQuery = 0;
            if (fault == -10)
                ++a.cuQuery;
            if (fault == -11)
                a.cuKv = UINT64_MAX - 3;
            if (fault == -12)
                a.output = uint64_t(data[0].data());
            if (fault == -13) {
                a.context.totalQ = 0;
                a.inferTotalQ = 1;
            }
            const bool built = BuildGqaMetadata(a);
            if (!built)
                for (auto x : record)
                    if (x != 0xA5A5A5A5)
                        return 4;
            if (fault >= 0)
                record.at(fault) = value;
            if (fault == -14)
                --c.workspaceBytes;
            if (fault == -15)
                ++data[3].at(0);
            if (fault == -16)
                ++data[0].at(1);
            std::memset(&schedule, 0xA5, sizeof(schedule));
            const auto before = schedule;
            const auto status = ReadGqaMetadata(Reader{record}, l.words, c, Arrays{data}, schedule);
            if (status != MlaMetadataReadStatus::Success && std::memcmp(&before, &schedule, sizeof(schedule)))
                return 3;
            std::cout << "{\"layout\":true,\"built\":" << built << ",\"status\":" << int(status)
                      << ",\"bytes\":" << l.bytes
                      << ",\"groups\":" << (status == MlaMetadataReadStatus::Success ? schedule.groups : 0)
                      << ",\"record\":[";
            for (uint32_t i = 0; i < l.words; ++i)
                std::cout << (i ? "," : "") << record[i];
            std::cout << "]}\n";
        } else if (mode == 2) {
            uint64_t group;
            uint32_t split;
            std::cin >> group >> split;
            MlaAutoSplitBsndTask t{};
            const bool ok = GqaTask(Reader{record}, c, l, schedule, group, split, t);
            std::cout << '[' << ok << ',' << t.batch << ',' << t.m << ',' << t.splitCount << ',' << t.rowBegin << ','
                      << t.rowEnd << ',' << t.s2Begin << ',' << t.s2End << "]\n";
        } else
            return 2;
        if (!std::cin)
            return 2;
        std::cout << std::flush;
    }
}
