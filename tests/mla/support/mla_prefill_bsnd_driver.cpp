// CPU-only production producer/consumer harness. No torch or accelerator runtime.
#include "mla_prefill_bsnd_consumer.h"
#include <cstring>
#include <iostream>
#include <vector>
using namespace flash_attn::mla;
struct Reader {
    const std::vector<uint32_t>& v;
    uint32_t operator()(uint32_t i) const
    {
        return v.at(i);
    }
};
struct Arrays {
    const std::vector<uint32_t>&q, &k;
    uint32_t operator()(uint32_t axis, uint32_t i) const
    {
        return (axis ? k : q).at(i);
    }
};
int main()
{
    MlaPrefillBsndContext c{};
    int fault;
    uint32_t value;
    while (std::cin >> c.batch >> c.sq >> c.sk >> c.heads >> c.kvHeads >> c.splits >> c.cores >> c.dtype >>
           c.usedFlags >> c.window.kind >> c.window.left >> c.window.right >> fault >> value) {
        std::vector<uint32_t> q(c.batch), k(c.batch);
        for (auto& x : q)
            std::cin >> x;
        for (auto& x : k)
            std::cin >> x;
        MlaPrefillBsndLayout l{};
        if (!PrefillBsndLayout(c, l)) {
            std::cout << "{\"layout\":false}\n" << std::flush;
            continue;
        }
        c.workspaceBytes = l.bytes;
        std::vector<uint32_t> record(l.words, 0xA5A5A5A5);
        MlaPrefillBsndMetadataArgs a{c, c.usedFlags & 1 ? uint64_t(q.data()) : 0,
                                     c.usedFlags & 2 ? uint64_t(k.data()) : 0, uint64_t(record.data()), l.words};
        if (fault == -2)
            a.output = 0;
        if (fault == -3)
            ++a.output;
        if (fault == -4)
            a.output = uint64_t(q.data());
        if (fault == -5)
            a.output = UINT64_MAX - 3;
        if (fault == -6)
            ++a.usedQuery;
        if (fault == -7)
            a.usedQuery = UINT64_MAX - 3;
        if (fault == -8)
            --a.words;
        bool built = BuildPrefillBsndMetadata(a);
        if (!built)
            for (auto x : record)
                if (x != 0xA5A5A5A5)
                    return 4;
        if (fault >= 0)
            record.at(fault) = value;
        MlaAutoSplitBsndSchedule schedule{}, before{};
        std::memset(&schedule, 0xA5, sizeof(schedule));
        before = schedule;
        const auto status = ReadPrefillBsndMetadata(Reader{record}, l.words, c, Arrays{q, k}, schedule);
        if (status != MlaMetadataReadStatus::Success && std::memcmp(&before, &schedule, sizeof(schedule)))
            return 3;
        std::cout << "{\"layout\":true,\"built\":" << built << ",\"status\":" << int(status) << ",\"bytes\":" << l.bytes
                  << ",\"record\":[";
        for (uint32_t i = 0; i < l.words; ++i)
            std::cout << (i ? "," : "") << record[i];
        std::cout << "]}\n" << std::flush;
    }
}
