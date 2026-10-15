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
    const std::vector<uint32_t>&q, &k, &cq, &ck;
    bool tnd;
    uint32_t operator()(uint32_t axis, uint32_t i) const
    {
        return (tnd ? (axis == 0 ? cq : axis == 1 ? ck : axis == 2 ? q : k) : (axis ? k : q)).at(i);
    }
};
int main(int argc, char** argv)
{
    MlaPrefillBsndContext c{};
    int fault;
    uint32_t value;
    while (std::cin >> c.batch >> c.sq >> c.sk >> c.heads >> c.kvHeads >> c.splits >> c.cores >> c.dtype >>
           c.usedFlags >> c.window.kind >> c.window.left >> c.window.right >> fault >> value) {
        c.tnd = argc > 1 ? (std::strcmp(argv[1], "--mixed") == 0 ? 2 : 1) : 0;
        if (c.tnd)
            std::cin >> c.totalQ >> c.totalK;
        std::vector<uint32_t> q(c.batch), k(c.batch), cq(c.batch + 1), ck(c.batch + 1);
        for (auto& x : q)
            std::cin >> x;
        for (auto& x : k)
            std::cin >> x;
        if (c.tnd) {
            for (auto& x : cq)
                std::cin >> x;
            for (auto& x : ck)
                std::cin >> x;
        }
        MlaPrefillBsndLayout l{};
        if (!PrefillBsndLayout(c, l)) {
            std::cout << "{\"layout\":false}\n" << std::flush;
            continue;
        }
        c.workspaceBytes = l.bytes;
        std::vector<uint32_t> record(l.words, 0xA5A5A5A5);
        MlaPrefillBsndMetadataArgs a{c, c.usedFlags & 1 ? uint64_t(q.data()) : 0,
                                     c.usedFlags & 2 ? uint64_t(k.data()) : 0, uint64_t(record.data()), l.words};
        if (c.tnd) {
            a.cuQuery = uint64_t(cq.data());
            a.cuKv = c.tnd == 1 ? uint64_t(ck.data()) : 0;
        }
        if (fault == -13)
            a.inferTotalQ = 1;
        if (fault == -14)
            a.context.totalQ = 0;
        if (fault == -9)
            a.cuQuery = 0;
        if (fault == -10)
            ++a.cuQuery;
        if (fault == -11)
            a.cuKv = UINT64_MAX - 3;
        if (fault == -12)
            a.output = uint64_t(cq.data());
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
        const auto status =
            ReadPrefillBsndMetadata(Reader{record}, l.words, c, Arrays{q, k, cq, ck, bool(c.tnd)}, schedule);
        if (status != MlaMetadataReadStatus::Success && std::memcmp(&before, &schedule, sizeof(schedule)))
            return 3;
        std::cout << "{\"layout\":true,\"built\":" << built << ",\"status\":" << int(status) << ",\"bytes\":" << l.bytes
                  << ",\"record\":[";
        for (uint32_t i = 0; i < l.words; ++i)
            std::cout << (i ? "," : "") << record[i];
        std::cout << "]}\n" << std::flush;
    }
}
