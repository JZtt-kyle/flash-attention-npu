// CPU protocol bridge; production builder and independent consumer run separately.
#include "mla_split_tnd_metadata.h"
#include "mla_split_tnd_consumer.h"
#include "mla_split_tnd_plan.h"
#include <iostream>
#include <vector>
#include <cstring>
using namespace flash_attn::mla;
int main()
{
    int64_t b, tq, tk, q, k, h, s, left, right;
    int causal, dtype, uq, uk, mut;
    uint32_t value;
    int mode;
    while (std::cin >> b >> tq >> tk >> q >> k >> h >> s >> left >> right >> causal >> dtype >> uq >> uk >> mut >>
           value >> mode) {
        if (b < 0 || b > 1000)
            return 2;
        std::vector<int32_t> cq(b + 1), ck(b + 1), qs(b), ks(b);
        for (auto& x : cq)
            std::cin >> x;
        for (auto& x : ck)
            std::cin >> x;
        for (auto& x : qs)
            std::cin >> x;
        for (auto& x : ks)
            std::cin >> x;
        MlaPlatformInfo p{};
        p.aicNum = 26;
        p.aivNum = 52;
        p.localMemory = {524288, 196608, 65536, 65536, 131072};
        const bool nonempty = tq && tk;
        p.blockDims.fill(nonempty ? 26 : 1);
        MlaUsedShape shape{b, h, {MlaSequenceLayout::Tnd, tq, q}, {MlaSequenceLayout::Tnd, tk, k}};
        uint32_t words = 0;
        SplitTndMetadataWords(b, words);
        std::vector<uint32_t> data(words, 0xdeadbeef);
        int32_t empty = 0;
        MlaUsedInputs in{{cq.data(), uint64_t(b + 1)},
                         {ck.data(), uint64_t(b + 1)},
                         {uq ? (b ? qs.data() : &empty) : nullptr, uq ? uint64_t(b) : 0},
                         {uk ? (b ? ks.data() : &empty) : nullptr, uk ? uint64_t(b) : 0}};
        if (mode == 1)
            in.usedKv = {reinterpret_cast<int32_t*>(data.data() + words - 1), 1};
        const MlaWindowConfig window{q, k, left, right, bool(causal)};
        const bool built = BuildSplitTndMetadata(shape, p, dtype, data.data(), words, in, window, s);
        const bool intact = data.front() == 0xdeadbeef;
        const uint32_t generated = data[5];
        if (mut >= 0 && uint32_t(mut) < words)
            data[mut] = value;
        MlaWindow w{};
        NormalizeWindow(window, w);
        const auto axis = [](int64_t total, int64_t cap) {
            return MlaMetadataAllocation{1, {uint32_t(total), uint32_t(uint64_t(total) >> 32)}, uint32_t(cap)};
        };
        const auto arr = [](MlaLengthArray a) {
            return MlaUsedMetadataArray{reinterpret_cast<uint64_t>(a.data), a.count};
        };
        MlaSplitTndContext c{{uint32_t(dtype), uint32_t(b), uint32_t(h), uint32_t(uq + 2 * uk), axis(tq, q),
                              axis(tk, k), 26, 52, uint32_t(nonempty ? 26 : 1), nonempty ? uint64_t(INT64_MAX) : 0, 0,
                              p.localMemory},
                             {uint32_t(w.kind), w.left, w.right},
                             uint32_t(s)};
        if (mode == 2)
            c.used.workspaceBytes = 1;
        if (mode == 3 && b)
            ks[0] = (ks[0] + 1) % (k + 1);
        if (mode == 4 && b)
            cq[b] -= 1;
        if (mode == 5 && b)
            ck[b] -= 1;
        auto read = [&](uint32_t i) { return data.at(i); };
        auto arrays = [&](uint32_t a, uint32_t i) {
            return uint32_t(a == 0 ? cq.at(i) : a == 1 ? ck.at(i) : a == 2 ? qs.at(i) : ks.at(i));
        };
        MlaSplitTndSchedule result{};
        result.groups = 12345;
        auto status = ReadSplitTndMetadata(
            read, words, c, {arr(in.cuQuery), arr(in.cuKv), arr(in.usedQuery), arr(in.usedKv)}, arrays, result);
        std::cout << "{\"built\":" << built << ",\"intact\":" << intact << ",\"generated\":" << generated
                  << ",\"read\":" << uint32_t(status) << ",\"groups\":" << result.groups
                  << ",\"tasks\":" << result.tasks << ",\"bytes\":" << result.bytes << ",\"record\":[";
        for (uint32_t i = 0; i < words; ++i)
            std::cout << (i ? "," : "") << data[i];
        std::cout << "],\"fragments\":[";
        if (status == MlaMetadataReadStatus::Success && nonempty) {
            bool first = true;
            const auto& t = result.used.tiling;
            for (uint32_t batch = 0; batch < uint32_t(b); ++batch) {
                const uint32_t qlen = uq ? qs[batch] : cq[batch + 1] - cq[batch],
                               klen = uk ? ks[batch] : ck[batch + 1] - ck[batch];
                for (uint32_t m = 0; m < (uint64_t(qlen) * h + t.mBaseSize - 1) / t.mBaseSize; ++m) {
                    for (uint32_t split = 0; split < uint32_t(s); ++split) {
                        uint32_t begin, end;
                        SplitBsndFragment(qlen, klen, h, m, t.mBaseSize, t.s2BaseSize, c.window, s, split, begin, end);
                        std::cout << (first ? "" : ",") << '[' << batch << ',' << m << ',' << split << ',' << begin
                                  << ',' << end << ']';
                        first = false;
                    }
                }
            }
        }
        std::cout << "]}" << std::endl;
    }
}
