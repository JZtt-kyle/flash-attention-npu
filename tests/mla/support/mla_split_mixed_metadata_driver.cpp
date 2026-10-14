// CPU protocol bridge; production builder and independent consumer run separately.
#include "mla_split_mixed_metadata.h"
#include "mla_split_mixed_consumer.h"
#include "mla_split_mixed_plan.h"
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
        if (b < 0 || b > 300000)
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
        MlaUsedShape shape{b, h, {MlaSequenceLayout::Tnd, tq, q}, {MlaSequenceLayout::Bsnd, tk, k}};
        uint32_t words = 0;
        SplitMixedMetadataWords(b, words);
        std::vector<uint32_t> data(words, 0xdeadbeef);
        int32_t empty = 0;
        MlaUsedInputs in{{cq.data(), uint64_t(b + 1)},
                         {nullptr, 0},
                         {uq ? (b ? qs.data() : &empty) : nullptr, uq ? uint64_t(b) : 0},
                         {uk ? (b ? ks.data() : &empty) : nullptr, uk ? uint64_t(b) : 0}};
        if (mode == 1)
            in.usedKv = {reinterpret_cast<int32_t*>(data.data() + words - 1), 1};
        const MlaWindowConfig window{q, k, left, right, bool(causal)};
        auto* destination = mode == 18   ? nullptr
                            : mode == 19 ? reinterpret_cast<uint32_t*>(reinterpret_cast<uintptr_t>(data.data()) + 1)
                                         : data.data();
        const bool built = BuildSplitMixedMetadata(shape, p, dtype, destination, mode == 17 ? words - 1 : words, in,
                                                   window, mode == 21 ? 2 : s);
        const bool intact = data.front() == 0xdeadbeef;
        const uint32_t generated = data[5];
        if (mode == 21)
            data[410 + 8 * b + 3] = s;
        if (mut >= 0 && uint32_t(mut) < words)
            data[mut] = value;
        MlaWindow w{};
        NormalizeWindow(window, w);
        const auto axis = [](uint32_t layout, int64_t total, int64_t cap) {
            return MlaMetadataAllocation{layout, {uint32_t(total), uint32_t(uint64_t(total) >> 32)}, uint32_t(cap)};
        };
        const auto arr = [](MlaLengthArray a) {
            return MlaUsedMetadataArray{reinterpret_cast<uint64_t>(a.data), a.count};
        };
        MlaSplitMixedContext c{{uint32_t(dtype), uint32_t(b), uint32_t(h), uint32_t(uq + 2 * uk), axis(1, tq, q),
                                axis(0, tk, k), 26, 52, uint32_t(nonempty ? 26 : 1), nonempty ? uint64_t(INT64_MAX) : 0,
                                0, p.localMemory},
                               {uint32_t(w.kind), w.left, w.right},
                               uint32_t(s)};
        if (mode == 2)
            c.used.workspaceBytes = 1;
        if (mode == 3 && b)
            ks[0] = (ks[0] + 1) % (k + 1);
        if (mode == 4 && b)
            cq[b] -= 1;
        if (mode == 5)
            in.cuKv = {ck.data(), uint64_t(b + 1)};
        if (mode == 6)
            c.used.kv.totalTokens.low ^= 1;
        if (mode == 7)
            c.used.kv.layout = 1;
        if (mode == 8)
            c.used.query.layout = 0;
        if (mode == 9)
            c.used.usedFlags ^= 2;
        if (mode == 10)
            in.cuQuery.count -= 1;
        if (mode == 11)
            in.cuQuery.data = nullptr;
        if (mode == 12)
            in.usedKv.count += 1;
        if (mode == 13)
            c.splits += 1;
        if (mode == 14)
            c.window.left += 1;
        if (mode == 15)
            c.used.dtype = 3 - c.used.dtype;
        if (mode == 16)
            c.used.localMemory.ubBytes = 1;
        if (mode == 23)
            c.used.batch = UINT32_MAX;
        if (mode == 24)
            in.usedKv.data = reinterpret_cast<const int32_t*>(UINTPTR_MAX - 3);
        auto read = [&](uint32_t i) { return data.at(i); };
        auto arrays = [&](uint32_t a, uint32_t i) {
            return uint32_t(a == 0 ? cq.at(i) : a == 1 ? ck.at(i) : a == 2 ? qs.at(i) : ks.at(i));
        };
        MlaSplitMixedSchedule result{};
        result.groups = 12345;
        const auto before = result;
        auto status =
            ReadSplitMixedMetadata(read, mode == 22 ? words - 1 : words, c,
                                   {arr(in.cuQuery), arr(in.cuKv), arr(in.usedQuery), arr(in.usedKv)}, arrays, result);
        std::cout << "{\"built\":" << built << ",\"intact\":" << intact << ",\"generated\":" << generated
                  << ",\"read\":" << uint32_t(status) << ",\"groups\":" << result.groups
                  << ",\"unchanged\":" << (std::memcmp(&before, &result, sizeof(result)) == 0)
                  << ",\"tasks\":" << result.tasks << ",\"bytes\":" << result.bytes << ",\"record\":[";
        for (uint32_t i = 0; mode != 20 && mode != 21 && i < words; ++i)
            std::cout << (i ? "," : "") << data[i];
        std::cout << "],\"fragments\":[";
        if (status == MlaMetadataReadStatus::Success && nonempty && mode != 20 && mode != 21) {
            bool first = true;
            const auto& t = result.used.tiling;
            for (uint32_t batch = 0; batch < uint32_t(b); ++batch) {
                const uint32_t qlen = uq ? qs[batch] : cq[batch + 1] - cq[batch], klen = uk ? ks[batch] : uint32_t(k);
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
        std::cout << "],\"tail\":[";
        for (uint32_t i = words - 4; i < words; ++i)
            std::cout << (i == words - 4 ? "" : ",") << data[i];
        std::cout << "],\"last_span\":[";
        if (b)
            for (uint32_t i = 410 + 8 * (b - 1); i < 410 + 8 * b; ++i)
                std::cout << (i == 410 + 8 * (b - 1) ? "" : ",") << data[i];
        std::cout << "]}" << std::endl;
    }
}
