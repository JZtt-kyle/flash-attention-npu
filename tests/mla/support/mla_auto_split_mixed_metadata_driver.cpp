// Independent wire consumer; never use the producer's plan as decoded state.
#include "mla_auto_split_mixed_metadata.h"
#include "mla_auto_split_mixed_consumer.h"
#include "mla_auto_split_tnd_consumer.h"
#include <cstring>
#include <iostream>
#include <vector>
using namespace flash_attn::mla;
template <typename T>
void Emit(const T& a)
{
    bool first = true;
    for (auto x : a) {
        std::cout << (first ? "" : ",") << x;
        first = false;
    }
}
int main(int argc, char** argv)
{
    std::vector<uint32_t> data;
    std::vector<int32_t> qs, ks, cq;
    MlaAutoSplitMixedContext c{};
    MlaUsedMetadataInputs inputs{};
    MlaAutoSplitMixedKernelSchedule schedule{};
    bool valid = false;
    auto read = [&](uint32_t i) { return data.at(i); };
    auto arrays = [&](uint32_t axis, uint32_t i) {
        return uint32_t(axis == 0   ? cq.at(i)
                        : axis == 1 ? throw std::out_of_range("mixed KV has no prefix")
                        : axis == 2 ? qs.at(i)
                                    : ks.at(i));
    };
    const auto consume = [&]() {
        MlaAutoSplitMixedKernelSchedule result;
        std::memset(&result, 0x5a, sizeof(result));
        const auto before = result;
        auto status = ReadAutoSplitMixedMetadata(read, data.size(), c, inputs, arrays, result);
        valid = status == MlaMetadataReadStatus::Success;
        if (valid)
            schedule = result;
        std::cout << "\"read\":" << uint32_t(status)
                  << ",\"unchanged\":" << (std::memcmp(&result, &before, sizeof(result)) == 0);
        if (valid) {
            std::cout << ",\"summary\":[" << schedule.groups << ',' << schedule.usedCoreNum << ',' << schedule.fdHeads
                      << ',' << schedule.maxSplits << ',' << schedule.vectors << ','
                      << schedule.partialPrefix[schedule.fdHeads] << ',' << schedule.outOffset << ','
                      << schedule.sumOffset << ',' << schedule.maxOffset << ',' << schedule.bytes << ']';
        }
    };
    int op;
    while (std::cin >> op) {
        if (op == 0) {
            int64_t b;
            std::cin >> b;
            uint32_t words = 12345;
            const bool ok = AutoSplitMixedMetadataWords(b, words);
            std::cout << "{\"ok\":" << ok << ",\"words\":" << words << "}\n";
        } else if (op == 1) {
            int64_t b, q, k, h, cores, ratio, left, right, tq, tk;
            uint32_t dtype, uq, uk, causal, mode;
            std::cin >> b >> q >> k >> h >> cores >> ratio >> dtype >> uq >> uk >> left >> right >> causal >> mode >>
                tq >> tk;
            uint32_t hasSoftcap = 0;
            if (argc > 1 && std::strcmp(argv[1], "--softcap") == 0)
                std::cin >> hasSoftcap;
            if (b < 0 || b > 300000)
                return 2;
            cq.resize(b + 1);
            for (auto& x : cq)
                std::cin >> x;
            qs.resize(b);
            ks.resize(b);
            for (auto& x : qs)
                std::cin >> x;
            for (auto& x : ks)
                std::cin >> x;
            MlaPlatformInfo platform{};
            platform.aicNum = cores;
            platform.aivNum = cores * ratio;
            platform.localMemory = {524288, 196608, 65536, 65536, 131072};
            const bool nonempty = tq && tk;
            platform.blockDims.fill(nonempty ? cores : 1);
            if (mode == 5)
                platform.blockDims.fill(1);
            if (mode == 6)
                platform.localMemory.ubBytes = 1;
            MlaUsedShape shape{b, h, {MlaSequenceLayout::Tnd, tq, q}, {MlaSequenceLayout::Bsnd, tk, k}};
            uint32_t words = 0;
            AutoSplitMixedMetadataWords(b, words);
            data.assign(words, 0xdeadbeef);
            int32_t empty = 0;
            MlaUsedInputs in{{cq.data(), uint64_t(b + 1)},
                             {nullptr, 0},
                             {uq ? (b ? qs.data() : &empty) : nullptr, uq ? uint64_t(b) : 0},
                             {uk ? (b ? ks.data() : &empty) : nullptr, uk ? uint64_t(b) : 0}};
            if (mode == 1)
                in.usedKv = {reinterpret_cast<int32_t*>(data.data() + words - 1), 1};
            if (mode == 2)
                in.usedKv = {reinterpret_cast<int32_t*>(UINTPTR_MAX - 3), 2};
            if (mode == 3)
                in.usedKv.data = nullptr;
            if (mode == 4)
                shape.query.layout = MlaSequenceLayout::Bsnd;
            if (mode == 8)
                in.cuKv = {cq.data(), uint64_t(b + 1)};
            if (mode == 9)
                shape.kv.layout = MlaSequenceLayout::Tnd;
            MlaWindowConfig window{q, k, left, right, bool(causal)};
            auto producerShape = shape;
            if (mode == 10)
                producerShape.query.totalTokens = 0;
            const bool built = BuildAutoSplitMixedMetadata(producerShape, platform, dtype, data.data(), words, in,
                                                           window, mode == 10, hasSoftcap);
            const uint32_t generated = data[5];
            MlaWindow w{};
            NormalizeWindow(window, w);
            if (mode == 7) {
                // Forge an otherwise valid all-zero-KV record from v5, without
                // using the auto producer/plan. Its accumulator capacity is
                // deliberately not representable; the independent reader must
                // reject it too, rather than relying on the producer's check.
                const uint32_t base = kMlaUsedMetadataPrefixWords + 8 * b;
                if (!BuildUsedMetadata(shape, platform, dtype, data.data(), base, in) || data[5] || !uk)
                    return 3;
                data[1] = data[2] = hasSoftcap ? kMlaSoftcapAutoSplitMixedVersion : kMlaAutoSplitMixedVersion;
                data[3] = words;
                data[base] = uint32_t(w.kind);
                data[base + 1] = uint32_t(w.left);
                data[base + 2] = uint32_t(w.right);
                data[base + 3] = 1;
                data[base + 4] = b;
                data[base + 85] = 8;
                uint64_t prefix = 0;
                for (uint32_t i = 0; i <= uint32_t(b); ++i) {
                    data[base + 346 + 2 * i] = uint32_t(prefix);
                    data[base + 347 + 2 * i] = uint32_t(prefix >> 32);
                    if (i < uint32_t(b) && nonempty) {
                        if (ks[i])
                            return 3;
                        const uint64_t rows = uint64_t(uq ? qs[i] : cq[i + 1] - cq[i]) * h;
                        prefix += (rows + data[15] - 1) / data[15];
                    }
                }
            }
            const auto axis = [](int64_t total, int64_t cap) {
                return MlaMetadataAllocation{1, {uint32_t(total), uint32_t(uint64_t(total) >> 32)}, uint32_t(cap)};
            };
            const auto arr = [](MlaLengthArray a) {
                return MlaUsedMetadataArray{reinterpret_cast<uint64_t>(a.data), a.count};
            };
            c = {{dtype,
                  uint32_t(b),
                  uint32_t(h),
                  uq + 2 * uk,
                  axis(tq, q),
                  {0, {uint32_t(tk), uint32_t(uint64_t(tk) >> 32)}, uint32_t(k)},
                  uint32_t(cores),
                  uint32_t(cores * ratio),
                  uint32_t(nonempty ? cores : 1),
                  nonempty ? uint64_t(INT64_MAX) : 0,
                  0,
                  platform.localMemory},
                 {uint32_t(w.kind), w.left, w.right},
                 hasSoftcap};
            inputs = {arr(in.cuQuery), arr(in.cuKv), arr(in.usedQuery), arr(in.usedKv)};
            std::cout << "{\"built\":" << built << ",\"generated\":" << generated
                      << ",\"intact\":" << (data.front() == 0xdeadbeef) << ',';
            consume();
            std::cout << ",\"record\":[";
            Emit(data);
            std::cout << "]}\n";
        } else if (op == 2 || op == 4) {
            uint64_t group;
            uint32_t split, row = 0;
            std::cin >> group >> split;
            if (op == 4)
                std::cin >> row;
            MlaAutoSplitMixedKernelTask task{};
            auto& t = task.native;
            t.group = 12345;
            bool ok = valid && AutoSplitMixedTask(read, schedule, c.window, group, split, task);
            if (op == 2) {
                const auto& v = task.sequence;
                std::cout << "{\"sequence\":[" << v.queryStart << ',' << v.kvStart << ',' << v.queryCapacity << ','
                          << v.kvCapacity << ',' << v.queryLength << ',' << v.kvLength << "],";
                std::cout << "\"ok\":" << ok << ",\"task\":[" << t.group << ',' << t.slot << ',' << t.batch << ','
                          << t.m << ',' << t.split << ',' << t.splitCount << ',' << t.owner << ',' << t.fdHead << ','
                          << t.kind << "],\"range\":[" << t.rowBegin << ',' << t.rowEnd << ',' << t.s2Begin << ','
                          << t.s2End << ',' << t.kvBegin << ',' << t.kvEnd << "],\"offsets\":[" << t.outOffset << ','
                          << t.sumOffset << ',' << t.maxOffset << "]}\n";
            } else {
                MlaAutoSplitMixedKernelRow r{};
                r.out = 12345;
                ok = ok && AutoSplitMixedRow(schedule, task, row, r);
                std::cout << "{\"ok\":" << ok << ",\"row\":[" << r.out << ',' << r.lse << ',' << r.accumulatorOut << ','
                          << r.accumulatorLse << ',' << r.partialOut << ',' << r.partialSum << ',' << r.partialMax
                          << "]}\n";
            }
        } else if (op == 3) {
            uint32_t head, part;
            std::cin >> head >> part;
            MlaAutoSplitMixedKernelMerge m{};
            m.group = 12345;
            const bool ok = valid && AutoSplitMixedMerge(read, schedule, head, part, m);
            std::cout << "{\"ok\":" << ok << ",\"merge\":[" << m.group << ',' << m.firstSlot << ',' << m.batch << ','
                      << m.m << ',' << m.head << ',' << m.part << ',' << m.owner << ',' << m.splitCount << ','
                      << m.rowBegin << ',' << m.rowEnd << "],\"offsets\":[" << m.outOffset << ',' << m.sumOffset << ','
                      << m.maxOffset << "]}\n";
        } else if (op == 5) {
            uint32_t index, value;
            std::cin >> index >> value;
            const auto old = data.at(index);
            data.at(index) = value;
            std::cout << '{';
            consume();
            std::cout << "}\n";
            data.at(index) = old;
        } else if (op == 6) {
            uint32_t mode;
            uint64_t value;
            std::cin >> mode >> value;
            const auto old = c;
            const auto oldInputs = inputs;
            auto oldKs = ks;
            auto oldQs = qs;
            auto oldCq = cq;
            if (mode == 0)
                c.used.workspaceBytes = value;
            if (mode == 1)
                ks.at(0) = value;
            if (mode == 2)
                c.window.kind = value;
            if (mode == 3)
                c.used.blockDim = value;
            if (mode == 4)
                inputs.usedKv.count = value;
            if (mode == 5)
                inputs.usedKv.address = value;
            if (mode == 6)
                c.used.batch = value;
            if (mode == 7)
                c.used.dtype = value;
            if (mode == 8)
                cq.at(cq.size() - 1) = value;
            if (mode == 9)
                c.used.kv.totalTokens.low = value;
            if (mode == 10)
                qs.at(0) = value;
            if (mode == 11)
                inputs.cuQuery.count = value;
            if (mode == 12)
                inputs.cuKv.count = value;
            if (mode == 13)
                inputs.cuQuery.address = value;
            if (mode == 14)
                inputs.cuKv.address = value;
            if (mode == 15)
                c.used.query.totalTokens.high = value;
            if (mode == 16)
                c.used.kv.totalTokens.high = value;
            if (mode == 17)
                c.used.query.layout = value;
            if (mode == 18)
                c.used.kv.layout = value;
            if (mode == 19)
                c.used.usedFlags = value;
            if (mode == 20)
                inputs.usedQuery.count = value;
            if (mode == 21)
                inputs.usedQuery.address = value;
            std::cout << '{';
            consume();
            std::cout << "}\n";
            c = old;
            inputs = oldInputs;
            ks = oldKs;
            qs = oldQs;
            cq = oldCq;
        } else if (op == 8) {
            uint32_t target, version;
            std::cin >> target >> version;
            auto old1 = data[1], old2 = data[2];
            data[1] = data[2] = version;
            MlaAutoSplitBsndContext other{c.used, c.window};
            MlaAutoSplitBsndSchedule r{};
            r.groups = 12345;
            const auto before = r;
            MlaAutoSplitTndContext tnd{c.used, c.window};
            const auto status = target ? ReadAutoSplitTndMetadata(read, data.size(), tnd, inputs, arrays, r)
                                       : ReadAutoSplitBsndMetadata(read, data.size(), other, inputs, arrays, r);
            std::cout << "{\"read\":" << uint32_t(status)
                      << ",\"unchanged\":" << (std::memcmp(&r, &before, sizeof(r)) == 0) << "}\n";
            data[1] = old1;
            data[2] = old2;
        } else
            return 2;
        std::cout << std::flush;
        if (!std::cin)
            return 2;
    }
}
