// CPU/sanitizer bridge to the actual v13 AICPU entry and shared consumer.
#include "mla_window_metadata.h"
#include "mla_used_metadata_consumer.h"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>
using namespace flash_attn::mla;
extern uint32_t ComputeMlaWindowTndMetadata(void*);

uint32_t Generate(const MlaUsedMetadataArgs& input, const MlaWindowConfig& window)
{
    MlaWindowMetadataArgs args{input, window};
    return ComputeMlaWindowTndMetadata(&args);
}

int main()
{
    int64_t batch, sq, heads, sk, tq, tk;
    uint32_t dtype, flags, cores;
    int64_t left, right;
    bool causal;
    assert(ComputeMlaWindowTndMetadata(nullptr) == 1);
    while (std::cin >> batch >> sq >> heads >> sk >> tq >> tk >> dtype >> flags >> cores >> left >> right >> causal) {
        assert(batch >= 0 && batch < 1000);
        std::vector<int32_t> cq(batch + 1), ck(batch + 1), uq(batch + 1), uk(batch + 1);
        for (auto* array : {&cq, &ck})
            for (auto& value : *array)
                std::cin >> value;
        for (int64_t b = 0; b < batch; ++b)
            std::cin >> uq[b] >> uk[b];
        assert(std::cin);
        MlaWindowConfig config{sq, sk, left, right, causal};
        MlaWindow normalized{};
        NormalizeWindow(config, normalized);
        MlaWindowKernelParams params{uint32_t(normalized.kind), normalized.left, normalized.right};
        MlaUsedMetadataArgs a{};
        a.shape = {batch, heads, {MlaSequenceLayout::Tnd, tq, sq}, {MlaSequenceLayout::Tnd, tk, sk}};
        a.dtype = dtype;
        a.platform = {cores, 2 * cores, {516096, 174080, 65536, 65536, 131072}, 4096, {}};
        const bool nonempty = batch && tq && tk;
        a.platform.blockDims.fill(cores);
        a.platform.blockDims[0] = nonempty ? cores : 1;
        const auto array = [](std::vector<int32_t>& values, uint64_t count) {
            return MlaUsedMetadataArray{reinterpret_cast<uint64_t>(values.data()), count};
        };
        a.cuQuery = array(cq, batch + 1);
        a.cuKv = array(ck, batch + 1);
        if (flags & 1)
            a.usedQuery = array(uq, batch);
        if (flags & 2)
            a.usedKv = array(uk, batch);
        assert(WindowMetadataWords(batch, a.outputWords));
        constexpr uint32_t guard = 0x12345678;
        std::vector<uint32_t> storage(a.outputWords + 2, guard);
        auto* words = storage.data() + 1;
        a.outputAddr = reinterpret_cast<uint64_t>(words);
        assert(Generate(a, config) == 0);
        assert(storage.front() == guard && storage.back() == guard);
        assert(words[1] == 13 && words[2] == 13);
        const auto original = storage;
        MlaUsedMetadataContext c{};
        c.dtype = dtype;
        c.batch = batch;
        c.heads = heads;
        c.usedFlags = flags;
        c.query = {1, {uint32_t(tq), uint32_t(uint64_t(tq) >> 32)}, uint32_t(sq)};
        c.kv = {1, {uint32_t(tk), uint32_t(uint64_t(tk) >> 32)}, uint32_t(sk)};
        c.aicNum = cores;
        c.aivNum = cores * 2;
        c.blockDim = nonempty ? cores : 1;
        c.localMemory = a.platform.localMemory;
        const uint64_t m = (std::min(uint64_t(sq * heads), uint64_t(256)) + 15) / 16 * 16;
        const uint64_t n = (std::min(uint64_t(sk), uint64_t(512)) + 31) / 32 * 32;
        c.workspaceBytes = nonempty ? uint64_t(2) * cores * (6 * m * n + 8 * m * 512 + 8 * 256) : 0;
        c.libApiBytes = nonempty ? 4096 : 0;
        MlaUsedMetadataInputs inputs{a.cuQuery, a.cuKv, a.usedQuery, a.usedKv};
        const auto read = [&](uint32_t i) {
            assert(i < a.outputWords);
            return words[i];
        };
        const auto arrays = [&](uint32_t axis, uint32_t i) {
            assert(axis < 4 && i < uint64_t(batch) + (axis < 2));
            return uint32_t((axis == 0 ? cq : axis == 1 ? ck : axis == 2 ? uq : uk)[i]);
        };
        const auto check = [&](bool success) {
            MlaUsedKernelSchedule result;
            std::memset(&result, 0xa5, sizeof(result));
            const auto before = result;
            auto status = ReadUsedMetadata<false, false, false, true, true>(read, a.outputWords, c, inputs, arrays,
                                                                            result, params);
            assert((status == MlaMetadataReadStatus::Success) == success);
            if (!success)
                assert(std::memcmp(&result, &before, sizeof(result)) == 0);
            if (words[1] == 13 && words[2] == 13) {
                assert(ReadUsedMetadata(read, a.outputWords, c, inputs, arrays, result) !=
                       MlaMetadataReadStatus::Success);
                assert(ReadUsedMetadata<true>(read, a.outputWords, c, inputs, arrays, result) !=
                       MlaMetadataReadStatus::Success);
            }
        };
        check(words[5] == 0);
        if (words[5] == 0) {
            for (auto* value : {&params.left, &params.right}) {
                *value ^= 1;
                check(false);
                *value ^= 1;
            }
            auto other = a;
            other.shape.query.layout = MlaSequenceLayout::Bsnd;
            other.shape.kv.layout = MlaSequenceLayout::Bsnd;
            assert(Generate(other, config) == 0 && words[5] != 0);
            std::copy(original.begin(), original.end(), storage.begin());
            for (uint32_t field : {0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 10U, 401U, 402U, 404U, 405U, 406U, 408U, 409U}) {
                words[field] ^= 0x40000000U;
                check(false);
                std::copy(original.begin(), original.end(), storage.begin());
            }
            for (uint32_t field = 410; field < a.outputWords; ++field) {
                words[field] ^= 1U;
                check(false);
                std::copy(original.begin(), original.end(), storage.begin());
            }
            for (auto* prefix : {&cq, &ck}) {
                for (uint32_t i = 0; i <= uint64_t(batch); ++i) {
                    (*prefix)[i] ^= 1;
                    check(false);
                    (*prefix)[i] ^= 1;
                }
            }
            for (uint32_t axis = 0; axis < 2; ++axis) {
                auto& values = axis == 0 ? uq : uk;
                if (flags & (1U << axis))
                    for (uint32_t i = 0; i < uint64_t(batch); ++i) {
                        values[i] ^= 1;
                        check(false);
                        values[i] ^= 1;
                    }
            }
            for (uint32_t version : {1U, 2U, 3U, 5U, 7U, 8U, 9U, 10U, 11U, 12U}) {
                words[1] = words[2] = version;
                check(false);
            }
            std::copy(original.begin(), original.end(), storage.begin());
            c.kv.layout = 0;
            check(false);
            c.kv.layout = 1;
            if (nonempty) {
                words[23 + words[20] - 1] = batch - 1;
                check(false);
                std::copy(original.begin(), original.end(), storage.begin());
            }
        } else {
            for (uint32_t i = 8; i < a.outputWords; ++i)
                assert(words[i] == 0);
        }
        for (uint32_t size : {0U, a.outputWords - 1, a.outputWords + 1}) {
            auto bad = a;
            bad.outputWords = size;
            assert(Generate(bad, config) == 1 && storage == original);
        }
        auto bad = a;
        bad.cuQuery.address = a.outputAddr;
        assert(Generate(bad, config) == 1 && storage == original);
        bad = a;
        bad.cuKv.address = a.outputAddr + (a.outputWords - 1) * 4;
        assert(Generate(bad, config) == 1 && storage == original);
        bad = a;
        bad.cuKv.address = UINT64_MAX - 1;
        assert(Generate(bad, config) == 0 && words[5] != 0);
        assert(std::all_of(words + 8, words + a.outputWords, [](auto x) { return x == 0; }));
        std::copy(original.begin(), original.end(), storage.begin());
        uint32_t count = guard;
        assert(!WindowMetadataWords((UINT32_MAX - 413L) / 8 + 1, count) && count == guard);
        assert(!WindowMetadataWords(-1, count) && count == guard);
        std::cout << '[';
        for (uint32_t i = 0; i < a.outputWords; ++i)
            std::cout << (i ? "," : "") << words[i];
        std::cout << "]\n";
    }
}
