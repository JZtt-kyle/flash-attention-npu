// CPU/sanitizer bridge to the actual v7 AICPU entry and device-independent consumer.
#include "mla_causal_metadata.h"
#include "mla_used_metadata_consumer.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>
using namespace flash_attn::mla;
extern uint32_t ComputeMlaCausalBsndMetadata(void*);

uint32_t Generate(const MlaUsedMetadataArgs& input, const MlaCausalFullSchedule& full = {})
{
    MlaCausalMetadataArgs args{input, full};
    return ComputeMlaCausalBsndMetadata(&args);
}

int main()
{
    int64_t batch, sq, heads, sk;
    uint32_t dtype, qPresent, kPresent, cores;
    while (std::cin >> batch >> sq >> heads >> sk >> dtype >> qPresent >> kPresent >> cores) {
        assert(batch >= 0 && batch < 1000);
        std::vector<int32_t> q(batch + 1), k(batch + 1);
        for (int64_t b = 0; b < batch; ++b)
            std::cin >> q[b] >> k[b];
        assert(std::cin);
        MlaUsedMetadataArgs a{};
        a.shape = {batch, heads, {MlaSequenceLayout::Bsnd, batch * sq, sq}, {MlaSequenceLayout::Bsnd, batch * sk, sk}};
        a.dtype = dtype;
        a.platform = {cores, 2 * cores, {516096, 174080, 65536, 65536, 131072}, 4096, {}};
        a.platform.blockDims.fill(cores);
        a.platform.blockDims[0] = batch && sq && sk ? cores : 1;
        a.usedQuery = {qPresent ? reinterpret_cast<uint64_t>(q.data()) : 0, qPresent ? uint64_t(batch) : 0};
        a.usedKv = {kPresent ? reinterpret_cast<uint64_t>(k.data()) : 0, kPresent ? uint64_t(batch) : 0};
        assert(UsedMetadataWords(batch, a.outputWords));
        constexpr uint32_t guard = 0x12345678;
        std::vector<uint32_t> storage(a.outputWords + 2, guard);
        auto* words = storage.data() + 1; // Deliberately only four-byte aligned.
        a.outputAddr = reinterpret_cast<uint64_t>(words);
        MlaCausalFullSchedule full{};
        BuildCausalFullSchedule(a.shape, a.platform, full);
        assert(Generate(a, full) == 0);
        const auto withCandidate = storage;
        assert(Generate(a) == 0 && storage == withCandidate);
        // The wire record must not change on a cache hit, partial lengths or
        // any mismatched candidate key. Validation failures retain zero payloads.
        for (uint32_t fault = 0; fault < 10; ++fault) {
            auto bad = full;
            switch (fault) {
            case 0:
                bad.enabled = 2;
                break;
            case 1:
                bad.batch ^= 1;
                break;
            case 2:
                bad.queryCapacity ^= 1;
                break;
            case 3:
                bad.kvCapacity ^= 1;
                break;
            case 4:
                bad.heads ^= 1;
                break;
            case 5:
                bad.aicNum ^= 1;
                break;
            case 6:
                bad.cvRatio = 3;
                break;
            case 7:
                bad.mBaseSize ^= 1;
                break;
            case 8:
                bad.s2BaseSize ^= 1;
                break;
            default:
                bad.usedCoreNum = 0;
                break;
            }
            assert(Generate(a, bad) == 0 && storage == withCandidate);
        }
        assert(storage.front() == guard && storage.back() == guard);
        assert(words[1] == 7 && words[2] == 7);
        // Tiles, allocation, scratch bounds and wire padding must agree with
        // the independently completed v5 schedule. Only the causal partition
        // (checked separately against the pinned source) is allowed to differ.
        std::vector<uint32_t> unmasked(a.outputWords);
        assert(BuildUsedMetadata(a.shape, a.platform, a.dtype, unmasked.data(), a.outputWords, UsedMetadataInputs(a)));
        if (words[5] == 0) {
            assert(unmasked[5] == 0);
            MlaUsedMetadataPrefix oldPrefix, newPrefix;
            std::memcpy(&oldPrefix, unmasked.data(), sizeof(oldPrefix));
            std::memcpy(&newPrefix, words, sizeof(newPrefix));
            const auto clearPartition = [](MlaUsedMetadataPrefix& p) {
                p.version = p.kind = 0;
                auto& s = p.schedule;
                s.usedCoreNum = s.blockDim = 0;
                std::fill(std::begin(s.bN2End), std::end(s.bN2End), 0);
                std::fill(std::begin(s.gS1End), std::end(s.gS1End), 0);
                std::fill(std::begin(s.s2End), std::end(s.s2End), 0);
                s.maxCost = {};
                auto& w = s.workspace;
                w.qkOffsetBytes = w.pOffsetBytes = w.pvOffsetBytes = w.vec2OffsetBytes = w.kernelBytes = {};
            };
            clearPartition(oldPrefix);
            clearPartition(newPrefix);
            assert(std::memcmp(&oldPrefix, &newPrefix, sizeof(oldPrefix)) == 0);
            assert(std::equal(words + kMlaUsedMetadataPrefixWords, words + a.outputWords,
                              unmasked.data() + kMlaUsedMetadataPrefixWords));
        }
        const auto original = storage;
        const auto originalQ = q, originalK = k;
        MlaUsedMetadataContext c{};
        c.dtype = dtype;
        c.batch = batch;
        c.heads = heads;
        c.usedFlags = qPresent | (kPresent << 1);
        const auto allocation = [](uint64_t total, uint32_t length) {
            return MlaMetadataAllocation{0, {uint32_t(total), uint32_t(total >> 32)}, length};
        };
        c.query = allocation(batch * sq, sq);
        c.kv = allocation(batch * sk, sk);
        c.aicNum = cores;
        c.aivNum = cores * 2;
        c.blockDim = batch && sq && sk ? cores : 1;
        c.localMemory = a.platform.localMemory;
        const uint64_t m = (std::min(uint64_t(sq * heads), uint64_t(256)) + 15) / 16 * 16;
        const uint64_t n = (std::min(uint64_t(sk), uint64_t(512)) + 31) / 32 * 32;
        c.workspaceBytes = batch && sq && sk ? uint64_t(2) * cores * (6 * m * n + 8 * m * 512 + 8 * 256) : 0;
        c.libApiBytes = batch && sq && sk ? 4096 : 0;
        MlaUsedMetadataInputs inputs{{}, {}, a.usedQuery, a.usedKv};
        const auto read = [&](uint32_t i) {
            assert(i < a.outputWords);
            return words[i];
        };
        const auto arrays = [&](uint32_t axis, uint32_t i) {
            assert(axis >= 2 && i < uint64_t(batch));
            return uint32_t(axis == 2 ? q[i] : k[i]);
        };
        const auto check = [&](bool success) {
            MlaUsedKernelSchedule result;
            std::memset(&result, 0xa5, sizeof(result));
            const auto before = result;
            auto status = ReadUsedMetadata<true>(read, a.outputWords, c, inputs, arrays, result);
            assert((status == MlaMetadataReadStatus::Success) == success);
            if (!success)
                assert(std::memcmp(&result, &before, sizeof(result)) == 0);
            MlaUsedKernelSchedule old{};
            assert(ReadUsedMetadata(read, a.outputWords, c, inputs, arrays, old) != MlaMetadataReadStatus::Success);
        };
        check(words[5] == 0);
        if (words[5] == 0) {
            // Diagnostic maxCost may vary, but version/geometry/length identity may not.
            for (uint32_t field : {0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 10U, 401U, 404U, 405U, 408U, 409U}) {
                words[field] ^= 0x40000000U;
                check(false);
                std::copy(original.begin(), original.end(), storage.begin());
            }
            if (batch) {
                words[416] ^= 1U;
                check(false);
                std::copy(original.begin(), original.end(), storage.begin());
                words[417] ^= 1U;
                check(false);
                std::copy(original.begin(), original.end(), storage.begin());
                if (qPresent) {
                    q[0] ^= 1;
                    check(false);
                    q = originalQ;
                }
                if (kPresent) {
                    k[0] ^= 1;
                    check(false);
                    k = originalK;
                }
            }
            if (batch && sq && sk) {
                const uint32_t begin = offsetof(MlaUsedMetadataPrefix, schedule.bN2IdxOfFdHead) / 4;
                const uint32_t end = offsetof(MlaUsedMetadataPrefix, schedule.workspace) / 4;
                for (uint32_t field = begin; field < end; ++field) {
                    words[field] = 1U << ((field - begin) % 32);
                    check(false);
                    std::copy(original.begin(), original.end(), storage.begin());
                }
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
            assert(Generate(bad, full) == 1 && storage == original);
        }
        auto overlap = a;
        overlap.usedKv = {a.outputAddr, uint64_t(batch) + 1};
        assert(Generate(overlap, full) == 1 && storage == original);
        uint32_t count = guard;
        assert(!UsedMetadataWords((UINT32_MAX - 410L) / 8 + 1, count) && count == guard);
        assert(!UsedMetadataWords(-1, count) && count == guard);
        std::cout << '[';
        for (uint32_t i = 0; i < a.outputWords; ++i)
            std::cout << (i ? "," : "") << words[i];
        std::cout << "]\n";
    }
}
