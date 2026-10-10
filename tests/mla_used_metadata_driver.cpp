// CPU/ASan bridge to the actual v5 AICPU entry with a test-only barrier shim.
// The independent fixed-source oracle and wire projection live in Python.
#include "mla_used_metadata.h"
#include "mla_used_metadata_args.h"
#include "mla_bsnd_metadata_consumer.h"
#include "mla_tnd_metadata_consumer.h"
#include "mla_tnd_bsnd_metadata_consumer.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace flash_attn::mla;
extern uint32_t ComputeMlaUsedMetadata(void*);
constexpr uint32_t guard = 0x12345678;

void CheckBuffers()
{
    constexpr uint32_t words = 426;
    const int64_t largest = (UINT32_MAX - 410) / 8;
    uint32_t count = guard;
    for (int64_t b : {-1L, largest + 1, int64_t(UINT32_MAX), INT64_MAX})
        assert(!UsedMetadataWords(b, count) && count == guard);
    assert(UsedMetadataWords(largest, count) && count == 410 + largest * 8);
    assert(UsedMetadataWords(0, count) && count == 410);
    alignas(8) std::array<uint32_t, words + 2> storage;
    storage.fill(guard);
    alignas(8) int32_t q[] = {int32_t(guard), 0, 3, 4, int32_t(guard)};
    alignas(8) int32_t k[] = {int32_t(guard), 0, 17, 50, int32_t(guard)};
    alignas(8) int32_t uq[] = {int32_t(guard), 2, 1, int32_t(guard)};
    alignas(8) int32_t uk[] = {int32_t(guard), 7, 0, int32_t(guard)};
    MlaUsedMetadataArgs args{};
    args.outputAddr = reinterpret_cast<uint64_t>(storage.data() + 1);
    args.outputWords = words;
    args.dtype = 1;
    args.shape = {2, 171, {MlaSequenceLayout::Tnd, 4, 7}, {MlaSequenceLayout::Tnd, 50, 513}};
    args.platform.aicNum = 20;
    args.platform.aivNum = 40;
    args.platform.localMemory = {516096, 174080, 65536, 65536, 131072};
    for (uint32_t c = 0; c < 26; ++c)
        args.platform.blockDims[c] = c + 1;
    args.cuQuery = {reinterpret_cast<uint64_t>(q + 1), 3};
    args.cuKv = {reinterpret_cast<uint64_t>(k + 1), 3};
    args.usedQuery = {reinterpret_cast<uint64_t>(uq + 1), 2};
    args.usedKv = {reinterpret_cast<uint64_t>(uk + 1), 2};
    assert(ComputeMlaUsedMetadata(&args) == 0 && storage[6] == 0);
    const auto good = storage;
    const auto noLengthRead = [](uint32_t) -> uint32_t {
        assert(false);
        return 0;
    };
    const auto readHeader = [&](uint32_t i) {
        assert(i < 8);
        return storage[i + 1];
    };
    MlaDenseMetadataContext context{};
    context.batch = 2;
    MlaDenseKernelSchedule decoded;
    std::memset(&decoded, 0xa5, sizeof(decoded));
    const auto before = decoded;
    for (uint32_t size : {401U, words})
        assert(ReadDenseMetadata(readHeader, size, context, decoded) != MlaMetadataReadStatus::Success);
    for (uint32_t size : {403U, words})
        assert(ReadBsndKvLengthMetadata(readHeader, size, context, noLengthRead, false, 0, decoded) !=
               MlaMetadataReadStatus::Success);
    assert(std::memcmp(&before, &decoded, sizeof(decoded)) == 0);
    MlaTndMetadataContext tndContext{};
    tndContext.batch = 2;
    MlaTndKernelSchedule tndDecoded;
    std::memset(&tndDecoded, 0xa5, sizeof(tndDecoded));
    const auto beforeTnd = tndDecoded;
    for (uint32_t size : {413U, words})
        assert(ReadTndMetadata(readHeader, size, tndContext, noLengthRead, 3, noLengthRead, 3, tndDecoded) !=
               MlaMetadataReadStatus::Success);
    assert(std::memcmp(&beforeTnd, &tndDecoded, sizeof(tndDecoded)) == 0);
    MlaTndBsndMetadataContext mixedContext{};
    mixedContext.batch = 2;
    MlaTndBsndKernelSchedule mixedDecoded;
    std::memset(&mixedDecoded, 0xa5, sizeof(mixedDecoded));
    const auto beforeMixed = mixedDecoded;
    for (uint32_t size : {416U, words})
        assert(ReadTndBsndMetadata(readHeader, size, mixedContext, noLengthRead, 3, noLengthRead, 2, mixedDecoded) !=
               MlaMetadataReadStatus::Success);
    assert(std::memcmp(&beforeMixed, &mixedDecoded, sizeof(mixedDecoded)) == 0);
    const auto unchanged = [&](MlaUsedMetadataArgs bad) {
        assert(ComputeMlaUsedMetadata(&bad) == 1 && storage == good);
    };
    assert(ComputeMlaUsedMetadata(nullptr) == 1);
    for (uint32_t size : {0U, 401U, 405U, 410U, 425U, 427U, UINT32_MAX}) {
        auto bad = args;
        bad.outputWords = size;
        unchanged(bad);
    }
    for (uint64_t address : {uint64_t(0), args.outputAddr + 1, args.outputAddr + 2, UINT64_MAX - 3}) {
        auto bad = args;
        bad.outputAddr = address;
        unchanged(bad);
    }
    for (int64_t b : {-1L, largest + 1, INT64_MAX}) {
        auto bad = args;
        bad.shape.batch = b;
        unchanged(bad);
    }
    const auto failure = [&](MlaUsedMetadataArgs bad, uint32_t status, uint32_t scheduleStatus) {
        assert(ComputeMlaUsedMetadata(&bad) == 0);
        assert(storage.front() == guard && storage.back() == guard);
        assert(storage[6] == status && storage[7] == scheduleStatus);
        for (uint32_t word = 8; word < words; ++word)
            assert(storage[word + 1] == 0);
        assert(ComputeMlaUsedMetadata(&args) == 0 && storage == good);
    };
    for (auto member : {&MlaUsedMetadataArgs::cuQuery, &MlaUsedMetadataArgs::cuKv, &MlaUsedMetadataArgs::usedQuery,
                        &MlaUsedMetadataArgs::usedKv}) {
        for (uint64_t address :
             {args.outputAddr - 4, args.outputAddr + 1, args.outputAddr, args.outputAddr + (words - 1) * 4}) {
            for (uint64_t size : {uint64_t(3), UINT64_MAX}) {
                auto bad = args;
                bad.*member = {address, size};
                unchanged(bad);
            }
        }
        for (uint64_t size : {uint64_t(0), (args.*member).count - 1, (args.*member).count + 1}) {
            auto bad = args;
            (bad.*member).count = size;
            failure(bad, 3, 0);
        }
        for (uint64_t address : {uint64_t(0), args.cuQuery.address + 1, UINT64_MAX - 3}) {
            auto bad = args;
            (bad.*member).address = address;
            failure(bad, 3, 0);
        }
        auto bad = args;
        bad.*member = {UINT64_MAX - 3, UINT64_MAX};
        failure(bad, 3, 0);
    }
    auto bad = args;
    bad.cuQuery.address = bad.cuKv.address = bad.usedQuery.address = bad.usedKv.address = 4;
    bad.dtype = 3; // Static rejection before any load, even from unreadable pointers.
    failure(bad, 1, 0);
    bad.dtype = 1;
    bad.shape.kv.maxLength = 16385;
    failure(bad, 2, 1);
    bad = args;
    bad.platform.libApiWorkspaceBytes = UINT64_MAX;
    failure(bad, 2, 5);
    // Read-only aliases: both prefixes and used arrays may share one backing store.
    bad = args;
    bad.shape.kv.totalTokens = 4;
    bad.cuKv = bad.cuQuery;
    bad.usedKv = bad.usedQuery;
    assert(ComputeMlaUsedMetadata(&bad) == 0 && storage[6] == 0);
    const std::array<uint32_t, 16> aliasSpans = {0, 0, 0, 0, 3, 3, 2, 2, 3, 0, 3, 0, 1, 1, 1, 1};
    assert(std::equal(aliasSpans.begin(), aliasSpans.end(), storage.begin() + 411));
    assert(q[0] == int32_t(guard) && q[4] == int32_t(guard) && q[1] == 0 && q[2] == 3 && q[3] == 4);
    assert(k[0] == int32_t(guard) && k[4] == int32_t(guard) && k[1] == 0 && k[2] == 17 && k[3] == 50);
    assert(uq[0] == int32_t(guard) && uq[3] == int32_t(guard) && uq[1] == 2 && uq[2] == 1);
    assert(uk[0] == int32_t(guard) && uk[3] == int32_t(guard) && uk[1] == 7 && uk[2] == 0);
}

int main(int argc, char** argv)
{
    if (argc == 2 && std::string(argv[1]) == "--check-buffers") {
        CheckBuffers();
        return 0;
    }
    const uint32_t dtype = argc >= 2 ? std::stoul(argv[1]) : 1;
    MlaUsedMetadataArgs args{};
    auto& s = args.shape;
    auto& p = args.platform;
    const auto readArray = [&](MlaUsedMetadataArray& array, std::vector<int32_t>& values) {
        uint32_t present, size;
        std::cin >> present >> array.count >> size;
        values.assign(size + 2, guard);
        for (uint32_t i = 1; i <= size; ++i)
            std::cin >> values[i];
        array.address = present ? reinterpret_cast<uint64_t>(values.data() + 1) : 0;
    };
    while (std::cin >> s.batch >> s.query.maxLength >> s.queryHeads >> s.kv.maxLength >> p.aicNum >> p.aivNum >>
           p.localMemory.l1Bytes >> p.localMemory.ubBytes >> p.localMemory.l0aBytes >> p.localMemory.l0bBytes >>
           p.localMemory.l0cBytes >> p.libApiWorkspaceBytes) {
        uint32_t blockDim, layout;
        std::cin >> blockDim;
        std::vector<int32_t> q, k, uq, uk;
        readArray(args.cuKv, k);
        std::cin >> s.query.totalTokens >> s.kv.totalTokens;
        readArray(args.cuQuery, q);
        std::cin >> layout;
        readArray(args.usedQuery, uq);
        readArray(args.usedKv, uk);
        assert(std::cin);
        s.query.layout =
            layout == 0 ? MlaSequenceLayout::Bsnd : (layout <= 2 ? MlaSequenceLayout::Tnd : MlaSequenceLayout(99));
        s.kv.layout = layout == 1 ? MlaSequenceLayout::Tnd : MlaSequenceLayout::Bsnd;
        const auto qBefore = q, kBefore = k, uqBefore = uq, ukBefore = uk;
        for (uint32_t i = 0; i < 26; ++i)
            p.blockDims[i] = blockDim ? blockDim : i + 1;
        if (!UsedMetadataWords(s.batch, args.outputWords)) {
            std::cout << "[-1]\n";
            continue;
        }
        args.dtype = dtype;
        std::vector<uint32_t> storage(args.outputWords + 2, guard);
        args.outputAddr = reinterpret_cast<uint64_t>(storage.data() + 1);
        assert(ComputeMlaUsedMetadata(&args) == 0);
        assert(storage.front() == guard && storage.back() == guard);
        assert(q == qBefore && k == kBefore && uq == uqBefore && uk == ukBefore);
        std::cout << '[';
        for (uint32_t i = 1; i <= args.outputWords; ++i)
            std::cout << (i == 1 ? "" : ",") << storage[i];
        std::cout << "]\n";
    }
}
