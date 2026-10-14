// CPU/ASan bridge to the real v4 AICPU entry (test-only barrier shim).
// The wire projection/source oracle is in Python, not this driver.
#include "mla_tnd_bsnd_metadata.h"
#include "mla_tnd_bsnd_metadata_args.h"
#include "mla_bsnd_metadata_consumer.h"
#include "mla_tnd_metadata_consumer.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace flash_attn::mla;
extern uint32_t ComputeMlaTndBsndMetadata(void*);
extern uint32_t ComputeMlaTndBsndMetadataFromPrefix(void*);
constexpr uint32_t guard = 0x12345678;

void CheckBuffers()
{
    constexpr uint32_t words = 416;
    uint32_t count = guard;
    const int64_t largest = (UINT32_MAX - 406) / 5;
    for (int64_t b : {-1L, largest + 1, int64_t(UINT32_MAX), INT64_MAX}) {
        assert(!TndBsndMetadataWords(b, count) && count == guard);
    }
    assert(TndBsndMetadataWords(largest, count) && count == 406 + largest * 5);
    assert(TndBsndMetadataWords(0, count) && count == 406);
    alignas(8) std::array<uint32_t, words + 2> storage;
    storage.fill(guard);
    alignas(8) int32_t q[] = {int32_t(guard), 0, 3, 4, int32_t(guard)};
    alignas(8) int32_t k[] = {int32_t(guard), 17, 33, int32_t(guard)};
    MlaTndBsndMetadataArgs args{};
    args.outputAddr = reinterpret_cast<uint64_t>(storage.data() + 1);
    args.outputWords = words;
    args.dtype = 1;
    args.shape = {2, 4, 171, 7, 513};
    args.platform.aicNum = 20;
    args.platform.aivNum = 40;
    args.platform.localMemory = {516096, 174080, 65536, 65536, 131072};
    for (uint32_t c = 0; c < 26; ++c)
        args.platform.blockDims[c] = c + 1;
    args.cuQueryAddr = reinterpret_cast<uint64_t>(q + 1);
    args.kvLengthsAddr = reinterpret_cast<uint64_t>(k + 1);
    args.queryCount = 3;
    args.kvCount = 2;
    assert(ComputeMlaTndBsndMetadata(&args) == 0 && storage[6] == 0);
    auto good = storage;
    // The official generator has no total-Q parameter. Its device-only mode
    // derives that value and uses all cores without changing the explicit ABI.
    auto inferred = args;
    inferred.shape.totalQuery = INT64_MAX;
    auto full = args;
    full.platform.blockDims.fill(full.platform.blockDims[full.platform.aicNum - 1]);
    assert(ComputeMlaTndBsndMetadata(&full) == 0);
    auto publicGood = storage;
    assert(ComputeMlaTndBsndMetadataFromPrefix(&inferred) == 0 && storage == publicGood);
    assert(ComputeMlaTndBsndMetadataFromPrefix(nullptr) == 1);
    for (bool query : {false, true}) {
        auto bad = inferred;
        (query ? bad.cuQueryAddr : bad.kvLengthsAddr) = 0;
        assert(ComputeMlaTndBsndMetadataFromPrefix(&bad) == 0 && storage[6] == 3);
        for (uint32_t i = 9; i <= words; ++i)
            assert(storage[i] == 0);
        bad = inferred;
        (query ? bad.queryCount : bad.kvCount) = 0;
        assert(ComputeMlaTndBsndMetadataFromPrefix(&bad) == 0 && storage[6] == 3);
        assert(ComputeMlaTndBsndMetadataFromPrefix(&inferred) == 0 && storage == publicGood);
        bad = inferred;
        (query ? bad.cuQueryAddr : bad.kvLengthsAddr) = args.outputAddr;
        assert(ComputeMlaTndBsndMetadataFromPrefix(&bad) == 1 && storage == publicGood);
    }
    assert(ComputeMlaTndBsndMetadata(&args) == 0 && storage == good);
    MlaDenseMetadataContext context{};
    context.batch = 2;
    MlaDenseKernelSchedule decoded;
    std::memset(&decoded, 0xa5, sizeof(decoded));
    const auto beforeDecode = decoded;
    const auto readHeader = [&](uint32_t i) {
        assert(i < 8); // Old consumers must never inspect the v4 payload.
        return storage[i + 1];
    };
    const auto noLengthRead = [](uint32_t) -> uint32_t {
        assert(false);
        return 0;
    };
    for (uint32_t size : {401U, words})
        assert(ReadDenseMetadata(readHeader, size, context, decoded) != MlaMetadataReadStatus::Success);
    for (uint32_t size : {403U, words})
        assert(ReadBsndKvLengthMetadata(readHeader, size, context, noLengthRead, false, 0, decoded) !=
               MlaMetadataReadStatus::Success);
    assert(std::memcmp(&decoded, &beforeDecode, sizeof(decoded)) == 0);
    MlaTndMetadataContext tndContext{};
    tndContext.batch = 2;
    MlaTndKernelSchedule tndDecoded{};
    std::memset(&tndDecoded, 0xa5, sizeof(tndDecoded));
    const auto beforeTnd = tndDecoded;
    for (uint32_t size : {413U, words})
        assert(ReadTndMetadata(readHeader, size, tndContext, noLengthRead, 3, noLengthRead, 3, tndDecoded) !=
               MlaMetadataReadStatus::Success);
    assert(std::memcmp(&tndDecoded, &beforeTnd, sizeof(tndDecoded)) == 0);
    const auto unchanged = [&](MlaTndBsndMetadataArgs bad) {
        assert(ComputeMlaTndBsndMetadata(&bad) == 1 && storage == good);
    };
    assert(ComputeMlaTndBsndMetadata(nullptr) == 1);
    for (uint32_t size : {0U, 401U, 405U, 406U, 413U, 415U, 417U, UINT32_MAX}) {
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
    // Check each input separately, including partial overlap at both ends,
    // unaligned overlap, wrong counts and overflowing descriptor spans.
    for (bool query : {false, true}) {
        for (uint64_t address :
             {args.outputAddr - 4, args.outputAddr + 1, args.outputAddr, args.outputAddr + (words - 1) * 4}) {
            for (uint64_t size : {uint64_t(3), UINT64_MAX}) {
                auto bad = args;
                (query ? bad.cuQueryAddr : bad.kvLengthsAddr) = address;
                (query ? bad.queryCount : bad.kvCount) = size;
                unchanged(bad);
            }
        }
    }
    const auto failure = [&](MlaTndBsndMetadataArgs bad, uint32_t status, uint32_t scheduleStatus) {
        assert(ComputeMlaTndBsndMetadata(&bad) == 0);
        assert(storage.front() == guard && storage.back() == guard);
        assert(storage[6] == status && storage[7] == scheduleStatus);
        for (uint32_t word = 8; word < words; ++word)
            assert(storage[word + 1] == 0);
        // Reuse the same buffer after failure; no stale payload may survive.
        assert(ComputeMlaTndBsndMetadata(&args) == 0 && storage == good);
    };
    for (bool query : {false, true}) {
        for (uint64_t size : {uint64_t(0), uint64_t(query ? 2 : 1), uint64_t(query ? 4 : 3)}) {
            auto bad = args;
            (query ? bad.queryCount : bad.kvCount) = size;
            failure(bad, 3, 0);
        }
        for (uint64_t address : {uint64_t(0), args.cuQueryAddr + 1, UINT64_MAX - 3}) {
            auto bad = args;
            (query ? bad.cuQueryAddr : bad.kvLengthsAddr) = address;
            failure(bad, 3, 0);
        }
        auto bad = args;
        (query ? bad.cuQueryAddr : bad.kvLengthsAddr) = UINT64_MAX - 3;
        (query ? bad.queryCount : bad.kvCount) = UINT64_MAX;
        failure(bad, 3, 0);
    }
    auto bad = args;
    bad.cuQueryAddr = bad.kvLengthsAddr = 4; // Unreadable; static rejection must precede loads.
    bad.dtype = 3;
    failure(bad, 1, 0);
    bad.dtype = 1;
    bad.shape.kvCapacity = 16385;
    failure(bad, 2, 1);
    bad = args;
    bad.platform.libApiWorkspaceBytes = UINT64_MAX;
    failure(bad, 2, 5);
    // Aliasing the read-only Q prefix and KV lengths is legal and retains independent spans.
    bad = args;
    bad.kvLengthsAddr = bad.cuQueryAddr;
    assert(ComputeMlaTndBsndMetadata(&bad) == 0 && storage[6] == 0);
    const std::array<uint32_t, 10> aliasSpans = {0, 0, 0, 3, 0, 3, 513, 0, 1, 3};
    assert(std::equal(aliasSpans.begin(), aliasSpans.end(), storage.begin() + 407));
    assert(q[0] == int32_t(guard) && q[4] == int32_t(guard) && q[1] == 0 && q[2] == 3 && q[3] == 4);
    assert(k[0] == int32_t(guard) && k[3] == int32_t(guard) && k[1] == 17 && k[2] == 33);
}

int main(int argc, char** argv)
{
    if (argc == 2 && std::string(argv[1]) == "--check-buffers") {
        CheckBuffers();
        return 0;
    }
    const uint32_t dtype = argc >= 2 ? std::stoul(argv[1]) : 1;
    const bool fromQueryPrefix = argc == 3 && std::string(argv[2]) == "--from-prefix";
    MlaTndBsndMetadataArgs args{};
    auto& s = args.shape;
    auto& p = args.platform;
    while (std::cin >> s.batch >> s.maxQueryLength >> s.queryHeads >> s.kvCapacity >> p.aicNum >> p.aivNum >>
           p.localMemory.l1Bytes >> p.localMemory.ubBytes >> p.localMemory.l0aBytes >> p.localMemory.l0bBytes >>
           p.localMemory.l0cBytes >> p.libApiWorkspaceBytes) {
        uint32_t blockDim, hasQ, hasK, ksize, qsize;
        std::cin >> blockDim >> hasK >> args.kvCount >> ksize;
        std::vector<int32_t> k(ksize + 2, guard);
        for (uint32_t i = 1; i <= ksize; ++i)
            std::cin >> k[i];
        int64_t unusedTotalKv;
        std::cin >> s.totalQuery >> unusedTotalKv >> hasQ >> args.queryCount >> qsize;
        std::vector<int32_t> q(qsize + 2, guard);
        for (uint32_t i = 1; i <= qsize; ++i)
            std::cin >> q[i];
        assert(std::cin);
        const auto qBefore = q, kBefore = k;
        for (uint32_t i = 0; i < 26; ++i)
            p.blockDims[i] = blockDim ? blockDim : i + 1;
        if (!TndBsndMetadataWords(s.batch, args.outputWords)) {
            std::cout << "[-1]\n";
            continue;
        }
        args.dtype = dtype;
        args.cuQueryAddr = hasQ ? reinterpret_cast<uint64_t>(q.data() + 1) : 0;
        args.kvLengthsAddr = hasK ? reinterpret_cast<uint64_t>(k.data() + 1) : 0;
        std::vector<uint32_t> storage(args.outputWords + 2, guard);
        args.outputAddr = reinterpret_cast<uint64_t>(storage.data() + 1);
        assert((fromQueryPrefix ? ComputeMlaTndBsndMetadataFromPrefix(&args) : ComputeMlaTndBsndMetadata(&args)) == 0);
        assert(storage.front() == guard && storage.back() == guard && q == qBefore && k == kBefore);
        std::cout << '[';
        for (uint32_t i = 1; i <= args.outputWords; ++i)
            std::cout << (i == 1 ? "" : ",") << storage[i];
        std::cout << "]\n";
    }
}
