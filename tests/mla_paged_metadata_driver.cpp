// Independent CPU bridge only. Wire decoding/source comparison live in Python.
#include "mla_paged_metadata.h"
#include "mla_bsnd_metadata_consumer.h"
#include "mla_tnd_metadata_consumer.h"
#include "mla_tnd_bsnd_metadata_consumer.h"
#include "mla_used_metadata_consumer.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace flash_attn::mla;
constexpr uint32_t guard = 0x12345678;

void CheckBuffers()
{
    constexpr uint32_t words = 418;
    const int64_t largest = (UINT32_MAX - 408) / 5;
    uint32_t count = guard;
    for (int64_t b : {-1L, largest + 1, int64_t(UINT32_MAX), INT64_MAX})
        assert(!PagedMetadataWords(b, count) && count == guard);
    assert(PagedMetadataWords(largest, count) && count == 408 + largest * 5);
    assert(PagedMetadataWords(0, count) && count == 408);
    alignas(8) std::array<uint32_t, words + 2> storage;
    storage.fill(guard);
    alignas(8) int32_t q[] = {int32_t(guard), 0, 3, 4, int32_t(guard)};
    alignas(8) int32_t uq[] = {int32_t(guard), 2, 1, int32_t(guard)};
    alignas(8) int32_t kv[] = {int32_t(guard), 17, 0, int32_t(guard)};
    const auto qBefore = std::array<int32_t, 5>{q[0], q[1], q[2], q[3], q[4]};
    auto* output = storage.data() + 1; // Deliberately only 4-byte aligned.
    const MlaPagedShape shape{2, 171, {MlaSequenceLayout::Tnd, 4, 7}, 224, 112};
    const MlaPagedInputs inputs{{q + 1, 3}, {uq + 1, 2}, {kv + 1, 2}};
    MlaPlatformInfo platform{};
    platform.aicNum = 20;
    platform.aivNum = 40;
    platform.localMemory = {516096, 174080, 65536, 65536, 131072};
    for (uint32_t c = 0; c < 26; ++c)
        platform.blockDims[c] = c + 1;
    assert(BuildPagedMetadata(shape, platform, 1, output, words, inputs) && storage[6] == 0);
    const auto good = storage;
    const auto noLengthRead = [](uint32_t) -> uint32_t {
        assert(false);
        return 0;
    };
    const auto readHeader = [&](uint32_t i) {
        assert(i < 8);
        return output[i];
    };
    MlaDenseMetadataContext dense{};
    dense.batch = 2;
    MlaDenseKernelSchedule decoded;
    std::memset(&decoded, 0xa5, sizeof(decoded));
    const auto before = decoded;
    for (uint32_t size : {401U, words})
        assert(ReadDenseMetadata(readHeader, size, dense, decoded) != MlaMetadataReadStatus::Success);
    for (uint32_t size : {403U, words})
        assert(ReadBsndKvLengthMetadata(readHeader, size, dense, noLengthRead, false, 0, decoded) !=
               MlaMetadataReadStatus::Success);
    assert(std::memcmp(&before, &decoded, sizeof(decoded)) == 0);
    MlaTndMetadataContext tnd{};
    tnd.batch = 2;
    MlaTndKernelSchedule tndDecoded{};
    const auto beforeTnd = tndDecoded;
    for (uint32_t size : {413U, words})
        assert(ReadTndMetadata(readHeader, size, tnd, noLengthRead, 3, noLengthRead, 3, tndDecoded) !=
               MlaMetadataReadStatus::Success);
    assert(std::memcmp(&beforeTnd, &tndDecoded, sizeof(tndDecoded)) == 0);
    MlaTndBsndMetadataContext mixed{};
    mixed.batch = 2;
    MlaTndBsndKernelSchedule mixedDecoded{};
    const auto beforeMixed = mixedDecoded;
    for (uint32_t size : {416U, words})
        assert(ReadTndBsndMetadata(readHeader, size, mixed, noLengthRead, 3, noLengthRead, 2, mixedDecoded) !=
               MlaMetadataReadStatus::Success);
    assert(std::memcmp(&beforeMixed, &mixedDecoded, sizeof(mixedDecoded)) == 0);
    MlaUsedMetadataContext used{};
    used.batch = 2;
    MlaUsedKernelSchedule usedDecoded{};
    const auto beforeUsed = usedDecoded;
    const auto noArrays = [](uint32_t, uint32_t) -> uint32_t {
        assert(false);
        return 0;
    };
    for (uint32_t size : {426U, words})
        assert(ReadUsedMetadata(readHeader, size, used, {}, noArrays, usedDecoded) != MlaMetadataReadStatus::Success);
    assert(std::memcmp(&beforeUsed, &usedDecoded, sizeof(usedDecoded)) == 0);

    for (uint32_t size : {0U, 401U, 408U, 417U, 419U, UINT32_MAX})
        assert(!BuildPagedMetadata(shape, platform, 1, output, size, inputs) && storage == good);
    for (uintptr_t address : {uintptr_t(0), reinterpret_cast<uintptr_t>(output) + 1,
                              reinterpret_cast<uintptr_t>(output) + 2, UINTPTR_MAX - 3})
        assert(!BuildPagedMetadata(shape, platform, 1, reinterpret_cast<uint32_t*>(address), words, inputs) &&
               storage == good);
    for (int64_t b : {-1L, largest + 1, INT64_MAX}) {
        auto bad = shape;
        bad.batch = b;
        assert(!BuildPagedMetadata(bad, platform, 1, output, words, inputs) && storage == good);
    }
    const auto failure = [&](const MlaPagedShape& s, const MlaPlatformInfo& p, uint32_t dtype,
                             const MlaPagedInputs& arrays, uint32_t status, uint32_t scheduleStatus) {
        assert(BuildPagedMetadata(s, p, dtype, output, words, arrays));
        assert(storage.front() == guard && storage.back() == guard);
        assert(output[0] == 0x4d4c4131 && output[1] == 6 && output[2] == 6 && output[3] == words);
        assert(output[4] == dtype && output[5] == status && output[6] == scheduleStatus && output[7] == 0);
        for (uint32_t word = 8; word < words; ++word)
            assert(output[word] == 0);
        assert(BuildPagedMetadata(shape, platform, 1, output, words, inputs) && storage == good);
    };
    for (auto member : {&MlaPagedInputs::cuQuery, &MlaPagedInputs::usedQuery, &MlaPagedInputs::kvLengths}) {
        for (uintptr_t address :
             {reinterpret_cast<uintptr_t>(output) - 4, reinterpret_cast<uintptr_t>(output) + 1,
              reinterpret_cast<uintptr_t>(output), reinterpret_cast<uintptr_t>(output + words - 1)}) {
            for (uint64_t size : {uint64_t(3), UINT64_MAX}) {
                auto bad = inputs;
                bad.*member = {reinterpret_cast<const int32_t*>(address), size};
                assert(!BuildPagedMetadata(shape, platform, 1, output, words, bad) && storage == good);
                // Unsafe output wins even when dtype would otherwise fail.
                assert(!BuildPagedMetadata(shape, platform, 3, output, words, bad) && storage == good);
            }
        }
        for (uint64_t size : {uint64_t(0), (inputs.*member).count - 1, (inputs.*member).count + 1, UINT64_MAX}) {
            auto bad = inputs;
            bad.*member = {reinterpret_cast<const int32_t*>(UINTPTR_MAX - 4095), size};
            failure(shape, platform, 1, bad, 3, 0);
        }
        for (uintptr_t address : {uintptr_t(0), reinterpret_cast<uintptr_t>(q + 1) + 1, UINTPTR_MAX - 3}) {
            auto bad = inputs;
            (bad.*member).data = reinterpret_cast<const int32_t*>(address);
            failure(shape, platform, 1, bad, 3, 0);
        }
    }
    auto unreadable = inputs;
    unreadable.cuQuery.data = unreadable.usedQuery.data = unreadable.kvLengths.data =
        reinterpret_cast<const int32_t*>(4);
    failure(shape, platform, 3, unreadable, 1, 0);
    auto badShape = shape;
    badShape.pageSize = 17;
    failure(badShape, platform, 1, unreadable, 2, 1);
    badShape = shape;
    badShape.query.layout = MlaSequenceLayout(99);
    failure(badShape, platform, 1, unreadable, 2, 1);
    auto badPlatform = platform;
    badPlatform.libApiWorkspaceBytes = UINT64_MAX;
    failure(shape, badPlatform, 1, inputs, 2, 5);
    // All read-only arrays may alias the same backing storage.
    auto alias = inputs;
    alias.usedQuery = {q + 1, 2};
    alias.kvLengths = alias.usedQuery;
    auto aliasShape = shape;
    aliasShape.query.totalTokens = 7;
    q[3] = 7;
    assert(BuildPagedMetadata(aliasShape, platform, 1, output, words, alias) && output[5] == 0);
    const std::array<uint32_t, 10> aliasSpans{0, 0, 3, 0, 0, 3, 0, 4, 3, 3};
    assert(std::equal(aliasSpans.begin(), aliasSpans.end(), output + 408));
    q[3] = 4;
    assert(std::equal(qBefore.begin(), qBefore.end(), q));
    assert(uq[0] == int32_t(guard) && uq[1] == 2 && uq[2] == 1 && uq[3] == int32_t(guard));
    assert(kv[0] == int32_t(guard) && kv[1] == 17 && kv[2] == 0 && kv[3] == int32_t(guard));
    // Adjacent inputs/outputs do not overlap; the guard slots can hold lengths.
    std::array<uint32_t, words + 2> adjacent{};
    adjacent[words] = 17;
    auto near = inputs;
    near.kvLengths = {reinterpret_cast<const int32_t*>(adjacent.data() + words), 2};
    assert(BuildPagedMetadata(shape, platform, 1, adjacent.data(), words, near) && adjacent[5] == 0);
    assert(adjacent[words] == 17 && adjacent[words + 1] == 0);
}

int main(int argc, char** argv)
{
    if (argc == 2 && std::string(argv[1]) == "--check-buffers") {
        CheckBuffers();
        return 0;
    }
    const uint32_t dtype = argc >= 2 ? std::stoul(argv[1]) : 1;
    MlaPagedShape shape{};
    MlaPlatformInfo p{};
    uint32_t layout, blockDim;
    while (std::cin >> shape.batch >> shape.queryHeads >> layout >> shape.query.totalTokens >> shape.query.maxLength >>
           shape.pageSize >> shape.logicalKvCapacity >> p.aicNum >> p.aivNum >> p.localMemory.l1Bytes >>
           p.localMemory.ubBytes >> p.localMemory.l0aBytes >> p.localMemory.l0bBytes >> p.localMemory.l0cBytes >>
           p.libApiWorkspaceBytes >> blockDim) {
        shape.query.layout = MlaSequenceLayout(layout);
        const auto readArray = [&]() {
            uint32_t present, size;
            uint64_t count;
            std::cin >> present >> count >> size;
            std::vector<int32_t> values(size + 2, guard);
            for (uint32_t i = 1; i <= size; ++i)
                std::cin >> values[i];
            // Descriptors may be malformed, but any count that could pass
            // validation must still refer to readable driver-owned values.
            assert(!present || count <= size || (count != uint64_t(shape.batch) && count != uint64_t(shape.batch) + 1));
            return std::make_pair(std::move(values), std::make_pair(present, count));
        };
        const auto q = readArray(), uq = readArray(), kv = readArray();
        assert(std::cin);
        const auto array = [](const auto& item) {
            return MlaLengthArray{item.second.first ? item.first.data() + 1 : nullptr, item.second.second};
        };
        const MlaPagedInputs inputs{array(q), array(uq), array(kv)};
        for (uint32_t i = 0; i < 26; ++i)
            p.blockDims[i] = blockDim ? blockDim : i + 1;
        uint32_t words;
        if (!PagedMetadataWords(shape.batch, words)) {
            std::cout << "[-1]\n";
            continue;
        }
        assert(words <= 1000000); // Harness allocation limit, not protocol limit.
        std::vector<uint32_t> storage(words + 2, guard);
        assert(BuildPagedMetadata(shape, p, dtype, storage.data() + 1, words, inputs));
        assert(storage.front() == guard && storage.back() == guard);
        for (const auto* item : {&q, &uq, &kv})
            assert(item->first.front() == int32_t(guard) && item->first.back() == int32_t(guard));
        std::cout << '[';
        for (uint32_t i = 1; i <= words; ++i)
            std::cout << (i == 1 ? "" : ",") << storage[i];
        std::cout << "]\n";
    }
}
