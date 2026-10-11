// Sanitizer bridge to the real AICPU entry, with only the device barrier stubbed.
#include "mla_paged_metadata.h"
#include "mla_paged_metadata_args.h"
#include <array>
#include <cassert>
#include <cstring>

extern uint32_t ComputeMlaPagedMetadata(void*);
using namespace flash_attn::mla;

int main()
{
    constexpr uint32_t guard = 0x12345678, words = 418;
    std::array<uint32_t, words + 2> storage;
    storage.fill(guard);
    int32_t q[] = {0, 3, 4}, used[] = {2, 1}, kv[] = {17, 0};
    MlaPagedMetadataArgs args{};
    args.outputAddr = reinterpret_cast<uintptr_t>(storage.data() + 1);
    args.outputWords = words;
    args.dtype = 1;
    args.shape = {2, 171, {MlaSequenceLayout::Tnd, INT64_MIN, 7}, 224, 112};
    args.platform.aicNum = 20;
    args.platform.aivNum = 40;
    args.platform.localMemory = {516096, 174080, 65536, 65536, 131072};
    for (uint32_t c = 0; c < 26; ++c)
        args.platform.blockDims[c] = c + 1;
    args.cuQuery = {reinterpret_cast<uintptr_t>(q), 3};
    args.usedQuery = {reinterpret_cast<uintptr_t>(used), 2};
    args.kvLengths = {reinterpret_cast<uintptr_t>(kv), 2};
    assert(ComputeMlaPagedMetadata(&args) == 0);
    assert(storage[6] == 0 && storage[403] == 4 && storage.front() == guard && storage.back() == guard);
    const auto good = storage;
    const auto beforeArgs = args;
    const auto unchanged = [&](MlaPagedMetadataArgs bad) {
        assert(ComputeMlaPagedMetadata(&bad) == 1 && storage == good);
    };
    assert(ComputeMlaPagedMetadata(nullptr) == 1 && storage == good);
    for (uint64_t address : {uint64_t(0), args.outputAddr + 1, args.outputAddr + 2, UINT64_MAX - 3}) {
        auto bad = args;
        bad.outputAddr = address;
        unchanged(bad);
    }
    for (uint32_t size : {0U, words - 1, words + 1, UINT32_MAX}) {
        auto bad = args;
        bad.outputWords = size;
        unchanged(bad);
    }
    for (int64_t batch : {-1L, int64_t((UINT32_MAX - 408) / 5 + 1), INT64_MAX}) {
        auto bad = args;
        bad.shape.batch = batch;
        unchanged(bad);
    }
    const auto failure = [&](MlaPagedMetadataArgs bad, uint32_t status, uint32_t scheduleStatus) {
        assert(ComputeMlaPagedMetadata(&bad) == 0);
        assert(storage.front() == guard && storage.back() == guard);
        assert(storage[6] == status && storage[7] == scheduleStatus && storage[8] == 0);
        for (uint32_t i = 9; i <= words; ++i)
            assert(storage[i] == 0);
        storage = good;
    };
    for (auto member :
         {&MlaPagedMetadataArgs::cuQuery, &MlaPagedMetadataArgs::usedQuery, &MlaPagedMetadataArgs::kvLengths}) {
        for (uint64_t address :
             {args.outputAddr - 4, args.outputAddr, args.outputAddr + 1, args.outputAddr + 4 * (words - 1)}) {
            for (uint64_t size : {uint64_t(3), UINT64_MAX}) {
                auto bad = args;
                bad.*member = {address, size};
                unchanged(bad);
                bad.dtype = 3;
                unchanged(bad);
            }
        }
        for (uint64_t address : {uint64_t(0), (args.*member).address + 1, UINT64_MAX - 3}) {
            auto bad = args;
            (bad.*member).address = address;
            failure(bad, 3, 0);
        }
        for (uint64_t count : {uint64_t(0), (args.*member).count - 1, (args.*member).count + 1, UINT64_MAX}) {
            auto bad = args;
            bad.*member = {UINT64_MAX - 4095, count};
            failure(bad, 3, 0);
        }
    }
    auto bad = args;
    bad.dtype = 3;
    bad.cuQuery.address = bad.usedQuery.address = bad.kvLengths.address = 4;
    failure(bad, 1, 0);
    for (int32_t tail : {-1, 2, 12}) {
        q[2] = tail;
        failure(args, 2, 1);
    }
    q[2] = 4;
    for (int32_t tail : {5, 6, 7}) {
        q[2] = tail;
        assert(ComputeMlaPagedMetadata(&args) == 0 && storage[403] == uint32_t(tail));
        auto explicitShape = args.shape;
        explicitShape.query.totalTokens = tail;
        std::array<uint32_t, words> cpu{};
        assert(BuildPagedMetadata(explicitShape, args.platform, 1, cpu.data(), words, PagedMetadataInputs(args)));
        assert(std::memcmp(cpu.data(), storage.data() + 1, words * 4) == 0);
    }
    q[2] = 4;
    // No launch descriptor mutation; input storage is read-only even on failures.
    assert(std::memcmp(&args, &beforeArgs, sizeof(args)) == 0);
    assert(q[0] == 0 && q[1] == 3 && used[0] == 2 && used[1] == 1 && kv[0] == 17 && kv[1] == 0);
}
