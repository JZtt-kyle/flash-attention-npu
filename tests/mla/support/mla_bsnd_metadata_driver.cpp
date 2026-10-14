// CPU/ASan bridge to the real AICPU entry, with a test-only barrier shim.
#include "mla_bsnd_metadata.h"
#include "mla_metadata_args.h"
#include <array>
#include <cassert>
#include <cstring>

using namespace flash_attn::mla;
extern uint32_t ComputeMlaBsndKvLengthMetadata(void*);

extern "C" uint32_t bsnd_entry(MlaMetadataArgs* args)
{
    return ComputeMlaBsndKvLengthMetadata(args);
}
extern "C" bool bsnd_words(int64_t batch, uint32_t* words)
{
    return BsndKvLengthMetadataWords(batch, *words);
}

#ifdef MLA_BSND_METADATA_SANITIZE
int main()
{
    constexpr uint32_t guard = 0x12345678, words = 403;
    uint32_t count = guard;
    for (int64_t batch : {int64_t(-1), int64_t(UINT32_MAX) - 400, INT64_MAX}) {
        assert(!BsndKvLengthMetadataWords(batch, count) && count == guard);
    }
    assert(BsndKvLengthMetadataWords(int64_t(UINT32_MAX) - 401, count) && count == UINT32_MAX);
    alignas(8) std::array<uint32_t, words + 2> storage;
    storage.fill(guard);
    alignas(8) int32_t lengths[3] = {int32_t(guard), 0, 513};
    MlaMetadataArgs args{};
    args.outputAddr = reinterpret_cast<uint64_t>(storage.data() + 1);
    args.outputWords = words;
    args.dtype = 1;
    args.shape = {2, 171, 3, 8193};
    args.platform.aicNum = 20;
    args.platform.aivNum = 40;
    args.platform.localMemory = {516096, 174080, 65536, 65536, 131072};
    for (uint32_t c = 0; c < 26; ++c)
        args.platform.blockDims[c] = c + 1;
    args.cacheSeqlensAddr = reinterpret_cast<uint64_t>(lengths + 1);
    args.cacheSeqlensCount = 2;
    assert(bsnd_entry(&args) == 0);
    assert(storage[1] == kMlaMetadataMagic && storage[2] == 2 && storage[3] == 2);
    assert(storage[4] == words && storage[6] == 0 && storage[7] == 0);
    assert(storage[402] == 0 && storage[403] == 513);
    auto before = storage;
    assert(bsnd_entry(nullptr) != 0);
    for (uint32_t size : {0U, 400U, 401U, 402U, 404U, UINT32_MAX}) {
        auto bad = args;
        bad.outputWords = size;
        assert(bsnd_entry(&bad) != 0 && storage == before);
    }
    for (uint64_t address : {uint64_t(0), args.outputAddr + 1, args.outputAddr + 2, UINT64_MAX - 3}) {
        auto bad = args;
        bad.outputAddr = address;
        assert(bsnd_entry(&bad) != 0 && storage == before);
    }
    for (int64_t batch : {int64_t(-1), int64_t(UINT32_MAX), INT64_MAX}) {
        auto bad = args;
        bad.shape.batch = batch;
        assert(bsnd_entry(&bad) != 0 && storage == before);
    }
    for (uint64_t address : {args.outputAddr, args.outputAddr + 4, args.outputAddr + 401 * 4}) {
        auto bad = args;
        bad.cacheSeqlensAddr = address;
        assert(bsnd_entry(&bad) != 0 && storage == before);
    }
    for (uint64_t size : {uint64_t(1), uint64_t(3)}) {
        auto bad = args;
        bad.cacheSeqlensAddr = args.outputAddr + 2;
        bad.cacheSeqlensCount = size;
        assert(bsnd_entry(&bad) != 0 && storage == before);
    }
    auto failure = [&](MlaMetadataArgs bad, uint32_t status, uint32_t scheduleStatus) {
        assert(bsnd_entry(&bad) == 0);
        assert(storage.front() == guard && storage.back() == guard);
        assert(storage[6] == status && storage[7] == scheduleStatus);
        for (uint32_t word = 8; word < words; ++word)
            assert(storage[word + 1] == 0);
    };
    for (uint64_t size : {uint64_t(0), uint64_t(1), uint64_t(3), UINT64_MAX}) {
        auto bad = args;
        bad.cacheSeqlensCount = size;
        failure(bad, 3, 0);
    }
    for (uint64_t address : {uint64_t(0), args.cacheSeqlensAddr + 1, args.cacheSeqlensAddr + 2, UINT64_MAX - 3}) {
        auto bad = args;
        bad.cacheSeqlensAddr = address;
        failure(bad, 3, 0);
    }
    for (int32_t value : {INT32_MIN, -1, 8194, INT32_MAX}) {
        lengths[2] = value;
        failure(args, 2, 1);
    }
    lengths[2] = 513;
    auto bad = args;
    bad.cacheSeqlensAddr = 4; // Aligned, unreadable: static validation must win.
    bad.shape.kvLength = 16385;
    failure(bad, 2, 1);
    bad = args;
    bad.cacheSeqlensAddr = 4;
    bad.dtype = 3;
    failure(bad, 1, 0);
    bad = args;
    bad.platform.libApiWorkspaceBytes = UINT64_MAX;
    failure(bad, 2, 5);
    assert(bsnd_entry(&args) == 0 && storage == before);
    assert(lengths[0] == int32_t(guard) && lengths[1] == 0 && lengths[2] == 513);
}
#endif
