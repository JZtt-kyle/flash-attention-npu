// ASan/UBSan harness for the actual AICPU entry compiled with a barrier shim.
#include "mla_metadata_args.h"
#include <array>
#include <cassert>
#include <cstring>
#include <limits>

extern uint32_t ComputeMlaDenseMetadata(void*);
using namespace flash_attn::mla;

int main()
{
    constexpr uint32_t guard = 0x12345678;
    alignas(8) std::array<uint32_t, kMlaDenseMetadataWords + 2> storage;
    storage.fill(guard);
    MlaMetadataArgs args{};
    args.outputAddr = reinterpret_cast<uint64_t>(storage.data() + 1); // 4-byte, not 8-byte aligned.
    args.outputWords = kMlaDenseMetadataWords;
    args.dtype = static_cast<uint32_t>(MlaMetadataDtype::Fp16);
    args.shape = {2, 1, 8, 257};
    args.platform.aicNum = 20;
    args.platform.aivNum = 40;
    args.platform.localMemory = {516096, 174080, 65536, 65536, 131072};
    for (uint32_t c = 0; c < kMlaMaxAic; ++c)
        args.platform.blockDims[c] = c + 1;
    auto read = [&]() {
        MlaDenseMetadata result;
        std::memcpy(&result, storage.data() + 1, sizeof(result));
        assert(storage.front() == guard && storage.back() == guard);
        return result;
    };
    assert(ComputeMlaDenseMetadata(&args) == 0);
    auto good = read();
    assert(good.magic == kMlaMetadataMagic && good.version == 1 && good.kind == 1);
    assert(good.wordCount == kMlaDenseMetadataWords && good.status == 0 && good.scheduleStatus == 0);
    assert(good.schedule.bSize == 2 && good.schedule.s2Size == 257);
    assert(ComputeMlaDenseMetadata(nullptr) != 0);
    auto before = storage;
    for (uint32_t size : {0U, kMlaDenseMetadataWords - 1, kMlaDenseMetadataWords + 1}) {
        auto bad = args;
        bad.outputWords = size;
        assert(ComputeMlaDenseMetadata(&bad) != 0 && storage == before);
    }
    for (uint64_t address : {uint64_t(0), args.outputAddr + 1, args.outputAddr + 2}) {
        auto bad = args;
        bad.outputAddr = address;
        assert(ComputeMlaDenseMetadata(&bad) != 0 && storage == before);
    }
    auto checkFailure = [&](MlaMetadataArgs bad, uint32_t expectedStatus, uint32_t expectedScheduleStatus) {
        assert(ComputeMlaDenseMetadata(&bad) == 0);
        auto failure = read();
        assert(failure.status == expectedStatus && failure.scheduleStatus == expectedScheduleStatus);
        for (size_t word = 8; word < kMlaDenseMetadataWords; ++word)
            assert(storage[word + 1] == 0);
        assert(ComputeMlaDenseMetadata(&args) == 0 && storage == before);
    };
    auto bad = args;
    bad.shape.kvLength = 513;
    checkFailure(bad, 2, 1);
    bad = args;
    bad.dtype = 0;
    checkFailure(bad, 1, 0);
    bad = args;
    bad.platform.libApiWorkspaceBytes = std::numeric_limits<uint64_t>::max();
    checkFailure(bad, 2, 5);
    uint64_t budget = good.schedule.workspace.userBytes.low | (uint64_t(good.schedule.workspace.userBytes.high) << 32);
    args.platform.libApiWorkspaceBytes = std::numeric_limits<uint64_t>::max() - budget;
    assert(ComputeMlaDenseMetadata(&args) == 0);
    auto wide = read();
    assert(wide.status == 0 && wide.schedule.workspace.totalBytes.low == UINT32_MAX &&
           wide.schedule.workspace.totalBytes.high == UINT32_MAX);
}
