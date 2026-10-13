// CPU/sanitizer bridge to the production v10 AICPU entry and release reader.
#include "mla_paged_metadata_consumer.h"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>
using namespace flash_attn::mla;
constexpr uint32_t guard = 0x12345678;
extern uint32_t ComputeMlaCausalPagedBsndMetadata(void*);

void CheckRecord(MlaPagedMetadataArgs& args, const MlaPagedInputs& source, std::vector<uint32_t>& storage)
{
    auto* data = storage.data() + 1;
    assert(data[1] == 10 && data[2] == 10);
    const auto original = storage;
    auto* output = data;
    for (uint32_t words : {0U, args.outputWords - 1, args.outputWords + 1, UINT32_MAX}) {
        assert(!BuildCausalPagedBsndMetadata(args.shape, args.platform, args.dtype, output, words, source));
        assert(storage == original);
    }
    for (auto member : {&MlaPagedInputs::cuQuery, &MlaPagedInputs::usedQuery, &MlaPagedInputs::kvLengths}) {
        auto overlap = source;
        overlap.*member = {reinterpret_cast<const int32_t*>(output), 1};
        assert(!BuildCausalPagedBsndMetadata(args.shape, args.platform, args.dtype, output, args.outputWords, overlap));
        assert(storage == original);
    }
    if (data[5] != 0) {
        for (uint32_t i = 8; i < args.outputWords; ++i)
            assert(data[i] == 0);
        return;
    }
    MlaPagedMetadataPrefix prefix;
    std::memcpy(&prefix, data, sizeof(prefix));
    MlaPagedMetadataContext c{args.dtype,
                              uint32_t(args.shape.batch),
                              uint32_t(args.shape.queryHeads),
                              prefix.usedFlags,
                              prefix.query,
                              prefix.pageSize,
                              prefix.logicalKvCapacity,
                              args.platform.aicNum,
                              args.platform.aivNum,
                              data[21],
                              uint64_t(data[385]) | (uint64_t(data[386]) << 32),
                              data[15] ? args.platform.libApiWorkspaceBytes : 0,
                              args.platform.localMemory};
    MlaPagedMetadataInputs inputs{args.cuQuery, args.usedQuery, args.kvLengths};
    std::vector<int32_t> pages(uint64_t(c.batch) * (c.logicalKvCapacity / c.pageSize), 0);
    MlaPagedCacheInputs cache{};
    auto& p = cache.layout;
    p.batchSize = c.batch;
    p.pageSize = c.pageSize;
    p.maxPages = c.logicalKvCapacity / c.pageSize;
    p.logicalCapacity = c.logicalKvCapacity;
    p.physicalPages = 1;
    p.tableRowStride = p.maxPages;
    p.tableSpanElements = pages.size();
    p.k = {c.pageSize * 64, 64, 64, c.pageSize * 64};
    p.v = {c.pageSize * 512, 512, 512, c.pageSize * 512};
    cache.table = {reinterpret_cast<uint64_t>(pages.data()), pages.size()};
    cache.k = {4096, p.k.spanElements};
    cache.v = {8192, p.v.spanElements};
    const auto read = [&](uint32_t i) {
        assert(i < args.outputWords);
        return data[i];
    };
    const auto arrays = [&](uint32_t axis, uint32_t i) {
        const MlaLengthArray a[] = {source.cuQuery, source.usedQuery, source.kvLengths};
        assert(axis < 3 && i < a[axis].count);
        return uint32_t(a[axis].data[i]);
    };
    const auto page = [&](uint64_t i) {
        assert(i < pages.size());
        return uint32_t(pages[i]);
    };
    const auto consume = [&](bool success) {
        MlaPagedKernelSchedule decoded;
        std::memset(&decoded, 0xa5, sizeof(decoded));
        const auto before = decoded;
        const auto result = ReadPagedMetadata<true>(read, args.outputWords, c, inputs, arrays, cache, page, decoded);
        assert((result == MlaMetadataReadStatus::Success) == success);
        if (!success)
            assert(std::memcmp(&before, &decoded, sizeof(decoded)) == 0);
    };
    consume(true);
    MlaPagedKernelSchedule decoded{};
    assert(ReadPagedMetadata(read, args.outputWords, c, inputs, arrays, cache, page, decoded) !=
           MlaMetadataReadStatus::Success);
    for (uint32_t i : {0U, 1U, 2U, 3U, 4U, 5U, 6U, 7U, 8U, 10U, 401U, 402U, 404U, 405U, 406U, 407U}) {
        data[i] ^= 0x40000000U;
        consume(false);
        data[i] ^= 0x40000000U;
    }
    for (uint32_t i = 408; i < args.outputWords; ++i) {
        data[i] ^= 1;
        consume(false);
        data[i] ^= 1;
    }
    data[1] = data[2] = 6;
    consume(false);
    data[1] = data[2] = 10;
    for (uint32_t b = 0; b < c.batch; ++b) {
        const uint32_t count = (source.kvLengths.data[b] + c.pageSize - 1) / c.pageSize;
        if (count) {
            auto& last = pages[b * p.maxPages + count - 1];
            last = -1;
            consume(false);
            last = 0;
        }
    }
    assert(storage == original);
}

int main()
{
    MlaPagedShape shape{};
    MlaPlatformInfo p{};
    uint32_t layout, blockDim;
    uint32_t dtype;
    assert(ComputeMlaCausalPagedBsndMetadata(nullptr) == 1);
    while (std::cin >> dtype >> shape.batch >> shape.queryHeads >> layout >> shape.query.totalTokens >>
           shape.query.maxLength >> shape.pageSize >> shape.logicalKvCapacity >> p.aicNum >> p.aivNum >>
           p.localMemory.l1Bytes >> p.localMemory.ubBytes >> p.localMemory.l0aBytes >> p.localMemory.l0bBytes >>
           p.localMemory.l0cBytes >> p.libApiWorkspaceBytes >> blockDim) {
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
        const auto descriptor = [](const MlaLengthArray& a) {
            return MlaPagedMetadataArray{reinterpret_cast<uint64_t>(a.data), a.count};
        };
        MlaPagedMetadataArgs args{reinterpret_cast<uint64_t>(storage.data() + 1),
                                  words,
                                  dtype,
                                  shape,
                                  p,
                                  descriptor(inputs.cuQuery),
                                  descriptor(inputs.usedQuery),
                                  descriptor(inputs.kvLengths)};
        assert(ComputeMlaCausalPagedBsndMetadata(&args) == 0);
        CheckRecord(args, inputs, storage);

        assert(storage.front() == guard && storage.back() == guard);
        for (const auto* item : {&q, &uq, &kv})
            assert(item->first.front() == int32_t(guard) && item->first.back() == int32_t(guard));
        std::cout << '[';
        for (uint32_t i = 1; i <= words; ++i)
            std::cout << (i == 1 ? "" : ",") << storage[i];
        std::cout << "]" << std::endl;
    }
}
