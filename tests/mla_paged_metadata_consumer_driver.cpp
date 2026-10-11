// CPU/sanitizer bridge for the same reader instantiated in the release probe.
#include "mla_paged_metadata_consumer.h"
#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>

using namespace flash_attn::mla;

struct Reader {
    const void* data;
    uint64_t size;
    uint64_t* reads;
    uint32_t operator()(uint64_t i) const
    {
        assert(i < size);
        if (*reads < i + 1)
            *reads = i + 1;
        uint32_t value;
        std::memcpy(&value, static_cast<const char*>(data) + i * 4, 4);
        return value;
    }
};
struct Arrays {
    Reader data[3];
    uint32_t operator()(uint32_t axis, uint32_t i) const
    {
        assert(axis < 3);
        return data[axis](i);
    }
};

extern "C" uint32_t consume_paged_metadata(const void* data, uint32_t words, const MlaPagedMetadataContext* context,
                                           const MlaPagedMetadataInputs* inputs, const MlaPagedCacheInputs* cache,
                                           MlaPagedKernelSchedule* output, uint64_t* reads)
{
    std::memset(reads, 0, 5 * sizeof(uint64_t));
    if (!data || !ValidPagedMetadataAddress(reinterpret_cast<uint64_t>(data), words))
        return uint32_t(MlaMetadataReadStatus::InvalidBuffer);
    Arrays arrays;
    const MlaPagedMetadataArray descriptors[] = {inputs->cuQuery, inputs->usedQuery, inputs->kvLengths};
    for (uint32_t i = 0; i < 3; ++i)
        arrays.data[i] = {reinterpret_cast<const void*>(descriptors[i].address), descriptors[i].count, reads + i + 1};
    const Reader pages{reinterpret_cast<const void*>(cache->table.address), cache->table.count, reads + 4};
    return uint32_t(
        ReadPagedMetadata(Reader{data, words, reads}, words, *context, *inputs, arrays, *cache, pages, *output));
}

extern "C" uint32_t build_paged_cache(const MlaPageTableView* table, const MlaBsndTensorView* k,
                                      const MlaBsndTensorView* v, MlaPagedCacheLayout* output)
{
    return uint32_t(BuildPagedCacheLayout(*table, *k, *v, *output));
}

#ifdef MLA_CONSUMER_SANITIZE
int main()
{
    MlaPlatformInfo platform{};
    platform.aicNum = 20;
    platform.aivNum = 40;
    platform.localMemory = {516096, 174080, 65536, 65536, 131072};
    for (uint32_t i = 0; i < 20; ++i)
        platform.blockDims[i] = i + 1;
    uint64_t calls = 0;
    for (uint32_t dtype : {1U, 2U}) {
        for (uint32_t layout : {0U, 1U}) {
            for (uint32_t page = 16; page <= 1024; page += 16) {
                for (uint32_t mode = 0; mode < 6; ++mode) {
                    const uint32_t batch = mode == 1 ? 0 : 3, qc = mode == 2 ? 0 : 7;
                    const uint32_t capacity = mode == 3 ? 0 : 3 * page;
                    std::vector<int32_t> q(batch + 1), uq(batch), kv(batch), table(batch * 5, -1);
                    for (uint32_t b = 0; b < batch; ++b) {
                        q[b + 1] = q[b] + qc;
                        uq[b] = mode == 4 ? 0 : qc;
                        kv[b] = capacity == 0 || mode == 5 ? 0 : page + 1;
                        table[b * 5] = 2;
                        table[b * 5 + 1] = 0;
                    }
                    int32_t empty = 0;
                    MlaPagedShape shape{
                        batch, 171, {MlaSequenceLayout(layout), int64_t(batch) * qc, qc}, capacity, page};
                    MlaPagedInputs source{{layout ? q.data() : nullptr, layout ? batch + 1 : 0},
                                          {batch ? uq.data() : &empty, batch},
                                          {kv.data(), batch}};
                    const uint32_t words = 408 + 5 * batch;
                    std::vector<uint32_t> record(words + 2, 0x12345678);
                    auto* data = record.data() + 1;
                    assert(BuildPagedMetadata(shape, platform, dtype, data, words, source));
                    assert(data[5] == 0);
                    MlaPagedMetadataPrefix prefix;
                    std::memcpy(&prefix, data, sizeof(prefix));
                    MlaPagedMetadataContext context{dtype,
                                                    batch,
                                                    171,
                                                    prefix.usedFlags,
                                                    prefix.query,
                                                    page,
                                                    capacity,
                                                    20,
                                                    40,
                                                    data[21],
                                                    uint64_t(data[385]) | (uint64_t(data[386]) << 32),
                                                    0,
                                                    platform.localMemory};
                    const auto desc = [](const MlaLengthArray& a) {
                        return MlaPagedMetadataArray{reinterpret_cast<uint64_t>(a.data), a.count};
                    };
                    MlaPagedMetadataInputs inputs{desc(source.cuQuery), desc(source.usedQuery), desc(source.kvLengths)};
                    // Cache contents are not read by this validation layer.
                    MlaPagedCacheInputs cache{};
                    auto& p = cache.layout;
                    p.batchSize = batch;
                    p.maxPages = capacity / page;
                    p.logicalCapacity = capacity;
                    p.physicalPages = 3;
                    p.pageSize = page;
                    p.tableRowStride = 5;
                    p.tableSpanElements = batch && capacity ? (batch - 1) * 5 + p.maxPages : 0;
                    p.k = {page * 64, 64, 64, 3 * page * 64};
                    p.v = {page * 512, 512, 512, 3 * page * 512};
                    cache.table = {reinterpret_cast<uint64_t>(table.data()), table.size()};
                    cache.k = {4096, p.k.spanElements};
                    cache.v = {8192, p.v.spanElements};
                    MlaPagedKernelSchedule output;
                    uint64_t reads[5];
                    const auto consume = [&](uint32_t size) {
                        std::memset(&output, 0xa5, sizeof(output));
                        const auto before = output;
                        const auto result =
                            consume_paged_metadata(data, size, &context, &inputs, &cache, &output, reads);
                        if (result != 0)
                            assert(std::memcmp(&before, &output, sizeof(output)) == 0);
                        assert(record.front() == 0x12345678 && record.back() == 0x12345678);
                        ++calls;
                        return result;
                    };
                    assert(consume(words) == 0);
                    for (uint32_t size : {0U, 8U, words - 1, words + 1, UINT32_MAX}) {
                        assert(consume(size) == 1 && reads[0] == 0);
                    }
                    for (uint32_t word = 0; word < words; ++word) {
                        const auto saved = data[word];
                        for (uint32_t bad : {0U, 1U, 0x7fffffffU, 0xffffffffU}) {
                            data[word] = bad;
                            const auto result = consume(words);
                            if (word < 8 && bad != saved)
                                assert(result != 0 && reads[0] <= 8 && reads[4] == 0);
                        }
                        data[word] = saved;
                    }
                    if (batch && kv[0]) {
                        for (uint32_t b = 0; b < batch; ++b) {
                            table[b * 5 + 1] = -1;
                            assert(consume(words) == 15);
                            table[b * 5 + 1] = 0;
                        }
                    }
                    for (auto* a :
                         {&inputs.cuQuery, &inputs.usedQuery, &inputs.kvLengths, &cache.table, &cache.k, &cache.v}) {
                        const auto saved = *a;
                        for (uint64_t address : {uint64_t(1), uint64_t(3), UINT64_MAX}) {
                            a->address = address;
                            assert(consume(words) != 0 && reads[4] == 0);
                        }
                        *a = saved;
                    }
                }
            }
        }
    }
    std::cout << "{\"calls\":" << calls << ",\"status\":\"passed\"}\n";
}
#endif
