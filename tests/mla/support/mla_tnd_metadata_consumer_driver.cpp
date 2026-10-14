// CPU/ASan/UBSan bridge for the exact decoder instantiated on 910B.
#include "mla_tnd_metadata_consumer.h"
#include "mla_tnd_schedule.h"
#include <cassert>
#include <cstring>
#include <vector>

using namespace flash_attn::mla;

struct Reader {
    const void* data;
    uint32_t size;
    uint32_t* reads;
    uint32_t operator()(uint32_t i) const
    {
        assert(i < size);
        if (*reads < i + 1)
            *reads = i + 1;
        uint32_t word;
        std::memcpy(&word, static_cast<const char*>(data) + uint64_t(i) * sizeof(word), sizeof(word));
        return word;
    }
};

extern "C" uint32_t consume_tnd_metadata(const void* data, uint32_t words, const MlaTndMetadataContext* context,
                                         const int32_t* query, uint64_t queryCount, const int32_t* kv, uint64_t kvCount,
                                         MlaTndKernelSchedule* output, uint32_t* reads)
{
    reads[0] = reads[1] = reads[2] = 0;
    const auto addr = reinterpret_cast<uintptr_t>(data);
    if (!data || addr % 4 || words > (UINTPTR_MAX - addr) / 4)
        return uint32_t(MlaMetadataReadStatus::InvalidBuffer);
    const auto qAddr = reinterpret_cast<uintptr_t>(query), kAddr = reinterpret_cast<uintptr_t>(kv);
    if (!query || !kv || qAddr % 4 || kAddr % 4 || queryCount > (UINTPTR_MAX - qAddr) / 4 ||
        kvCount > (UINTPTR_MAX - kAddr) / 4)
        return uint32_t(MlaMetadataReadStatus::InvalidPrefixes);
    return uint32_t(ReadTndMetadata(Reader{data, words, reads}, words, *context,
                                    Reader{query, uint32_t(queryCount), reads + 1}, queryCount,
                                    Reader{kv, uint32_t(kvCount), reads + 2}, kvCount, *output));
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
    for (uint32_t dtype : {1U, 2U}) {
        for (uint32_t batch : {0U, 1U, 3U, 10U, 21U, 41U}) {
            for (uint32_t mode = 0; mode < 5; ++mode) {
                std::vector<int32_t> q(batch + 1, 0), k(batch + 1, 0);
                for (uint32_t b = 0; b < batch; ++b) {
                    q[b + 1] = q[b] + (mode == 1 || (mode == 4 && b % 2) ? 0 : 3);
                    k[b + 1] = k[b] + (mode == 2 || (mode == 4 && b % 2 == 0) ? 0 : (b % 2 ? 16384 : 513));
                }
                MlaTndShape shape{batch, q.back(), k.back(), 171, 7, 16384};
                const uint32_t words = 405 + 4 * batch;
                std::vector<uint32_t> storage(words + 2, 0x12345678);
                auto* data = storage.data() + 1;
                assert(BuildTndMetadata(shape, platform, dtype, data, words, q.data(), q.size(), k.data(), k.size()));
                assert(data[5] == 0);
                MlaTndMetadataContext c{dtype,
                                        batch,
                                        uint32_t(q.back()),
                                        uint32_t(k.back()),
                                        171,
                                        7,
                                        16384,
                                        20,
                                        40,
                                        data[21],
                                        uint64_t(data[385]) | (uint64_t(data[386]) << 32),
                                        0,
                                        platform.localMemory};
                MlaTndKernelSchedule output;
                std::memset(&output, 0xa5, sizeof(output));
                const auto before = output;
                uint32_t reads[3];
                uint32_t header[8];
                std::memcpy(header, data, sizeof(header));
                for (uint32_t i = 0; i < 8; ++i) {
                    header[i] ^= 0xffffffffU;
                    assert(consume_tnd_metadata(header, words, &c, q.data(), q.size(), k.data(), k.size(), &output,
                                                reads) != 0);
                    assert(reads[0] <= 8 && reads[1] == 0 && reads[2] == 0);
                    assert(std::memcmp(&output, &before, sizeof(output)) == 0);
                    header[i] ^= 0xffffffffU;
                }
                for (uint32_t size : {0U, 8U, words - 1, words + 1, UINT32_MAX}) {
                    assert(consume_tnd_metadata(header, size, &c, q.data(), q.size(), k.data(), k.size(), &output,
                                                reads) == 1);
                    assert(reads[0] == 0 && reads[1] == 0 && reads[2] == 0);
                }
                assert(consume_tnd_metadata(data, words, &c, q.data(), q.size(), k.data(), k.size(), &output, reads) ==
                       0);
                for (uint32_t i = 0; i < words; ++i) {
                    const auto saved = data[i];
                    for (uint32_t bad : {0U, 1U, 0x7fffffffU, 0xffffffffU}) {
                        data[i] = bad;
                        output = before;
                        if (consume_tnd_metadata(data, words, &c, q.data(), q.size(), k.data(), k.size(), &output,
                                                 reads) != 0)
                            assert(std::memcmp(&output, &before, sizeof(output)) == 0);
                    }
                    data[i] = saved;
                }
                for (auto* prefix : {&q, &k}) {
                    for (uint32_t b = 0; b <= batch; ++b) {
                        const auto saved = (*prefix)[b];
                        (*prefix)[b] = -1;
                        output = before;
                        assert(consume_tnd_metadata(data, words, &c, q.data(), q.size(), k.data(), k.size(), &output,
                                                    reads) == 12);
                        assert(std::memcmp(&output, &before, sizeof(output)) == 0);
                        (*prefix)[b] = saved;
                    }
                }
                assert(storage.front() == 0x12345678 && storage.back() == 0x12345678);
                c.batch = UINT32_MAX;
                assert(consume_tnd_metadata(header, UINT32_MAX, &c, q.data(), q.size(), k.data(), k.size(), &output,
                                            reads) == 1);
                assert(reads[0] == 0 && reads[1] == 0 && reads[2] == 0);
            }
        }
    }
}
#endif
