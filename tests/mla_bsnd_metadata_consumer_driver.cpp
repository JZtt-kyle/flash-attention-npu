// CPU/ASan/UBSan bridge for the exact v2 decoder instantiated on the device.
#include "mla_bsnd_metadata_consumer.h"
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

extern "C" uint32_t consume_bsnd_metadata(const void* data, uint32_t words, const MlaDenseMetadataContext* context,
                                          const int32_t* lengths, uint64_t count, MlaDenseKernelSchedule* output,
                                          uint32_t* reads)
{
    reads[0] = reads[1] = 0;
    const auto addr = reinterpret_cast<uintptr_t>(data), lenAddr = reinterpret_cast<uintptr_t>(lengths);
    if (!data || addr % 4 || words > (UINTPTR_MAX - addr) / 4)
        return uint32_t(MlaMetadataReadStatus::InvalidBuffer);
    if (lenAddr % 4 || count > (UINTPTR_MAX - lenAddr) / 4)
        return uint32_t(MlaMetadataReadStatus::InvalidCacheLengths);
    return uint32_t(ReadBsndKvLengthMetadata(Reader{data, words, reads}, words, *context,
                                             Reader{lengths, context->batch, reads + 1}, lengths != nullptr, count,
                                             *output));
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
    for (MlaDenseShape shape : {MlaDenseShape{5, 3, 171, 16384}, {5, 0, 171, 16384}, {5, 3, 171, 0}, {0, 3, 171, 1}}) {
        const uint32_t words = 401 + shape.batch;
        std::vector<int32_t> lengths(shape.batch, 0);
        std::vector<uint32_t> storage(words + 2, 0x12345678);
        auto* data = storage.data() + 1;
        assert(BuildBsndKvLengthMetadata(shape, platform, 1, data, words, lengths.data(), lengths.size()));
        assert(data[5] == 0);
        MlaDenseMetadataContext c{1,
                                  uint32_t(shape.batch),
                                  uint32_t(shape.queryLength),
                                  uint32_t(shape.queryHeads),
                                  uint32_t(shape.kvLength),
                                  20,
                                  40,
                                  data[21],
                                  uint64_t(data[385]) | (uint64_t(data[386]) << 32),
                                  0,
                                  platform.localMemory};
        MlaDenseKernelSchedule output;
        std::memset(&output, 0xa5, sizeof(output));
        const auto before = output;
        uint32_t reads[2];
        uint32_t header[8];
        std::memcpy(header, data, sizeof(header));
        for (uint32_t i = 0; i < 8; ++i) {
            header[i] ^= 0xffffffffU;
            assert(consume_bsnd_metadata(header, words, &c, lengths.data(), lengths.size(), &output, reads) != 0);
            assert(reads[0] <= 8 && reads[1] == 0);
            assert(std::memcmp(&output, &before, sizeof(output)) == 0);
            header[i] ^= 0xffffffffU;
        }
        for (uint32_t size : {0U, 8U, words - 1, words + 1, UINT32_MAX}) {
            assert(consume_bsnd_metadata(header, size, &c, lengths.data(), lengths.size(), &output, reads) == 1);
            assert(reads[0] == 0 && reads[1] == 0);
        }
        assert(consume_bsnd_metadata(data, words, &c, lengths.data(), lengths.size(), &output, reads) == 0);
        for (uint32_t i = 0; i < words; ++i) {
            const auto saved = data[i];
            for (uint32_t bad : {0U, 1U, 0x7fffffffU, 0xffffffffU}) {
                data[i] = bad;
                output = before;
                if (consume_bsnd_metadata(data, words, &c, lengths.data(), lengths.size(), &output, reads) != 0)
                    assert(std::memcmp(&output, &before, sizeof(output)) == 0);
            }
            data[i] = saved;
        }
        for (uint32_t b = 0; b < c.batch; ++b) {
            lengths[b] = -1;
            output = before;
            assert(consume_bsnd_metadata(data, words, &c, lengths.data(), lengths.size(), &output, reads) == 11);
            assert(std::memcmp(&output, &before, sizeof(output)) == 0);
            lengths[b] = 0;
        }
        assert(storage.front() == 0x12345678 && storage.back() == 0x12345678);
        c.batch = UINT32_MAX;
        assert(consume_bsnd_metadata(header, UINT32_MAX, &c, nullptr, 0, &output, reads) == 1);
        assert(reads[0] == 0 && reads[1] == 0);
    }
}
#endif
