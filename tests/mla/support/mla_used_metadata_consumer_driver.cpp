// CPU/ASan/UBSan bridge for the exact decoder instantiated on 910B.
#include "mla_used_metadata_consumer.h"
#include <cassert>
#include <cstring>
#include <vector>

using namespace flash_attn::mla;

struct Reader {
    const void* data;
    uint64_t size;
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
struct Arrays {
    Reader data[4];
    uint32_t operator()(uint32_t axis, uint32_t i) const
    {
        assert(axis < 4);
        return data[axis](i);
    }
};

extern "C" uint32_t consume_used_metadata(const void* data, uint32_t words, const MlaUsedMetadataContext* context,
                                          const MlaUsedMetadataInputs* inputs, MlaUsedKernelSchedule* output,
                                          uint32_t* reads)
{
    std::memset(reads, 0, 5 * sizeof(uint32_t));
    if (!data || !ValidUsedMetadataAddress(reinterpret_cast<uintptr_t>(data), words))
        return uint32_t(MlaMetadataReadStatus::InvalidBuffer);
    Arrays arrays;
    const MlaUsedMetadataArray descriptors[] = {inputs->cuQuery, inputs->cuKv, inputs->usedQuery, inputs->usedKv};
    for (uint32_t i = 0; i < 4; ++i)
        arrays.data[i] = {reinterpret_cast<const void*>(descriptors[i].address), descriptors[i].count, reads + i + 1};
    return uint32_t(ReadUsedMetadata(Reader{data, words, reads}, words, *context, *inputs, arrays, *output));
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
        for (uint32_t layout = 0; layout < 3; ++layout) {
            for (uint32_t batch : {0U, 1U, 3U, 10U, 21U}) {
                for (uint32_t mode = 0; mode < 6; ++mode) {
                    const uint32_t qc = mode == 1 ? 0 : 7, kc = mode == 2 ? 0 : 513;
                    std::vector<int32_t> q(batch + 3, 0), k(batch + 3, 0), uq(batch + 2, 0), uk(batch + 2, 0);
                    for (uint32_t b = 0; b < batch; ++b) {
                        q[b + 2] = q[b + 1] + qc;
                        k[b + 2] = k[b + 1] + kc;
                        uq[b + 1] = mode == 3 || (mode == 4 && b % 2) ? 0 : qc;
                        uk[b + 1] = mode == 3 || (mode == 4 && b % 2 == 0) ? 0 : kc;
                    }
                    const bool explicitUsed = mode != 5;
                    MlaUsedShape shape{
                        batch,
                        171,
                        {layout == 0 ? MlaSequenceLayout::Bsnd : MlaSequenceLayout::Tnd, q[batch + 1], qc},
                        {layout == 1 ? MlaSequenceLayout::Tnd : MlaSequenceLayout::Bsnd, k[batch + 1], kc}};
                    MlaUsedInputs source{{layout != 0 ? q.data() + 1 : nullptr, layout != 0 ? batch + 1 : 0},
                                         {layout == 1 ? k.data() + 1 : nullptr, layout == 1 ? batch + 1 : 0},
                                         {explicitUsed ? uq.data() + 1 : nullptr, explicitUsed ? batch : 0},
                                         {explicitUsed ? uk.data() + 1 : nullptr, explicitUsed ? batch : 0}};
                    const uint32_t words = 410 + 8 * batch;
                    std::vector<uint32_t> storage(words + 2, 0x12345678);
                    auto* data = storage.data() + 1;
                    assert(BuildUsedMetadata(shape, platform, dtype, data, words, source));
                    assert(data[5] == 0);
                    MlaUsedMetadataPrefix prefix;
                    std::memcpy(&prefix, data, sizeof(prefix));
                    MlaUsedMetadataContext c{dtype,
                                             batch,
                                             171,
                                             prefix.usedFlags,
                                             prefix.query,
                                             prefix.kv,
                                             20,
                                             40,
                                             data[21],
                                             uint64_t(data[385]) | (uint64_t(data[386]) << 32),
                                             0,
                                             platform.localMemory};
                    const auto desc = [](const MlaLengthArray& a) {
                        return MlaUsedMetadataArray{reinterpret_cast<uint64_t>(a.data), a.count};
                    };
                    MlaUsedMetadataInputs inputs{desc(source.cuQuery), desc(source.cuKv), desc(source.usedQuery),
                                                 desc(source.usedKv)};
                    MlaUsedKernelSchedule output;
                    std::memset(&output, 0xa5, sizeof(output));
                    const auto before = output;
                    uint32_t reads[5];
                    uint32_t header[8];
                    std::memcpy(header, data, sizeof(header));
                    for (uint32_t i = 0; i < 8; ++i) {
                        header[i] ^= 0xffffffffU;
                        assert(consume_used_metadata(header, words, &c, &inputs, &output, reads) != 0);
                        assert(reads[0] <= 8 && reads[1] == 0 && reads[2] == 0 && reads[3] == 0 && reads[4] == 0);
                        assert(std::memcmp(&output, &before, sizeof(output)) == 0);
                        header[i] ^= 0xffffffffU;
                    }
                    for (uint32_t size : {0U, 8U, words - 1, words + 1, UINT32_MAX}) {
                        assert(consume_used_metadata(header, size, &c, &inputs, &output, reads) == 1);
                        assert(reads[0] == 0);
                    }
                    assert(consume_used_metadata(data, words, &c, &inputs, &output, reads) == 0);
                    for (uint32_t i = 0; i < words; ++i) {
                        const auto saved = data[i];
                        for (uint32_t bad : {0U, 1U, 0x7fffffffU, 0xffffffffU}) {
                            data[i] = bad;
                            output = before;
                            if (consume_used_metadata(data, words, &c, &inputs, &output, reads) != 0)
                                assert(std::memcmp(&output, &before, sizeof(output)) == 0);
                        }
                        data[i] = saved;
                    }
                    MlaUsedMetadataArray* descriptors[] = {&inputs.cuQuery, &inputs.cuKv, &inputs.usedQuery,
                                                           &inputs.usedKv};
                    for (auto* a : descriptors) {
                        const auto saved = *a;
                        for (const MlaUsedMetadataArray bad :
                             {MlaUsedMetadataArray{0, 1}, {1, 0}, {4, UINT64_MAX}, {UINT64_MAX - 3, 2}}) {
                            *a = bad;
                            output = before;
                            assert(consume_used_metadata(data, words, &c, &inputs, &output, reads) != 0);
                            assert(reads[1] == 0 && reads[2] == 0 && reads[3] == 0 && reads[4] == 0);
                            assert(std::memcmp(&output, &before, sizeof(output)) == 0);
                        }
                        *a = saved;
                    }
                    for (auto* array : {&q, &k, &uq, &uk}) {
                        const uint32_t axis = array == &q ? 0 : array == &k ? 1 : array == &uq ? 2 : 3;
                        if (!descriptors[axis]->address)
                            continue;
                        for (uint32_t b = 1; b <= descriptors[axis]->count; ++b) {
                            const auto saved = (*array)[b];
                            (*array)[b] = INT32_MIN;
                            output = before;
                            assert(consume_used_metadata(data, words, &c, &inputs, &output, reads) != 0);
                            assert(std::memcmp(&output, &before, sizeof(output)) == 0);
                            (*array)[b] = saved;
                        }
                    }
                    assert(storage.front() == 0x12345678 && storage.back() == 0x12345678);
                    c.batch = UINT32_MAX;
                    assert(consume_used_metadata(header, UINT32_MAX, &c, &inputs, &output, reads) == 1);
                    assert(reads[0] == 0);
                }
            }
        }
    }
}
#endif
