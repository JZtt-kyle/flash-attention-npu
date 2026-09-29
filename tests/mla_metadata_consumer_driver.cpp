// CPU boundary/ASan/UBSan bridge for the same decoder instantiated on device.
#include "mla_metadata_consumer.h"
#include "mla_metadata_args.h"
#include <cassert>
#include <cstring>
#include <limits>

using namespace flash_attn::mla;

struct Reader {
    const void* data;
    uint32_t* reads;
    uint32_t operator()(uint32_t i) const
    {
        assert(i < kMlaDenseMetadataWords);
        if (reads && *reads < i + 1)
            *reads = i + 1;
        uint32_t word;
        std::memcpy(&word, static_cast<const char*>(data) + i * sizeof(word), sizeof(word));
        return word;
    }
};

extern "C" uint32_t consume_metadata(const void* data, uint32_t words, const MlaDenseMetadataContext* context,
                                     MlaDenseKernelSchedule* schedule, uint32_t* reads)
{
    *reads = 0;
    if (!data || reinterpret_cast<uintptr_t>(data) % sizeof(uint32_t) != 0) {
        return uint32_t(MlaMetadataReadStatus::InvalidBuffer);
    }
    return uint32_t(ReadDenseMetadata(Reader{data, reads}, words, *context, *schedule));
}

extern "C" void build_metadata(const MlaMetadataArgs* args, MlaDenseMetadata* output, MlaDenseMetadataContext* c)
{
    BuildDenseMetadata(args->shape, args->platform, args->dtype, *output);
    const auto& s = output->schedule;
    auto wide = [](MlaMetadataWide v) { return uint64_t(v.low) | (uint64_t(v.high) << 32); };
    *c = {args->dtype,
          s.bSize,
          s.s1Size,
          s.gSize,
          s.s2Size,
          s.aicNum,
          s.aivNum,
          s.blockDim,
          wide(s.workspace.userBytes),
          wide(s.workspace.libApiBytes),
          args->platform.localMemory};
}

#ifdef MLA_CONSUMER_SANITIZE
int main()
{
    MlaMetadataArgs args{};
    args.dtype = 1;
    args.shape = {2, 1, 8, 257};
    args.platform.aicNum = 20;
    args.platform.aivNum = 40;
    args.platform.localMemory = {516096, 174080, 65536, 65536, 131072};
    for (uint32_t i = 0; i < 20; ++i)
        args.platform.blockDims[i] = i + 1;
    MlaDenseMetadata metadata;
    MlaDenseMetadataContext context;
    build_metadata(&args, &metadata, &context);
    MlaDenseKernelSchedule output;
    std::memset(&output, 0xa5, sizeof(output));
    auto before = output;
    uint32_t reads;
    // Physically only an 8-word header: ASan proves no payload access on failure.
    uint32_t header[8];
    std::memcpy(header, &metadata, sizeof(header));
    for (uint32_t i = 0; i < 8; ++i) {
        auto saved = header[i];
        header[i] ^= 0xffffffffU;
        assert(consume_metadata(header, kMlaDenseMetadataWords, &context, &output, &reads) != 0);
        assert(reads <= 8 && std::memcmp(&before, &output, sizeof(output)) == 0);
        header[i] = saved;
    }
    for (uint32_t words : {0U, 8U, kMlaDenseMetadataWords - 1, kMlaDenseMetadataWords + 1, UINT32_MAX}) {
        assert(consume_metadata(header, words, &context, &output, &reads) == 1 && reads == 0);
    }
    assert(consume_metadata(nullptr, kMlaDenseMetadataWords, &context, &output, &reads) == 1);
    alignas(8) uint32_t storage[kMlaDenseMetadataWords + 2];
    storage[0] = storage[kMlaDenseMetadataWords + 1] = 0x12345678;
    std::memcpy(storage + 1, &metadata, sizeof(metadata));
    for (uint32_t i = 0; i < kMlaDenseMetadataWords; ++i) {
        const auto saved = storage[i + 1];
        for (uint32_t bad : {0U, 1U, 0x7fffffffU, 0xffffffffU}) {
            storage[i + 1] = bad;
            output = before;
            if (consume_metadata(storage + 1, kMlaDenseMetadataWords, &context, &output, &reads) != 0)
                assert(std::memcmp(&before, &output, sizeof(output)) == 0);
        }
        storage[i + 1] = saved;
    }
    assert(storage[0] == 0x12345678 && storage[kMlaDenseMetadataWords + 1] == 0x12345678);
    assert(consume_metadata(storage + 1, kMlaDenseMetadataWords, &context, &output, &reads) == 0);
    assert(output.batch == 2 && output.sk == 257 && output.usedCoreNum == 2);
    context.libApiBytes = UINT64_MAX;
    assert(consume_metadata(storage + 1, kMlaDenseMetadataWords, &context, &output, &reads) != 0);
}
#endif
