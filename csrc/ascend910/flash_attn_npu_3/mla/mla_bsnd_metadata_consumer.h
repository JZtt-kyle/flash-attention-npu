// v2 BSND length snapshot validation. Schedule/resource checks and their CANN
// provenance are shared with mla_metadata_consumer.h. No attention launch here.
#pragma once
#include "mla_bsnd_metadata.h"
#include "mla_metadata_consumer.h"

#if defined(__NPU_ARCH__)
#define MLA_BSND_INLINE __aicore__ inline
#else
#define MLA_BSND_INLINE inline
#endif

namespace flash_attn::mla {

template <typename Reader, typename LengthReader>
struct BsndMetadataLengths {
    const Reader& record;
    const LengthReader& current;
    bool explicitLengths;

    MLA_BSND_INLINE bool Check(uint32_t batch, uint32_t capacity) const
    {
        for (uint32_t b = 0; b < batch; ++b) {
            const uint32_t snapshot = record(kMlaBsndKvLengthMetadataPrefixWords + b);
            // Unsigned comparison also rejects negative int32 lengths.
            if (snapshot > capacity || snapshot != (explicitLengths ? current(b) : capacity)) {
                return false;
            }
        }
        return true;
    }

    MLA_BSND_INLINE bool Covers(uint64_t end, uint64_t tasks, uint32_t mBlocks, uint32_t batch) const
    {
        if (end == tasks) {
            return true;
        }
        // The pinned source may omit a zero-KV suffix. It must start at a
        // sample boundary and every omitted sample must be empty; checking
        // just the last sample would allow a hole with nonzero KV work.
        if (end % mBlocks != 0) {
            return false;
        }
        for (uint32_t b = end / mBlocks; b < batch; ++b) {
            if (record(kMlaBsndKvLengthMetadataPrefixWords + b) != 0) {
                return false;
            }
        }
        return true;
    }
};

// Both readers and the static context belong to this call. The caller validates
// their addresses/alignment and keeps buffers immutable until all cores finish.
// Missing current lengths means full capacity, not "trust the snapshot".
// Success returns geometry only: a short-length schedule cannot drive MlaDense.
template <typename Reader, typename LengthReader>
MLA_BSND_INLINE MlaMetadataReadStatus ReadBsndKvLengthMetadata(const Reader& read, uint32_t words,
                                                               const MlaDenseMetadataContext& context,
                                                               const LengthReader& current, bool explicitLengths,
                                                               uint64_t lengthCount, MlaDenseKernelSchedule& output)
{
    using Status = MlaMetadataReadStatus;
    if (context.batch > UINT32_MAX - kMlaBsndKvLengthMetadataPrefixWords) {
        return Status::InvalidBuffer;
    }
    const uint32_t expected = kMlaBsndKvLengthMetadataPrefixWords + context.batch;
    const auto status = ReadMetadataHeader(read, words, expected, kMlaBsndKvLengthMetadataVersion,
                                           kMlaMetadataBsndKvLengths, context.dtype);
    if (status != Status::Success) {
        return status;
    }
    if (lengthCount != (explicitLengths ? uint64_t(context.batch) : 0)) {
        return Status::InvalidCacheLengths;
    }
    return ReadMetadataSchedule(read, context,
                                BsndMetadataLengths<Reader, LengthReader>{read, current, explicitLengths}, output);
}
} // namespace flash_attn::mla
#undef MLA_BSND_INLINE
