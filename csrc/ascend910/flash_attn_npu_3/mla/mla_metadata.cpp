// Serialization only. The source-derived algorithm and its copyright/license
// remain in mla_schedule.cpp and detail/cann_split_core.cpp (LICENSE.cann).
#include "mla_metadata.h"
#include <algorithm>
#include <limits>

namespace flash_attn::mla {
namespace {
MlaMetadataStatus CheckCacheLengths(const MlaDenseShape& shape, const int32_t* lengths, uint64_t count)
{
    if (lengths == nullptr && count == 0) {
        return MlaMetadataStatus::Success;
    }
    const auto address = reinterpret_cast<uintptr_t>(lengths);
    if (lengths == nullptr || address % alignof(int32_t) != 0 || count != uint64_t(shape.batch) ||
        count > (std::numeric_limits<uintptr_t>::max() - address) / sizeof(int32_t)) {
        return MlaMetadataStatus::InvalidCacheLengths;
    }
    for (uint64_t b = 0; b < count; ++b) {
        if (lengths[b] != shape.kvLength) {
            return MlaMetadataStatus::UnsupportedCacheLength;
        }
    }
    return MlaMetadataStatus::Success;
}

MlaMetadataWide Wide(uint64_t value)
{
    return {static_cast<uint32_t>(value), static_cast<uint32_t>(value >> 32)};
}

MlaMetadataSchedule Serialize(const MlaSchedule& s)
{
    MlaMetadataSchedule out{};
    out.bSize = s.bSize;
    out.n2Size = s.n2Size;
    out.gSize = s.gSize;
    out.s1Size = s.s1Size;
    out.s2Size = s.s2Size;
    out.headDim = s.headDim;
    out.headDimRope = s.headDimRope;
    out.mBaseSize = s.mBaseSize;
    out.s2BaseSize = s.s2BaseSize;
    out.aicNum = s.aicNum;
    out.aivNum = s.aivNum;
    out.cvRatio = s.cvRatio;
    out.usedCoreNum = s.usedCoreNum;
    out.blockDim = s.blockDim;
    out.scheduleMode = s.scheduleMode;
    std::copy(s.bN2End.begin(), s.bN2End.end(), out.bN2End);
    std::copy(s.gS1End.begin(), s.gS1End.end(), out.gS1End);
    std::copy(s.s2End.begin(), s.s2End.end(), out.s2End);
    out.maxCost = Wide(static_cast<uint64_t>(s.maxCost));
    out.numOfFdHead = s.fd.numOfFdHead;
    out.maxS2SplitNum = s.fd.maxS2SplitNum;
    out.usedVecNumOfFd = s.fd.usedVecNumOfFd;
    out.gS1BaseSizeOfFd = s.fd.gS1BaseSizeOfFd;
    std::copy(s.fd.bN2IdxOfFdHead.begin(), s.fd.bN2IdxOfFdHead.end(), out.bN2IdxOfFdHead);
    std::copy(s.fd.gS1IdxOfFdHead.begin(), s.fd.gS1IdxOfFdHead.end(), out.gS1IdxOfFdHead);
    std::copy(s.fd.s2SplitNumOfFdHead.begin(), s.fd.s2SplitNumOfFdHead.end(), out.s2SplitNumOfFdHead);
    std::copy(s.fd.s2SplitStartIdxOfCore.begin(), s.fd.s2SplitStartIdxOfCore.end(), out.s2SplitStartIdxOfCore);
    std::copy(s.fd.gS1SplitNumOfFdHead.begin(), s.fd.gS1SplitNumOfFdHead.end(), out.gS1SplitNumOfFdHead);
    std::copy(s.fd.gS1LastPartSizeOfFdHead.begin(), s.fd.gS1LastPartSizeOfFdHead.end(), out.gS1LastPartSizeOfFdHead);
    std::copy(s.fd.gS1IdxEndOfFdHead.begin(), s.fd.gS1IdxEndOfFdHead.end(), out.gS1IdxEndOfFdHead);
    std::copy(s.fd.gS1IdxEndOfFdHeadSplit.begin(), s.fd.gS1IdxEndOfFdHeadSplit.end(), out.gS1IdxEndOfFdHeadSplit);
    const auto& w = s.workspace;
    out.workspace = {Wide(w.mm1ResSize),    Wide(w.mm2ResSize),   Wide(w.fdAccumOutSize), Wide(w.fdLogSumExpSize),
                     Wide(w.qkOffsetBytes), Wide(w.pOffsetBytes), Wide(w.pvOffsetBytes),  Wide(w.vec2OffsetBytes),
                     Wide(w.kernelBytes),   Wide(w.userBytes),    Wide(w.libApiBytes),    Wide(w.totalBytes)};
    const auto& m = s.requiredLocalMemory;
    out.requiredLocalMemory = {Wide(m.l1Bytes), Wide(m.ubBytes), Wide(m.l0aBytes), Wide(m.l0bBytes), Wide(m.l0cBytes)};
    return out;
}
} // namespace

void BuildDenseMetadata(const MlaDenseShape& shape, const MlaPlatformInfo& platform, uint32_t dtype,
                        MlaDenseMetadata& output, const int32_t* cacheSeqlens, uint64_t cacheSeqlensCount)
{
    MlaDenseMetadata metadata{};
    metadata.magic = kMlaMetadataMagic;
    metadata.version = kMlaMetadataVersion;
    metadata.kind = kMlaMetadataDenseSharedKV;
    metadata.wordCount = kMlaDenseMetadataWords;
    metadata.dtype = dtype;
    if (dtype != static_cast<uint32_t>(MlaMetadataDtype::Fp16) &&
        dtype != static_cast<uint32_t>(MlaMetadataDtype::Bf16)) {
        metadata.status = static_cast<uint32_t>(MlaMetadataStatus::UnsupportedDtype);
    } else {
        MlaSchedule schedule{};
        auto status = BuildDenseSchedule(shape, platform, schedule);
        metadata.scheduleStatus = static_cast<uint32_t>(status);
        if (status == MlaScheduleStatus::Success) {
            // Validate static scheduling bounds before dereferencing lengths.
            // Failure leaves the entire payload zero, including after reuse.
            metadata.status = static_cast<uint32_t>(CheckCacheLengths(shape, cacheSeqlens, cacheSeqlensCount));
            if (metadata.status == static_cast<uint32_t>(MlaMetadataStatus::Success)) {
                metadata.schedule = Serialize(schedule);
            }
        } else {
            metadata.status = static_cast<uint32_t>(MlaMetadataStatus::ScheduleFailure);
        }
    }
    output = metadata;
}
} // namespace flash_attn::mla
