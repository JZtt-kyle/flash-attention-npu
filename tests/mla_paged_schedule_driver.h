// Included only by the independent CPU schedule driver with PAGED_KV.
#include "mla_paged_schedule.h"
#include <cstring>

void Run(const Input& in)
{
    using namespace flash_attn::mla;
    MlaPlatformInfo platform{};
    platform.aicNum = in.aic;
    platform.aivNum = in.aiv;
    platform.localMemory = {in.l1, in.ub, in.l0a, in.l0b, in.l0c};
    platform.libApiWorkspaceBytes = in.libapi;
    for (uint32_t i = 0; i < kMlaMaxAic; ++i) {
        platform.blockDims[i] = in.blockDimOverride ? in.blockDimOverride : i + 1;
    }
    const int32_t sentinel = 1234;
    const auto array = [&](bool present, uint64_t count, const std::vector<int32_t>& values) {
        return MlaLengthArray{present ? (values.empty() ? &sentinel : values.data()) : nullptr, count};
    };
    // Reject impossible harness inputs instead of letting a driver-owned
    // buffer overread masquerade as an API validation failure.
    for (const auto& pair : {std::make_pair(in.hasQuery && in.queryCount == uint64_t(in.batch) + 1,
                                            std::make_pair(in.queryCount, in.cuQuery.size())),
                             std::make_pair(in.hasUsedQ && in.usedQCount == uint64_t(in.batch),
                                            std::make_pair(in.usedQCount, in.usedQ.size())),
                             std::make_pair(in.hasUsedKv && in.usedKvCount == uint64_t(in.batch),
                                            std::make_pair(in.usedKvCount, in.usedKv.size()))}) {
        if (pair.first && pair.second.first > pair.second.second) {
            std::abort();
        }
    }
    const auto layout =
        in.layout == 0 ? MlaSequenceLayout::Bsnd : (in.layout == 2 ? MlaSequenceLayout::Tnd : MlaSequenceLayout(99));
    const MlaPagedShape shape{in.batch, in.heads, {layout, in.totalQuery, in.sq}, in.sk, in.pageSize};
    const MlaPagedInputs inputs{array(in.hasQuery, in.queryCount, in.cuQuery),
                                array(in.hasUsedQ, in.usedQCount, in.usedQ),
                                array(in.hasUsedKv, in.usedKvCount, in.usedKv)};
    MlaPagedSchedule result;
    result.pageSize = 17;
    result.logicalKvCapacity = 19;
    result.lengths = {101, 99, 77, {{11, 33, 55, 66}}};
    auto& s = result.schedule;
    std::memset(&s, 0xa5, sizeof(s));
    std::array<unsigned char, sizeof(s)> before;
    std::memcpy(before.data(), &s, sizeof(s));
    const bool parseOnly = std::getenv("MLA_TND_PARSE_ONLY") != nullptr;
    const auto status = parseOnly ? ParsePagedLengths(shape, inputs, result.lengths)
                                  : BuildPagedSchedule(shape, platform, inputs, result);
    if (status != MlaScheduleStatus::Success) {
        if (std::memcmp(before.data(), &s, sizeof(s)) || result.pageSize != 17 || result.logicalKvCapacity != 19 ||
            result.lengths.maxQueryCapacity != 101 || result.lengths.maxQueryLength != 99 ||
            result.lengths.maxKvLength != 77 || result.lengths.sequences.size() != 1 ||
            result.lengths.sequences[0].queryStart != 11 || result.lengths.sequences[0].queryCapacity != 33 ||
            result.lengths.sequences[0].queryLength != 55 || result.lengths.sequences[0].kvLength != 66) {
            std::abort();
        }
        std::cout << '[' << uint32_t(status) << "]\n";
        return;
    }
    const auto emitLengths = [&]() {
        Emit(result.lengths.maxQueryLength);
        Emit(result.lengths.maxKvLength);
        Emit(result.lengths.maxQueryCapacity);
        for (const auto& seq : result.lengths.sequences) {
            Emit(seq.queryStart);
            Emit(seq.queryCapacity);
            Emit(seq.queryLength);
            Emit(seq.kvLength);
        }
    };
    std::cout << "[0";
    if (!parseOnly) {
        const auto& w = s.workspace;
        for (uint64_t value :
             {uint64_t(s.mBaseSize), uint64_t(s.s2BaseSize), uint64_t(s.usedCoreNum), uint64_t(s.maxCost),
              uint64_t(s.fd.numOfFdHead), uint64_t(s.fd.maxS2SplitNum), uint64_t(s.fd.usedVecNumOfFd),
              uint64_t(s.fd.gS1BaseSizeOfFd), w.mm1ResSize, w.mm2ResSize, w.fdAccumOutSize, w.fdLogSumExpSize,
              w.userBytes, w.libApiBytes, w.totalBytes, uint64_t(s.blockDim), uint64_t(s.scheduleMode)}) {
            Emit(value);
        }
        EmitArray(s.bN2End, 26);
        EmitArray(s.gS1End, 26);
        EmitArray(s.s2End, 26);
        EmitArray(s.fd.bN2IdxOfFdHead, 26);
        EmitArray(s.fd.gS1IdxOfFdHead, 26);
        EmitArray(s.fd.s2SplitNumOfFdHead, 26);
        EmitArray(s.fd.s2SplitStartIdxOfCore, 26);
        EmitArray(s.fd.gS1SplitNumOfFdHead, 26);
        EmitArray(s.fd.gS1LastPartSizeOfFdHead, 26);
        EmitArray(s.fd.gS1IdxEndOfFdHead, 52);
        EmitArray(s.fd.gS1IdxEndOfFdHeadSplit, 52);
        // Check device segment accounting too; the pinned host exposes its
        // budget, while these are the migrated launch offsets inside it.
        const uint64_t slots = uint64_t(2) * s.blockDim;
        if (w.qkOffsetBytes != 0 || w.pOffsetBytes != slots * w.mm1ResSize * 4 ||
            w.pvOffsetBytes != w.pOffsetBytes + slots * w.mm1ResSize * 2 ||
            w.vec2OffsetBytes != w.pvOffsetBytes + slots * w.mm2ResSize * 4 ||
            w.kernelBytes != w.vec2OffsetBytes + slots * w.mm2ResSize * 4 || w.kernelBytes > w.userBytes) {
            std::abort();
        }
    }
    emitLengths();
    if (!parseOnly) {
        Emit(result.pageSize);
        Emit(result.logicalKvCapacity);
        Emit(s.s1Size);
        Emit(s.s2Size);
    }
    std::cout << "]\n";
}
