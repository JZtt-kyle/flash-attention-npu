// CPU-only protocol harness. No torch, gert, AICPU or production entry points.
#include "mla_split_kv.h"
#include <iostream>
using namespace flash_attn::mla;

template <typename T>
void Emit(const T& items)
{
    for (const auto& item : items) {
        std::cout << ',' << item;
    }
}
void EmitTensor(const MlaAccumulatorTensor& t)
{
    std::cout << ',' << t.rank;
    Emit(t.shape);
    Emit(t.strides);
    std::cout << ',' << t.elements << ',' << t.bytes;
}
int main()
{
    int mode;
    while (std::cin >> mode) {
        if (mode == 0 || mode == 1) {
            MlaWindowConfig w{};
            std::cin >> w.maxQueryLength >> w.maxKvLength >> w.windowLeft >> w.windowRight >> w.causal;
            if (mode == 0) {
                MlaCausalScheduleInput in{};
                uint32_t count;
                std::cin >> in.queryHeads >> in.mBaseSize >> in.s2BaseSize >> in.aicNum >> in.cvRatio >> count;
                for (uint32_t i = 0; i < count; ++i) {
                    MlaCausalSequence seq{};
                    std::cin >> seq.queryLength >> seq.kvLength;
                    in.sequences.push_back(seq);
                }
                MlaNativeSplitKvSchedule out{};
                out.usedCoreNum = 987;
                auto status = BuildNativeSplitKvSchedule(in, w, out);
                std::cout << '[' << int(status);
                if (status == MlaCausalStatus::Success) {
                    const auto& fd = out.fd;
                    std::cout << ',' << out.usedCoreNum << ',' << out.maxCost << ',' << fd.numOfFdHead << ','
                              << fd.maxS2SplitNum << ',' << fd.usedVecNumOfFd << ',' << out.totalCost << ','
                              << out.totalS2Blocks;
                    Emit(out.bN2End);
                    Emit(out.gS1End);
                    Emit(out.s2End);
                    Emit(fd.bN2IdxOfFdHead);
                    Emit(fd.gS1IdxOfFdHead);
                    Emit(fd.s2SplitNumOfFdHead);
                    Emit(fd.s2SplitStartIdxOfCore);
                    Emit(fd.gS1SplitNumOfFdHead);
                    Emit(fd.gS1LastPartSizeOfFdHead);
                    Emit(fd.gS1IdxEndOfFdHead);
                    Emit(fd.gS1IdxEndOfFdHeadSplit);
                    std::cout << ',' << out.totalMBlocks << ',' << fd.gS1BaseSizeOfFd;
                } else if (out.usedCoreNum != 987) {
                    return 3;
                }
            } else {
                MlaCausalBlockInput in{};
                int64_t count, index;
                std::cin >> in.queryLength >> in.kvLength >> in.queryHeads >> in.mBaseSize >> in.s2BaseSize >>
                    in.mBlock >> count >> index;
                MlaSplitKvBlock out{};
                out.cost = -987;
                auto status = BuildSplitKvBlock(in, w, count, index, out);
                std::cout << '[' << int(status);
                if (status == MlaCausalStatus::Success) {
                    std::cout << ',' << out.s2Begin << ',' << out.s2End << ',' << out.kvBegin << ',' << out.kvEnd << ','
                              << out.cost << ',' << out.whole.s2Begin << ',' << out.whole.s2End << ','
                              << out.whole.kvBegin << ',' << out.whole.kvEnd << ',' << out.whole.cost;
                } else if (out.cost != -987) {
                    return 3;
                }
            }
        } else if (mode == 2) {
            int64_t requested, automatic;
            std::cin >> requested >> automatic;
            uint32_t out = 987;
            auto status = ResolveSplitKvRequest(requested, automatic, out);
            std::cout << '[' << int(status) << ',' << out;
        } else if (mode == 3) {
            MlaSplitKvWorkspaceInput in{};
            std::cin >> in.mBaseSize >> in.s2BaseSize >> in.maxQueryRows >> in.aicNum >> in.blockDim >>
                in.flashDecode >> in.libApiBytes;
            MlaSplitKvWorkspace out{};
            out.attention.totalBytes = 987;
            auto status = BuildSplitKvWorkspace(in, out);
            std::cout << '[' << int(status);
            if (status == MlaScheduleStatus::Success) {
                const auto& w = out.attention;
                std::cout << ',' << w.mm1ResSize << ',' << w.mm2ResSize << ',' << w.fdAccumOutSize << ','
                          << w.fdLogSumExpSize << ',' << w.qkOffsetBytes << ',' << w.pOffsetBytes << ','
                          << w.pvOffsetBytes << ',' << w.vec2OffsetBytes << ',' << out.fdAccumOffsetBytes << ','
                          << out.fdSumOffsetBytes << ',' << out.fdMaxOffsetBytes << ',' << w.kernelBytes << ','
                          << w.userBytes << ',' << w.libApiBytes << ',' << w.totalBytes;
            } else if (out.attention.totalBytes != 987) {
                return 3;
            }
        } else if (mode == 4 || mode == 5) {
            MlaAccumulatorInput in{};
            int layout;
            std::cin >> layout >> in.batch >> in.queryCapacity >> in.totalQuery >> in.queryHeads >> in.numSplits;
            in.layout = MlaAccumulatorQueryLayout(layout);
            if (mode == 4) {
                MlaAccumulatorLayout out{};
                out.out.bytes = 987;
                auto status = BuildSplitAccumulatorLayout(in, out);
                std::cout << '[' << int(status);
                if (status == MlaScheduleStatus::Success) {
                    EmitTensor(out.out);
                    EmitTensor(out.lse);
                } else if (out.out.bytes != 987) {
                    return 3;
                }
            } else {
                int64_t split, batch, head, query;
                std::cin >> split >> batch >> head >> query;
                uint64_t o = 987, l = 987;
                auto status = GetSplitAccumulatorOffsets(in, split, batch, head, query, o, l);
                std::cout << '[' << int(status) << ',' << o << ',' << l;
            }
        } else {
            return 2;
        }
        std::cout << "]\n" << std::flush;
        if (!std::cin) {
            return 2;
        }
    }
}
