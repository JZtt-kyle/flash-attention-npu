// CPU-only causal adapter driver. No torch, NPU runtime or production dispatch.
#include "mla_causal.h"
#include "mla_paged_schedule.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>

using namespace flash_attn::mla;

template <typename T>
void Emit(T value)
{
    std::cout << ',' << value;
}
template <typename T>
void EmitArray(const T& values)
{
    for (const auto value : values) {
        Emit(value);
    }
}

template <typename T>
void CheckFailure(const T& before, const T& after)
{
    if (std::memcmp(&before, &after, sizeof(T)) != 0) {
        std::abort();
    }
}

void Block()
{
    MlaCausalBlockInput in{};
    std::cin >> in.queryLength >> in.kvLength >> in.queryHeads >> in.mBaseSize >> in.s2BaseSize >> in.mBlock;
    MlaCausalBlock out;
    std::memset(&out, 0xa5, sizeof(out));
    const auto before = out;
    auto status = BuildCausalBlock(in, out);
    std::cout << '[' << int(status);
    if (status == MlaCausalStatus::Success) {
        Emit(out.nextTokenLeftUp);
        Emit(out.rowBegin);
        Emit(out.rowEnd);
        Emit(out.queryBegin);
        Emit(out.queryEnd);
        Emit(out.kvBegin);
        Emit(out.kvEnd);
        Emit(out.s2Begin);
        Emit(out.s2End);
        Emit(out.leadingEmptyRows);
        Emit(out.cost);
        Emit(out.normalS2Cost);
        Emit(out.lastS2Cost);
    } else {
        CheckFailure(before, out);
    }
    std::cout << "]\n";
}

std::vector<int32_t> Array(size_t count)
{
    std::vector<int32_t> values(count);
    for (auto& value : values) {
        std::cin >> value;
    }
    return values;
}

void Schedule(int mode)
{
    MlaCausalScheduleInput in{};
    size_t batch;
    std::cin >> in.queryHeads >> in.mBaseSize >> in.s2BaseSize >> in.aicNum >> in.cvRatio >> batch;
    std::vector<uint64_t> spans;
    if (mode == 1) {
        for (size_t b = 0; b < batch; ++b) {
            MlaCausalSequence seq{};
            std::cin >> seq.queryLength >> seq.kvLength;
            in.sequences.push_back(seq);
        }
    } else {
        int layout, page, hasQ, hasK;
        std::cin >> layout >> page;
        const auto q = Array(batch), k = Array(batch);
        std::cin >> hasQ;
        const auto uq = Array(hasQ ? batch : 0);
        std::cin >> hasK;
        const auto uk = Array(hasK ? batch : 0);
        std::vector<int32_t> cq{0}, ck{0};
        for (size_t b = 0; b < batch; ++b) {
            // Driver cases are small physical allocations; wide inputs use mode 1.
            cq.push_back(cq.back() + q[b]);
            ck.push_back(ck.back() + k[b]);
        }
        const auto array = [](const std::vector<int32_t>& v, bool present) {
            static const int32_t empty = 0;
            return MlaLengthArray{present ? (v.empty() ? &empty : v.data()) : nullptr, present ? v.size() : 0};
        };
        const auto qlayout = layout == 0 ? MlaSequenceLayout::Bsnd : MlaSequenceLayout::Tnd;
        const auto klayout = layout == 1 ? MlaSequenceLayout::Tnd : MlaSequenceLayout::Bsnd;
        const int64_t qmax = q.empty() ? 0 : *std::max_element(q.begin(), q.end());
        const int64_t kmax = k.empty() ? 0 : *std::max_element(k.begin(), k.end());
        const MlaSequenceAllocation qa{qlayout, cq.back(), qmax};
        if (page != 0) {
            MlaPagedLengths lengths;
            const auto status = ParsePagedLengths({int64_t(batch), in.queryHeads, qa, kmax, page},
                                                  {array(cq, layout != 0), array(uq, hasQ), array(uk, hasK)}, lengths);
            if (status != MlaScheduleStatus::Success) {
                std::cout << '[' << 100 + int(status) << "]\n";
                return;
            }
            for (const auto& seq : lengths.sequences) {
                in.sequences.push_back({seq.queryLength, seq.kvLength});
                spans.insert(spans.end(), {seq.queryStart, seq.queryCapacity, seq.queryLength, seq.kvLength});
            }
        } else {
            MlaUsedLengths lengths;
            const auto status = ParseUsedLengths(
                {int64_t(batch), in.queryHeads, qa, {klayout, ck.back(), kmax}},
                {array(cq, layout != 0), array(ck, layout == 1), array(uq, hasQ), array(uk, hasK)}, lengths);
            if (status != MlaScheduleStatus::Success) {
                std::cout << '[' << 100 + int(status) << "]\n";
                return;
            }
            for (const auto& seq : lengths.sequences) {
                in.sequences.push_back({seq.queryLength, seq.kvLength});
                spans.insert(spans.end(), {seq.queryStart, seq.kvStart, seq.queryCapacity, seq.kvCapacity,
                                           seq.queryLength, seq.kvLength});
            }
        }
    }
    MlaCausalSchedule out;
    std::memset(&out, 0xa5, sizeof(out));
    const auto before = out;
    auto status = BuildCausalSchedule(in, out);
    std::cout << '[' << int(status);
    if (status == MlaCausalStatus::Success) {
        Emit(out.usedCoreNum);
        Emit(out.maxCost);
        Emit(out.totalCost);
        Emit(out.totalMBlocks);
        Emit(out.totalS2Blocks);
        EmitArray(out.bN2End);
        EmitArray(out.gS1End);
        EmitArray(out.s2End);
        EmitArray(out.coreCost);
        EmitArray(spans);
    } else {
        CheckFailure(before, out);
    }
    std::cout << "]\n";
}

int main()
{
    int mode;
    while (std::cin >> mode) {
        if (mode == 0) {
            Block();
        } else {
            Schedule(mode);
        }
        std::cout << std::flush;
        if (!std::cin) {
            return 2;
        }
    }
}
