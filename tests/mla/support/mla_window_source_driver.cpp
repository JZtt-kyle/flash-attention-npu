// Independent fixed-source CPU oracle. This executable never links the adapter
// or a migrated length parser. Python supplies independently decoded lengths.
// MIGRATED_SPLIT builds a second executable to compare every native FD field.
#ifdef MIGRATED_SPLIT
#include "detail/cann_split_core.h"
namespace source = flash_attn::mla::detail;
#else
#include "split_core.h"
namespace source = optiling;
#endif
#include <iostream>

using namespace source;

template <typename T>
void Emit(const T& values)
{
    for (const auto value : values) {
        std::cout << ',' << value;
    }
}

int main()
{
    int mode, mask;
    int64_t left, right;
    uint32_t heads, m, n, cores, ratio, count;
    while (std::cin >> mode >> mask >> left >> right >> heads >> m >> n >> cores >> ratio >> count) {
        BaseInfo base{};
        base.bSize = count;
        base.n2Size = 1;
        base.gSize = heads;
        base.attenMaskFlag = mask != 0;
        base.sparseMode = mask;
        base.preToken = left;
        base.nextToken = right;
        base.actualLenQDims = base.actualLenKvDims = count;
        for (uint32_t b = 0; b < count; ++b) {
            int64_t q, k;
            std::cin >> q >> k;
            base.actualSeqS1Size.push_back(q);
            base.actualSeqS2Size.push_back(k);
        }
        SplitParam param{};
        param.mBaseSize = m;
        param.s2BaseSize = n;
        SplitContext context(base, param);
        CalcSplitInfo(context);
        if (mode == 0) {
            uint32_t index;
            std::cin >> index;
            BatchCache batch{};
            CalcBatchCache(0, context, batch);
            S1GCache block{};
            CalcS1GCache(index, context, batch, block);
            std::cout << '[' << batch.preTokenLeftUp << ',' << batch.nextTokenLeftUp << ',' << block.s2Start << ','
                      << block.s2End << ',' << block.s1GCost << ',' << block.s1GNormalBlockCost << ','
                      << block.s1GLastBlockCost << "]\n";
        } else {
            CalcCostInfo(context);
            SplitResult split{cores, ratio};
            SplitCore(cores, base, param, split);
            std::cout << '[' << split.usedCoreNum << ',' << split.maxCost << ',' << split.numOfFdHead << ','
                      << split.maxS2SplitNum << ',' << split.usedVecNumOfFd << ',' << context.costInfo.totalCost << ','
                      << context.costInfo.totalBlockNum;
            Emit(split.bN2End);
            Emit(split.gS1End);
            Emit(split.s2End);
            Emit(split.fdRes.bN2IdxOfFdHead);
            Emit(split.fdRes.gS1IdxOfFdHead);
            Emit(split.fdRes.s2SplitNumOfFdHead);
            Emit(split.fdRes.s2SplitStartIdxOfCore);
            Emit(split.fdRes.gS1SplitNumOfFdHead);
            Emit(split.fdRes.gS1LastPartSizeOfFdHead);
            Emit(split.fdRes.gS1IdxEndOfFdHead);
            Emit(split.fdRes.gS1IdxEndOfFdHeadSplit);
            std::cout << "]\n";
        }
        std::cout << std::flush;
        if (!std::cin) {
            return 2;
        }
    }
}
