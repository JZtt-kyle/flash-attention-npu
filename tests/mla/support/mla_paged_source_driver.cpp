// CPU trace adapter for verbatim pinned CANN offset/parser/copy helpers.
// No migrated header or object is included. Device copies are recorded, not
// executed; this verifies addresses/segments, not device instructions or timing.
#include <cstdint>
#include <iostream>
#include <tuple>
#include <type_traits>
#include <vector>

#define __aicore__
#define __gm__
enum class GmFormat {
    BSND,
    BNSD,
    NTD,
    TND,
    PA_BnBsND,
    PA_BnNBsD,
    PA_NZ
};
enum class L1Format {
    NZ
};
enum class FormatCategory {
    GM_KV_PA_BNBD
};
struct int4b_t {};
template <typename A, typename B>
using IsSameType = std::is_same<A, B>;

struct Address {
    uint64_t tag, offset;
    Address operator[](uint64_t n) const
    {
        return {tag, offset + n};
    }
};

uint64_t pageReads = 0;
namespace AscendC {
namespace Std = std;
template <typename T>
struct GlobalTensor {
    const T* values = nullptr;
    void SetGlobalBuffer(const T* data)
    {
        values = data;
    }
    T GetValue(uint64_t n) const
    {
        ++pageReads;
        return values[n];
    }
};
} // namespace AscendC
using AscendC::GlobalTensor;

template <GmFormat FORMAT>
struct GmLayout {
    std::tuple<uint64_t, uint64_t, uint64_t> shape;
    std::tuple<uint64_t, uint64_t, uint64_t, uint64_t> stride;
    void MakeLayout(uint32_t heads, uint32_t page, uint32_t dim, uint64_t pageStride, uint64_t headStride)
    {
        shape = {heads, page, dim};
        stride = {pageStride, headStride, dim, 1};
    }
};
template <GmFormat FORMAT, FormatCategory CATEGORY, typename ACTLEN_T>
struct OffsetCalculatorImpl;

template <typename T, L1Format FORMAT>
struct FaL1Tensor {
    Address tensor;
    uint64_t rowCount;
};
struct GmKvCoord {
    uint32_t bIdx, n2Idx, s2Idx, dIdx, s2DealSize, dDealSize;
};
struct DataCopyParams {
    uint32_t blockCount, blockLen, dstStride, srcStride;
};
void DataCopy(Address, Address, DataCopyParams); // Uninstantiated NZ-only branch.

struct Trace {
    Address dst, src;
    uint64_t rows, cols, stride, dstRows;
};
std::vector<Trace> traces;
void CopySingleMatrixNDToNZ(Address dst, Address src, uint64_t rows, uint64_t cols, uint64_t stride, uint64_t dstRows)
{
    traces.push_back({dst, src, rows, cols, stride, dstRows});
}

// Generated from git show of the fixed revision, with copyright retained.
#include "source_helpers.inc"

template <GmFormat FORMAT>
struct SourceTensor {
    OffsetCalculatorImpl<FORMAT, FormatCategory::GM_KV_PA_BNBD, int64_t> offsetCalculator;
    Address gmTensor;
};

int main()
{
    uint32_t page, tableStride, batch, start, count, kd, vd, kw, vw, tableCount;
    uint64_t kp, kr, vp, vr;
    while (std::cin >> page >> tableStride >> kp >> kr >> vp >> vr >> batch >> start >> count >> kd >> vd >> kw >> vw >>
           tableCount) {
        std::vector<int32_t> table(tableCount);
        for (auto& x : table) {
            std::cin >> x;
        }
        if (!std::cin) {
            return 2;
        }
        SourceTensor<GmFormat::PA_BnBsND> rope{}, content{};
        rope.gmTensor = {0, 0};
        content.gmTensor = {1, 0};
        for (auto* x : {&rope, &content}) {
            x->offsetCalculator.Init(1, page, x == &rope ? 64 : 512, {table.data()}, tableStride);
        }
        // Supply validated FA3 strides directly to the source offset getters.
        // This permits testing the new outer-stride adapter as well as native
        // ND contiguous rows; it does not claim native FIA accepts every view.
        rope.offsetCalculator.gmLayout.stride = {kp, 0, kr, 1};
        content.offsetCalculator.gmLayout.stride = {vp, 0, vr, 1};
        FaL1Tensor<uint16_t, L1Format::NZ> dstK{{1, 0}, 512}, dstRope{{0, 0}, 512};
        GmKvCoord kc{batch, 0, start, kd, count, kw}, vc{batch, 0, start, vd, count, vw};
        traces.clear();
        pageReads = 0;
        CopyKKropePAGmToL1<uint16_t, GmFormat::PA_BnBsND>{}(dstK, dstRope, content, rope, vc, kc);
        const auto qkReads = pageReads;
        CopyKvGmToL1<uint16_t, GmFormat::PA_BnBsND>{}(dstK, content, vc);
        std::cout << qkReads << ',' << pageReads - qkReads << ',' << traces.size();
        for (const auto& t : traces) {
            std::cout << ',' << t.dst.tag << ',' << t.dst.offset << ',' << t.src.tag << ',' << t.src.offset << ','
                      << t.rows << ',' << t.cols << ',' << t.stride << ',' << t.dstRows;
        }
        std::cout << '\n';
    }
    return std::cin.eof() ? 0 : 2;
}
