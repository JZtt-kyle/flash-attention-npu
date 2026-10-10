// Standalone CPU harness: no torch, CANN, or device runtime linkage.
#include <cstdint>
#include <iostream>
#include <type_traits>
#include <vector>

#include "mla_paged_layout.h"

using namespace flash_attn::mla;
static_assert(std::is_trivially_copyable_v<MlaPagedCacheLayout>);
static_assert(std::is_standard_layout_v<MlaPagedCopySegment>);

void ReadView(MlaBsndTensorView& view)
{
    for (auto& n : view.sizes) {
        std::cin >> n;
    }
    for (auto& n : view.strides) {
        std::cin >> n;
    }
    std::cin >> view.storageOffset >> view.storageBytes;
}

void PrintTensor(const MlaBsndTensorLayout& x)
{
    std::cout << ',' << x.batchStride << ',' << x.rowStride << ',' << x.headStride << ',' << x.spanElements;
}

int main()
{
    int operation;
    while (std::cin >> operation) {
        MlaPageTableView table{};
        MlaBsndTensorView k{}, v{};
        for (auto& n : table.sizes) {
            std::cin >> n;
        }
        for (auto& n : table.strides) {
            std::cin >> n;
        }
        std::cin >> table.storageOffset >> table.storageBytes;
        ReadView(k);
        ReadView(v);
        MlaPagedCacheLayout layout{17, 17, 17, 17, 17, 17, 17, {17, 17, 17, 17}, {17, 17, 17, 17}};
        const auto status = BuildPagedCacheLayout(table, k, v, layout);
        if (operation == 0) {
            std::cout << int(status) << ',' << layout.batchSize << ',' << layout.maxPages << ','
                      << layout.logicalCapacity << ',' << layout.physicalPages << ',' << layout.pageSize << ','
                      << layout.tableRowStride << ',' << layout.tableSpanElements;
            PrintTensor(layout.k);
            PrintTensor(layout.v);
        } else {
            if (status != MlaLayoutStatus::Success) {
                return 2;
            }
            uint64_t actualSize, available;
            bool hasTable;
            std::cin >> actualSize >> available >> hasTable;
            if (actualSize > 1000000 || available > actualSize) {
                return 2;
            }
            std::vector<int32_t> values(actualSize);
            for (auto& n : values) {
                std::cin >> n;
            }
            const int32_t* data = hasTable ? values.data() : nullptr;
            if (operation == 1) {
                bool hasLengths;
                uint64_t lengthCount;
                std::cin >> actualSize >> lengthCount >> hasLengths;
                if (actualSize > 1000000 || lengthCount > actualSize) {
                    return 2;
                }
                std::vector<int32_t> lengths(actualSize);
                for (auto& n : lengths) {
                    std::cin >> n;
                }
                std::cout << int(ValidatePagedCachePages(layout, data, available, hasLengths ? lengths.data() : nullptr,
                                                         lengthCount));
            } else if (operation == 2) {
                uint64_t batch, start, count;
                int64_t length;
                std::cin >> batch >> length >> start >> count;
                MlaPagedCopySegment segment{17, 17, 17, 17, 17, 17};
                const auto result = GetPagedCopySegment(layout, data, available, batch, length, start, count, segment);
                std::cout << int(result) << ',' << segment.tableOffset << ',' << segment.physicalPage << ','
                          << segment.pageRow << ',' << segment.rowCount << ',' << segment.kOffset << ','
                          << segment.vOffset;
            } else {
                return 2;
            }
        }
        if (!std::cin) {
            return 2;
        }
        std::cout << '\n';
    }
    return std::cin.eof() ? 0 : 2;
}
