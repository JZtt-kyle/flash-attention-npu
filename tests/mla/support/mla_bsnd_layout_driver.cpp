// Standalone CPU harness: no torch, CANN, or device runtime linkage.
#include <cstdint>
#include <iostream>
#include <type_traits>

#include "mla_bsnd_layout.h"

using namespace flash_attn::mla;
static_assert(std::is_trivially_copyable_v<MlaBsndTensorLayout>);
static_assert(std::is_standard_layout_v<MlaBsndTensorLayout>);
static_assert(sizeof(MlaBsndTensorLayout) == 4 * sizeof(uint64_t));

int main()
{
    MlaBsndTensorView view{};
    while (std::cin >> view.sizes[0]) {
        for (uint32_t axis = 1; axis < 4; ++axis) {
            std::cin >> view.sizes[axis];
        }
        for (auto& stride : view.strides) {
            std::cin >> stride;
        }
        uint32_t coordinateCount = 0;
        std::cin >> view.storageOffset >> view.storageBytes >> coordinateCount;
        if (!std::cin) {
            return 2;
        }
        MlaBsndTensorLayout layout{11, 22, 33, 44};
        const auto status = BuildBsndTensorLayout(view, layout);
        std::cout << static_cast<int>(status) << ',' << layout.batchStride << ',' << layout.rowStride << ','
                  << layout.headStride << ',' << layout.spanElements;
        for (uint32_t i = 0; i < coordinateCount; ++i) {
            int64_t coord[4];
            for (uint32_t axis = 0; axis < 4; ++axis) {
                if (!(std::cin >> coord[axis]) || coord[axis] < 0 || coord[axis] >= view.sizes[axis]) {
                    return 2;
                }
            }
            if (status != MlaLayoutStatus::Success) {
                return 2;
            }
            // Simulate addressing from data_ptr(), not from the storage base.
            std::cout << ','
                      << uint64_t(coord[0]) * layout.batchStride + uint64_t(coord[1]) * layout.rowStride +
                             uint64_t(coord[2]) * layout.headStride + uint64_t(coord[3]);
        }
        std::cout << '\n';
    }
    return std::cin.eof() ? 0 : 2;
}
