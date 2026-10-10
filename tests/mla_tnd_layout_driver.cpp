// Standalone CPU harness: no torch, CANN, or device runtime linkage.
#include <cstdint>
#include <iostream>
#include <type_traits>

#include "mla_tnd_layout.h"

using namespace flash_attn::mla;
static_assert(std::is_trivially_copyable_v<MlaTndTensorLayout>);
static_assert(std::is_standard_layout_v<MlaTndTensorLayout>);
static_assert(sizeof(MlaTndTensorLayout) == 3 * sizeof(uint64_t));

int main()
{
    MlaTndTensorView view{};
    while (std::cin >> view.sizes[0]) {
        for (uint32_t axis = 1; axis < 3; ++axis) {
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
        MlaTndTensorLayout layout{11, 22, 33};
        const auto status = BuildTndTensorLayout(view, layout);
        std::cout << static_cast<int>(status) << ',' << layout.tokenStride << ',' << layout.headStride << ','
                  << layout.spanElements;
        for (uint32_t i = 0; i < coordinateCount; ++i) {
            int64_t coord[3];
            for (uint32_t axis = 0; axis < 3; ++axis) {
                if (!(std::cin >> coord[axis]) || coord[axis] < 0 || coord[axis] >= view.sizes[axis]) {
                    return 2;
                }
            }
            if (status != MlaLayoutStatus::Success) {
                return 2;
            }
            // Simulate device addressing from data_ptr(), not storage base.
            std::cout << ','
                      << uint64_t(coord[0]) * layout.tokenStride + uint64_t(coord[1]) * layout.headStride +
                             uint64_t(coord[2]);
        }
        std::cout << '\n';
    }
    return std::cin.eof() ? 0 : 2;
}
