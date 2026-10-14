// Independent source-only oracle: generated snippets are byte-for-byte from
// pinned CANN/FA3 sources. Never includes/links the migrated adapter.
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

struct Tensor {
    std::vector<int64_t> shape;
    int64_t stride(int axis) const
    {
        int64_t stride = 1;
        for (size_t i = axis + 1; i < shape.size(); ++i) {
            stride *= std::max(int64_t(1), shape[i]);
        }
        return stride;
    }
};
namespace at {
using Tensor = ::Tensor;
enum class ScalarType {
    Float
};
constexpr ScalarType kFloat = ScalarType::Float;
} // namespace at
struct Options {
    Options dtype(at::ScalarType) const
    {
        return {};
    }
};
namespace torch {
Tensor empty(std::initializer_list<int64_t> shape, Options)
{
    return {shape};
}
} // namespace torch
struct Params {
    int num_splits;
    int64_t oaccum_batch_stride, lseaccum_batch_stride;
};
int automatic;
int get_num_splits(const Params&)
{
    return automatic;
}
#define TORCH_CHECK(condition, ...)                                                                                    \
    if (!(condition)) {                                                                                                \
        throw std::runtime_error("invalid");                                                                           \
    }

// Minimal state consumed by verbatim CANN workspace methods.
constexpr uint64_t PRE_LOAD_NUM_MLA = 2;
constexpr uint32_t BYTE_BLOCK = 32;
struct FiaInfo {
    uint32_t gSize = 1, s1Size = 0, n2Size = 1;
};
struct WorkspaceParams {
    uint64_t mm1ResSize, mm2ResSize, fdAccumOutSize, fdLogSumExpSize;
    void set_mm1ResSize(uint64_t v)
    {
        mm1ResSize = v;
    }
    void set_mm2ResSize(uint64_t v)
    {
        mm2ResSize = v;
    }
    void set_fdAccumOutSize(uint64_t v)
    {
        fdAccumOutSize = v;
    }
    void set_fdLogSumExpSize(uint64_t v)
    {
        fdLogSumExpSize = v;
    }
};
struct FiaTilingNonQuantMla {
    FiaInfo* fiaInfo_;
    uint32_t mBaseSize_, sInnerSizeAlign_, headDimAlign_ = 512, aicNum_, coreNum_;
    int64_t mm1ResSize_, mm2ResSize_;
    uint64_t libapiSize_, workspaceSize_;
    bool splitKVFlag_;
    struct {
        WorkspaceParams workspaceParams;
    } tilingData_;
    void CalcMmResSize();
    void FillTilingWorkspaceParams();
    uint32_t CalcFlashDecodeParamNums(uint32_t) const;
    uint64_t CalcNormalWorkspaceSize(uint32_t, int64_t, int64_t, uint32_t) const;
    uint64_t CalcFlashDecodeWorkspace(uint32_t) const;
    void CalcWorkspaceSize();
};
#include "workspace_functions.inc"
#include "fa3_split_functions.inc"

int main()
{
    int mode;
    while (std::cin >> mode) {
        if (mode == 1) {
            int start, end, count, index;
            std::cin >> start >> end >> count >> index;
            auto range = ExplicitRange(start, end, count, index);
            std::cout << '[' << range.first << ',' << range.second;
        } else if (mode == 2) {
            int64_t requested;
            std::cin >> requested >> automatic;
            try {
                const auto value = Request(requested);
                std::cout << "[0," << value;
            } catch (const std::runtime_error&) {
                std::cout << "[1";
            }
        } else if (mode == 3) {
            uint32_t m, n, cores, blockDim;
            int64_t rows;
            bool fd;
            uint64_t lib;
            std::cin >> m >> n >> rows >> cores >> blockDim >> fd >> lib;
            FiaInfo info{};
            info.s1Size = rows;
            FiaTilingNonQuantMla tiler{};
            tiler.fiaInfo_ = &info;
            tiler.mBaseSize_ = m;
            tiler.sInnerSizeAlign_ = (n + 31) / 32 * 32;
            tiler.aicNum_ = tiler.coreNum_ = cores;
            tiler.libapiSize_ = lib;
            tiler.splitKVFlag_ = fd;
            tiler.CalcMmResSize();
            tiler.FillTilingWorkspaceParams();
            tiler.CalcWorkspaceSize();
            const auto& w = tiler.tilingData_.workspaceParams;
            std::cout << '[' << w.mm1ResSize << ',' << w.mm2ResSize << ',' << w.fdAccumOutSize << ','
                      << w.fdLogSumExpSize << ',' << tiler.workspaceSize_;
        } else if (mode == 4) {
            int layout;
            int64_t b, q, t, h, s;
            std::cin >> layout >> b >> q >> t >> h >> s;
            auto tensors = Accumulators(layout, b, q, t, h, s);
            std::cout << '[';
            bool comma = false;
            for (const auto& tensor : tensors) {
                if (comma) {
                    std::cout << ',';
                }
                comma = true;
                std::cout << tensor.shape.size();
                for (auto dim : tensor.shape) {
                    std::cout << ',' << dim;
                }
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
