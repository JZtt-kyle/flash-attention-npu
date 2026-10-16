// Fixed CANN ca18aee CalcInnerSize/CalcMBaseSize body is included verbatim.
// Copyright (c) 2025 Huawei Technologies Co., Ltd.
// CANN Open Software License Agreement Version 2.0; see LICENSE.cann.
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <cstring>
#define OP_LOGI(...) ((void)0)
constexpr uint32_t S1_SIZE_16 = 16, MAX_SPLIT_SIZE = 8192, BYTE_BLOCK = 32;
constexpr uint32_t S_INNER_SIZE_512 = 512, S_INNER_SIZE_1024 = 1024;
constexpr uint32_t S_INNER_SIZE_ALIGN_512 = 512, S_INNER_SIZE_ALIGN_1024 = 1024, S_INNER_SIZE_ALIGN_2048 = 2048,
                   S_INNER_SIZE_ALIGN_4096 = 4096;
constexpr uint32_t M_BASE_SIZE_512 = 512, M_BASE_SIZE_256 = 256, M_BASE_SIZE_128 = 128, M_BASE_SIZE_64 = 64,
                   M_BASE_SIZE_32 = 32, SPARSE_MODE_2 = 2, SPARSE_MODE_3 = 3, SPARSE_MODE_4 = 4;
enum class TilingKeyLayout {
    BSH_BSND,
    TND,
    NTD
};
template <class T>
T Align(T v, T a)
{
    return (v + a - 1) / a * a;
}
struct Info {
    TilingKeyLayout inputLayout = TilingKeyLayout::BSH_BSND;
    uint32_t s1Size, gSize, sparseMode = 0, blockSize = 0;
    bool attenMaskFlag = false, pageAttentionFlag = false;
};
struct FiaTilingNonQuant {
    Info* fiaInfo_;
    uint32_t sInnerSize_ = 0, sInnerLoopTimes_ = 0, sInnerSizeTail_ = 0, sInnerSizeAlign_ = 0, mBaseSize_ = 0;
    bool softmaxWithBrcbFlag_ = false;
    void CalcInnerSize(uint32_t);
    void CalcMBaseSize();
};
#include "generic_tiling.inc"
int main(int argc, char** argv)
{
    Info info;
    const bool pagedTnd = argc > 1 && std::strcmp(argv[1], "--paged-tnd") == 0;
    const bool paged = pagedTnd || (argc > 1 && std::strcmp(argv[1], "--paged") == 0);
    if (pagedTnd || (argc > 1 && !paged))
        info.inputLayout = TilingKeyLayout::TND;
    FiaTilingNonQuant tiling{&info};
    uint32_t sk, mask;
    while (std::cin >> info.s1Size >> sk >> info.gSize >> mask) {
        if (paged)
            std::cin >> info.blockSize;
        info.pageAttentionFlag = paged;
        info.attenMaskFlag = mask != 0;
        info.sparseMode = mask == 1 ? 3 : mask == 2 ? 4 : 0;
        tiling.CalcInnerSize(sk);
        std::cout << '[' << tiling.mBaseSize_ << ',' << tiling.sInnerSize_ << ']' << std::endl;
    }
}
