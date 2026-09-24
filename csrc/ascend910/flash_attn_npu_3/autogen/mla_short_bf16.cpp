/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Modified by Minghua Shen, 2026.
 */
#include "../mla/mla_short_kernel.hpp"

template void flash_attn::mla::LaunchShort<bfloat16_t>(
    const flash_attn::mla::DenseParams &, uint32_t, uint64_t, aclrtStream);
