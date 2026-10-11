/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Invocation-only PA descriptors shared by host preparation and device
// validation. Addresses/counts start at data_ptr(), after storage offset.
#pragma once
#include "mla_paged_layout.h"
#include "mla_paged_metadata_args.h"

namespace flash_attn::mla {
struct MlaPagedMetadataInputs {
    MlaPagedMetadataArray cuQuery, usedQuery, kvLengths;
};

// Host validates dtype/device and builds the complete physical layout. The
// device independently checks geometry and spans before reading any data.
// table counts int32 entries; k/v count FP16/BF16 elements, including padding.
struct MlaPagedCacheInputs {
    MlaPagedCacheLayout layout;
    MlaPagedMetadataArray table, k, v;
};
static_assert(std::is_standard_layout_v<MlaPagedCacheInputs> && std::is_trivially_copyable_v<MlaPagedCacheInputs>);
} // namespace flash_attn::mla
