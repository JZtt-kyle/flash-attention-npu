/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Address/segment mapping derived from CANN ops-transformer
// ca18aee610729ca5270a44e2d84c751bc687fbf0, attention/common/op_kernel/memcopy/:
// offset_calculator_v2.h (GM_KV_PA_BNBD), parser.h (BlockTableParser),
// attn_copy_gm_to_l1.h (CopyKKropePAGmToL1 / CopyKvGmToL1).
// The upstream license is reproduced in LICENSE.cann next to this file.
// Changes: independent CPU validation, signed page checks, checked 64-bit
// geometry and full storage bounds, arbitrary nonnegative outer strides.
// FA3 k maps to source RoPE K; FA3 v maps to both source content K and V.

#include "mla_paged_layout.h"

#include <algorithm>

namespace flash_attn::mla {
namespace {

MlaLayoutStatus BuildPageTableLayout(const MlaPageTableView& table, MlaPagedCacheLayout& layout)
{
    if (table.sizes[0] < 0 || table.sizes[1] < 0) {
        return MlaLayoutStatus::InvalidShape;
    }
    if (table.strides[0] < 0 || table.strides[1] != 1) {
        return MlaLayoutStatus::InvalidStride;
    }
    if (table.storageOffset < 0) {
        return MlaLayoutStatus::InvalidStorage;
    }
    layout.batchSize = table.sizes[0];
    layout.maxPages = table.sizes[1];
    layout.tableRowStride = table.strides[0];
    if (layout.batchSize != 0 && layout.maxPages != 0) {
        constexpr uint64_t maxElements = INT64_MAX / sizeof(int32_t);
        const uint64_t rows = layout.batchSize - 1;
        if (layout.maxPages > maxElements ||
            (rows != 0 && layout.tableRowStride > (maxElements - layout.maxPages) / rows)) {
            return MlaLayoutStatus::AddressOverflow;
        }
        const uint64_t span = rows * layout.tableRowStride + layout.maxPages;
        const uint64_t offset = table.storageOffset;
        if (offset > maxElements - span) {
            return MlaLayoutStatus::AddressOverflow;
        }
        if ((offset + span) * sizeof(int32_t) > table.storageBytes) {
            return MlaLayoutStatus::InvalidStorage;
        }
        layout.tableSpanElements = span;
    }
    return MlaLayoutStatus::Success;
}

bool ValidTableBuffer(const MlaPagedCacheLayout& layout, const int32_t* table, uint64_t elements)
{
    return elements >= layout.tableSpanElements && (layout.tableSpanElements == 0 || table != nullptr);
}

bool ValidLength(const MlaPagedCacheLayout& layout, int64_t length)
{
    return length >= 0 && uint64_t(length) <= layout.logicalCapacity;
}

bool ValidPage(const MlaPagedCacheLayout& layout, int32_t page)
{
    return page >= 0 && uint64_t(page) < layout.physicalPages;
}

} // namespace

namespace {
template <bool Int8>
MlaLayoutStatus BuildCacheLayout(const MlaPageTableView& table, const MlaBsndTensorView& k, const MlaBsndTensorView& v,
                                 MlaPagedCacheLayout& output, bool prefill)
{
    if (Int8 && (prefill || k.sizes[1] != 128)) {
        return MlaLayoutStatus::InvalidShape;
    }
    if (k.sizes[0] < 0 || uint64_t(k.sizes[0]) > uint64_t(INT32_MAX) + 1 || k.sizes[1] < 16 || k.sizes[1] > 1024 ||
        k.sizes[1] % 16 != 0 || (prefill ? k.sizes[2] <= 0 : k.sizes[2] != 1) || k.sizes[3] != (prefill ? 192 : 64) ||
        v.sizes[0] != k.sizes[0] || v.sizes[1] != k.sizes[1] || v.sizes[2] != k.sizes[2] ||
        v.sizes[3] != (prefill ? 128 : 512)) {
        return MlaLayoutStatus::InvalidShape;
    }
    MlaPagedCacheLayout layout{};
    auto status = BuildPageTableLayout(table, layout);
    if (status != MlaLayoutStatus::Success) {
        return status;
    }
    layout.physicalPages = k.sizes[0];
    layout.pageSize = k.sizes[1];
    if (layout.maxPages > uint64_t(INT64_MAX) / layout.pageSize) {
        return MlaLayoutStatus::AddressOverflow;
    }
    layout.logicalCapacity = layout.maxPages * layout.pageSize;
    status = BuildBsndTensorLayout(k, layout.k);
    if (status != MlaLayoutStatus::Success) {
        return status;
    }
    status = Int8 ? BuildInt8BsndTensorLayout(v, layout.v) : BuildBsndTensorLayout(v, layout.v);
    if (status != MlaLayoutStatus::Success) {
        return status;
    }
    output = layout;
    return MlaLayoutStatus::Success;
}
} // namespace

MlaLayoutStatus BuildPagedCacheLayout(const MlaPageTableView& table, const MlaBsndTensorView& k,
                                      const MlaBsndTensorView& v, MlaPagedCacheLayout& output, bool prefill)
{
    return BuildCacheLayout<false>(table, k, v, output, prefill);
}

MlaLayoutStatus BuildInt8PagedCacheLayout(const MlaPageTableView& table, const MlaBsndTensorView& k,
                                          const MlaBsndTensorView& v, MlaPagedCacheLayout& output)
{
    return BuildCacheLayout<true>(table, k, v, output, false);
}

MlaPagedStatus ValidatePagedCachePages(const MlaPagedCacheLayout& layout, const int32_t* table, uint64_t tableElements,
                                       const int32_t* lengths, uint64_t lengthCount)
{
    if (!ValidTableBuffer(layout, table, tableElements) || lengthCount != layout.batchSize ||
        (lengthCount != 0 && lengths == nullptr)) {
        return MlaPagedStatus::InvalidBuffer;
    }
    for (uint64_t batch = 0; batch < lengthCount; ++batch) {
        if (!ValidLength(layout, lengths[batch])) {
            return MlaPagedStatus::InvalidLength;
        }
        const uint64_t length = lengths[batch];
        const uint64_t pages = length / layout.pageSize + (length % layout.pageSize != 0);
        for (uint64_t page = 0; page < pages; ++page) {
            if (!ValidPage(layout, table[batch * layout.tableRowStride + page])) {
                return MlaPagedStatus::InvalidPage;
            }
        }
    }
    return MlaPagedStatus::Success;
}

MlaPagedStatus GetPagedCopySegment(const MlaPagedCacheLayout& layout, const int32_t* table, uint64_t tableElements,
                                   uint64_t batch, int64_t length, uint64_t start, uint64_t count,
                                   MlaPagedCopySegment& output)
{
    if (!ValidTableBuffer(layout, table, tableElements)) {
        return MlaPagedStatus::InvalidBuffer;
    }
    if (!ValidLength(layout, length)) {
        return MlaPagedStatus::InvalidLength;
    }
    if (batch >= layout.batchSize || start > uint64_t(length) || count > uint64_t(length) - start) {
        return MlaPagedStatus::InvalidRange;
    }
    MlaPagedCopySegment segment{};
    if (count != 0) {
        segment.tableOffset = batch * layout.tableRowStride + start / layout.pageSize;
        const int32_t page = table[segment.tableOffset];
        if (!ValidPage(layout, page)) {
            return MlaPagedStatus::InvalidPage;
        }
        segment.physicalPage = page;
        segment.pageRow = start % layout.pageSize;
        segment.rowCount = std::min(count, layout.pageSize - segment.pageRow);
        // Full-view validation bounds both products and their sum, including
        // the rest of the page and all 64/512 D elements, before these calls.
        segment.kOffset = segment.physicalPage * layout.k.batchStride + segment.pageRow * layout.k.rowStride;
        segment.vOffset = segment.physicalPage * layout.v.batchStride + segment.pageRow * layout.v.rowStride;
    }
    output = segment;
    return MlaPagedStatus::Success;
}

} // namespace flash_attn::mla
