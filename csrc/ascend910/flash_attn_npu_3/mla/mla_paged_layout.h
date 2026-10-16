// Independent CPU address adapter for compressed MLA ND paged caches.
// This component does not enable PA in a launcher, metadata or public entry.
#pragma once

#include "mla_bsnd_layout.h"

namespace flash_attn::mla {

struct MlaPageTableView {
    int64_t sizes[2];
    int64_t strides[2]; // int32 elements, not bytes; last stride must be one.
    int64_t storageOffset;
    uint64_t storageBytes;
};

struct MlaPagedCacheLayout {
    uint64_t batchSize;
    uint64_t maxPages;        // Logical page-table width, independent of row stride.
    uint64_t logicalCapacity; // maxPages * pageSize, independent of physicalPages.
    uint64_t physicalPages;
    uint64_t pageSize;
    uint64_t tableRowStride;
    uint64_t tableSpanElements;
    MlaBsndTensorLayout k, v; // batchStride denotes physical page stride.
};

// K:[Np,P,1,64], V:[Np,P,1,512], P=16..1024 in multiples of 16.
// All physical pages, including unused ones, must fit their own storage.
// Np <= INT32_MAX+1 makes every physical page representable by an int32 ID.
// Read-only broadcasts/overlaps are legal. Empty views follow BSND storage
// rules. All addresses are relative to data_ptr(), which includes the offset.
// This address layer does not impose schedule limits (e.g. capacity<=16384).
// Callers separately validate Q/qv with BuildBsnd/BuildTndTensorLayout, dtype,
// device and sequence geometry. Failure leaves output unchanged.
MlaLayoutStatus BuildPagedCacheLayout(const MlaPageTableView& table, const MlaBsndTensorView& k,
                                      const MlaBsndTensorView& v, MlaPagedCacheLayout& output, bool prefill = false);

enum class MlaPagedStatus {
    Success,
    InvalidBuffer,
    InvalidLength,
    InvalidPage,
    InvalidRange,
};

// CPU host buffers only, for independent validation/oracles; never pass a
// device pointer or copy device lengths/pages to the host for this API.
// Layout must come from BuildPagedCacheLayout. Counts are accessible elements
// starting at data_ptr(), excluding storageOffset. Validate all batches even
// when their Q-used is zero. Only ceil(length/P) entries in each row are read;
// padding may contain -1 or any other poison. No cumulative KV lengths in PA.
MlaPagedStatus ValidatePagedCachePages(const MlaPagedCacheLayout& layout, const int32_t* table, uint64_t tableElements,
                                       const int32_t* lengths, uint64_t lengthCount);

struct MlaPagedCopySegment {
    uint64_t tableOffset;
    uint64_t physicalPage;
    uint64_t pageRow;
    uint64_t rowCount;
    uint64_t kOffset; // Row start, before adding the independently selected D.
    uint64_t vOffset;
};

// Return the first page-contained segment of [start,start+count), within this
// batch's effective length. Iterate by rowCount; even adjacent equal page IDs
// remain separate segments. count=0 succeeds with a zero descriptor, without
// reading a page. Invalid range/page/buffer leaves output unchanged.
// Requires a built layout; checks the selected page itself. Validate all pages
// separately before consuming segments, and keep table/lengths immutable.
MlaPagedStatus GetPagedCopySegment(const MlaPagedCacheLayout& layout, const int32_t* table, uint64_t tableElements,
                                   uint64_t batch, int64_t length, uint64_t start, uint64_t count,
                                   MlaPagedCopySegment& output);

} // namespace flash_attn::mla
