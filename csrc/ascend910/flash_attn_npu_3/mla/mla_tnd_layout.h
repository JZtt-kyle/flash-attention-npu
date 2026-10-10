// FA3 input-view adapter for compressed MLA (FP16/BF16, D=64 or 512).
// Pointer/stride convention: TriDao FA3 d15f1531a460ba456f41b01a774f33ab2db8febf,
// hopper/flash_api.cpp::set_params_fprop and mha_fwd's q_v handling.
#pragma once

#include "mla_bsnd_layout.h"

namespace flash_attn::mla {

struct MlaTndTensorView {
    int64_t sizes[3];
    int64_t strides[3];    // Elements, in logical TND order (including NTD views).
    int64_t storageOffset; // Elements from the beginning of storage.
    uint64_t storageBytes;
};

// Relative to data_ptr(), which already includes storageOffset. Each of Q,
// qv, K and V has an independent descriptor; mixed BSND K/V use the BSND one.
// A sequence starts at allocationPrefix[b] * tokenStride, never a sum of used
// lengths or b * maxLength. No address or stride belongs in scheduler metadata.
struct MlaTndTensorLayout {
    uint64_t tokenStride;
    uint64_t headStride;
    uint64_t spanElements; // One past the last reachable element, or zero if empty.
};

// CPU-only adapter to the shared BSND storage/checked-byte validation, with a
// singleton batch. Requires T>=0, H>0, D=64/512, nonnegative token/head strides
// and a unit last stride. Validates the entire allocation, even with zero used.
// Empty T accesses no storage; unused singleton/empty strides remain 64-bit.
// Callers separately validate dtype/device, shape compatibility, allocation
// prefixes, used lengths and schedule limits. Failure leaves output unchanged.
// This independent component does not enable production TND/mixed strides.
MlaLayoutStatus BuildTndTensorLayout(const MlaTndTensorView& view, MlaTndTensorLayout& output);

} // namespace flash_attn::mla
