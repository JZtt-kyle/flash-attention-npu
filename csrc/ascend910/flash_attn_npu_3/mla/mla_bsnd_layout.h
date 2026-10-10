// FA3 input-view adapter for compressed MLA (FP16/BF16, D=64 or 512).
// Pointer/stride convention: TriDao FA3 d15f1531a460ba456f41b01a774f33ab2db8febf,
// hopper/flash_api.cpp::set_params_fprop and mha_fwd's q_v handling.
#pragma once

#include <cstdint>

namespace flash_attn::mla {

struct MlaBsndTensorView {
    int64_t sizes[4];
    int64_t strides[4];    // Elements, in logical BSND order.
    int64_t storageOffset; // Elements from the beginning of storage.
    uint64_t storageBytes;
};

// Trivially copyable launch data, relative to the tensor's data_ptr().
// data_ptr() already includes storageOffset; never add that offset again.
// Q, qv, K and V each need their own descriptor, even when their shapes match.
// No address or stride belongs in reusable scheduler metadata.
struct MlaBsndTensorLayout {
    uint64_t batchStride;
    uint64_t rowStride;
    uint64_t headStride;
    uint64_t spanElements; // One past the last reachable element, or zero if empty.
};

enum class MlaLayoutStatus {
    Success,
    InvalidShape,
    InvalidStride,
    InvalidStorage,
    AddressOverflow,
};

// CPU-only address validation, independent of torch/CANN and schedule limits.
// Accept nonnegative outer strides (including read-only overlap/broadcast),
// a unit last stride, B/S >= 0, H > 0, and D=64/512. The entire allocated view
// must fit storage and signed 64-bit byte offsets, regardless of used lengths.
// Empty views access no storage; like PyTorch, their offset may exceed storage.
// Singleton/empty axes do not impose a byte limit on an unused stride.
// Callers must separately check dtype, device, shape compatibility and schedule
// limits. No production entry point consumes this descriptor yet.
// Failure leaves output unchanged.
MlaLayoutStatus BuildBsndTensorLayout(const MlaBsndTensorView& view, MlaBsndTensorLayout& output);

} // namespace flash_attn::mla
