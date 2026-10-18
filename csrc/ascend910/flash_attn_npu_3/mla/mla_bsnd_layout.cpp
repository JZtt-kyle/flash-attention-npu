#include "mla_bsnd_layout.h"

namespace flash_attn::mla {

namespace {
template <uint64_t elementBytes>
MlaLayoutStatus BuildLayout(const MlaBsndTensorView& view, MlaBsndTensorLayout& output)
{
    if (view.sizes[0] < 0 || view.sizes[1] < 0 || view.sizes[2] <= 0 ||
        (view.sizes[3] != 64 && view.sizes[3] != 128 && view.sizes[3] != 192 && view.sizes[3] != 512)) {
        return MlaLayoutStatus::InvalidShape;
    }
    if (view.strides[0] < 0 || view.strides[1] < 0 || view.strides[2] < 0 || view.strides[3] != 1) {
        return MlaLayoutStatus::InvalidStride;
    }
    if (view.storageOffset < 0) {
        return MlaLayoutStatus::InvalidStorage;
    }

    MlaBsndTensorLayout layout{uint64_t(view.strides[0]), uint64_t(view.strides[1]), uint64_t(view.strides[2]), 0};
    if (view.sizes[0] != 0 && view.sizes[1] != 0) {
        // Check each product/addition before
        // computing it; neither a signed intermediate nor the final byte span
        // may overflow. Storage offset is checked here, not sent to the device.
        constexpr uint64_t maxElements = INT64_MAX / elementBytes;
        uint64_t span = view.sizes[3];
        for (uint32_t axis = 0; axis < 3; ++axis) {
            const uint64_t count = view.sizes[axis] - 1;
            const uint64_t stride = view.strides[axis];
            if (count != 0 && stride > (maxElements - span) / count) {
                return MlaLayoutStatus::AddressOverflow;
            }
            span += count * stride;
        }
        const uint64_t offset = view.storageOffset;
        if (offset > maxElements - span) {
            return MlaLayoutStatus::AddressOverflow;
        }
        if ((offset + span) * elementBytes > view.storageBytes) {
            return MlaLayoutStatus::InvalidStorage;
        }
        layout.spanElements = span;
    }
    output = layout;
    return MlaLayoutStatus::Success;
}
} // namespace

MlaLayoutStatus BuildBsndTensorLayout(const MlaBsndTensorView& view, MlaBsndTensorLayout& output)
{
    return BuildLayout<2>(view, output);
}

MlaLayoutStatus BuildInt8BsndTensorLayout(const MlaBsndTensorView& view, MlaBsndTensorLayout& output)
{
    return BuildLayout<1>(view, output);
}

} // namespace flash_attn::mla
