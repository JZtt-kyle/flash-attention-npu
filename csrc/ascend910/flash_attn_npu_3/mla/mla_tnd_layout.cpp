#include "mla_tnd_layout.h"

namespace flash_attn::mla {

MlaLayoutStatus BuildTndTensorLayout(const MlaTndTensorView& view, MlaTndTensorLayout& output)
{
    // TND is a single BSND batch for address validation. Do not infer physical
    // strides from token counts, per-sample maxima, or the last dimension.
    const MlaBsndTensorView bsnd{{1, view.sizes[0], view.sizes[1], view.sizes[2]},
                                 {0, view.strides[0], view.strides[1], view.strides[2]},
                                 view.storageOffset,
                                 view.storageBytes};
    MlaBsndTensorLayout layout{};
    const auto status = BuildBsndTensorLayout(bsnd, layout);
    if (status == MlaLayoutStatus::Success) {
        output = {layout.rowStride, layout.headStride, layout.spanElements};
    }
    return status;
}

} // namespace flash_attn::mla
