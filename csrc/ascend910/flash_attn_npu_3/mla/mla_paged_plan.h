// Static host preparation for compressed BSND MLA paged attention.
// No tensor data, addresses, torch types or device runtime calls enter this API.
#pragma once

#include "mla_paged_layout.h"
#include "mla_paged_metadata.h"

namespace flash_attn::mla {

struct MlaPagedBsndPlan {
    MlaPagedShape shape;
    // Pass this fixed launch policy to the AICPU generator, including when
    // device-only lengths turn a nonempty allocation into an empty schedule.
    MlaPlatformInfo platform;
    MlaPagedMetadataContext context;
    MlaBsndInputLayouts layouts;
    MlaPagedCacheLayout cache;
    uint32_t metadataWords;
    uint64_t totalWorkspaceBytes; // context.workspaceBytes + context.libApiBytes.
};

enum class MlaPagedPlanStatus {
    Success,
    InvalidShape,
    InvalidLayout,
    UnsupportedDtype,
    InvalidPlatform,
    InsufficientLocalMemory,
    WorkspaceOverflow,
};

// q:[B,S,H,64], qv:[B,S,H,512], k:[Np,P,1,64], v:[Np,P,1,512],
// table:[B,M]. C=M*P, not Np*P. All allocated views must fit their storage,
// including unused query slots, table columns and physical pages. Read-only
// zero/overlapping strides are legal; offsets are already accounted for by
// BuildBsndTensorLayout/BuildPagedCacheLayout and never enter device layouts.
//
// dtype is MlaMetadataDtype::Fp16/Bf16. The caller still checks actual tensor
// dtype/device/rank, input addresses and length/metadata descriptors. It must
// materialize None/integer KV lengths and keep every asynchronous input alive.
// Dynamic lengths, effective pages and explicit metadata are checked by the
// device consumer on every invocation; success here does not validate them.
//
// Static B/S/C selects one or all C1V2 cores, independently of Np or Q-used.
// The budget bounds every successful v6 schedule at that geometry; it does
// not assert that arbitrary dynamic lengths satisfy source tile/field limits.
// It also does not select endpoints or a source schedule. Failure preserves
// output. This adapter alone enables neither a device nor a public forward.
MlaPagedPlanStatus BuildPagedBsndPlan(const MlaBsndTensorView& q, const MlaBsndTensorView& qv,
                                      const MlaBsndTensorView& k, const MlaBsndTensorView& v,
                                      const MlaPageTableView& table, const MlaPlatformInfo& platform, uint32_t dtype,
                                      bool hasUsedQuery, MlaPagedBsndPlan& output);

} // namespace flash_attn::mla
