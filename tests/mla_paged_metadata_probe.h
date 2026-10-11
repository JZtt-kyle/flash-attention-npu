// Diagnostic wrapper only: expose the actual producer return code without
// poisoning the stream, so rejection can be checked against guarded storage.
#pragma once
#include "../csrc/ascend910/flash_attn_npu_3/mla/mla_paged_metadata_args.h"

struct MlaPagedMetadataProbeArgs {
    flash_attn::mla::MlaPagedMetadataArgs input;
    uint64_t statusAddr;
    uint32_t nullArgs;
};
