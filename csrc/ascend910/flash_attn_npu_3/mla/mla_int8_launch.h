#pragma once
#include "acl/acl.h"
#include "mla_int8_schedule.h"
#include "mla_paged_layout.h"
namespace flash_attn::mla::Int8 {
struct LaunchArgs {
    aclrtStream stream;
    uint64_t ffts, q, qv, k, v, qs, ks, vs, table, lengths, metadata, mask, out, lse, accum, accumLse, workspace;
    Config config;
    MlaPagedCacheLayout cache;
    float scale;
};
aclError Launch(const LaunchArgs&);
} // namespace flash_attn::mla::Int8
