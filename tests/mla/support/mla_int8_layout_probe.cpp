// Test-only C ABI for the mixed-byte-width extension of existing PA validation.
#include "mla/mla_paged_layout.h"
using namespace flash_attn::mla;
extern "C" int int8_cache_layout(const MlaPageTableView* table, const MlaBsndTensorView* k, const MlaBsndTensorView* v,
                                 MlaPagedCacheLayout* out)
{
    return int(BuildInt8PagedCacheLayout(*table, *k, *v, *out));
}
extern "C" int int8_pages(const MlaPagedCacheLayout* layout, const int32_t* table, uint64_t elements,
                          const int32_t* lengths, uint64_t count)
{
    return int(ValidatePagedCachePages(*layout, table, elements, lengths, count));
}
extern "C" int int8_segment(const MlaPagedCacheLayout* layout, const int32_t* table, uint64_t elements, uint64_t batch,
                            int64_t length, uint64_t start, uint64_t count, MlaPagedCopySegment* out)
{
    return int(GetPagedCopySegment(*layout, table, elements, batch, length, start, count, *out));
}
