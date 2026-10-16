// Test-only storage/page validation with independent padded K/V descriptors.
#include "mla_prefill_bsnd_consumer.h"
#include <iostream>
#include <vector>
using namespace flash_attn::mla;
struct Words {
    const std::vector<uint32_t>& data;
    uint32_t operator()(uint64_t i) const
    {
        return data.at(i);
    }
};
int main()
{
    uint32_t fault, heads, page;
    while (std::cin >> fault >> heads >> page) {
        constexpr uint32_t batches = 2, logicalPages = 3, physicalPages = 4;
        const uint64_t ks = 208, vs = 160, kr = heads * ks + 32, vr = heads * vs + 64;
        const uint64_t kp = page * kr + 96, vp = page * vr + 128;
        MlaBsndTensorView k{{physicalPages, page, heads, 192},
                            {int64_t(kp), int64_t(kr), int64_t(ks), 1},
                            11,
                            (physicalPages * kp + 11) * 2};
        MlaBsndTensorView v{{physicalPages, page, heads, 128},
                            {int64_t(vp), int64_t(vr), int64_t(vs), 1},
                            13,
                            (physicalPages * vp + 13) * 2};
        MlaPageTableView table{{batches, logicalPages}, {5, 1}, 3, 11 * 4};
        if (fault == 1)
            k.storageBytes = 2;
        if (fault == 2)
            v.storageBytes = 2;
        if (fault == 3)
            table.storageBytes = 4;
        if (fault == 4)
            k.strides[2] = INT64_MAX;
        if (fault == 5)
            v.strides[0] = INT64_MAX;
        if (fault == 6)
            table.strides[0] = INT64_MAX;
        if (fault == 7)
            k.strides[1] = -1;
        if (fault == 8)
            table.strides[1] = 0;
        if (fault == 9)
            v.sizes[2] += 1;
        if (fault == 10)
            k.storageOffset = INT64_MAX;
        if (fault == 11)
            table.storageOffset = -1;
        if (fault == 12)
            k.strides[0] = k.strides[1] = k.strides[2] = 0;
        if (fault == 13)
            table.strides[0] = 0;
        if (fault == 14)
            k.sizes[0] = v.sizes[0] = 0;
        MlaPagedCacheLayout layout{};
        const auto built = BuildPagedCacheLayout(table, k, v, layout, true);
        std::cout << "{\"layout\":" << int(built);
        if (built != MlaLayoutStatus::Success) {
            std::cout << "}\n" << std::flush;
            continue;
        }
        std::vector<uint32_t> pages{2, 0, 2, UINT32_MAX, UINT32_MAX, 1, UINT32_MAX, UINT32_MAX};
        std::vector<uint32_t> words(28, 0);
        words[25] = fault == 14 ? 0 : 3 * page - 1;
        words[27] = fault == 14 ? 0 : page;
        if (fault == 15)
            pages[2] = UINT32_MAX;
        if (fault == 16)
            pages[5] = physicalPages;
        MlaPrefillBsndContext c{};
        c.batch = batches;
        c.sk = 3 * page;
        c.pageSize = page;
        c.kvHeads = heads;
        MlaPagedCacheInputs cache{
            layout, {4, layout.tableSpanElements}, {2, layout.k.spanElements}, {2, layout.v.spanElements}};
        if (fault == 17)
            --cache.k.count;
        if (fault == 18)
            --cache.v.count;
        if (fault == 19)
            --cache.table.count;
        if (fault == 20)
            ++cache.layout.k.spanElements;
        if (fault == 21)
            cache.layout.k.headStride = UINT64_MAX;
        if (fault == 22)
            cache.table.address = UINT64_MAX - 3;
        if (fault == 23)
            cache.layout.physicalPages = uint64_t(INT32_MAX) + 2;
        if (fault == 24)
            ++words[25], ++words[25];
        auto status = ReadPrefillPagedCache(Words{words}, c, cache, Words{pages});
        std::cout << ",\"status\":" << int(status) << ",\"segments\":[";
        if (fault == 0) {
            for (uint32_t i = 0; i < 3; ++i) {
                const uint32_t start = i == 0 ? 0 : i == 1 ? page - 1 : 2 * page + 1;
                MlaPagedCopySegment segment{};
                const auto checked = GetPagedCopySegment(layout, reinterpret_cast<int32_t*>(pages.data()), pages.size(),
                                                         0, words[25], start, words[25] - start, segment);
                if (checked != MlaPagedStatus::Success)
                    return 2;
                std::cout << (i ? "," : "") << '[' << segment.physicalPage << ',' << segment.pageRow << ','
                          << segment.rowCount << ',' << segment.kOffset + (heads - 1) * layout.k.headStride << ','
                          << segment.vOffset + (heads - 1) * layout.v.headStride << ']';
            }
        }
        std::cout << "]}\n" << std::flush;
    }
}
