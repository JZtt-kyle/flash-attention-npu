// CPU consumer bridge. Synthetic cache descriptors are never dereferenced;
// page callbacks enumerate current effective pages and reject padding reads.
#include "mla_auto_split_paged_bsnd_driver.h"
#include "mla_auto_split_paged_bsnd_consumer.h"
#include <algorithm>

int main(int argc, char** argv)
{
    int command;
    while (std::cin >> command) {
        if (command == 1) {
            int64_t batch;
            std::cin >> batch;
            uint32_t words = 12345;
            const bool ok = AutoSplitPagedBsndMetadataWords(batch, words);
            std::cout << "{\"ok\":" << ok << ",\"words\":" << words << "}" << std::endl;
            continue;
        }
        if (command != 0)
            return 2;
        const auto r = Config();
        const auto& shape = r.shape;
        int64_t left, right;
        int causal, dtype, mode, mutation;
        uint32_t value;
        uint64_t taskIndex;
        uint32_t splitIndex, rowIndex, mergeHead, mergePart;
        std::cin >> left >> right >> causal >> dtype >> mode >> mutation >> value >> taskIndex >> splitIndex >>
            rowIndex >> mergeHead >> mergePart;
        uint32_t hasSoftcap = 0;
        if (argc > 1 && std::strcmp(argv[1], "--softcap") == 0)
            std::cin >> hasSoftcap;
        Lengths lengths;
        lengths.Read();
        auto inputs = lengths.Inputs();
        uint32_t words = 0;
        if (!AutoSplitPagedBsndMetadataWords(shape.batch, words) || words > 3000000)
            return 2;
        std::vector<uint32_t> record(words, 0xdeadbeef);
        const auto initial = record;
        if (mode == 1)
            inputs.kvLengths = {reinterpret_cast<int32_t*>(record.data() + words - 1), 1};
        const MlaWindowConfig window{shape.query.maxLength, shape.logicalKvCapacity, left, right, bool(causal)};
        auto* destination = mode == 18   ? nullptr
                            : mode == 19 ? reinterpret_cast<uint32_t*>(reinterpret_cast<uintptr_t>(record.data()) + 1)
                                         : record.data();
        const bool built = BuildAutoSplitPagedBsndMetadata(shape, r.platform, dtype, destination,
                                                           mode == 17 ? words - 1 : words, inputs, window, hasSoftcap);
        const bool intact = record == initial;
        uint32_t generated = record[5];
        const bool cleared = std::all_of(record.begin() + 8, record.end(), [](uint32_t v) { return v == 0; });
        if (mode == 25) {
            // Independent forged all-zero-KV record: base PA does not check
            // the new FP32 accumulator capacity. The consumer must do so.
            const uint32_t base = 408 + 5 * shape.batch;
            if (!BuildPagedMetadata(shape, r.platform, dtype, record.data(), base, inputs) || record[5])
                return 3;
            std::fill(record.begin() + base, record.end(), 0);
            MlaWindow norm{};
            NormalizeWindow(window, norm);
            record[1] = record[2] = 24;
            record[3] = words;
            record[base] = uint32_t(norm.kind);
            record[base + 1] = uint32_t(norm.left);
            record[base + 2] = uint32_t(norm.right);
            record[base + 3] = 1;
            record[base + 4] = shape.batch;
            record[base + 85] = 8;
            uint64_t prefix = 0;
            for (uint32_t i = 0; i <= uint32_t(shape.batch); ++i) {
                record[base + 346 + 2 * i] = uint32_t(prefix);
                record[base + 347 + 2 * i] = uint32_t(prefix >> 32);
                if (i < uint32_t(shape.batch) && shape.logicalKvCapacity) {
                    if (lengths.values[2][i])
                        return 3;
                    const uint64_t q = inputs.usedQuery.data ? lengths.values[1][i] : shape.query.maxLength;
                    prefix += (q * shape.queryHeads + 255) / 256;
                }
            }
            generated = 0;
        }
        if (mutation >= 0 && uint32_t(mutation) < words)
            record[mutation] = value;
        MlaWindow w{};
        NormalizeWindow(window, w);
        const uint64_t total = shape.query.totalTokens;
        const bool active = shape.batch && total && shape.logicalKvCapacity;
        MlaAutoSplitPagedBsndBudget budget{};
        const auto budgetStatus = BuildAutoSplitPagedBsndBudget(shape, r.platform, budget);
        MlaAutoSplitPagedBsndContext c{
            {uint32_t(dtype),
             uint32_t(shape.batch),
             uint32_t(shape.queryHeads),
             uint32_t(inputs.usedQuery.data != nullptr),
             {uint32_t(shape.query.layout), {uint32_t(total), uint32_t(total >> 32)}, uint32_t(shape.query.maxLength)},
             uint32_t(shape.pageSize),
             uint32_t(shape.logicalKvCapacity),
             r.platform.aicNum,
             r.platform.aivNum,
             r.platform.blockDims[0],
             budgetStatus == MlaScheduleStatus::Success ? budget.userBytes : uint64_t(INT64_MAX),
             active ? r.platform.libApiWorkspaceBytes : 0,
             r.platform.localMemory},
            {uint32_t(w.kind), w.left, w.right},
            hasSoftcap};
        // Bound the virtual buffers without allocating O(B*C) or physical KV.
        const uint64_t page = shape.pageSize > 0 ? shape.pageSize : 16;
        const uint64_t columns = shape.logicalKvCapacity > 0 ? shape.logicalKvCapacity / page : 0;
        const uint64_t batch = shape.batch;
        const uint64_t stride = columns + (mode == 33 ? 7 : 3);
        const uint64_t span = batch && columns ? (batch - 1) * stride + columns : 0;
        MlaPagedCacheInputs cache{};
        auto& p = cache.layout;
        p.batchSize = batch;
        p.pageSize = page;
        p.logicalCapacity = shape.logicalKvCapacity;
        p.maxPages = columns;
        p.physicalPages = 17;
        p.tableRowStride = stride;
        p.tableSpanElements = span;
        p.k = {page * 80 + 32, 80, 0, 16 * (page * 80 + 32) + (page - 1) * 80 + 64};
        p.v = {page * 528 + 64, 528, 0, 16 * (page * 528 + 64) + (page - 1) * 528 + 512};
        cache.table = {0x1000, span};
        cache.k = {0x2000, p.k.spanElements};
        cache.v = {0x4000, p.v.spanElements};
        if (mode == 33) {
            p.k = {page * 128 + 64, 128, 0, 16 * (page * 128 + 64) + (page - 1) * 128 + 64};
            p.v = {page * 640 + 96, 640, 0, 16 * (page * 640 + 96) + (page - 1) * 640 + 512};
            cache.k = {0x8000, p.k.spanElements};
            cache.v = {0x10000, p.v.spanElements};
        }
        if (mode == 2)
            c.paged.workspaceBytes = budget.userBytes - 1;
        if (mode == 3 && batch)
            lengths.values[2][0] ^= 1;
        if (mode == 4 && batch)
            lengths.values[1][0] ^= 1;
        if (mode == 5)
            inputs.kvLengths.count += 1;
        if (mode == 6)
            c.paged.query.totalTokens.low ^= 1;
        if (mode == 7)
            inputs.usedQuery.data = nullptr;
        if (mode == 8)
            c.paged.usedFlags ^= 1;
        if (mode == 10)
            p.physicalPages = 1;
        if (mode == 11)
            cache.table.count = span ? span - 1 : 1;
        if (mode == 12)
            cache.k.count -= 1;
        if (mode == 13)
            p.v.rowStride = UINT64_MAX;
        if (mode == 14)
            c.paged.pageSize += 16;
        if (mode == 15)
            c.paged.logicalKvCapacity += page;
        if (mode == 16)
            c.paged.blockDim = 0;
        if (mode == 23)
            c.paged.batch = UINT32_MAX;
        if (mode == 24)
            inputs.kvLengths.data = reinterpret_cast<const int32_t*>(UINTPTR_MAX - 3);
        if (mode == 26)
            c.window.left ^= 1;
        if (mode == 27)
            c.paged.aicNum -= 1;
        if (mode == 28)
            c.paged.localMemory.ubBytes = 1;
        if (mode == 29)
            c.paged.libApiBytes = UINT64_MAX;
        if (mode == 30)
            inputs.cuQuery = {&lengths.empty, 1};
        if (mode == 31) {
            cache.k.address = 1;
            cache.v.address = 1;
        }
        uint64_t reads = 0;
        auto pages = [&](uint64_t offset) {
            ++reads;
            const uint64_t b = offset / stride, j = offset % stride;
            if (b >= batch || j >= (uint64_t(lengths.values[2].at(b)) + page - 1) / page)
                std::abort();
            return mode == 9 ? UINT32_MAX : uint32_t((b + j * 3 + (mode == 32 ? 5 : 0)) % 17);
        };
        auto read = [&](uint32_t i) { return record.at(i); };
        auto arrays = [&](uint32_t a, uint32_t i) { return uint32_t(lengths.values.at(a).at(i)); };
        const auto array = [](MlaLengthArray a) {
            return MlaPagedMetadataArray{reinterpret_cast<uint64_t>(a.data), a.count};
        };
        MlaAutoSplitPagedBsndSchedule output{};
        output.native.groups = 12345;
        const auto before = output;
        const auto status = ReadAutoSplitPagedBsndMetadata(
            read, mode == 22 ? words - 1 : words, c,
            {array(inputs.cuQuery), array(inputs.usedQuery), array(inputs.kvLengths)}, arrays, cache, pages, output);
        std::cout << "{\"built\":" << built << ",\"intact\":" << intact << ",\"generated\":" << generated
                  << ",\"cleared\":" << cleared << ",\"read\":" << uint32_t(status)
                  << ",\"unchanged\":" << (std::memcmp(&before, &output, sizeof(output)) == 0)
                  << ",\"groups\":" << output.native.groups
                  << ",\"slots\":" << output.native.partialPrefix[output.native.fdHeads]
                  << ",\"bytes\":" << output.native.bytes << ",\"budget\":" << output.budgetBytes
                  << ",\"page_reads\":" << reads << ",\"record\":[";
        for (uint32_t i = 0; mode != 20 && mode != 25 && i < words; ++i)
            std::cout << (i ? "," : "") << record[i];
        std::cout << "],\"task\":[";
        MlaAutoSplitBsndTask task{};
        task.group = 987;
        const auto taskBefore = task;
        bool found = false;
        if (status == MlaMetadataReadStatus::Success)
            found = AutoSplitPagedBsndTask(read, output, c.window, taskIndex, splitIndex, task);
        if (found) {
            const auto& t = task;
            std::cout << t.group << ',' << t.slot << ',' << t.batch << ',' << t.m << ',' << t.split << ','
                      << t.splitCount << ',' << t.owner << ',' << t.fdHead << ',' << t.kind;
        }
        std::cout << "],\"task_unchanged\":" << (std::memcmp(&taskBefore, &task, sizeof(task)) == 0) << ",\"bases\":["
                  << output.native.used.tiling.mBaseSize << ',' << output.native.used.tiling.s2BaseSize
                  << "],\"range\":";
        Values(std::cout, task.rowBegin, task.rowEnd, task.s2Begin, task.s2End, task.kvBegin, task.kvEnd);
        std::cout << ",\"offsets\":";
        Values(std::cout, task.outOffset, task.sumOffset, task.maxOffset);
        MlaAutoSplitBsndRow row{};
        row.out = 987;
        const auto rowBefore = row;
        const bool rowOk = found && AutoSplitPagedBsndRow(output, task, rowIndex, row);
        std::cout << ",\"row_ok\":" << rowOk
                  << ",\"row_unchanged\":" << (std::memcmp(&rowBefore, &row, sizeof(row)) == 0) << ",\"row\":";
        Values(std::cout, row.out, row.lse, row.accumulatorOut, row.accumulatorLse, row.partialOut, row.partialSum,
               row.partialMax);
        MlaAutoSplitBsndMerge merge{};
        merge.group = 987;
        const auto mergeBefore = merge;
        const bool mergeOk = status == MlaMetadataReadStatus::Success &&
                             AutoSplitPagedBsndMerge(read, output, mergeHead, mergePart, merge);
        std::cout << ",\"merge_ok\":" << mergeOk
                  << ",\"merge_unchanged\":" << (std::memcmp(&mergeBefore, &merge, sizeof(merge)) == 0)
                  << ",\"merge\":";
        Values(std::cout, merge.group, merge.firstSlot, merge.batch, merge.m, merge.head, merge.part, merge.owner,
               merge.splitCount, merge.rowBegin, merge.rowEnd);
        std::cout << ",\"merge_offsets\":";
        Values(std::cout, merge.outOffset, merge.sumOffset, merge.maxOffset);
        std::cout << "}" << std::endl;
    }
}
