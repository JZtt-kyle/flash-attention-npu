// Static-plan harness: no torch, CANN runtime, device pointers or tensor data.
#include <cstring>
#include <iostream>
#include <type_traits>

#include "mla_paged_plan.h"

using namespace flash_attn::mla;

static_assert(std::is_trivially_copyable_v<MlaPagedBsndPlan>);

template <typename T>
void ReadView(T& view)
{
    for (auto& n : view.sizes) {
        std::cin >> n;
    }
    for (auto& n : view.strides) {
        std::cin >> n;
    }
    std::cin >> view.storageOffset >> view.storageBytes;
}

void PrintTensor(const MlaBsndTensorLayout& x)
{
    std::cout << '[' << x.batchStride << ',' << x.rowStride << ',' << x.headStride << ',' << x.spanElements << ']';
}

int main(int argc, char** argv)
{
    const bool tnd = argc == 2 && std::strcmp(argv[1], "--tnd") == 0;
    if (argc != (tnd ? 2 : 1)) {
        return 2;
    }
    uint32_t dtype;
    bool used;
    while (std::cin >> dtype >> used) {
        MlaBsndTensorView q{}, qv{}, k{}, v{};
        MlaTndTensorView tq{}, tqv{};
        int64_t maxQuery = 0;
        MlaPageTableView table{};
        if (tnd) {
            std::cin >> maxQuery;
            ReadView(tq);
            ReadView(tqv);
        } else {
            ReadView(q);
            ReadView(qv);
        }
        ReadView(k);
        ReadView(v);
        ReadView(table);
        MlaPlatformInfo platform{};
        auto& mem = platform.localMemory;
        std::cin >> platform.aicNum >> platform.aivNum >> mem.l1Bytes >> mem.ubBytes >> mem.l0aBytes >> mem.l0bBytes >>
            mem.l0cBytes >> platform.libApiWorkspaceBytes;
        for (auto& n : platform.blockDims) {
            std::cin >> n;
        }
        if (!std::cin) {
            return 2;
        }
        MlaPagedPlan plan;
        std::memset(&plan, 0xa5, sizeof(plan));
        unsigned char before[sizeof(plan)];
        std::memcpy(before, &plan, sizeof(plan));
        const auto status = tnd ? BuildPagedTndPlan(tq, tqv, k, v, table, maxQuery, platform, dtype, used, plan)
                                : BuildPagedBsndPlan(q, qv, k, v, table, platform, dtype, used, plan);
        const bool unchanged = std::memcmp(before, &plan, sizeof(plan)) == 0;
        std::cout << "{\"status\":" << int(status) << ",\"unchanged\":" << (unchanged ? "true" : "false");
        if (status == MlaPagedPlanStatus::Success) {
            const auto& c = plan.context;
            const auto& s = plan.shape;
            const auto& p = plan.cache;
            std::cout << ",\"words\":" << plan.metadataWords << ",\"workspace\":" << plan.totalWorkspaceBytes
                      << ",\"shape\":[" << s.batch << ',' << s.queryHeads << ',' << int(s.query.layout) << ','
                      << s.query.totalTokens << ',' << s.query.maxLength << ',' << s.logicalKvCapacity << ','
                      << s.pageSize << "],\"context\":[" << c.dtype << ',' << c.batch << ',' << c.heads << ','
                      << c.usedFlags << ',' << c.query.layout << ',' << c.query.totalTokens.low << ','
                      << c.query.totalTokens.high << ',' << c.query.maxLength << ',' << c.pageSize << ','
                      << c.logicalKvCapacity << ',' << c.aicNum << ',' << c.aivNum << ',' << c.blockDim << ','
                      << c.workspaceBytes << ',' << c.libApiBytes << ',' << c.localMemory.l1Bytes << ','
                      << c.localMemory.ubBytes << ',' << c.localMemory.l0aBytes << ',' << c.localMemory.l0bBytes << ','
                      << c.localMemory.l0cBytes << "],\"blocks\":[";
            for (uint32_t i = 0; i < kMlaMaxAic; ++i) {
                std::cout << (i ? "," : "") << plan.platform.blockDims[i];
            }
            std::cout << "],\"platform\":[" << plan.platform.aicNum << ',' << plan.platform.aivNum << ','
                      << plan.platform.localMemory.l1Bytes << ',' << plan.platform.localMemory.ubBytes << ','
                      << plan.platform.localMemory.l0aBytes << ',' << plan.platform.localMemory.l0bBytes << ','
                      << plan.platform.localMemory.l0cBytes << ',' << plan.platform.libApiWorkspaceBytes
                      << "],\"layouts\":[";
            PrintTensor(plan.layouts.q);
            std::cout << ',';
            PrintTensor(plan.layouts.qv);
            std::cout << ',';
            PrintTensor(plan.layouts.k);
            std::cout << ',';
            PrintTensor(plan.layouts.v);
            std::cout << "],\"cache\":[" << p.batchSize << ',' << p.maxPages << ',' << p.logicalCapacity << ','
                      << p.physicalPages << ',' << p.pageSize << ',' << p.tableRowStride << ',' << p.tableSpanElements
                      << ',';
            PrintTensor(p.k);
            std::cout << ',';
            PrintTensor(p.v);
            std::cout << ']';
        }
        std::cout << "}\n";
    }
    return std::cin.eof() ? 0 : 2;
}
