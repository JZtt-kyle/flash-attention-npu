#include <iostream>
#include <vector>
#include "mla/mla_int8_schedule.h"
int main()
{
    using namespace flash_attn::mla::Int8;
    Config c{};
    int64_t total;
    while (std::cin >> c.batch >> c.query >> c.heads >> c.capacity >> c.cores >> c.splits >> c.causal >> c.reserved >>
           total) {
        uint32_t count, qcount;
        std::cin >> count;
        std::vector<int64_t> lengths(count);
        for (auto& n : lengths)
            std::cin >> n;
        std::cin >> qcount;
        std::vector<int64_t> prefix(qcount);
        for (auto& n : prefix)
            std::cin >> n;
        Int8TilingParams p{};
        uint32_t mode = 0;
        bool ok =
            ValidTnd(c) && count == c.batch && uint64_t(qcount) == uint64_t(c.batch) + 1 &&
            ScheduleTnd(
                c, [&](uint32_t b) { return lengths.at(b); }, [&](uint32_t b) { return prefix.at(b); }, total, p, mode);
        std::cout << "[" << ok << "," << mode;
        const auto* words = reinterpret_cast<const uint32_t*>(&p);
        for (uint32_t i = 0; i < sizeof(p) / 4; ++i)
            std::cout << "," << words[i];
        std::cout << "]" << std::endl;
    }
}
