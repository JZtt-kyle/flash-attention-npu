#include <iostream>
#include <vector>
#include "mla/mla_int8_schedule.h"
int main()
{
    using namespace flash_attn::mla::Int8;
    Config c{};
    while (std::cin >> c.batch >> c.query >> c.heads >> c.capacity >> c.cores >> c.splits >> c.causal >> c.reserved) {
        uint32_t count;
        std::cin >> count;
        std::vector<int64_t> lengths(count);
        for (auto& n : lengths)
            std::cin >> n;
        Int8TilingParams p{};
        uint32_t mode = 0;
        bool ok = Valid(c) && count == c.batch && Schedule(c, [&](uint32_t b) { return lengths.at(b); }, p, mode);
        std::cout << "[" << ok << "," << mode;
        const auto* words = reinterpret_cast<const uint32_t*>(&p);
        for (uint32_t i = 0; i < sizeof(p) / 4; ++i)
            std::cout << "," << words[i];
        std::cout << "]" << std::endl;
    }
}
