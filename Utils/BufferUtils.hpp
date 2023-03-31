#pragma once

#include <random>
#include <type_traits>

static std::random_device rd;
static std::mt19937 gen(rd());
static std::uniform_real_distribution<float> dist(-1.f, 1.f);

template<typename Iter>
void randomize(Iter beg, Iter end) {
    using ValueType = typename std::remove_pointer<Iter>::type;
    for (auto i = beg; i != end; ++i) {
        *i = dist(gen);
    }
}
