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

template<typename T>
void toIdentity(T *data, size_t numRows, size_t numCols) {
    for (size_t i = 0; i < numCols; ++i) {
        for (size_t j = 0; j < numRows; ++j) {
            T val = (i == j ? T{1} : T{0});
            data[j + i * numRows] = val;
        }
    }
}

template<typename T>
void printMultiDim(const T *data, size_t numRows, size_t numCols) {
    std::cout << "[";
    for (size_t i = 0; i < numCols; ++i) {
        std::cout << "[";
        for (size_t j = 0; j < numRows; ++j) {
            std::cout << data[j + i * numRows] << ", ";
        }
        std::cout << "]\n";
    }
    std::cout << "]\n";
}
