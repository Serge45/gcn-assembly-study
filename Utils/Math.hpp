#pragma once
#include <type_traits>
#include <limits>

template<typename T, typename=std::enable_if_t<std::is_floating_point<T>::value>>
bool almostEqual(T a, T b, T delta = std::numeric_limits<T>::epsilon()) {
    return std::abs(a - b) < delta;
}
