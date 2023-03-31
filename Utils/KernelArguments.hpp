#pragma once
#include <cstdint>
#include <cstring>
#include <vector>
#include <type_traits>
#include <iterator>

class KernelArguments {
    using BufferElemType = std::uint8_t;
    using BufferElemPtrType = std::add_pointer<BufferElemType>::type;
    using BufferType = std::vector<BufferElemType>;

public:
    template<typename T>
    void append(const T &arg) {
        constexpr auto numBytesRequired = sizeof(T);
        const T *rawPtr = &arg;
        BufferType data(numBytesRequired, 0);
        std::memcpy(data.data(), rawPtr, data.size());
        std::copy(begin(data), end(data), std::back_inserter(bytes));
    }

    template<std::size_t Alignment = 8>
    void applyAlignment() {
        if (const auto rem = bytes.size() % Alignment) {
            std::fill_n(std::back_inserter(bytes), Alignment - rem, BufferElemType(0));
        }
        assert(size() % Alignment == 0);
    }

    void *buffer() {
        return bytes.data();
    }

    const void *buffer() const {
        return bytes.data();
    }

    std::size_t size() const {
        return bytes.size();
    }
private:
    BufferType bytes;
};
