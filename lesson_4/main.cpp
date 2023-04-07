#include <algorithm>
#include <cassert>
#include <chrono>
#include <iostream>
#include <limits>
#include <string>
#include <numeric>
#include <hip/hip_runtime.h>
#include <hip/device_functions.h>
#include <hip/hip_ext.h>
#include <hip/math_functions.h>
#include "../Utils/KernelArguments.hpp"
#include "../Utils/Math.hpp"
#include "../Utils/BufferUtils.hpp"

constexpr std::uint32_t NUM_WORKITEM_PER_WORKGROUP = 1024;

template<typename DType, std::uint32_t Cols, std::uint32_t RowTile>
__global__ void hipGpuSoftmaxKernel(DType *m, const DType *a, std::uint32_t numRows) {
    static_assert(Cols % 2 == 0, "Cols must be power of 2");
    constexpr uint32_t numRowsPerIter = NUM_WORKITEM_PER_WORKGROUP / Cols;
    constexpr uint32_t numRowsInBlock = RowTile * numRowsPerIter;
    constexpr uint32_t prepad = 0;
    __shared__ DType localBuf[numRowsInBlock][Cols + prepad];
    const uint32_t tId = threadIdx.x;
    const auto row = tId / Cols;
    const auto col = tId & (Cols - 1);
    const uint32_t ldsPad = 0;
    const uint32_t paddedCol = col + ldsPad;
    DType tVal[RowTile];

    #pragma unroll
    for (std::uint32_t j = 0; j < RowTile; ++j) {
        const auto procRow = row + numRowsPerIter * j;
        const auto readOffset = blockIdx.x * blockDim.x * RowTile + procRow * Cols + col;
        tVal[j] = a[readOffset];
        localBuf[procRow][paddedCol] = tVal[j];
    }

    __syncthreads();

    #pragma unroll
    for (std::uint32_t s = (Cols >> 1); s > 0; s >>= 1) {
        if (col < s) {
            #pragma unroll
            for (std::uint32_t j = 0; j < RowTile; ++j) {
                const auto procRow = row + j * numRowsPerIter;
                localBuf[procRow][paddedCol] = std::max(localBuf[procRow][paddedCol], localBuf[procRow][paddedCol + s]);
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (std::uint32_t j = 0; j < RowTile; ++j) {
        const auto procRow = row + j * numRowsPerIter;
        const DType rowMax = localBuf[procRow][ldsPad];
        tVal[j] -= rowMax;
        tVal[j] = expf(tVal[j]);
        localBuf[procRow][paddedCol] = tVal[j];
    }
    __syncthreads();

    #pragma unroll
    for (std::uint32_t s = (Cols >> 1); s > 0; s >>= 1) {
        if (col < s) {
            #pragma unroll
            for (std::uint32_t j = 0; j < RowTile; ++j) {
                const auto procRow = row + j * numRowsPerIter;
                localBuf[procRow][paddedCol] += localBuf[procRow][paddedCol + s];
            }
        }
        __syncthreads();
    }

    for (std::uint32_t j = 0; j < RowTile; ++j) {
        const auto procRow = row + j * numRowsPerIter;
        const auto writeOffset = blockIdx.x * blockDim.x * RowTile + procRow * Cols + col;
        const DType rowSum = localBuf[procRow][ldsPad];
        m[writeOffset] = tVal[j] / rowSum;
    }
}

template<typename DType, std::uint32_t Cols, std::uint32_t RowTile>
void hipGpuSoftmax(DType *m, DType *a, std::uint32_t numRows) {
    std::uint32_t numElements = numRows * Cols;
    const std::uint32_t numWorkgroups = std::ceil(numElements / static_cast<float>(RowTile) / NUM_WORKITEM_PER_WORKGROUP);
    hipGpuSoftmaxKernel<DType, Cols, RowTile><<<numWorkgroups, NUM_WORKITEM_PER_WORKGROUP>>>(m, a, numRows);
    return;
}

template<typename DType, std::uint32_t Cols>
void cpuSoftmax(DType *m, DType *a, std::uint32_t numRows) {
    for (std::uint32_t i = 0; i < numRows; ++i) {
        auto rowMax = a[i * Cols];

        for (std::uint32_t j = 1; j < Cols; ++j) {
            rowMax = std::max(a[i * Cols + j], rowMax);
        }

        auto rowSum = 0.f;

        for (std::uint32_t j = 0; j < Cols; ++j) {
            const auto v = std::exp(a[i * Cols + j] - rowMax);
            m[i * Cols + j] = v;
            rowSum += v;
        }

        for (std::uint32_t j = 0; j < Cols; ++j) {
            m[i * Cols + j] /= rowSum;
        }
    }
}

int main(int argc, char **argv) {
    hipDevice_t dev{};
    auto err = hipDeviceGet(&dev, 0);
    hipModule_t module;
    assert(argc == 3);
    constexpr uint32_t n = 16;
    const std::string coPath(argv[1]);
    const std::uint32_t m(std::atoi(argv[2]));
    const std::uint32_t numElements = m * n;
    float *gpuMem{};
    std::vector<float> cpuMem(numElements, 0);
    randomize(begin(cpuMem), end(cpuMem));
    err = hipMalloc(&gpuMem, sizeof(float) * numElements);
    err = hipMemcpyHtoD(gpuMem, cpuMem.data(), cpuMem.size() * sizeof(float));
    float *hipResult{};
    err = hipMalloc(&hipResult, sizeof(float) * numElements);

    hipEvent_t beg, end;
    err = hipEventCreate(&beg);
    err = hipEventCreate(&end);

    err = hipEventRecord(beg);
    std::size_t numRuns = 100;
    for (std::size_t i = 0; i < numRuns; ++i) {
        hipGpuSoftmax<float, n, 2>(hipResult, gpuMem, m);
    }
    err = hipDeviceSynchronize();
    err = hipEventRecord(end);
    float hipDur{};
    err = hipEventElapsedTime(&hipDur, beg, end);
    std::cout << "HIP Softmax func: " << std::to_string(hipDur / numRuns) << " ms ~= " << 2 * numElements * sizeof(float) * 1e3 / std::pow(1024.f, 3) / hipDur << " GB/s\n";

    auto cpuBeg = std::chrono::steady_clock::now();
    std::vector<float> cpuOut(numElements);
    for (std::size_t i = 0; i < numRuns; ++i) {
        cpuSoftmax<float, n>(cpuOut.data(), cpuMem.data(), m);
    }
    auto cpuEnd = std::chrono::steady_clock::now();
    std::cout << "CPU Softmax func: " << std::chrono::duration<float, std::milli>(cpuEnd - cpuBeg).count() / numRuns << " ms\n";
    std::vector<float> hipReturnedResult(numElements);
    err = hipMemcpyDtoH(hipReturnedResult.data(), hipResult, sizeof(float) * numElements);
    std::size_t numMismatched{};

    for (size_t i = 0; i < numElements; ++i) {
        if (!almostEqual(hipResult[i], cpuOut[i])) {
            //std::cout << "Check HIP vs CPU failed at: " << i << ", " << std::to_string(hipReturnedResult[i]) << " : " << std::to_string(cpuOut[i]) << '\n';
            ++numMismatched;
        }
    }

    std::cout << "Mismatched: " << numMismatched << '\n';

    err = hipFree(gpuMem);
    err = hipModuleUnload(module);
    return 0;
}
