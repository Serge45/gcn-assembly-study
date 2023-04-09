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

template<typename DType, std::uint32_t Cols, std::uint32_t RowTile, std::uint32_t RowStride, std::uint32_t LdsPad>
__device__ void globalRead(volatile DType localBuf[][Cols + LdsPad], 
                           DType *tVal,
                           const DType *a,
                           std::uint32_t row,
                           std::uint32_t col,
                           std::uint32_t rowShift,
                           std::uint32_t ldsPad) {
    static_assert(Cols % RowTile == 0, "Cols % RowTile must be 0");

    if constexpr (RowTile <= 2) {
        #pragma unroll
        for (std::uint32_t j = 0; j < RowTile; ++j) {
            const auto procRow = row + RowStride * j;
            const auto readOffset = blockIdx.x * blockDim.x * RowTile + procRow * Cols - rowShift * Cols + col;
            tVal[j] = a[readOffset];
            localBuf[procRow][col + ldsPad] = tVal[j];
        }
    } else if constexpr (RowTile <= 4) {
        constexpr auto numRowComp = Cols / RowTile;
        row = threadIdx.x / numRowComp;
        col = (threadIdx.x % numRowComp) * RowTile;
        const auto readOffset = blockIdx.x * blockDim.x * RowTile + row * Cols - rowShift * Cols + col;
        float2 tVals = make_float2(a[readOffset + 0], a[readOffset + 1]);
        tVal[0] = tVals.x;
        tVal[1] = tVals.y;
        //float2 *localBufPtr = (float2 *)(&localBuf[row][ldsPad]);
        //*localBufPtr = tVals;
        localBuf[row][col + ldsPad] = tVals.x;
        localBuf[row][col + ldsPad + 1] = tVals.y;
    } else {
        constexpr auto numRowComp = Cols / RowTile;
        row = threadIdx.x / numRowComp;
        col = (threadIdx.x % numRowComp) * RowTile;
        const auto readOffset = blockIdx.x * blockDim.x * RowTile + row * Cols - rowShift * Cols + col;
        #pragma unroll
        for (std::uint32_t j = 0; j < RowTile; ++j) {
            tVal[j] = a[readOffset + j];
        }

        #pragma unroll
        for (std::uint32_t j = 0; j < RowTile; ++j) {
            localBuf[row][col + j] = tVal[j];
        }
    }
}

template<typename DType, std::uint32_t Cols, std::uint32_t RowTile>
__global__ void hipGpuSoftmaxKernel(DType *m, const DType *a, std::uint32_t numRows) {
    static_assert((Cols & (Cols - 1)) == 0, "Cols must be power of 2");
    constexpr uint32_t numRowsPerIter = NUM_WORKITEM_PER_WORKGROUP / Cols;
    constexpr uint32_t numRowsInBlock = RowTile * numRowsPerIter;
    constexpr uint32_t prepad = 32;
    __shared__ DType localBuf[numRowsInBlock][Cols + prepad];
    const uint32_t tId = threadIdx.x;
    const auto lastRowIdx = numRowsInBlock * (blockIdx.x + 1);
    const auto globalReadRowShift = (lastRowIdx <= numRows) ? 0 : (lastRowIdx - numRows);
    const auto row = tId / Cols;
    const auto col = tId & (Cols - 1);
    const uint32_t ldsPad = row % prepad;
    const uint32_t paddedCol = col + ldsPad;
    DType tVal[RowTile];
    globalRead<DType, Cols, RowTile, numRowsPerIter, prepad>(localBuf, tVal, a, row, col, globalReadRowShift, ldsPad);
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

    #pragma unroll
    for (std::uint32_t j = 0; j < RowTile; ++j) {
        const auto procRow = row + j * numRowsPerIter;
        const auto writeOffset = blockIdx.x * blockDim.x * RowTile + procRow * Cols - globalReadRowShift * Cols + col;
        if (procRow >= globalReadRowShift) {
            const DType rowSum = localBuf[procRow][ldsPad];
            m[writeOffset] = tVal[j] / rowSum;
        }
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
        const auto rowMax = *std::max_element(a + i * Cols, a + i * Cols + Cols);
        auto rowSum = 0.f;
        std::transform(a + i * Cols, a + i * Cols + Cols, m + i * Cols, [&rowSum] (auto v) {
            const auto u = std::exp(v);
            rowSum += u;
            return u;
        });

        std::transform(m + i * Cols, m + i * Cols + Cols, m + i * Cols, [rowSum] (auto v) {
            return v / rowSum;
        });
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

    constexpr std::size_t numWarmups = 50;
    constexpr std::size_t rowTile = 1;

    for (std::size_t i = 0; i < numWarmups; ++i) {
        hipGpuSoftmax<float, n, rowTile>(hipResult, gpuMem, m);
    }

    const std::size_t numRuns = 300;
    for (std::size_t i = 0; i < numRuns; ++i) {
        hipGpuSoftmax<float, n, rowTile>(hipResult, gpuMem, m);
    }
    err = hipDeviceSynchronize();
    err = hipEventRecord(end);
    float hipDur{};
    err = hipEventElapsedTime(&hipDur, beg, end);
    std::cout << "HIP Softmax func: " << std::to_string(hipDur / numRuns) << " ms ~= " << 2 * numRuns * numElements * sizeof(float) * 1e3 / std::pow(1024.f, 3) / hipDur << " GB/s\n";

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
            // std::cout << "Check HIP vs CPU failed at: " << i
            //     << ", " << std::to_string(hipReturnedResult[i])
            //     << " : " << std::to_string(cpuOut[i])
            //     << ", diff: "
            //     << std::to_string(std::abs(hipReturnedResult[i] - cpuOut[i]))
            //     << '\n';
            ++numMismatched;
        }
    }

    std::cout << "Mismatched: " << numMismatched << '\n';

    err = hipFree(gpuMem);
    err = hipModuleUnload(module);
    return 0;
}
