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
#include "../Utils/KernelArguments.hpp"
#include "../Utils/Math.hpp"
#include "../Utils/BufferUtils.hpp"

void cpuGemm(
    const float *a, const float *b, const float *c, float *d,
    float alpha, float beta,
    std::uint32_t m, std::uint32_t n, std::uint32_t k
) {
    for (std::uint32_t i = 0; i < m; ++i) {
        for (std::uint32_t j = 0; j < n; ++j) {
            float acc{};

            for (std::uint32_t l = 0; l < k; ++l) {
                acc += a[i + l * m] * b[l + j * k];
            }

            const auto dstIdx = i + j * m;
            d[dstIdx] = beta * c[dstIdx] + alpha * acc;
        }
    }
}

template<size_t TileM, size_t TileN>
__global__ void naiveGemm(
    const float *a, const float *b, const float *c, float *d,
    float alpha, float beta,
    std::uint32_t m, std::uint32_t n, std::uint32_t k) {
    const auto blockRow = blockIdx.x * TileM;
    const auto blockCol = blockIdx.y * TileN;
    const auto blockOffset = blockCol * m + blockRow;
    const auto tId = threadIdx.x;
    const auto tRow = tId % TileM;
    const auto tCol = tId / TileM;
    float acc{};

    for (uint32_t i = 0; i < k; ++i) {
        acc += a[blockRow + tRow + m * i] * b[i + (tCol + blockCol) * k];
    }

    const uint64_t dstOffset = blockRow + tRow + (blockCol + tCol) * m;
    acc *= alpha;
    acc += beta * c[dstOffset];
    d[dstOffset] = acc;
}

void launchGpuGemm(
    const float *a, const float *b, const float *c, float *d,
    float alpha, float beta,
    std::uint32_t m, std::uint32_t n, std::uint32_t k) {
    constexpr size_t TileM = 16;
    constexpr size_t TileN = 16;
    const auto numWgM = (m / TileM) + !!(m % TileM);
    const auto numWgN = (n / TileN) + !!(n % TileN);
    naiveGemm<TileM, TileN><<<dim3(numWgM, numWgN, 1), 256>>>(a, b, c, d, alpha, beta, m, n, k);
}

hipError_t prepareASMKernel(const std::string &funcName, const std::string &coPath, hipModule_t *module, hipFunction_t *func) {
    auto err = hipModuleLoad(module, coPath.c_str());
    err = hipModuleGetFunction(func, *module, funcName.c_str());
    return err;
}

double gflops(uint32_t m, uint32_t n, uint32_t k, float durMs) {
    return 2 * m * n * k / durMs * 1e-6;
}

hipError_t launchASMKernel(hipFunction_t func, const float *a, const float *b, const float *c, float *d, float alpha, float beta, uint32_t m, uint32_t n, uint32_t k) {
    KernelArguments kArgs;
    kArgs.append(a);
    kArgs.append(b);
    kArgs.append(c);
    kArgs.append(d);
    kArgs.append(m);
    kArgs.append(n);
    kArgs.append(k);
    kArgs.append(alpha);
    kArgs.append(beta);
    kArgs.applyAlignment();
    std::size_t argSize = kArgs.size();
    void *args[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER,
        kArgs.buffer(),
        HIP_LAUNCH_PARAM_BUFFER_SIZE,
        &argSize,
        HIP_LAUNCH_PARAM_END
    };
    const auto numWorkgroups0 = m / 16 + !!(m % 16);
    const auto numWorkgroups1 = n / 16 + !!(n % 16);
    return hipExtModuleLaunchKernel(func, numWorkgroups0 * 256, numWorkgroups1, 1, 256, 1, 1, 0, nullptr, nullptr, args);
}

int main(int argc, char **argv) {
    hipError_t err{};
    hipModule_t mod;
    hipFunction_t func;
    err = prepareASMKernel("gemm", argv[1], &mod, &func);

    if (argc < 4) {
        return -1;
    }

    const uint32_t m = std::atoi(argv[2]);
    const uint32_t n = std::atoi(argv[3]);
    const uint32_t k = std::atoi(argv[4]);
    std::vector<float> cpuA(m * k, 1);
    std::vector<float> cpuB(k * n, 0);
    std::vector<float> cpuC(m * n, 1);
    std::vector<float> cpuD(m * n, 0);
    // std::iota(begin(cpuA), end(cpuA), 0.f);
    // std::iota(begin(cpuB), end(cpuB), 0.f);
    // std::iota(begin(cpuC), end(cpuC), 0.f);
    randomize(begin(cpuA), end(cpuA));
    randomize(begin(cpuB), end(cpuB));
    randomize(begin(cpuC), end(cpuC));
    float alpha{2.f};
    float beta{3.f};
    const uint32_t numRuns = 10;
    auto cpuBeg = std::chrono::steady_clock::now();

    for (uint32_t i = 0; i < numRuns; ++i) {
        cpuGemm(cpuA.data(), cpuB.data(), cpuC.data(), cpuD.data(), alpha, beta, m, n, k);
    }

    auto cpuEnd = std::chrono::steady_clock::now();
    std::cout << "cpuGemm func: " << std::chrono::duration<float, std::milli>(cpuEnd - cpuBeg).count() / numRuns << " ms\n";
    float *gpuA{};
    float *gpuB{};
    float *gpuC{};
    float *gpuD{};
    err = hipMalloc(&gpuA, m * k * sizeof(float));
    err = hipMalloc(&gpuB, n * k * sizeof(float));
    err = hipMalloc(&gpuC, m * n * sizeof(float));
    err = hipMalloc(&gpuD, m * n * sizeof(float));
    err = hipMemcpyHtoD(gpuA, cpuA.data(), m * k * sizeof(float));
    err = hipMemcpyHtoD(gpuB, cpuB.data(), n * k * sizeof(float));
    err = hipMemcpyHtoD(gpuC, cpuC.data(), m * n * sizeof(float));
    hipEvent_t start, stop;
    err = hipEventCreate(&start);
    err = hipEventCreate(&stop);
    //warmup for HIP kernel
    launchGpuGemm(gpuA, gpuB, gpuC, gpuD, alpha, beta, m, n, k);
    err = hipDeviceSynchronize();

    err = hipEventRecord(start);

    for (uint32_t i = 0; i < numRuns; ++i) {
        launchGpuGemm(gpuA, gpuB, gpuC, gpuD, alpha, beta, m, n, k);
    }

    err = hipEventRecord(stop);
    err = hipDeviceSynchronize();

    float dur{};
    err = hipEventElapsedTime(&dur, start, stop);
    std::cout << "HIP gemm: " << dur / numRuns << " ms\n"
              << "Gflops: " << gflops(m, n, k, dur / numRuns) << '\n';

    (void)hipMemset(gpuD, 0, sizeof(float) * m * n);
    //warmup
    (void)launchASMKernel(func, gpuA, gpuB, gpuC, gpuD, alpha, beta, m, n, k);
    err = hipEventRecord(start);
    for (uint32_t i = 0; i < numRuns; ++i) {
        (void)launchASMKernel(func, gpuA, gpuB, gpuC, gpuD, alpha, beta, m, n, k);
    }
    err = hipEventRecord(stop);
    err = hipDeviceSynchronize();
    err = hipEventElapsedTime(&dur, start, stop);
    std::cout << "ASM gemm: " << dur / numRuns << " ms\n"
              << "Gflops: " << gflops(m, n, k, dur / numRuns) << '\n';

    err = hipEventDestroy(start);
    err = hipEventDestroy(stop);
    std::vector<float> gpuResult(m * n, 0);
    err = hipMemcpyDtoH(gpuResult.data(), gpuD, m * n * sizeof(float));

    for (size_t i = 0; i < gpuResult.size(); ++i) {
        if (!almostEqual(gpuResult[i], cpuD[i], 1e-3f)) {
            std::cout << "gpu & cpu results mismatched at index: " << i << '\n';
            std::cout << gpuResult[i] << " != " << cpuD[i] << '\n';
            break;
        }
    }

    err = hipModuleUnload(mod);
    err = hipFree(gpuA);
    err = hipFree(gpuB);
    err = hipFree(gpuC);
    err = hipFree(gpuD);
    err = hipModuleUnload(mod);
    return 0;
}