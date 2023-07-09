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

template<typename T>
void gemv(T *dst, const T *a, const T *x, const T *y, uint32_t m, uint32_t n) {
    for (uint32_t i = 0; i < m; ++i) {
        dst[i] = y[i];
        for (uint32_t j = 0; j < n; ++j) {
            dst[i] += a[i + m * j] * x[j];
        }
    }
}

hipError_t prepareASMKernel(const std::string &funcName, const std::string &coPath, hipModule_t *module, hipFunction_t *func) {
    auto err = hipModuleLoad(module, coPath.c_str());
    err = hipModuleGetFunction(func, *module, funcName.c_str());
    return err;
}

hipError_t launchASMKernel(hipFunction_t func, float *a, float *x, float *y, float *out, uint32_t m, uint32_t n) {
    KernelArguments kArgs;
    kArgs.append(a);
    kArgs.append(x);
    kArgs.append(y);
    kArgs.append(m);
    kArgs.append(n);
    kArgs.append(1.f);
    kArgs.append(1.f);
    kArgs.append(out);
    kArgs.applyAlignment();
    std::size_t argSize = kArgs.size();
    void *args[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER,
        kArgs.buffer(),
        HIP_LAUNCH_PARAM_BUFFER_SIZE,
        &argSize,
        HIP_LAUNCH_PARAM_END
    };
    const auto numWorkgroups = m / 256 + !!(m % 256);
    return hipExtModuleLaunchKernel(func, numWorkgroups * 256, 1, 1, 256, 1, 1, 0, nullptr, nullptr, args);
}

int main(int argc, char **argv) {
    hipModule_t mod;
    hipFunction_t func;
    auto err = prepareASMKernel("gemv", argv[1], &mod, &func);
    constexpr uint32_t m = 256;
    constexpr uint32_t n = 8192;
    constexpr uint32_t numElements = m * n;
    std::vector<float> cpuA(numElements, 0);
    std::vector<float> cpuX(n, 1);
    std::vector<float> cpuY(m, 1);
    std::vector<float> cpuOut(m, 0);
    // std::iota(begin(cpuA), end(cpuA), 0.f);
    // std::iota(begin(cpuX), end(cpuX), 0.f);
    // std::iota(begin(cpuY), end(cpuY), 0.f);
    randomize(begin(cpuA), end(cpuA));
    randomize(begin(cpuX), end(cpuX));
    randomize(begin(cpuY), end(cpuY));
    auto cpuBeg = std::chrono::steady_clock::now();
    gemv(cpuOut.data(), cpuA.data(), cpuX.data(), cpuY.data(), m, n);
    auto cpuEnd = std::chrono::steady_clock::now();
    std::cout << "CPU gemv func: " << std::chrono::duration<float, std::milli>(cpuEnd - cpuBeg).count() << " ms\n";
    float *gpuA{};
    float *gpuX{};
    float *gpuY{};
    float *gpuOut{};
    err = hipMalloc(&gpuA, numElements * sizeof(float));
    err = hipMalloc(&gpuX, n * sizeof(float));
    err = hipMalloc(&gpuY, m * sizeof(float));
    err = hipMalloc(&gpuOut, m * sizeof(float));
    err = hipMemcpyHtoD(gpuA, cpuA.data(), numElements * sizeof(float));
    err = hipMemcpyHtoD(gpuX, cpuX.data(), n * sizeof(float));
    err = hipMemcpyHtoD(gpuY, cpuY.data(), m * sizeof(float));
    hipEvent_t start, stop;
    err = hipEventCreate(&start);
    err = hipEventCreate(&stop);
    err = hipEventRecord(start);
    err = launchASMKernel(func, gpuA, gpuX, gpuY, gpuOut, m, n);
    err = hipEventRecord(stop);
    err = hipDeviceSynchronize();
    float dur{};
    err = hipEventElapsedTime(&dur, start, stop);
    std::cout << "ASM gemv: " << dur << " ms\n";
    err = hipEventDestroy(start);
    err = hipEventDestroy(stop);
    std::vector<float> gpuResult(m, 0);
    err = hipMemcpyDtoH(gpuResult.data(), gpuOut, m * sizeof(float));

    for (size_t i = 0; i < gpuResult.size(); ++i) {
        if (!almostEqual(gpuResult[i], cpuOut[i], 1e-3f)) {
            std::cout << "gpu & cpu results mismatched!\n";
            std::cout << gpuResult[i] << " != " << cpuOut[i] << '\n';
            break;
        }
    }

    err = hipModuleUnload(mod);
    err = hipFree(gpuA);
    err = hipFree(gpuX);
    err = hipFree(gpuY);
    err = hipFree(gpuOut);
    return 0;
}