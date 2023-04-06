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

constexpr std::uint32_t NUM_WORKITEM_PER_WORKGROUP = 256;

template<typename DType>
__global__ __launch_bounds__(NUM_WORKITEM_PER_WORKGROUP, 4) void hipGpuMaxKernel(DType *m, const DType *a, std::uint32_t numElements) {
    __shared__ DType localBuf[NUM_WORKITEM_PER_WORKGROUP];
    const auto tId = threadIdx.x;
    const auto readOffset = (blockIdx.x * blockDim.x + tId);
    const auto writeOffset = blockIdx.x;
    localBuf[tId] = a[readOffset];
    std::uint32_t i = 0;

    for (std::uint32_t s = (blockDim.x >> 1); s > 0; s >>= 1) {
        if (tId < s) {
            localBuf[tId] = std::max(localBuf[tId], localBuf[tId + s]);
        }
        __syncthreads();
    }

    if (tId == 0) {
        m[writeOffset] = localBuf[0];
    }
}

template<typename DType>
hipError_t hipGpuMax(DType *m, DType *a, std::uint32_t numElements) {
    DType *gpuBuf{};
    DType *gpuM{};
    auto err = hipMalloc(&gpuBuf, sizeof(DType) * numElements);
    err = hipMemcpyHtoD(gpuBuf, a, sizeof(DType) * numElements);
    err = hipMalloc(&gpuM, sizeof(DType) * (numElements / NUM_WORKITEM_PER_WORKGROUP));
    hipEvent_t beg, end;
    err = hipEventCreate(&beg);
    err = hipEventCreate(&end);
    err = hipEventRecord(beg);
    hipGpuMaxKernel<DType><<<numElements / NUM_WORKITEM_PER_WORKGROUP, NUM_WORKITEM_PER_WORKGROUP>>>(gpuM, gpuBuf, numElements);
    numElements /= NUM_WORKITEM_PER_WORKGROUP;
    err = hipDeviceSynchronize();

    while (numElements > 1) {
        hipGpuMaxKernel<DType><<<numElements / NUM_WORKITEM_PER_WORKGROUP, NUM_WORKITEM_PER_WORKGROUP>>>(gpuM, gpuM, numElements);
        numElements /= NUM_WORKITEM_PER_WORKGROUP;
        if (numElements == 1) {
            err = hipEventRecord(end);
        }
        err = hipDeviceSynchronize();
    }
    float dur{};
    err = hipEventElapsedTime(&dur, beg, end);
    std::cout << "HIP func: " << std::to_string(dur) << " ms\n";
    err = hipMemcpyDtoH(m, gpuM, sizeof(DType));
    err = hipFree(gpuBuf);
    err = hipFree(gpuM);
    return err;
}

hipError_t gpuMax(float *m, float *a, std::size_t numElements, hipFunction_t kernelFunc) {
    float *gpuBuf{};
    auto err = hipMalloc(&gpuBuf, sizeof(float) * numElements);
    err = hipMemcpyHtoD(gpuBuf, a, sizeof(float) * numElements);
    std::size_t workspaceSize = numElements / NUM_WORKITEM_PER_WORKGROUP;
    float *workspace{};
    err = hipMalloc(&workspace, sizeof(float) * workspaceSize);

    hipEvent_t beg, end;
    err = hipEventCreate(&beg);
    err = hipEventCreate(&end);

    err = hipEventRecord(beg);

    KernelArguments kArgs;
    kArgs.reserve(24);
    kArgs.append(gpuBuf);
    kArgs.append<std::uint32_t>(numElements);
    kArgs.append(workspace);
    kArgs.applyAlignment();
    std::size_t argSize = kArgs.size();
    void *kernelArgs[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER,
        kArgs.buffer(),
        HIP_LAUNCH_PARAM_BUFFER_SIZE,
        &argSize,
        HIP_LAUNCH_PARAM_END
    };

    err = hipExtModuleLaunchKernel(kernelFunc, numElements, 1, 1,
        NUM_WORKITEM_PER_WORKGROUP, 1, 1,
        sizeof(float) * NUM_WORKITEM_PER_WORKGROUP,
        nullptr,
        nullptr, kernelArgs);

    numElements /= NUM_WORKITEM_PER_WORKGROUP;
    err = hipDeviceSynchronize();

    while (numElements > 1) {
        KernelArguments kArgs;
        kArgs.reserve(24);
        kArgs.append(workspace);
        kArgs.append<std::uint32_t>(numElements);
        kArgs.append(workspace);
        kArgs.applyAlignment();
        std::size_t argSize = kArgs.size();
        void *kernelArgs[] = {
            HIP_LAUNCH_PARAM_BUFFER_POINTER,
            kArgs.buffer(),
            HIP_LAUNCH_PARAM_BUFFER_SIZE,
            &argSize,
            HIP_LAUNCH_PARAM_END
        };
        err = hipExtModuleLaunchKernel(kernelFunc, numElements, 1, 1,
            NUM_WORKITEM_PER_WORKGROUP, 1, 1,
            sizeof(float) * NUM_WORKITEM_PER_WORKGROUP,
            nullptr,
            nullptr, kernelArgs);
        numElements /= NUM_WORKITEM_PER_WORKGROUP;

        if (numElements == 1) {
            err = hipEventRecord(end);
        }
        err = hipDeviceSynchronize();
    }
    float dur{};
    err = hipEventElapsedTime(&dur, beg, end);
    const auto gpuBandWidth = (2 * numElements * sizeof(float) / std::pow(1024.f, 3)) * 1e3 / (dur);
    std::cout << "GPU Max func: " << std::to_string(dur) << " ms, " << std::to_string(gpuBandWidth) << " GB/s\n";
    err = hipEventDestroy(beg);
    err = hipEventDestroy(end);
    err = hipMemcpyDtoH(m, workspace, sizeof(float));
    err = hipFree(gpuBuf);
    err = hipFree(workspace);
    return err;
}

int main(int argc, char **argv) {
    hipDevice_t dev{};
    auto err = hipDeviceGet(&dev, 0);
    hipModule_t module;
    assert(argc == 3);
    const std::string coPath(argv[1]);
    const std::uint32_t numElements(std::atoi(argv[2]));
    err = hipModuleLoad(&module, coPath.c_str());
    assert(err == HIP_SUCCESS);
    hipFunction_t gpuFunc;
    err = hipModuleGetFunction(&gpuFunc, module, "max_func");
    assert(err == HIP_SUCCESS);
    float *gpuMem{};
    std::vector<float> cpuMem(numElements, 0);
    randomize(begin(cpuMem), end(cpuMem));
    err = hipMalloc(&gpuMem, sizeof(float) * numElements);
    err = hipMemcpyHtoD(gpuMem, cpuMem.data(), cpuMem.size() * sizeof(float));

    float gpuResult{};

    for (std::size_t i = 0; i < 10; ++i) {
        err = gpuMax(&gpuResult, cpuMem.data(), numElements, gpuFunc);
    }

    float hipResult{};

    for (std::size_t i = 0; i < 10; ++i) {
        err = hipGpuMax(&hipResult, cpuMem.data(), numElements);
    }

    auto cpuBeg = std::chrono::steady_clock::now();
    float cpuMax{};
    for (std::size_t i = 0; i < 10; ++i) {
        cpuMax = *std::max_element(begin(cpuMem), end(cpuMem));
    }
    auto cpuEnd = std::chrono::steady_clock::now();
    std::cout << "CPU Max func: " << std::chrono::duration<float, std::milli>(cpuEnd - cpuBeg).count() / 10 << " ms\n";

    assert(cpuMax == gpuResult);
    std::cout << "Check ASM vs CPU: " << almostEqual(cpuMax, gpuResult) << '\n';
    std::cout << "Check HIP vs CPU: " << almostEqual(cpuMax, hipResult) << '\n';
    err = hipFree(gpuMem);
    err = hipModuleUnload(module);
    return 0;
}
