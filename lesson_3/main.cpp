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
#include "../Utils/BufferUtils.hpp"

constexpr std::uint32_t NUM_WORKITEM_PER_WORKGROUP = 256;

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
        err = hipDeviceSynchronize();
    }
    err = hipEventRecord(end);
    float dur{};
    err = hipEventElapsedTime(&dur, beg, end);
    std::cout << "GPU Max func: " << std::to_string(dur) << " ms\n";
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
    err = gpuMax(&gpuResult, cpuMem.data(), numElements, gpuFunc);
    auto cpuBeg = std::chrono::steady_clock::now();
    auto cpuMax = *std::max_element(begin(cpuMem), end(cpuMem));
    auto cpuEnd = std::chrono::steady_clock::now();
    std::cout << "CPU Max func: " << std::chrono::duration<float, std::milli>(cpuEnd - cpuBeg).count() << " ms\n";

    assert(cpuMax == gpuResult);
    std::cout << "Check: " << int(cpuMax == gpuResult) << '\n';
    err = hipFree(gpuMem);
    err = hipModuleUnload(module);
    return 0;
}
