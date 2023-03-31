#include <algorithm>
#include <cassert>
#include <iostream>
#include <limits>
#include <string>
#include <numeric>
#include <hip/hip_runtime.h>
#include <hip/device_functions.h>
#include <hip/hip_ext.h>
#include "../Utils/KernelArguments.hpp"
#include "../Utils/BufferUtils.hpp"

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
    float *maxMem{};
    std::vector<float> cpuMem(numElements, 0);
    randomize(begin(cpuMem), end(cpuMem));
    err = hipMalloc(&gpuMem, sizeof(float) * numElements);
    err = hipMalloc(&maxMem, sizeof(float));
    err = hipMemcpyHtoD(gpuMem, cpuMem.data(), cpuMem.size() * sizeof(float));
    KernelArguments kArgs;
    kArgs.append(gpuMem);
    kArgs.append(numElements);
    kArgs.append(maxMem);
    kArgs.applyAlignment();
    std::size_t argSize = kArgs.size();

    void *kernelArgs[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER,
        reinterpret_cast<void *>(kArgs.buffer()),
        HIP_LAUNCH_PARAM_BUFFER_SIZE,
        reinterpret_cast<void *>(&argSize),
        HIP_LAUNCH_PARAM_END};

    err = hipExtModuleLaunchKernel(gpuFunc, 256, 1, 1, 256, 1, 1, 256 * sizeof(float), nullptr, nullptr, kernelArgs);
    err = hipDeviceSynchronize();
    float gpuResult{};
    err = hipMemcpyDtoH(&gpuResult, maxMem, sizeof(float));
    auto cpuMax = *std::max_element(begin(cpuMem), end(cpuMem));
    assert(cpuMax == gpuResult);
    err = hipFree(gpuMem);
    err = hipFree(maxMem);
    err = hipModuleUnload(module);
    return 0;
}
