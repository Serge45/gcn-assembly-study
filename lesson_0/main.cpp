#include <iostream>
#include <string>
#include <cassert>
#include <limits>
#include <hip/hip_runtime.h>
#include <hip/device_functions.h>
#include <hip/hip_ext.h>

int main(int argc, char **argv) {
    hipDevice_t dev{};
    auto err = hipDeviceGet(&dev, 0);
    hipModule_t module;
    assert(argc == 2);
    std::string coPath(argv[1]);
    err = hipModuleLoad(&module, coPath.c_str());
    assert(err == HIP_SUCCESS);
    hipFunction_t gpuFunc;
    err = hipModuleGetFunction(&gpuFunc, module, "set_func");
    assert(err == HIP_SUCCESS);
    float *gpuMem{};
    err = hipMalloc(&gpuMem, sizeof(float));
    err = hipMemset(gpuMem, 0, sizeof(float));
    std::size_t argSize = sizeof(gpuMem);

    void *kernelArgs[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER,
        reinterpret_cast<void *>(&gpuMem),
        HIP_LAUNCH_PARAM_BUFFER_SIZE,
        reinterpret_cast<void *>(&argSize),
        HIP_LAUNCH_PARAM_END
    };

    err = hipExtModuleLaunchKernel(gpuFunc, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, kernelArgs);
    err = hipDeviceSynchronize();
    float cpuMem{};
    err = hipMemcpyDtoH(&cpuMem, gpuMem, sizeof(float));
    assert(std::abs(cpuMem - 55.66f) < std::numeric_limits<float>::epsilon());
    err = hipFree(gpuMem);
    err = hipModuleUnload(module);
    return 0;
}