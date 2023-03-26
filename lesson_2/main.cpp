#include <cassert>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>
#include <hip/hip_runtime.h>
#include <hip/device_functions.h>
#include <hip/hip_ext.h>

int main(int argc, char **argv) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    hipDevice_t dev{};
    auto err = hipDeviceGet(&dev, 0);
    hipModule_t module;
    assert(argc == 2);
    std::string coPath(argv[1]);
    err = hipModuleLoad(&module, coPath.c_str());
    assert(err == HIP_SUCCESS);
    hipFunction_t gpuFunc;
    err = hipModuleGetFunction(&gpuFunc, module, "relu");
    assert(err == HIP_SUCCESS);
    std::uint32_t numElements = 1000;
    float *a{};
    float *b{};
    std::vector<float> cpuA(numElements);

    for (std::size_t i = 0; i < numElements; ++i) {
        cpuA[i] = dist(gen);
    }

    err = hipMalloc(&a, sizeof(float) * numElements);
    err = hipMalloc(&b, sizeof(float) * numElements);
    err = hipMemcpyHtoD(a, cpuA.data(), cpuA.size() * sizeof(float));
    err = hipMemset(b, 0, sizeof(float) * numElements);
    std::size_t argSize = sizeof(float *) * 2 + sizeof(numElements);
    constexpr std::size_t alignment = 8;

    if (argSize % alignment) {
        argSize += alignment - (argSize % alignment);
    }

    std::vector<std::uint8_t> args(argSize, 0);
    std::size_t cursor{};
    std::memcpy(args.data() + cursor, &a, sizeof(a));
    cursor += sizeof(a);
    std::memcpy(args.data() + cursor, &b, sizeof(b));
    cursor += sizeof(b);
    std::memcpy(args.data() + cursor, &numElements, sizeof(numElements));
    cursor += sizeof(numElements);

    void *kernelArgs[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER,
        reinterpret_cast<void *>(args.data()),
        HIP_LAUNCH_PARAM_BUFFER_SIZE,
        reinterpret_cast<void *>(&argSize),
        HIP_LAUNCH_PARAM_END
    };

    constexpr auto wavefrontSize = 64ull;
    constexpr auto numWavesPerWorkgroup = 4ull;
    constexpr auto workgroupSizes = wavefrontSize * numWavesPerWorkgroup;
    const std::size_t numWorkgroups = (numElements / workgroupSizes) + !!(numElements % workgroupSizes);

    err = hipExtModuleLaunchKernel(gpuFunc, numWorkgroups * workgroupSizes, 1, 1, workgroupSizes, 1, 1, 0, nullptr, nullptr, kernelArgs);
    err = hipDeviceSynchronize();
    std::vector<float> cpuB(numElements, 0.f);
    std::vector<float> ans(numElements, 0.f);

    for (std::size_t i = 0; i < numElements; ++i) {
        ans[i] = std::max(cpuA[i], 0.f);
    }

    err = hipMemcpyDtoH(cpuB.data(), b, sizeof(float) * numElements);
    assert(cpuB == ans);
    err = hipFree(a);
    err = hipFree(b);
    err = hipModuleUnload(module);
    return 0;
}