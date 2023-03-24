#include <cstring>
#include <iostream>
#include <string>
#include <cassert>
#include <limits>
#include <vector>
#include <random>
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
    err = hipModuleGetFunction(&gpuFunc, module, "vector_add");
    assert(err == HIP_SUCCESS);
    std::uint32_t numElements = 64;
    float *a{};
    float *b{};
    float *c{};
    std::vector<float> cpuA(numElements);
    std::vector<float> cpuB(numElements);

    for (std::size_t i = 0; i < numElements; ++i) {
        cpuA[i] = dist(gen);
        cpuB[i] = dist(gen);
    }

    err = hipMalloc(&a, sizeof(float) * numElements);
    err = hipMalloc(&b, sizeof(float) * numElements);
    err = hipMalloc(&c, sizeof(float) * numElements);
    err = hipMemcpyHtoD(a, cpuA.data(), cpuA.size() * sizeof(float));
    err = hipMemcpyHtoD(b, cpuB.data(), cpuB.size() * sizeof(float));
    err = hipMemset(c, 0, sizeof(float) * numElements);
    std::size_t argSize = sizeof(float *) * 3 + sizeof(numElements);
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
    std::memcpy(args.data() + cursor, &c, sizeof(c));
    cursor += sizeof(c);
    std::memcpy(args.data() + cursor, &numElements, sizeof(numElements));
    cursor += sizeof(numElements);

    void *kernelArgs[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER,
        reinterpret_cast<void *>(args.data()),
        HIP_LAUNCH_PARAM_BUFFER_SIZE,
        reinterpret_cast<void *>(&argSize),
        HIP_LAUNCH_PARAM_END
    };

    err = hipExtModuleLaunchKernel(gpuFunc, numElements, 1, 1, numElements, 1, 1, 0, nullptr, nullptr, kernelArgs);
    err = hipDeviceSynchronize();
    std::vector<float> cpuC(numElements, 0.f);
    std::vector<float> ans(numElements, 0.f);

    for (std::size_t i = 0; i < numElements; ++i) {
        ans[i] = cpuA[i] + cpuB[i];
    }

    err = hipMemcpyDtoH(cpuC.data(), c, sizeof(float) * numElements);
    assert(cpuC == ans);
    err = hipFree(a);
    err = hipFree(b);
    err = hipFree(c);
    err = hipModuleUnload(module);
    return 0;
}