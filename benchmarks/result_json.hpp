#pragma once

#include <cuda_runtime_api.h>
#include <cuda/devices>
#include <cuda/stream>

#include "host_result_json.hpp"

inline json benchmark_system() {
    auto system = benchmark_host_system();
    system["cuda_compile_version"] = CUDART_VERSION;

    int version{};
    if (cudaRuntimeGetVersion(&version) == cudaSuccess) {
        system["cuda_runtime_version"] = version;
    }
    if (cudaDriverGetVersion(&version) == cudaSuccess) {
        system["cuda_driver_version"] = version;
    }

    int device_ordinal{};
    if (cudaGetDevice(&device_ordinal) != cudaSuccess) {
        return system;
    }
    try {
        auto const device = cuda::devices[device_ordinal];
        auto const name = device.name();
        auto const major = device.attribute(cuda::device_attributes::compute_capability_major);
        auto const minor = device.attribute(cuda::device_attributes::compute_capability_minor);
        system.update({
            {"gpu", std::string(name.data(), name.size())},
            {"compute_capability", std::to_string(major) + "." + std::to_string(minor)},
            {"sm_count", device.attribute(cuda::device_attributes::multiprocessor_count)},
            {"gpu_ram_bytes", device.attribute(cuda::device_attributes::total_global_memory)},
        });
    } catch (cuda::cuda_error const&) {
        // CPU-only benchmarks still report host metadata when CUDA is unavailable.
    }
    return system;
}

inline json make_benchmark_result(
    std::string name,
    std::string operation,
    std::string scope,
    json measurements,
    json datasets = json::object()
) {
    return {
        {"schema", "cuddl-benchmark/v1"},
        {"name", std::move(name)},
        {"operation", std::move(operation)},
        {"scope", std::move(scope)},
        {"datasets", std::move(datasets)},
        {"system", benchmark_system()},
        {"measurements", std::move(measurements)},
    };
}

inline void write_benchmark_result(std::string const& path, json const& result) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("cannot open JSON output: " + path);
    }
    output << result.dump(2) << '\n';
}
