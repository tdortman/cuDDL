#pragma once

#include <cuda_runtime_api.h>

#include <cstdint>
#include <fstream>
#include <istream>
#include <stdexcept>
#include <string>
#include <utility>

#include <sys/utsname.h>
#include <unistd.h>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

inline std::string cpu_model(std::istream& input, std::string fallback) {
    std::string implementer;
    std::string part;
    for (std::string line; std::getline(input, line);) {
        auto const colon = line.find(':');
        if (colon == std::string::npos) {
            continue;
        }
        auto const key_end = line.find_last_not_of(" \t", colon - 1);
        auto const first = line.find_first_not_of(" \t", colon + 1);
        if (key_end == std::string::npos || first == std::string::npos) {
            continue;
        }
        auto const key = line.substr(0, key_end + 1);
        auto const value = line.substr(first);
        if (key == "model name" || key == "Processor" || key == "Hardware" || key == "cpu" ||
            key == "uarch") {
            return value;
        }
        if (key == "CPU implementer") {
            implementer = value;
        } else if (key == "CPU part") {
            part = value;
        }
    }
    if (!implementer.empty() || !part.empty()) {
        return "ARM implementer " + (implementer.empty() ? "unknown" : implementer) + " part " +
               (part.empty() ? "unknown" : part);
    }
    return fallback;
}

inline std::string cpu_model(std::string fallback) {
    std::ifstream input("/proc/cpuinfo");
    return cpu_model(input, std::move(fallback));
}

inline json benchmark_host_system() {
    utsname host{};
    if (uname(&host) != 0) {
        throw std::runtime_error("cannot determine operating system information");
    }
    auto const logical_cpu_count = sysconf(_SC_NPROCESSORS_ONLN);
    auto const physical_pages = sysconf(_SC_PHYS_PAGES);
    auto const page_size = sysconf(_SC_PAGESIZE);
    if (logical_cpu_count < 1 || physical_pages < 1 || page_size < 1) {
        throw std::runtime_error("cannot determine host processor or memory information");
    }

    return {
        {"os", host.sysname},
        {"kernel", host.release},
        {"architecture", host.machine},
        {"cpu", cpu_model(host.machine)},
        {"logical_cpu_count", logical_cpu_count},
        {"ram_bytes", static_cast<uint64_t>(physical_pages) * static_cast<uint64_t>(page_size)},
    };
}

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

    int device{};
    cudaDeviceProp properties{};
    if (cudaGetDevice(&device) == cudaSuccess &&
        cudaGetDeviceProperties(&properties, device) == cudaSuccess) {
        system.update({
            {"gpu", properties.name},
            {"compute_capability",
             std::to_string(properties.major) + "." + std::to_string(properties.minor)},
            {"sm_count", properties.multiProcessorCount},
            {"gpu_ram_bytes", properties.totalGlobalMem},
        });
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
