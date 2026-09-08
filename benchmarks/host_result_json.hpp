#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <vector>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <istream>
#include <stdexcept>
#include <string>
#include <utility>

#include <sys/utsname.h>
#include <unistd.h>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

// Consecutive timestamps partition one application run, including local teardown.
template <typename Function>
json measure_pipeline(int samples, int warmups, Function&& function) {
    using clock = std::chrono::steady_clock;
    constexpr std::array names{
        "prepare_wall", "query_output_wall", "teardown_wall", "end_to_end_wall"
    };
    std::array<std::vector<double>, names.size()> times;
    for (auto& values : times) {
        values.reserve(samples);
    }
    for (int sample = -warmups; sample < samples; ++sample) {
        std::array<clock::time_point, 4> stamps;
        size_t next = 1;
        stamps[0] = clock::now();
        function([&] { stamps.at(next++) = clock::now(); });
        stamps[3] = clock::now();
        if (next != 3) {
            throw std::runtime_error("pipeline must record preparation and output boundaries");
        }
        if (sample < 0) {
            continue;
        }
        for (size_t i = 0; i < 3; ++i) {
            times[i].push_back(
                std::chrono::duration<double, std::milli>(stamps[i + 1] - stamps[i]).count()
            );
        }
        times[3].push_back(
            std::chrono::duration<double, std::milli>(stamps[3] - stamps[0]).count()
        );
    }
    json result;
    for (size_t i = 0; i < names.size(); ++i) {
        auto& values = times[i];
        std::sort(values.begin(), values.end());
        auto middle = values.size() / 2;
        result[names[i]] = {
            {"samples", values.size()},
            {"median_ms",
             values.size() % 2 ? values[middle] : (values[middle - 1] + values[middle]) / 2},
            {"min_ms", values.front()},
            {"max_ms", values.back()},
            {"source", "steady_clock_cpu_wall"},
        };
    }
    return result;
}

// Provenance is collected outside every timing interval.
inline std::string command_output(std::string const& command) {
    auto* pipe = popen(command.c_str(), "r");
    if (pipe == nullptr) {
        throw std::runtime_error("cannot start provenance command");
    }
    std::string output;
    char buffer[4096];
    while (std::fgets(buffer, sizeof(buffer), pipe)) {
        output += buffer;
    }
    if (pclose(pipe) != 0) {
        throw std::runtime_error("provenance command failed: " + command);
    }
    while (!output.empty() && (output.back() == '\n' || output.back() == '\r')) {
        output.pop_back();
    }
    return output;
}
inline std::string quote(std::string const& value) {
    std::string output = "'";
    for (char c : value) {
        output += c == '\'' ? "'\\''" : std::string(1, c);
    }
    return output + "'";
}

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
