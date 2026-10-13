// Integer-ALU throughput one GPU sustains under its power limit.
//
// Every thread runs eight independent chains of LOP3 with two IADDs per round, the integer-ALU
// work of the exhaustive search, which is about 80% LOP3, without its memory traffic or stalls.
// Enough warps run per SM that the ALU pipes, not latency, set the rate. Kernels repeat for
// --seconds while NVML samples the SM clock, board power and clock-limit reasons; the first
// --warmup-seconds are left out, so the clock has settled at whatever the power limit allows.
//
// The report gives ALU warp instructions per second over the measured launches, and the SM
// clock that would issue them at --alu-per-clock warp instructions per SM sub-partition per
// clock. LOP3 issues at 0.5 on Ampere GA10x and on Blackwell GB20x, whose ALU Heavy sub-pipe
// runs it. Dividing the rate by the search's ALU warp instructions per pair gives a pairs/s
// ceiling that already accounts for the power limit.

#include <nvml.h>
#include <CLI/CLI.hpp>
#include <cuda/devices>
#include <cuda/stream>
#include <cuddl/cuda_error.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;

/// LOP3 and IADD per round of alu_load.
constexpr uint32_t lop3_per_round = 16;
constexpr uint32_t iadd_per_round = 2;
constexpr uint32_t chains = 8;

__device__ __forceinline__ uint32_t lop3(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t d;
    asm("lop3.b32 %0, %1, %2, %3, 0x96;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

__global__ void alu_load(uint32_t* out, uint32_t rounds) {
    // Chains are independent, and each step reads its own chain and two loop-invariant values,
    // so a warp always has ready instructions and latency never limits the ALU pipe.
    uint32_t a[chains], k[chains];
    _Pragma("unroll")
    for (uint32_t j = 0; j < chains; ++j) {
        a[j] = threadIdx.x * (j + 1U) + blockIdx.x;
        k[j] = threadIdx.x ^ (0x9e3779b9U * (j + 1U));
    }
    _Pragma("unroll 4")
    for (uint32_t round = 0; round < rounds; ++round) {
        _Pragma("unroll")
        for (uint32_t pass = 0; pass < lop3_per_round / chains; ++pass) {
            _Pragma("unroll")
            for (uint32_t j = 0; j < chains; ++j) {
                a[j] = lop3(a[j], k[j], k[(j + pass + 1U) % chains]);
            }
        }
        a[0] += round;
        a[4] += round;
    }
    uint32_t folded = 0;
    _Pragma("unroll")
    for (uint32_t j = 0; j < chains; ++j) folded ^= a[j];
    // Never true for real inputs; keeps the chains from being optimised away.
    if (folded == 0x9e3779b9U) out[0] = folded;
}

void nvml_check(nvmlReturn_t status) {
    if (status != NVML_SUCCESS) throw std::runtime_error(nvmlErrorString(status));
}

nvmlDevice_t nvml_device(cuda::device_ref device) {
    char address[32];
    CUDDL_CUDA_CALL(cudaDeviceGetPCIBusId(address, sizeof(address), device.get()));
    nvmlDevice_t handle;
    nvml_check(nvmlDeviceGetHandleByPciBusId_v2(address, &handle));
    return handle;
}

double median(std::vector<double> values) {
    std::ranges::sort(values);
    return values[values.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
    CLI::App app{"Integer-ALU throughput one GPU sustains under its power limit"};
    int device_index = 0;
    double seconds = 30, warmup_seconds = 10, alu_per_clock = 0.5;
    uint32_t rounds = 200000;
    std::string output;
    app.add_option("--device", device_index, "CUDA device");
    app.add_option("--seconds", seconds, "Total run time")->check(CLI::PositiveNumber);
    app.add_option("--warmup-seconds", warmup_seconds, "Leading time left out of the report");
    app.add_option("--rounds", rounds, "Rounds per thread per launch")->check(CLI::PositiveNumber);
    app.add_option(
        "--alu-per-clock",
        alu_per_clock,
        "ALU warp instructions per SM sub-partition per clock (0.5 for LOP3 on GA10x and GB20x)"
    );
    app.add_option("-o,--output", output, "JSON report destination");
    CLI11_PARSE(app, argc, argv);
    if (warmup_seconds >= seconds) throw std::invalid_argument("warmup must end before the run");

    cuda::device_ref const device = cuda::devices[device_index];
    cuda::stream stream{device};
    nvml_check(nvmlInit_v2());
    auto const nvml = nvml_device(device);
    unsigned limit_mw = 0;
    nvml_check(nvmlDeviceGetEnforcedPowerLimit(nvml, &limit_mw));

    auto const sms =
        static_cast<uint32_t>(device.attribute(cuda::device_attributes::multiprocessor_count));
    constexpr uint32_t threads = 256;
    auto const blocks = sms * 8U;
    uint32_t* out = nullptr;
    CUDDL_CUDA_CALL(cudaMalloc(&out, sizeof(uint32_t)));

    struct sample {
        double at_s, clock_mhz, power_w;
        unsigned long long reasons;
    };
    std::vector<sample> samples;
    std::atomic<bool> running{true};
    auto const start = clock_type::now();
    auto const since_start = [&] {
        return std::chrono::duration<double>(clock_type::now() - start).count();
    };
    std::jthread sampler([&] {
        while (running) {
            unsigned clock = 0, power = 0;
            unsigned long long reasons = 0;
            nvmlDeviceGetClockInfo(nvml, NVML_CLOCK_SM, &clock);
            nvmlDeviceGetPowerUsage(nvml, &power);
            nvmlDeviceGetCurrentClocksEventReasons(nvml, &reasons);
            samples.push_back({since_start(), double(clock), power / 1000.0, reasons});
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    });

    uint64_t measured_launches = 0;
    double measured_s = 0;
    while (since_start() < seconds) {
        auto const launched = since_start();
        alu_load<<<blocks, threads, 0, stream.get()>>>(out, rounds);
        CUDDL_CUDA_CALL(cudaGetLastError());
        stream.sync();
        if (launched >= warmup_seconds) {
            ++measured_launches;
            measured_s += since_start() - launched;
        }
    }
    running = false;
    sampler.join();
    CUDDL_CUDA_CALL(cudaFree(out));
    if (measured_launches == 0) throw std::runtime_error("no launch finished after the warmup");

    std::vector<double> clocks, powers;
    unsigned long long power_capped = 0, measured_samples = 0;
    for (auto const& s : samples) {
        if (s.at_s < warmup_seconds) continue;
        ++measured_samples;
        clocks.push_back(s.clock_mhz);
        powers.push_back(s.power_w);
        power_capped += (s.reasons & nvmlClocksEventReasonSwPowerCap) != 0U;
    }
    auto const warps_per_launch = double(blocks) * threads / 32.0;
    auto const alu_per_launch = warps_per_launch * rounds * (lop3_per_round + iadd_per_round);
    auto const alu_rate = alu_per_launch * measured_launches / measured_s;
    auto const sub_partitions = 4.0 * sms;
    nlohmann::json report{
        {"gpu", std::string(device.name())},
        {"sms", sms},
        {"power_limit_w", limit_mw / 1000.0},
        {"measured_s", measured_s},
        {"launches", measured_launches},
        {"alu_warp_instructions_per_second", alu_rate},
        {"alu_per_clock", alu_per_clock},
        {"equivalent_sm_clock_mhz", alu_rate / (sub_partitions * alu_per_clock) / 1e6},
        {"median_sm_clock_mhz", median(clocks)},
        {"median_power_w", median(powers)},
        {"power_capped_fraction", double(power_capped) / double(measured_samples)},
    };
    std::cout << report.dump(2) << '\n';
    if (!output.empty()) std::ofstream(output) << report.dump(2) << '\n';
}
