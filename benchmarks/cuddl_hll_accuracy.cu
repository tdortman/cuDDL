#include <CLI/CLI.hpp>
#include <cuco/hyperloglog.cuh>
#include <cuddl/cuddl.cuh>

#include <cuda/std/cstdint>

#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr size_t k_bucket_count = 2048;
constexpr size_t k_hll_precision = 11;
std::vector<uint64_t> make_inputs(size_t count, uint64_t seed) {
    std::vector<uint64_t> inputs;
    inputs.reserve(count);
    auto const mask = (1ULL << 50) - 1ULL;
    auto const multiplier = cuddl::detail::splitmix64(seed) | 1ULL;
    for (size_t i = 0; i < count; ++i) {
        inputs.push_back((multiplier * i + seed) & mask);
    }
    return inputs;
}

void emit(
    std::ofstream& csv,
    char const* estimator,
    size_t count,
    uint32_t trial,
    size_t exact,
    double estimate
) {
    auto const signed_error = (estimate - static_cast<double>(exact)) / static_cast<double>(exact);
    csv << estimator << ',' << count << ',' << trial << ',' << exact << ',' << estimate << ','
        << signed_error << ',' << std::abs(signed_error) << '\n';
}


}  // namespace

int main(int argc, char** argv) {
    try {
        CLI::App app{"Paired cuDDL and cuco HLL cardinality accuracy experiment"};

        std::string csv_path;
        uint32_t trials = 32;
        uint32_t min_power = 8;
        uint32_t max_power = 24;
        uint64_t root_seed = 42;

        app.add_option("--csv", csv_path, "Output CSV path")->required();
        app.add_option("--trials", trials, "Independent trials per cardinality")
            ->check(CLI::PositiveNumber);
        app.add_option("--min-power", min_power, "Smallest base-2 cardinality exponent")
            ->check(CLI::Range(1U, 28U));
        app.add_option("--max-power", max_power, "Largest base-2 cardinality exponent")
            ->check(CLI::Range(1U, 28U));
        app.add_option("--seed", root_seed, "Root seed");

        CLI11_PARSE(app, argc, argv);
        
        if (min_power > max_power) {
            throw std::runtime_error("--min-power must not exceed --max-power");
        }

        std::ofstream csv(csv_path);
        if (!csv) {
            throw std::runtime_error("cannot open CSV output: " + csv_path);
        }
        csv << "Estimator,Items,Trial,Exact,Estimate,Signed Error,Absolute Error\n";

        for (uint32_t power = min_power; power <= max_power; ++power) {
            auto const count = size_t{1} << power;
            for (uint32_t trial = 0; trial < trials; ++trial) {
                auto const seed = cuddl::detail::splitmix64(
                    root_seed ^ (static_cast<uint64_t>(power) << 32U) ^ trial
                );
                auto const host = make_inputs(count, seed);
                auto const exact = count;

                uint64_t* device = nullptr;
                CUDDL_CUDA_CALL(cudaMalloc(&device, count * sizeof(uint64_t)));
                CUDDL_CUDA_CALL(cudaMemcpy(
                    device, host.data(), count * sizeof(uint64_t), cudaMemcpyHostToDevice
                ));

                cuddl::sketch<25, k_bucket_count> cuddl;
                CUDDL_UNWRAP(cuddl.add({device, count}));
                emit(csv, "cuddl", count, trial, exact, CUDDL_UNWRAP(cuddl.cardinality()));
                auto const hybrids = CUDDL_UNWRAP(cuddl.hybrid_cardinality());
                emit(csv, "cuddl_bbtools", count, trial, exact, hybrids.bbtools);
                emit(csv, "cuddl_paper", count, trial, exact, hybrids.paper);

                cuco::hyperloglog<uint64_t> hll(cuco::precision{k_hll_precision});
                hll.add(device, device + count);
                emit(csv, "cuco_hll", count, trial, exact, hll.estimate());

                CUDDL_CUDA_CALL(cudaFree(device));
            }
            std::cerr << "Completed 2^" << power << '\n';
        }
    } catch (std::exception const& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
