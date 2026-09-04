#include <CLI/CLI.hpp>
#include <cuco/hyperloglog.cuh>
#include <cuddl/cuddl.cuh>

#include <cuda/std/cstdint>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "CLI/CLI.hpp"
#include "result_json.hpp"

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
    json& measurements,
    char const* estimator,
    size_t count,
    uint32_t trial,
    size_t exact,
    double estimate
) {
    auto const signed_error = (estimate - static_cast<double>(exact)) / static_cast<double>(exact);
    measurements.push_back({
        {"implementation", {{"name", estimator}}},
        {"case", {{"phase", "accuracy"}, {"items", count}, {"trial", trial}}},
        {"metrics",
         {
             {"exact", exact},
             {"estimate", estimate},
             {"signed_error", signed_error},
             {"absolute_error", std::abs(signed_error)},
         }},
    });
}

}  // namespace

int main(int argc, char** argv) {
    try {
        CLI::App app{"Paired cuDDL and cuco HLL cardinality accuracy experiment"};

        std::string output_path;
        uint32_t trials = 32;
        uint32_t min_power = 8;
        uint32_t max_power = 24;
        uint64_t root_seed = 42;

        app.add_option("--output", output_path, "Output JSON path")->required();
        app.add_option("--trials", trials, "Independent trials per cardinality")
            ->check(CLI::PositiveNumber);
        app.add_option("--min-power", min_power, "Smallest base-2 cardinality exponent")
            ->check(CLI::PositiveNumber);
        app.add_option("--max-power", max_power, "Largest base-2 cardinality exponent")
            ->check(CLI::PositiveNumber);
        app.add_option("--seed", root_seed, "Root seed");

        CLI11_PARSE(app, argc, argv);

        if (min_power > max_power) {
            throw std::runtime_error("--min-power must not exceed --max-power");
        }

        cuda::stream setup_stream{cuda::devices[0]};
        json measurements = json::array();

        for (uint32_t power = min_power; power <= max_power; ++power) {
            auto const count = size_t{1} << power;
            for (uint32_t trial = 0; trial < trials; ++trial) {
                auto const seed = cuddl::detail::splitmix64(
                    root_seed ^ (static_cast<uint64_t>(power) << 32U) ^ trial
                );
                auto const host = make_inputs(count, seed);
                auto const exact = count;
                auto device_storage =
                    cuda::make_device_buffer<uint64_t>(setup_stream, setup_stream.device(), host);
                auto* device = device_storage.data();
                setup_stream.sync();

                cuddl::sketch<25, k_bucket_count> cuddl(setup_stream);
                CUDDL_UNWRAP(cuddl.add({device, count}, setup_stream));
                emit(
                    measurements,
                    "cuddl",
                    count,
                    trial,
                    exact,
                    CUDDL_UNWRAP(cuddl.cardinality(setup_stream))
                );
                auto const hybrids = CUDDL_UNWRAP(cuddl.hybrid_cardinality(setup_stream));
                emit(measurements, "cuddl_bbtools", count, trial, exact, hybrids.bbtools);
                emit(measurements, "cuddl_paper", count, trial, exact, hybrids.paper);

                cuco::hyperloglog<uint64_t> hll(
                    cuco::precision{k_hll_precision}, {}, {}, setup_stream
                );
                hll.add(device, device + count, setup_stream);
                emit(measurements, "cuco_hll", count, trial, exact, hll.estimate(setup_stream));
            }
            std::cerr << "Completed 2^" << power << '\n';
        }
        write_benchmark_result(
            output_path,
            make_benchmark_result(
                "HLL cardinality accuracy", "hll_comparison", "kernel", std::move(measurements)
            )
        );
    } catch (std::exception const& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
