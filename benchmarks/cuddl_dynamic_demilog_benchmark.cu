#include <cuddl/cuddl.cuh>

#include <CLI/CLI.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

#include "result_json.hpp"

namespace {

using clock_type = std::chrono::steady_clock;
constexpr size_t bucket_count = 2048;

std::vector<uint64_t> make_inputs(size_t count, int64_t trial) {
    std::vector<uint64_t> values;
    values.reserve(count);
    auto const mask = (1ULL << 50) - 1ULL;
    auto const offset = static_cast<uint64_t>(trial) * 0x9e3779b97f4a7c15ULL;
    for (size_t i = 0; i < count; ++i) {
        values.push_back(cuddl::detail::splitmix64(42 + i + offset) & mask);
    }
    return values;
}

double run(std::vector<uint64_t> const& host, double& estimate) {
    uint64_t* device = nullptr;
    CUDDL_CUDA_CALL(cudaMalloc(&device, host.size() * sizeof(uint64_t)));
    CUDDL_CUDA_CALL(
        cudaMemcpy(device, host.data(), host.size() * sizeof(uint64_t), cudaMemcpyHostToDevice)
    );
    cuddl::sketch<25, bucket_count> sketch;
    auto const start = clock_type::now();
    CUDDL_UNWRAP(sketch.add({device, host.size()}));
    estimate = CUDDL_UNWRAP(sketch.cardinality());
    auto const seconds = std::chrono::duration<double>(clock_type::now() - start).count();
    CUDDL_CUDA_CALL(cudaFree(device));
    return seconds;
}

}  // namespace

int main(int argc, char** argv) {
    CLI::App app{"Direct cuDDL construction and cardinality benchmark"};
    size_t count = 1U << 20;
    int warmup = 3;
    int runs = 10;
    std::string output_path;
    app.add_option("--count", count)->check(CLI::PositiveNumber);
    app.add_option("--warmup", warmup)->check(CLI::NonNegativeNumber);
    app.add_option("--runs", runs)->check(CLI::PositiveNumber);
    app.add_option("--output", output_path, "Output JSON path")->required();
    CLI11_PARSE(app, argc, argv);

    for (int i = 0; i < warmup; ++i) {
        auto const host = make_inputs(count, -1 - i);
        double estimate = 0;
        run(host, estimate);
    }
    json measurements = json::array();
    for (int i = 0; i < runs; ++i) {
        auto const host = make_inputs(count, i);
        double estimate = 0;
        auto const seconds = run(host, estimate);
        auto const bounded_estimate = std::min(estimate, static_cast<double>(count));
        measurements.push_back({
            {"implementation", {{"name", "cuddl"}}},
            {"case", {{"count", count}, {"buckets", bucket_count}, {"trial", i}}},
            {"metrics",
             {
                 {"seconds", seconds},
                 {"adds_per_second", count / seconds},
                 {"estimate", estimate},
                 {"bounded_estimate", bounded_estimate},
             }},
            {"timings",
             {{"construction",
               {
                   {"samples", 1},
                   {"median_ms", seconds * 1'000.0},
                   {"source", "wall_clock"},
               }}}},
        });
    }
    write_benchmark_result(
        output_path,
        make_benchmark_result(
            "DynamicDemiLog construction and cardinality",
            "dynamic_demilog",
            "end_to_end",
            std::move(measurements)
        )
    );
}
