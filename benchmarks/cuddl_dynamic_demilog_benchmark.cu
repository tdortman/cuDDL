#include <cuddl/cuddl.cuh>

#include <CLI/CLI.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

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
    CUDDL_UNWRAP(sketch.add(device, device + host.size()));
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
    app.add_option("--count", count)->check(CLI::PositiveNumber);
    app.add_option("--warmup", warmup)->check(CLI::NonNegativeNumber);
    app.add_option("--runs", runs)->check(CLI::PositiveNumber);
    CLI11_PARSE(app, argc, argv);

    for (int i = 0; i < warmup; ++i) {
        auto const host = make_inputs(count, -1 - i);
        double estimate = 0;
        run(host, estimate);
    }
    for (int i = 0; i < runs; ++i) {
        auto const host = make_inputs(count, i);
        double estimate = 0;
        auto const seconds = run(host, estimate);
        std::printf(
            "cuddl,%zu,%zu,%d,%.9f,%.3f,%.0f\n",
            count,
            bucket_count,
            i,
            seconds,
            count / seconds,
            estimate
        );
    }
}
