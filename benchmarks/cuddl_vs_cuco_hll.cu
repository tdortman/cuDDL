// nvbench comparison of cuDDL construction/cardinality against cuCollections HyperLogLog.
//
// Both are GPU cardinality estimators. cuDDL uses a DynamicDemiLog register sketch
// (`BucketCount` packed uint32 registers); cuCollections HyperLogLog uses `2^precision` byte
// registers. This benchmark sweeps item counts and times each library's `add` throughput at a
// matched register count (2048), as context for the acceptance gate. It also reports each
// estimate's relative error against the generating truth for the cardinality comparison.

#include <cuco/hyperloglog.cuh>
#include <cuddl/cuddl.cuh>

#include <nvbench/nvbench.cuh>

#include <cuda/std/cstdint>

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

constexpr size_t k_bucket_count = 2048;
constexpr size_t k_hll_precision = 11;  // 2^11 = 2048 HLL registers

/// @brief Generates deterministic packed k-mers and their distinct count via library hashing.
std::vector<uint64_t> make_inputs(size_t count, size_t& distinct) {
    std::vector<uint64_t> out;
    out.reserve(count);
    auto const mask = (1ULL << 50) - 1ULL;  // k=25 packed space
    for (size_t i = 0; i < count; ++i) {
        out.push_back(cuddl::detail::splitmix64(42 + i) & mask);
    }
    auto seen = out;
    std::sort(seen.begin(), seen.end());
    distinct = static_cast<size_t>(std::unique(seen.begin(), seen.end()) - seen.begin());
    return out;
}

}  // namespace

// cuDDL construction

void cuddl_construction(nvbench::state& state) {
    auto const count = static_cast<size_t>(state.get_int64("Iters"));
    size_t distinct = 0;
    auto const host = make_inputs(count, distinct);

    uint64_t* d_input = nullptr;
    cudaMalloc(&d_input, count * sizeof(uint64_t));
    cudaMemcpy(d_input, host.data(), count * sizeof(uint64_t), cudaMemcpyHostToDevice);

    cuddl::sketch<25, k_bucket_count> sketch;
    state.add_element_count(count, "Iters");
    state.add_global_memory_reads<uint64_t>(count);

    // Reset the sketch each trial outside the timed add so throughput reflects only construction.
    state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
        auto const stream = cuda::stream_ref{launch.get_stream()};
        cuddl::require_void(sketch.clear_async(stream));

        timer.start();
        cuddl::require_void(sketch.add_async(d_input, d_input + count, stream));
        timer.stop();
    });

    // Report the cardinality estimate accuracy as a side-channel the plotter can read.
    auto const estimate = sketch.cardinality().value();
    auto const rel_err =
        std::fabs(estimate - static_cast<double>(distinct)) / static_cast<double>(distinct);
    std::printf(
        "cuddl,count=%zu,distinct=%zu,estimate=%.1f,rel_error=%.6f\n",
        count,
        distinct,
        estimate,
        rel_err
    );

    cudaFree(d_input);
}
NVBENCH_BENCH(cuddl_construction)
    .add_int64_power_of_two_axis("Iters", {20, 21, 22, 23, 24, 25, 26, 27, 28});

// cuCollections HyperLogLog construction

void cuco_hll_construction(nvbench::state& state) {
    auto const count = static_cast<size_t>(state.get_int64("Iters"));
    size_t distinct = 0;
    auto const host = make_inputs(count, distinct);

    uint64_t* d_input = nullptr;
    cudaMalloc(&d_input, count * sizeof(uint64_t));
    cudaMemcpy(d_input, host.data(), count * sizeof(uint64_t), cudaMemcpyHostToDevice);

    cuco::hyperloglog<uint64_t> hll(cuco::precision{k_hll_precision});
    state.add_element_count(count, "Iters");
    state.add_global_memory_reads<uint64_t>(count);

    state.exec(nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
        auto const stream = cuda::stream_ref{launch.get_stream()};
        hll.clear_async(stream);
        timer.start();
        hll.add_async(d_input, d_input + count, stream);
        timer.stop();
    });

    auto const estimate = static_cast<double>(hll.estimate());
    auto const rel_err =
        std::fabs(estimate - static_cast<double>(distinct)) / static_cast<double>(distinct);
    std::printf(
        "cuco_hll,count=%zu,distinct=%zu,estimate=%.1f,rel_error=%.6f\n",
        count,
        distinct,
        estimate,
        rel_err
    );

    cudaFree(d_input);
}

NVBENCH_BENCH(cuco_hll_construction)
    .add_int64_power_of_two_axis("Iters", {20, 21, 22, 23, 24, 25, 26, 27, 28});
