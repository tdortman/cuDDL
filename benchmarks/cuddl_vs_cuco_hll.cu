// nvbench comparison of cuDDL against cuCollections HyperLogLog.
//
// Both are GPU cardinality estimators. cuDDL uses a DynamicDemiLog register sketch
// (`BucketCount` packed uint32 registers); cuCollections HyperLogLog uses `2^precision` byte
// registers. Construction and cardinality estimation are timed separately at a matched register
// count (2048).

#include <cuco/hyperloglog.cuh>
#include <cuddl/cuddl.cuh>

#include <nvbench/nvbench.cuh>

#include <cuda/std/cstdint>

#include <vector>

namespace {

constexpr size_t k_bucket_count = 2048;
constexpr size_t k_hll_precision = 11;  // 2^11 = 2048 HLL registers

template <typename T>
__forceinline__ void do_not_optimise(T& value) {
    asm volatile("" : "+m,r"(value) : : "memory");
}

/// @brief Generates deterministic packed k-mers.
std::vector<uint64_t> make_inputs(size_t count) {
    std::vector<uint64_t> out;
    out.reserve(count);
    auto const mask = (1ULL << 50) - 1ULL;  // k=25 packed space
    for (size_t i = 0; i < count; ++i) {
        out.push_back(cuddl::detail::splitmix64(42 + i) & mask);
    }
    return out;
}

}  // namespace

// cuDDL construction

void cuddl_construction(nvbench::state& state) {
    auto const count = static_cast<size_t>(state.get_int64("Iters"));
    auto const host = make_inputs(count);

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

    cudaFree(d_input);
}
NVBENCH_BENCH(cuddl_construction)
    .add_int64_power_of_two_axis("Iters", {20, 21, 22, 23, 24, 25, 26, 27, 28});

void cuddl_cardinality(nvbench::state& state) {
    auto const count = static_cast<size_t>(state.get_int64("Iters"));
    auto const host = make_inputs(count);

    cuddl::sketch<25, k_bucket_count> sketch;
    cuddl::require_void(sketch.add(host.data(), host.data() + host.size()));

    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
        auto estimate = CUDDL_UNWRAP(sketch.cardinality());
        do_not_optimise(estimate);
    });
}
NVBENCH_BENCH(cuddl_cardinality)
    .add_int64_power_of_two_axis("Iters", {20, 21, 22, 23, 24, 25, 26, 27, 28});

// cuCollections HyperLogLog construction

void cuco_hll_construction(nvbench::state& state) {
    auto const count = static_cast<size_t>(state.get_int64("Iters"));
    auto const host = make_inputs(count);

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

    cudaFree(d_input);
}

NVBENCH_BENCH(cuco_hll_construction)
    .add_int64_power_of_two_axis("Iters", {20, 21, 22, 23, 24, 25, 26, 27, 28});

void cuco_hll_cardinality(nvbench::state& state) {
    auto const count = static_cast<size_t>(state.get_int64("Iters"));
    auto const host = make_inputs(count);

    cuco::hyperloglog<uint64_t> hll(cuco::precision{k_hll_precision});
    hll.add(host.begin(), host.end());

    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
        auto estimate = hll.estimate();
        do_not_optimise(estimate);
    });
}

NVBENCH_BENCH(cuco_hll_cardinality)
    .add_int64_power_of_two_axis("Iters", {20, 21, 22, 23, 24, 25, 26, 27, 28});
