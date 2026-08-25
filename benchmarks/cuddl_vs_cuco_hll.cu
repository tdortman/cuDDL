// nvbench comparison of cuDDL against cuCollections HyperLogLog.
//
// Both are GPU cardinality estimators. cuDDL uses a DynamicDemiLog register sketch
// (`BucketCount` packed uint32 registers); cuCollections HyperLogLog also uses uint32
// registers. Construction and cardinality estimation are timed separately at a matched register
// count (2048).

#include <cuco/hyperloglog.cuh>
#include <cuddl/cuddl.cuh>

#include <nvbench/nvbench.cuh>

#include <cuda/std/cstdint>

#include <vector>
#include "common.cuh"

namespace {

constexpr size_t k_bucket_count = 2048;
constexpr size_t k_hll_precision = 11;  // 2^11 = 2048 HLL registers

std::vector<nvbench::int64_t> const construction_powers{20, 21, 22, 23, 24, 25, 26, 27, 28};
std::vector<nvbench::int64_t> const cardinality_powers{
    8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
};

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
        CUDDL_UNWRAP(sketch.clear_async(stream));

        timer.start();
        CUDDL_UNWRAP(sketch.add_async({d_input, count}, stream));
        timer.stop();
    });
    add_time_stats(state);

    cudaFree(d_input);
}

void cuddl_cardinality(nvbench::state& state) {
    auto const count = static_cast<size_t>(state.get_int64("Iters"));
    auto const host = make_inputs(count);
    uint64_t* d_input = nullptr;
    cudaMalloc(&d_input, count * sizeof(uint64_t));
    cudaMemcpy(d_input, host.data(), count * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cuddl::sketch<25, k_bucket_count> sketch;
    CUDDL_UNWRAP(sketch.add({d_input, count}));

    // Host-visible API path: kernel launch, stream synchronisation, and device-to-host copy.
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
        auto estimate = CUDDL_UNWRAP(sketch.cardinality(cuda::stream_ref{launch.get_stream()}));
        do_not_optimise(estimate);
    });
    add_time_stats(state);
    add_value(state, "Exact", static_cast<double>(count));
    add_value(state, "Estimate", CUDDL_UNWRAP(sketch.cardinality()));
    cudaFree(d_input);
}

template <cuddl::detail::hybrid_variant Variant>
void cuddl_hybrid_cardinality(nvbench::state& state) {
    auto const count = static_cast<size_t>(state.get_int64("Iters"));
    auto const host = make_inputs(count);
    uint64_t* d_input = nullptr;
    cudaMalloc(&d_input, count * sizeof(uint64_t));
    cudaMemcpy(d_input, host.data(), count * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cuddl::sketch<25, k_bucket_count> sketch;
    CUDDL_UNWRAP(sketch.add({d_input, count}));

    // Host-visible API path: kernel launch, stream synchronisation, and device-to-host copy.
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
        auto const hybrids =
            CUDDL_UNWRAP(sketch.hybrid_cardinality(cuda::stream_ref{launch.get_stream()}));
        double estimate = 0.0;
        if constexpr (Variant == cuddl::detail::hybrid_variant::bbtools) {
            estimate = hybrids.bbtools;
        } else {
            estimate = hybrids.paper;
        }
        do_not_optimise(estimate);
    });
    add_time_stats(state);
    cudaFree(d_input);
}

void cuddl_bbtools_cardinality(nvbench::state& state) {
    cuddl_hybrid_cardinality<cuddl::detail::hybrid_variant::bbtools>(state);
}

void cuddl_paper_cardinality(nvbench::state& state) {
    cuddl_hybrid_cardinality<cuddl::detail::hybrid_variant::paper>(state);
}

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
    add_time_stats(state);

    cudaFree(d_input);
}

void cuco_hll_cardinality(nvbench::state& state) {
    auto const count = static_cast<size_t>(state.get_int64("Iters"));
    auto const host = make_inputs(count);
    uint64_t* d_input = nullptr;
    cudaMalloc(&d_input, count * sizeof(uint64_t));
    cudaMemcpy(d_input, host.data(), count * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cuco::hyperloglog<uint64_t> hll(cuco::precision{k_hll_precision});
    hll.add(d_input, d_input + count);

    // Public API path: cuco copies the registers to the host, synchronises, and finalises there.
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
        auto estimate = hll.estimate(cuda::stream_ref{launch.get_stream()});
        do_not_optimise(estimate);
    });
    add_time_stats(state);
    add_value(state, "Exact", static_cast<double>(count));
    add_value(state, "Estimate", static_cast<double>(hll.estimate()));
    cudaFree(d_input);
}

void cuddl_similarity(nvbench::state& state) {
    auto const count = static_cast<size_t>(state.get_int64("Iters"));
    auto const shared = count / 2;
    auto const left = make_inputs(count);
    auto right = make_inputs(shared);
    auto const unique = make_inputs(count + shared);
    right.insert(right.end(), unique.begin() + count, unique.end());

    uint64_t* d_left = nullptr;
    uint64_t* d_right = nullptr;
    cudaMalloc(&d_left, left.size() * sizeof(uint64_t));
    cudaMalloc(&d_right, right.size() * sizeof(uint64_t));
    cudaMemcpy(d_left, left.data(), left.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_right, right.data(), right.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cuddl::sketch<25, k_bucket_count> left_sketch;
    cuddl::sketch<25, k_bucket_count> right_sketch;
    CUDDL_UNWRAP(left_sketch.add({d_left, left.size()}));
    CUDDL_UNWRAP(right_sketch.add({d_right, right.size()}));

    auto const summary = CUDDL_UNWRAP(left_sketch.compare(right_sketch.ref()));
    auto similarity = *left_sketch.ref().containment(summary);
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
        auto timed_summary = CUDDL_UNWRAP(
            left_sketch.compare(right_sketch.ref(), cuda::stream_ref{launch.get_stream()})
        );
        auto equal = timed_summary.counts.equal;
        do_not_optimise(equal);
    });
    add_time_stats(state);
    add_value(state, "Exact Similarity", 0.5);
    add_value(state, "Similarity", similarity);
    cudaFree(d_right);
    cudaFree(d_left);
}

void cuco_hll_similarity(nvbench::state& state) {
    auto const count = static_cast<size_t>(state.get_int64("Iters"));
    auto const shared = count / 2;
    auto const left = make_inputs(count);
    auto right = make_inputs(shared);
    auto const unique = make_inputs(count + shared);
    right.insert(right.end(), unique.begin() + count, unique.end());

    uint64_t* d_left = nullptr;
    uint64_t* d_right = nullptr;
    cudaMalloc(&d_left, left.size() * sizeof(uint64_t));
    cudaMalloc(&d_right, right.size() * sizeof(uint64_t));
    cudaMemcpy(d_left, left.data(), left.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_right, right.data(), right.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cuco::hyperloglog<uint64_t> left_hll(cuco::precision{k_hll_precision});
    cuco::hyperloglog<uint64_t> right_hll(cuco::precision{k_hll_precision});
    left_hll.add(d_left, d_left + left.size());
    right_hll.add(d_right, d_right + right.size());
    cuco::hyperloglog<uint64_t> union_hll(cuco::precision{k_hll_precision});

    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
        auto const stream = cuda::stream_ref{launch.get_stream()};
        auto const left_estimate = static_cast<double>(left_hll.estimate(stream));
        auto const right_estimate = static_cast<double>(right_hll.estimate(stream));
        union_hll.clear(stream);
        union_hll.merge(left_hll, stream);
        union_hll.merge(right_hll, stream);
        auto const union_estimate = static_cast<double>(union_hll.estimate(stream));
        auto similarity = (left_estimate + right_estimate - union_estimate) / left_estimate;
        do_not_optimise(similarity);
    });
    add_time_stats(state);
    auto const left_estimate = static_cast<double>(left_hll.estimate());
    auto const right_estimate = static_cast<double>(right_hll.estimate());
    union_hll.clear();
    union_hll.merge(left_hll);
    union_hll.merge(right_hll);
    auto const union_estimate = static_cast<double>(union_hll.estimate());
    auto const similarity = (left_estimate + right_estimate - union_estimate) / left_estimate;
    add_value(state, "Exact Similarity", 0.5);
    add_value(state, "Similarity", similarity);
    cudaFree(d_right);
    cudaFree(d_left);
}

NVBENCH_BENCH(cuddl_construction).add_int64_power_of_two_axis("Iters", construction_powers);
NVBENCH_BENCH(cuco_hll_construction).add_int64_power_of_two_axis("Iters", construction_powers);

NVBENCH_BENCH(cuddl_similarity).add_int64_power_of_two_axis("Iters", construction_powers);
NVBENCH_BENCH(cuco_hll_similarity).add_int64_power_of_two_axis("Iters", construction_powers);

NVBENCH_BENCH(cuddl_cardinality).add_int64_power_of_two_axis("Iters", cardinality_powers);
NVBENCH_BENCH(cuddl_bbtools_cardinality).add_int64_power_of_two_axis("Iters", cardinality_powers);
NVBENCH_BENCH(cuddl_paper_cardinality).add_int64_power_of_two_axis("Iters", cardinality_powers);
NVBENCH_BENCH(cuco_hll_cardinality).add_int64_power_of_two_axis("Iters", cardinality_powers);
