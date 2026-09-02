#include <CLI/CLI.hpp>

#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

#include <cstdint>
#include <iostream>

#include <cuddl/cuddl.cuh>

int main(int argc, char** argv) {
    uint64_t count = 1ULL << 27;
    uint64_t seed = 42;

    CLI::App app{"cuDDL kernel profiling driver"};
    app.add_option("--count", count, "Number of packed k-mers")->default_val(count);
    app.add_option("--seed", seed, "Input seed")->default_val(seed);
    CLI11_PARSE(app, argc, argv);

    thrust::device_vector<uint64_t> input(count);
    auto const mask = (1ULL << 50) - 1ULL;
    thrust::transform(
        thrust::counting_iterator<uint64_t>(0),
        thrust::counting_iterator<uint64_t>(count),
        input.begin(),
        [seed, mask] __device__(uint64_t i) { return cuddl::detail::splitmix64(seed + i) & mask; }
    );

    using sketch_type = cuddl::sketch<25, 2048>;
    sketch_type left;
    sketch_type right;
    auto const middle = count / 2;

    CUDDL_UNWRAP(left.clear());
    CUDDL_UNWRAP(left.add(input));
    CUDDL_UNWRAP(right.clear_async());
    CUDDL_UNWRAP(right.add_async(
        cuddl::device_span<uint64_t const>{
            thrust::raw_pointer_cast(input.data()) + middle, input.size() - middle
        }
    ));
    CUDDL_CUDA_CALL(cudaDeviceSynchronize());

    CUDDL_UNWRAP(left.clear_async());
    CUDDL_UNWRAP(left.add_async({thrust::raw_pointer_cast(input.data()), input.size()}));

    thrust::device_vector<cuddl::pairwise_summary> summaries(3);
    auto* summary_output = thrust::raw_pointer_cast(summaries.data());
    CUDDL_UNWRAP(left.summary_async(right, summary_output[0]));
    CUDDL_UNWRAP(left.summary_async<true>(right, summary_output[1]));
    CUDDL_UNWRAP(left.compare_async(right, summary_output[2]));

    thrust::device_vector<uint64_t> empty(1);
    thrust::device_vector<double> estimate(1);
    CUDDL_UNWRAP(left.cardinality_async(
        thrust::raw_pointer_cast(empty.data()), thrust::raw_pointer_cast(estimate.data())
    ));
    thrust::device_vector<uint16_t> counts(left.bucket_count());
    thrust::device_vector<uint32_t> saturated(1);
    CUDDL_UNWRAP(left.winner_counts_async(
        thrust::raw_pointer_cast(counts.data()), thrust::raw_pointer_cast(saturated.data())
    ));
    CUDDL_CUDA_CALL(cudaDeviceSynchronize());

    auto const summary = CUDDL_UNWRAP(left.summary(right));
    auto const reverse = CUDDL_UNWRAP(right.compare(left));
    auto const cardinality = CUDDL_UNWRAP(left.cardinality());
    auto const winner_counts = CUDDL_UNWRAP(left.winner_counts());

    auto const wkid = left.wkid(summary);
    auto const ani = left.ani(summary);
    auto const containment = right.containment(reverse);
    auto const completeness = right.completeness(reverse);

    std::cout << "k: " << left.kmer_length() << '\n'
              << "Buckets: " << left.bucket_count() << '\n'
              << "Left cardinality: " << cardinality << '\n'
              << "Left/right WKID: " << wkid.value_or(0.0) << '\n'
              << "Left/right ANI: " << ani.value_or(0.0) << '\n'
              << "Right containment in left: " << containment.value_or(0.0) << '\n'
              << "Left content present in right: " << completeness.value_or(0.0) << '\n'
              << "Left saturated: " << winner_counts.second << '\n';
}
