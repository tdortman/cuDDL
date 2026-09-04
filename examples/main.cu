#include <CLI/CLI.hpp>

#include <cub/device/device_transform.cuh>
#include <cuda/buffer>
#include <cuda/iterator>

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

    cuda::stream stream{cuda::devices[0]};
    auto input = cuda::make_device_buffer<uint64_t>(stream, stream.device(), count, cuda::no_init);
    auto const mask = (1ULL << 50) - 1ULL;
    CUDDL_CUDA_CALL(
        cub::DeviceTransform::Transform(
            cuda::make_counting_iterator(uint64_t{0}),
            input.data(),
            count,
            [seed, mask] __device__(uint64_t i) -> uint64_t {
                return cuddl::detail::splitmix64(seed + i) & mask;
            },
            stream
        )
    );

    using sketch_type = cuddl::sketch<25, 2048>;
    sketch_type left(stream);
    sketch_type right(stream);
    auto const middle = count / 2;

    CUDDL_UNWRAP(left.clear(stream));
    CUDDL_UNWRAP(left.add(input, stream));
    CUDDL_UNWRAP(right.clear_async(stream));
    CUDDL_UNWRAP(right.add_async(
        cuddl::device_span<uint64_t const>{input.data() + middle, input.size() - middle}, stream
    ));
    stream.sync();

    CUDDL_UNWRAP(left.clear_async(stream));
    CUDDL_UNWRAP(left.add_async({input.data(), input.size()}, stream));

    auto summaries = cuda::make_device_buffer<cuddl::pairwise_summary>(
        stream, stream.device(), 3, cuda::no_init
    );
    auto* summary_output = summaries.data();
    CUDDL_UNWRAP(left.summary_async(right, summary_output[0], stream));
    CUDDL_UNWRAP(left.summary_async<true>(right, summary_output[1], stream));
    CUDDL_UNWRAP(left.compare_async(right, summary_output[2], stream));

    auto empty = cuda::make_device_buffer<uint64_t>(stream, stream.device(), 1, cuda::no_init);
    auto estimate = cuda::make_device_buffer<double>(stream, stream.device(), 1, cuda::no_init);
    CUDDL_UNWRAP(left.cardinality_async(empty.data(), estimate.data(), stream));
    auto counts = cuda::make_device_buffer<uint16_t>(
        stream, stream.device(), left.bucket_count(), cuda::no_init
    );
    auto saturated = cuda::make_device_buffer<uint32_t>(stream, stream.device(), 1, cuda::no_init);
    CUDDL_UNWRAP(left.winner_counts_async(counts.data(), saturated.data(), stream));
    stream.sync();

    auto const summary = CUDDL_UNWRAP(left.summary(right, stream));
    auto const reverse = CUDDL_UNWRAP(right.compare(left, stream));
    auto const cardinality = CUDDL_UNWRAP(left.cardinality(stream));
    auto const winner_counts = CUDDL_UNWRAP(left.winner_counts(stream));

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
