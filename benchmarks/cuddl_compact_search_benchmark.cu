#include <CLI/CLI.hpp>
#include <cuddl/cuddl.cuh>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/memory.h>

#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

constexpr uint32_t k_kmer_length = 25;
constexpr size_t k_bucket_count = 2048;

uint16_t make_score(uint64_t value) {
    auto const hash = cuddl::detail::splitmix64(value);
    return (hash & 7U) == 0U ? 0U : static_cast<uint16_t>((hash >> 48U) | 1U);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        CLI::App app{"Compact exhaustive cuDDL database-search smoke benchmark"};
        uint32_t reference_count = 4096;
        uint32_t iterations = 20;
        uint64_t seed = 42;
        app.add_option("--references", reference_count, "Reference rows")
            ->check(CLI::PositiveNumber);
        app.add_option("--iterations", iterations, "Timed query repetitions")
            ->check(CLI::PositiveNumber);
        app.add_option("--seed", seed, "Deterministic score seed");
        CLI11_PARSE(app, argc, argv);

        std::vector<uint16_t> rows(static_cast<size_t>(reference_count) * k_bucket_count);
        std::vector<uint16_t> query(k_bucket_count);
        for (size_t bucket = 0; bucket < k_bucket_count; ++bucket) {
            query[bucket] = make_score(seed + bucket);
        }
        for (size_t offset = 0; offset < rows.size(); ++offset) {
            rows[offset] = make_score(seed + k_bucket_count + offset);
        }

        thrust::device_vector<uint16_t> device_rows(rows);
        thrust::device_vector<uint16_t> device_query(query);
        thrust::device_vector<cuddl::reference_search_result> results(reference_count);
        thrust::device_vector<uint8_t> workspace;
        auto const compatibility =
            cuddl::score_compatibility::current<k_kmer_length, k_bucket_count>();
        auto database =
            CUDDL_UNWRAP((cuddl::reference_database<k_kmer_length, k_bucket_count>::build_async(
                device_rows, compatibility
            )));
        CUDDL_CUDA_CALL(cudaDeviceSynchronize());

        auto const search = [&] {
            CUDDL_UNWRAP(
                database.search_async(device_query, compatibility, workspace, results)
            );
        };
        search();
        CUDDL_CUDA_CALL(cudaDeviceSynchronize());

        cudaEvent_t start{};
        cudaEvent_t stop{};
        CUDDL_CUDA_CALL(cudaEventCreate(&start));
        CUDDL_CUDA_CALL(cudaEventCreate(&stop));
        CUDDL_CUDA_CALL(cudaEventRecord(start));
        for (uint32_t iteration = 0; iteration < iterations; ++iteration) {
            search();
        }
        CUDDL_CUDA_CALL(cudaEventRecord(stop));
        CUDDL_CUDA_CALL(cudaEventSynchronize(stop));
        float elapsed_ms = 0.0F;
        CUDDL_CUDA_CALL(cudaEventElapsedTime(&elapsed_ms, start, stop));
        CUDDL_CUDA_CALL(cudaEventDestroy(stop));
        CUDDL_CUDA_CALL(cudaEventDestroy(start));

        cuddl::reference_search_result first{};
        CUDDL_CUDA_CALL(cudaMemcpy(
            &first, thrust::raw_pointer_cast(results.data()), sizeof(first), cudaMemcpyDeviceToHost
        ));
        auto const bucket_total = first.summary.counts.lower + first.summary.counts.equal +
                                  first.summary.counts.higher + first.summary.counts.both_empty;
        if (first.reference_id != 0U || bucket_total != k_bucket_count) {
            throw std::runtime_error("compact search produced an invalid exact result");
        }

        auto const exact_comparisons = static_cast<uint64_t>(reference_count) * iterations;
        auto const throughput =
            static_cast<double>(exact_comparisons) * 1000.0 / static_cast<double>(elapsed_ms);
        std::cout << "references=" << reference_count << '\n'
                  << "iterations=" << iterations << '\n'
                  << "exact_comparisons=" << exact_comparisons << '\n'
                  << std::fixed << std::setprecision(3) << "elapsed_ms=" << elapsed_ms << '\n'
                  << "throughput_comparisons_per_second=" << throughput << '\n';
    } catch (std::exception const& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
