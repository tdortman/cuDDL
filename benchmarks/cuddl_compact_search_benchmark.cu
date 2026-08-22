#include <cuddl/cuddl.cuh>

#include <nvbench/nvbench.cuh>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {

constexpr uint32_t k_kmer_length = 25;
constexpr size_t k_bucket_count = 2048;
std::vector<nvbench::int64_t> const reference_powers{8, 10, 12, 14, 16, 18, 20};

uint16_t make_score(uint64_t value) {
    auto const hash = cuddl::detail::splitmix64(value);
    return (hash & 7U) == 0U ? 0U : static_cast<uint16_t>((hash >> 48U) | 1U);
}

cuddl::pairwise_summary score_row_oracle(
    std::vector<uint16_t> const& query, std::vector<uint16_t> const& reference
) {
    cuddl::pairwise_summary summary{};
    for (size_t bucket = 0; bucket < query.size(); ++bucket) {
        if (query[bucket] == 0U && reference[bucket] == 0U) {
            ++summary.counts.both_empty;
        } else if (query[bucket] < reference[bucket]) {
            ++summary.counts.lower;
        } else if (query[bucket] > reference[bucket]) {
            ++summary.counts.higher;
        } else {
            ++summary.counts.equal;
        }
    }
    return summary;
}

void add_value(nvbench::state& state, char const* name, double value) {
    auto& summary = state.add_summary(name);
    summary.set_string("name", name);
    summary.set_float64("value", value);
}

}  // namespace

void compact_exhaustive_search(nvbench::state& state) {
    auto const reference_count = static_cast<size_t>(state.get_int64("References"));
    constexpr uint64_t seed = 42;

    std::vector<uint16_t> rows(reference_count * k_bucket_count);
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
    auto const compatibility = cuddl::score_compatibility::current<k_kmer_length, k_bucket_count>();
    auto database =
        CUDDL_UNWRAP((cuddl::reference_database<k_kmer_length, k_bucket_count>::build_async(
            device_rows, compatibility
        )));
    CUDDL_CUDA_CALL(cudaDeviceSynchronize());

    CUDDL_UNWRAP(database.search_async(device_query, compatibility, workspace, results));
    CUDDL_CUDA_CALL(cudaDeviceSynchronize());
    cuddl::reference_search_result first{};
    CUDDL_CUDA_CALL(cudaMemcpy(
        &first, thrust::raw_pointer_cast(results.data()), sizeof(first), cudaMemcpyDeviceToHost
    ));
    if (first.reference_id != 0U || first.summary != score_row_oracle(query, rows)) {
        throw std::runtime_error("compact search disagrees with the scalar oracle");
    }

    state.add_element_count(reference_count, "Exact Comparisons");
    state.add_global_memory_reads<uint16_t>(2 * reference_count * k_bucket_count);
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
        CUDDL_UNWRAP(database.search_async(
            device_query, compatibility, workspace, results, cuda::stream_ref{launch.get_stream()}
        ));
    });

    auto const median = state.get_summary("nv/cold/time/gpu/median").get_float64("value");
    add_value(state, "Median GPU Time", median);
    add_value(state, "Median Throughput", static_cast<double>(reference_count) / median);
}

NVBENCH_BENCH(compact_exhaustive_search)
    .add_int64_power_of_two_axis("References", reference_powers);
