#include <cuddl/cuddl.cuh>

#include <nvbench/nvbench.cuh>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
#include "common.cuh"

namespace {

constexpr uint32_t k_kmer_length = 25;
constexpr size_t k_bucket_count = 2048;
std::vector<nvbench::int64_t> const reference_powers{8, 10, 12, 14, 16, 18, 20};
std::vector<nvbench::int64_t> const indexed_reference_powers{8, 10, 12, 14, 16, 18};

uint16_t make_score(uint64_t value) {
    auto const hash = cuddl::detail::splitmix64(value);
    return (hash & 7U) == 0U ? 0U : static_cast<uint16_t>((hash >> 48U) | 1U);
}
struct indexed_fixture {
    std::vector<uint16_t> rows;
    std::vector<uint16_t> query;
    size_t posting_work{};
    size_t candidate_count{};
};

indexed_fixture make_indexed_fixture(size_t reference_count) {
    constexpr uint64_t seed = 4242;
    indexed_fixture fixture{
        .rows = std::vector<uint16_t>(reference_count * k_bucket_count),
        .query = std::vector<uint16_t>(k_bucket_count),
    };
    for (size_t bucket = 0; bucket < k_bucket_count; ++bucket) {
        fixture.query[bucket] = make_score(seed + bucket);
        if (bucket < 6U) {
            fixture.query[bucket] |= 1U;
        }
    }
    for (size_t reference_id = 0; reference_id < reference_count; ++reference_id) {
        auto const matches = reference_id % 7U;
        fixture.posting_work += matches;
        fixture.candidate_count += matches >= 5U;
        for (size_t bucket = 0; bucket < k_bucket_count; ++bucket) {
            auto score = make_score(seed + k_bucket_count + reference_id * k_bucket_count + bucket);
            if (score == fixture.query[bucket]) {
                score = score == std::numeric_limits<uint16_t>::max()
                            ? 1U
                            : static_cast<uint16_t>(score + 1U);
            }
            fixture.rows[reference_id * k_bucket_count + bucket] =
                bucket < matches ? fixture.query[bucket] : score;
        }
    }
    return fixture;
}

cuddl::pairwise_summary score_row_oracle(
    std::vector<uint16_t> const& query,
    std::vector<uint16_t> const& references,
    size_t reference_id = 0
) {
    cuddl::pairwise_summary summary{};
    auto const reference_offset = reference_id * query.size();
    for (size_t bucket = 0; bucket < query.size(); ++bucket) {
        auto const reference = references[reference_offset + bucket];
        if (query[bucket] == 0U && reference == 0U) {
            ++summary.counts.both_empty;
        } else if (query[bucket] < reference) {
            ++summary.counts.lower;
        } else if (query[bucket] > reference) {
            ++summary.counts.higher;
        } else {
            ++summary.counts.equal;
        }
    }
    return summary;
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

void compact_indexed_build(nvbench::state& state) {
    auto const reference_count = static_cast<size_t>(state.get_int64("References"));
    auto const fixture = make_indexed_fixture(reference_count);
    thrust::device_vector<uint16_t> device_rows(fixture.rows);
    auto const compatibility = cuddl::score_compatibility::current<k_kmer_length, k_bucket_count>();

    size_t resident_bytes = 0;
    {
        auto database = CUDDL_UNWRAP(
            (cuddl::reference_database<k_kmer_length, k_bucket_count>::build_indexed_async(
                device_rows, compatibility
            ))
        );
        CUDDL_CUDA_CALL(cudaDeviceSynchronize());
        resident_bytes = database.persistent_row_bytes() + database.persistent_index_bytes();
    }

    state.add_element_count(fixture.rows.size(), "Scores Indexed");
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
        auto database = CUDDL_UNWRAP(
            (cuddl::reference_database<k_kmer_length, k_bucket_count>::build_indexed_async(
                device_rows, compatibility, cuda::stream_ref{launch.get_stream()}
            ))
        );
        do_not_optimise(database);
    });

    auto const median = state.get_summary("nv/cold/time/gpu/median").get_float64("value");
    add_value(state, "Median GPU Time", median);
    add_value(state, "Median Throughput", static_cast<double>(fixture.rows.size()) / median);
    add_value(state, "Resident Bytes", static_cast<double>(resident_bytes));
}

void compact_indexed_search_impl(nvbench::state& state, uint32_t minimum_matches) {
    auto const reference_count = static_cast<size_t>(state.get_int64("References"));
    auto const fixture = make_indexed_fixture(reference_count);
    thrust::device_vector<uint16_t> device_rows(fixture.rows);
    thrust::device_vector<uint16_t> device_query(fixture.query);
    thrust::device_vector<cuddl::reference_search_result> results(reference_count);
    thrust::device_vector<uint32_t> result_count(1);
    auto const compatibility = cuddl::score_compatibility::current<k_kmer_length, k_bucket_count>();
    auto database =
        CUDDL_UNWRAP((cuddl::reference_database<k_kmer_length, k_bucket_count>::build_indexed_async(
            device_rows, compatibility
        )));
    auto const workspace_bytes = CUDDL_UNWRAP(database.indexed_single_query_workspace_bytes());
    thrust::device_vector<uint8_t> workspace(workspace_bytes);
    CUDDL_CUDA_CALL(cudaDeviceSynchronize());

    CUDDL_UNWRAP(database.search_indexed_async(
        device_query,
        compatibility,
        workspace,
        results,
        result_count,
        {.minimum_matches = minimum_matches}
    ));
    CUDDL_CUDA_CALL(cudaDeviceSynchronize());
    uint32_t observed_count = 0;
    CUDDL_CUDA_CALL(cudaMemcpy(
        &observed_count,
        thrust::raw_pointer_cast(result_count.data()),
        sizeof(observed_count),
        cudaMemcpyDeviceToHost
    ));
    auto const expected_count = minimum_matches == 0U ? reference_count : fixture.candidate_count;
    if (observed_count != expected_count) {
        throw std::runtime_error("indexed search returned the wrong candidate count");
    }
    std::vector<cuddl::reference_search_result> observed_results(expected_count);
    CUDDL_CUDA_CALL(cudaMemcpy(
        observed_results.data(),
        thrust::raw_pointer_cast(results.data()),
        observed_results.size() * sizeof(observed_results.front()),
        cudaMemcpyDeviceToHost
    ));

    size_t true_positives = 0;
    size_t result_index = 0;
    for (size_t reference_id = 0; reference_id < reference_count; ++reference_id) {
        if (minimum_matches != 0U && reference_id % 7U < minimum_matches) {
            continue;
        }
        if (observed_results[result_index].reference_id != reference_id) {
            throw std::runtime_error("indexed search returned the wrong candidate IDs");
        }
        ++true_positives;
        ++result_index;
    }
    if (observed_results.front().summary !=
        score_row_oracle(fixture.query, fixture.rows, observed_results.front().reference_id)) {
        throw std::runtime_error("indexed search disagrees with the scalar oracle");
    }

    state.add_element_count(reference_count, "References Searched");
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
        CUDDL_UNWRAP(database.search_indexed_async(
            device_query,
            compatibility,
            workspace,
            results,
            result_count,
            {.minimum_matches = minimum_matches},
            cuda::stream_ref{launch.get_stream()}
        ));
    });

    auto const median = state.get_summary("nv/cold/time/gpu/median").get_float64("value");
    add_value(state, "Median GPU Time", median);
    add_value(state, "Median Throughput", static_cast<double>(reference_count) / median);
    add_value(state, "Exact Comparisons", static_cast<double>(expected_count));
    add_value(
        state,
        "Posting Work",
        static_cast<double>(minimum_matches == 0U ? 0U : fixture.posting_work)
    );
    add_value(
        state,
        "Candidate Recall",
        static_cast<double>(true_positives) / static_cast<double>(expected_count)
    );
    add_value(
        state,
        "Resident Bytes",
        static_cast<double>(database.persistent_row_bytes() + database.persistent_index_bytes())
    );
}

void compact_indexed_search(nvbench::state& state) {
    compact_indexed_search_impl(state, 5U);
}

void compact_indexed_zero_threshold_search(nvbench::state& state) {
    compact_indexed_search_impl(state, 0U);
}

NVBENCH_BENCH(compact_exhaustive_search)
    .add_int64_power_of_two_axis("References", reference_powers);
NVBENCH_BENCH(compact_indexed_build)
    .add_int64_power_of_two_axis("References", indexed_reference_powers);
NVBENCH_BENCH(compact_indexed_search)
    .add_int64_power_of_two_axis("References", indexed_reference_powers);
NVBENCH_BENCH(compact_indexed_zero_threshold_search)
    .add_int64_power_of_two_axis("References", indexed_reference_powers);
