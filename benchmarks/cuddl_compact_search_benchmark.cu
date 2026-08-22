#include <cuddl/cuddl.cuh>

#include <nvbench/nvbench.cuh>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <algorithm>
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

cuddl::pairwise_summary score_row_oracle_rows(uint16_t const* query, uint16_t const* reference) {
    cuddl::pairwise_summary summary{};
    for (size_t bucket = 0; bucket < k_bucket_count; ++bucket) {
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

cuddl::pairwise_summary score_row_oracle(
    std::vector<uint16_t> const& query,
    std::vector<uint16_t> const& references,
    size_t reference_id = 0
) {
    return score_row_oracle_rows(query.data(), references.data() + reference_id * query.size());
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

void compact_batch_and_all_to_all_search(nvbench::state& state) {
    constexpr uint32_t reference_count = 8U;
    constexpr uint32_t batch_query_count = 3U;
    constexpr uint32_t all_to_all_tile_count = 4U;
    constexpr uint32_t batch_query_id_offset = 100U;
    constexpr uint32_t first_all_to_all_tile = 0U;
    constexpr uint32_t second_all_to_all_tile = all_to_all_tile_count;

    std::vector<uint16_t> rows(static_cast<size_t>(reference_count) * k_bucket_count);
    for (size_t offset = 0; offset < rows.size(); ++offset) {
        rows[offset] = make_score(7000U + offset);
    }
    std::vector<uint16_t> queries(static_cast<size_t>(batch_query_count) * k_bucket_count);
    for (uint32_t query_id = 0; query_id < batch_query_count; ++query_id) {
        for (size_t bucket = 0; bucket < k_bucket_count; ++bucket) {
            queries[static_cast<size_t>(query_id) * k_bucket_count + bucket] =
                make_score(11000U + static_cast<uint64_t>(query_id) * k_bucket_count + bucket);
        }
    }

    thrust::device_vector<uint16_t> device_rows(rows);
    thrust::device_vector<uint16_t> device_queries(queries);
    auto const compatibility = cuddl::score_compatibility::current<k_kmer_length, k_bucket_count>();
    auto database =
        CUDDL_UNWRAP((cuddl::reference_database<k_kmer_length, k_bucket_count>::build_async(
            device_rows, compatibility
        )));
    CUDDL_CUDA_CALL(cudaDeviceSynchronize());

    auto const batch_requirements =
        CUDDL_UNWRAP(database.batch_search_requirements(batch_query_count));
    auto const first_tile_requirements = CUDDL_UNWRAP(
        database.all_to_all_search_requirements(first_all_to_all_tile, all_to_all_tile_count)
    );
    auto const second_tile_requirements = CUDDL_UNWRAP(
        database.all_to_all_search_requirements(second_all_to_all_tile, all_to_all_tile_count)
    );
    auto const workspace_bytes = std::max(
        batch_requirements.workspace_bytes,
        std::max(first_tile_requirements.workspace_bytes, second_tile_requirements.workspace_bytes)
    );
    auto const maximum_pair_count = std::max(
        static_cast<size_t>(batch_requirements.maximum_pair_count),
        std::max(
            static_cast<size_t>(first_tile_requirements.maximum_pair_count),
            static_cast<size_t>(second_tile_requirements.maximum_pair_count)
        )
    );
    auto const result_bytes = std::max(
        batch_requirements.result_bytes,
        std::max(first_tile_requirements.result_bytes, second_tile_requirements.result_bytes)
    );
    auto const match_count_bytes = std::max(
        batch_requirements.match_count_bytes,
        std::max(
            first_tile_requirements.match_count_bytes, second_tile_requirements.match_count_bytes
        )
    );
    auto const result_capacity = std::max(
        maximum_pair_count,
        (result_bytes + sizeof(cuddl::batch_search_result) - 1U) /
            sizeof(cuddl::batch_search_result)
    );
    auto const match_count_capacity = std::max(
        maximum_pair_count, (match_count_bytes + sizeof(uint32_t) - 1U) / sizeof(uint32_t)
    );

    thrust::device_vector<uint8_t> workspace(workspace_bytes);
    thrust::device_vector<cuddl::batch_search_result> results(result_capacity);
    thrust::device_vector<uint32_t> result_count(1);
    thrust::device_vector<uint32_t> result_match_counts(match_count_capacity);
    std::vector<cuddl::batch_search_result> observed_results(result_capacity);
    std::vector<uint32_t> observed_match_counts(match_count_capacity);

    auto read_result_count = [&] {
        uint32_t count = 0;
        CUDDL_CUDA_CALL(cudaMemcpy(
            &count,
            thrust::raw_pointer_cast(result_count.data()),
            sizeof(count),
            cudaMemcpyDeviceToHost
        ));
        return count;
    };
    auto validate_batch = [&] {
        auto const observed_count = read_result_count();
        auto const expected_count = batch_query_count * reference_count;
        if (observed_count != expected_count) {
            throw std::runtime_error("batch search returned the wrong pair count");
        }
        CUDDL_CUDA_CALL(cudaMemcpy(
            observed_results.data(),
            thrust::raw_pointer_cast(results.data()),
            static_cast<size_t>(observed_count) * sizeof(observed_results.front()),
            cudaMemcpyDeviceToHost
        ));
        CUDDL_CUDA_CALL(cudaMemcpy(
            observed_match_counts.data(),
            thrust::raw_pointer_cast(result_match_counts.data()),
            static_cast<size_t>(observed_count) * sizeof(observed_match_counts.front()),
            cudaMemcpyDeviceToHost
        ));
        for (uint32_t result_index = 0; result_index < observed_count; ++result_index) {
            auto const query_index = result_index / reference_count;
            auto const reference_id = result_index % reference_count;
            auto const expected_summary = score_row_oracle_rows(
                queries.data() + static_cast<size_t>(query_index) * k_bucket_count,
                rows.data() + static_cast<size_t>(reference_id) * k_bucket_count
            );
            auto const& observed = observed_results[result_index];
            if (observed.query_id != batch_query_id_offset + query_index ||
                observed.reference_id != reference_id) {
                throw std::runtime_error("batch search returned the wrong stable IDs");
            }
            if (observed.summary != expected_summary ||
                observed_match_counts[result_index] != expected_summary.counts.equal) {
                throw std::runtime_error("batch search disagrees with the scalar oracle");
            }
        }
    };
    auto validate_all_to_all = [&](uint32_t first_query_id) {
        auto const observed_count = read_result_count();
        auto const expected_count = [&] {
            uint32_t count = 0;
            for (uint32_t query_id = first_query_id;
                 query_id < first_query_id + all_to_all_tile_count;
                 ++query_id) {
                count += reference_count > query_id + 1U ? reference_count - query_id - 1U : 0U;
            }
            return count;
        }();
        if (observed_count != expected_count) {
            throw std::runtime_error("all-to-all search returned the wrong pair count");
        }
        CUDDL_CUDA_CALL(cudaMemcpy(
            observed_results.data(),
            thrust::raw_pointer_cast(results.data()),
            static_cast<size_t>(observed_count) * sizeof(observed_results.front()),
            cudaMemcpyDeviceToHost
        ));
        CUDDL_CUDA_CALL(cudaMemcpy(
            observed_match_counts.data(),
            thrust::raw_pointer_cast(result_match_counts.data()),
            static_cast<size_t>(observed_count) * sizeof(observed_match_counts.front()),
            cudaMemcpyDeviceToHost
        ));
        uint32_t result_index = 0;
        for (uint32_t query_id = first_query_id; query_id < first_query_id + all_to_all_tile_count;
             ++query_id) {
            for (uint32_t reference_id = query_id + 1U; reference_id < reference_count;
                 ++reference_id) {
                auto const expected_summary = score_row_oracle_rows(
                    rows.data() + static_cast<size_t>(query_id) * k_bucket_count,
                    rows.data() + static_cast<size_t>(reference_id) * k_bucket_count
                );
                auto const& observed = observed_results[result_index];
                if (observed.query_id != query_id || observed.reference_id != reference_id) {
                    throw std::runtime_error("all-to-all search returned the wrong stable IDs");
                }
                if (observed.summary != expected_summary ||
                    observed_match_counts[result_index] != expected_summary.counts.equal) {
                    throw std::runtime_error("all-to-all search disagrees with the scalar oracle");
                }
                ++result_index;
            }
        }
    };

    CUDDL_UNWRAP(database.search_batch_async(
        device_queries,
        compatibility,
        batch_query_id_offset,
        workspace,
        results,
        result_count,
        result_match_counts
    ));
    CUDDL_CUDA_CALL(cudaDeviceSynchronize());
    validate_batch();

    CUDDL_UNWRAP(database.search_all_to_all_async(
        first_all_to_all_tile,
        all_to_all_tile_count,
        workspace,
        results,
        result_count,
        result_match_counts
    ));
    CUDDL_CUDA_CALL(cudaDeviceSynchronize());
    validate_all_to_all(first_all_to_all_tile);

    CUDDL_UNWRAP(database.search_all_to_all_async(
        second_all_to_all_tile,
        all_to_all_tile_count,
        workspace,
        results,
        result_count,
        result_match_counts
    ));
    CUDDL_CUDA_CALL(cudaDeviceSynchronize());
    validate_all_to_all(second_all_to_all_tile);

    auto const total_pairs = static_cast<size_t>(batch_query_count) * reference_count +
                             static_cast<size_t>(reference_count) * (reference_count - 1U) / 2U;
    state.add_element_count(total_pairs, "Pair Comparisons");
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
        auto const stream = cuda::stream_ref{launch.get_stream()};
        CUDDL_UNWRAP(database.search_batch_async(
            device_queries,
            compatibility,
            batch_query_id_offset,
            workspace,
            results,
            result_count,
            result_match_counts,
            stream
        ));
        CUDDL_UNWRAP(database.search_all_to_all_async(
            first_all_to_all_tile,
            all_to_all_tile_count,
            workspace,
            results,
            result_count,
            result_match_counts,
            stream
        ));
        CUDDL_UNWRAP(database.search_all_to_all_async(
            second_all_to_all_tile,
            all_to_all_tile_count,
            workspace,
            results,
            result_count,
            result_match_counts,
            stream
        ));
    });

    auto const median = state.get_summary("nv/cold/time/gpu/median").get_float64("value");
    add_value(state, "Median GPU Time", median);
    add_value(state, "Median Throughput", static_cast<double>(total_pairs) / median);
}

NVBENCH_BENCH(compact_exhaustive_search)
    .add_int64_power_of_two_axis("References", reference_powers);
NVBENCH_BENCH(compact_indexed_build)
    .add_int64_power_of_two_axis("References", indexed_reference_powers);
NVBENCH_BENCH(compact_indexed_search)
    .add_int64_power_of_two_axis("References", indexed_reference_powers);
NVBENCH_BENCH(compact_indexed_zero_threshold_search)
    .add_int64_power_of_two_axis("References", indexed_reference_powers);
NVBENCH_BENCH(compact_batch_and_all_to_all_search);
