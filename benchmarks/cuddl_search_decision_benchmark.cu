#include <cuddl/cuddl.cuh>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>
#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "common.cuh"

namespace {

using clock_type = std::chrono::steady_clock;
using database_type = cuddl::reference_database<25, 2048>;

constexpr uint32_t k_kmer_length = 25;
constexpr size_t k_bucket_count = 2048;
constexpr uint64_t k_fixture_seed = 4242;
constexpr nvbench::int64_t k_max_hot_percentage = 50;
constexpr uint32_t k_minimum_matches = 5;
constexpr uint32_t k_score_period = std::numeric_limits<uint16_t>::max();

std::vector<nvbench::int64_t> const reference_counts{1024, 16384, 200687};
std::vector<nvbench::int64_t> const fill_permilles{250, 1000};
std::vector<nvbench::int64_t> const hot_percentages{0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50};
std::vector<nvbench::int64_t> const query_counts{1, 32};
std::vector<std::string> const query_profiles{"copied", "boundary"};
std::vector<nvbench::int64_t> const indexed_bucket_counts{1024, 2048};
std::vector<nvbench::int64_t> const key_bits{15, 16};

struct workload {
    uint32_t reference_count{};
    double fill_ratio{};
    double hot_fraction{};
    uint32_t query_count{};
    bool boundary_probe{};
};

struct index_mode {
    uint32_t indexed_bucket_count{};
    uint16_t key_mask{};
    uint32_t key_bits{};
};

struct fixture {
    std::vector<uint16_t> rows;
    std::vector<uint16_t> queries;
    std::vector<uint64_t> bucket_top_bit_counts;
    std::vector<uint8_t> bucket_filled;
    std::vector<uint8_t> bucket_hot;
    std::vector<uint32_t> bucket_shifts;
    uint64_t populated_posting_lists{};
    uint64_t posting_entries{};
};

struct indexed_fixture_metrics {
    uint64_t populated_posting_lists{};
    uint64_t posting_entries{};
    uint64_t posting_visits{};
    uint64_t top_bit_count{};
    uint64_t nonzero_scores{};
};

struct normalised_search_result {
    uint32_t query_id{};
    uint32_t reference_id{};
    cuddl::pairwise_summary summary{};
};

struct search_output {
    std::vector<normalised_search_result> results;
    size_t workspace_bytes{};
};

struct minimum_matches_predicate {
    uint32_t minimum_matches{};

    template <typename Result>
    [[nodiscard]] __host__ __device__ bool operator()(Result const& result) const {
        return result.summary.counts.equal >= minimum_matches;
    }
};

[[nodiscard]] workload read_workload(nvbench::state& state, bool includes_queries) {
    auto const references = state.get_int64("References");
    auto const fill_permille = state.get_int64("FillPermille");
    auto const hot_percentage = state.get_int64("HotPercent");
    auto const queries = includes_queries ? state.get_int64("Queries") : 1;
    auto const query_profile = includes_queries ? state.get_string("QueryProfile") : "copied";
    if (references <= 0 || references > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("reference count exceeds the supported 32-bit range");
    }
    if (fill_permille <= 0 || fill_permille > 1000) {
        throw std::runtime_error("fill permille must be in [1, 1000]");
    }
    if (hot_percentage < 0 || hot_percentage > k_max_hot_percentage) {
        throw std::runtime_error("hot skew must be in [0%, 50%]");
    }
    if (queries <= 0 || queries > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("query count exceeds the supported 32-bit range");
    }
    if (query_profile != "copied" && query_profile != "boundary") {
        throw std::runtime_error("query profile must be copied or boundary");
    }
    if (static_cast<uint64_t>(references) * static_cast<uint64_t>(queries) >
        std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("query/reference pairs exceed the supported 32-bit range");
    }
    return {
        .reference_count = static_cast<uint32_t>(references),
        .fill_ratio = static_cast<double>(fill_permille) / 1000.0,
        .hot_fraction = static_cast<double>(hot_percentage) / 100.0,
        .query_count = static_cast<uint32_t>(queries),
        .boundary_probe = query_profile == "boundary",
    };
}
[[nodiscard]] index_mode read_index_mode(nvbench::state& state) {
    auto const buckets = state.get_int64("IndexedBuckets");
    auto const bits = state.get_int64("KeyBits");
    if (buckets != static_cast<nvbench::int64_t>(k_bucket_count / 2U) &&
        buckets != static_cast<nvbench::int64_t>(k_bucket_count)) {
        throw std::runtime_error("indexed bucket count must be half or all buckets");
    }
    if (bits != 15 && bits != 16) {
        throw std::runtime_error("indexed key width must be 15 or 16 bits");
    }
    return {
        .indexed_bucket_count = static_cast<uint32_t>(buckets),
        .key_mask = static_cast<uint16_t>((uint32_t{1} << bits) - 1U),
        .key_bits = static_cast<uint32_t>(bits),
    };
}

[[nodiscard]] cuddl::score_compatibility indexed_compatibility(index_mode const& mode) {
    auto compatibility = cuddl::score_compatibility::current<k_kmer_length, k_bucket_count>();
    compatibility.indexed_bucket_count = mode.indexed_bucket_count;
    compatibility.key_mask = mode.key_mask;
    return compatibility;
}

void add_string(nvbench::state& state, char const* name, std::string value) {
    auto& summary = state.add_summary(name);
    summary.set_string("name", name);
    summary.set_string("value", std::move(value));
}
void add_common_metadata(
    nvbench::state& state,
    workload const& value,
    size_t free_memory_bytes,
    cuddl::score_compatibility const& compatibility
) {
    int runtime_version = 0;
    int driver_version = 0;
    CUDDL_CUDA_CALL(cudaRuntimeGetVersion(&runtime_version));
    CUDDL_CUDA_CALL(cudaDriverGetVersion(&driver_version));

    add_value(state, "fixture_seed", static_cast<double>(k_fixture_seed));
    add_value(state, "hot_fraction", value.hot_fraction);
    add_value(state, "minimum_matches", k_minimum_matches);
    add_value(state, "free_memory_before_bytes", static_cast<double>(free_memory_bytes));
    add_value(state, "cuda_runtime_version", runtime_version);
    add_value(state, "cuda_driver_version", driver_version);
    add_value(state, "cuda_compile_version", CUDART_VERSION);
    add_string(state, "compiler", __VERSION__);
#ifdef __OPTIMIZE__
    add_string(state, "build_configuration", "optimised");
#else
    add_string(state, "build_configuration", "unoptimised");
#endif
    add_value(state, "kmer_length", compatibility.kmer_length);
    add_value(state, "bucket_count", compatibility.bucket_count);
    add_value(state, "indexed_bucket_count", compatibility.indexed_bucket_count);
    add_value(state, "score_encoder_identity", compatibility.score_encoder_identity);
    add_value(state, "exponent_bits", compatibility.exponent_bits);
    add_value(state, "mantissa_bits", compatibility.mantissa_bits);
    add_value(state, "hash_identity", compatibility.hash_identity);
    add_string(state, "hash_seed", std::to_string(compatibility.hash_seed));
    add_value(state, "canonicalisation_policy", compatibility.canonicalisation_policy);
    add_string(state, "blacklist_identity", std::to_string(compatibility.blacklist_identity));
    add_value(state, "blacklist_version", compatibility.blacklist_version);
    add_value(state, "key_mask", compatibility.key_mask);
    add_value(
        state,
        "filled_cells_per_row",
        std::clamp<size_t>(
            static_cast<size_t>(std::llround(value.fill_ratio * k_bucket_count)), 1U, k_bucket_count
        )
    );
}

[[nodiscard]] size_t align_up(size_t value, size_t alignment) {
    return (value + alignment - 1U) & ~(alignment - 1U);
}

struct indexed_storage_bytes {
    size_t offset_bytes{};
    size_t posting_bytes{};
};

[[nodiscard]] size_t indexed_cell_count(cuddl::score_compatibility const& compatibility) {
    return static_cast<size_t>(compatibility.indexed_bucket_count) *
           (static_cast<size_t>(compatibility.key_mask) + 1U);
}

[[nodiscard]] indexed_storage_bytes
indexed_storage(uint32_t reference_count, cuddl::score_compatibility const& compatibility) {
    auto const cells = indexed_cell_count(compatibility);
    return {
        .offset_bytes = (cells + 1U) * sizeof(uint32_t),
        .posting_bytes = static_cast<size_t>(reference_count) * compatibility.indexed_bucket_count *
                         sizeof(uint32_t),
    };
}

[[nodiscard]] size_t indexed_build_temporary_bytes(
    cuddl::score_compatibility const& compatibility
) {
    auto const cells = indexed_cell_count(compatibility);
    size_t scan_bytes = 0;
    uint32_t* offsets = nullptr;
    CUDDL_CUDA_CALL(
        cub::DeviceScan::ExclusiveSum(
            nullptr, scan_bytes, offsets, static_cast<int64_t>(cells + 1U), cudaStream_t{nullptr}
        )
    );
    return align_up(scan_bytes, alignof(uint32_t)) + cells * sizeof(uint32_t);
}

[[nodiscard]] size_t
indexed_resident_bytes(uint32_t reference_count, cuddl::score_compatibility const& compatibility) {
    auto const storage = indexed_storage(reference_count, compatibility);
    return database_type::persistent_row_bytes(reference_count) + storage.offset_bytes +
           storage.posting_bytes;
}

[[nodiscard]] uint64_t congruent_reference_count(uint32_t reference_count, uint32_t reference_id) {
    auto const residue = reference_id % k_score_period;
    if (residue >= reference_count) {
        return 0U;
    }
    return 1U + (static_cast<uint64_t>(reference_count) - 1U - residue) / k_score_period;
}

[[nodiscard]] uint32_t query_reference_id(workload const& settings, uint32_t query_id) {
    return static_cast<uint32_t>(
        cuddl::detail::splitmix64(k_fixture_seed + 0x9e3779b97f4a7c15ULL + query_id) %
        settings.reference_count
    );
}
[[nodiscard]] fixture make_fixture(workload const& settings) {
    auto const fill_count = std::clamp<size_t>(
        static_cast<size_t>(std::llround(settings.fill_ratio * k_bucket_count)), 1U, k_bucket_count
    );
    auto const hot_count = std::clamp<size_t>(
        static_cast<size_t>(std::llround(settings.hot_fraction * fill_count)), 0U, fill_count
    );

    std::vector<size_t> buckets(fill_count);
    std::vector<uint32_t> shifts(fill_count);
    auto const offset =
        static_cast<size_t>(cuddl::detail::splitmix64(k_fixture_seed)) & (k_bucket_count - 1U);
    auto const stride = (static_cast<size_t>(cuddl::detail::splitmix64(k_fixture_seed + 1U)) | 1U) &
                        (k_bucket_count - 1U);
    for (size_t index = 0; index < fill_count; ++index) {
        buckets[index] = (offset + index * stride) & (k_bucket_count - 1U);
        shifts[index] = static_cast<uint32_t>(
            cuddl::detail::splitmix64(k_fixture_seed + 2U + buckets[index]) % k_score_period
        );
    }

    std::array<uint8_t, k_bucket_count> probe_matches{};
    if (settings.boundary_probe) {
        size_t first_half_matches = 0;
        size_t second_half_matches = 0;
        for (size_t index = hot_count; index < fill_count; ++index) {
            auto const bucket = buckets[index];
            if (bucket < k_bucket_count / 2U && first_half_matches < 2U) {
                probe_matches[bucket] = 1U;
                ++first_half_matches;
            } else if (bucket >= k_bucket_count / 2U && second_half_matches < 3U) {
                probe_matches[bucket] = 1U;
                ++second_half_matches;
            }
        }
        if (first_half_matches != 2U || second_half_matches != 3U) {
            throw std::runtime_error(
                "fixture cannot place threshold matches across both bucket halves"
            );
        }
    }

    fixture value{
        .rows =
            std::vector<uint16_t>(static_cast<size_t>(settings.reference_count) * k_bucket_count),
        .queries =
            std::vector<uint16_t>(static_cast<size_t>(settings.query_count) * k_bucket_count),
        .bucket_top_bit_counts = std::vector<uint64_t>(k_bucket_count),
        .bucket_filled = std::vector<uint8_t>(k_bucket_count),
        .bucket_hot = std::vector<uint8_t>(k_bucket_count),
        .bucket_shifts = std::vector<uint32_t>(k_bucket_count),
        .populated_posting_lists =
            hot_count +
            (fill_count - hot_count) * std::min<uint64_t>(settings.reference_count, k_score_period),
        .posting_entries = static_cast<uint64_t>(settings.reference_count) * fill_count,
    };
    for (size_t index = 0; index < fill_count; ++index) {
        auto const bucket = buckets[index];
        value.bucket_filled[bucket] = 1U;
        value.bucket_hot[bucket] = index < hot_count ? 1U : 0U;
        value.bucket_shifts[bucket] = shifts[index];
    }

    for (uint32_t reference_id = 0; reference_id < settings.reference_count; ++reference_id) {
        for (size_t index = 0; index < fill_count; ++index) {
            auto const bucket = buckets[index];
            auto const score =
                index < hot_count ? uint16_t{1U}
                                  : static_cast<uint16_t>(
                                        1U + (static_cast<uint64_t>(reference_id) + shifts[index]) %
                                                 k_score_period
                                    );
            value.rows[static_cast<size_t>(reference_id) * k_bucket_count + bucket] = score;
            value.bucket_top_bit_counts[bucket] += (score & 0x8000U) != 0U;
        }
    }

    for (uint32_t query_id = 0; query_id < settings.query_count; ++query_id) {
        auto const reference_id = query_reference_id(settings, query_id);
        auto* query = value.queries.data() + static_cast<size_t>(query_id) * k_bucket_count;
        std::memcpy(
            query,
            value.rows.data() + static_cast<size_t>(reference_id) * k_bucket_count,
            k_bucket_count * sizeof(uint16_t)
        );
        if (!settings.boundary_probe || query_id != 0U) {
            continue;
        }
        for (size_t index = 0; index < fill_count; ++index) {
            auto const bucket = buckets[index];
            if (probe_matches[bucket] != 0U) {
                continue;
            }
            auto replacement = static_cast<uint16_t>(
                1U + cuddl::detail::splitmix64(k_fixture_seed + 0xd1b54a32d192ed03ULL + bucket) %
                         k_score_period
            );
            if (replacement == query[bucket]) {
                replacement = replacement == k_score_period ? 1U : replacement + 1U;
            }
            query[bucket] = replacement;
        }
    }
    return value;
}

[[nodiscard]] uint64_t
distinct_index_keys(uint32_t reference_count, uint32_t shift, uint16_t key_mask) {
    if (key_mask == std::numeric_limits<uint16_t>::max()) {
        return std::min<uint64_t>(reference_count, k_score_period);
    }
    auto const key_count = static_cast<size_t>(key_mask) + 1U;
    if (reference_count >= k_score_period) {
        return key_count;
    }
    std::vector<uint8_t> seen(key_count);
    for (uint32_t reference_id = 0; reference_id < reference_count; ++reference_id) {
        auto const score = static_cast<uint16_t>(
            1U + (static_cast<uint64_t>(reference_id) + shift) % k_score_period
        );
        seen[score & key_mask] = 1U;
    }
    return static_cast<uint64_t>(std::count(seen.begin(), seen.end(), uint8_t{1}));
}

[[nodiscard]] indexed_fixture_metrics make_indexed_fixture_metrics(
    fixture const& data,
    workload const& settings,
    index_mode const& mode
) {
    indexed_fixture_metrics metrics{};
    auto posting_count = [&](uint16_t query_score, uint32_t shift) {
        auto count_score = [&](uint16_t score) {
            if (score == 0U) {
                return uint64_t{0};
            }
            auto const residue = static_cast<uint32_t>(
                (static_cast<uint64_t>(score - 1U) + k_score_period - (shift % k_score_period)) %
                k_score_period
            );
            return congruent_reference_count(settings.reference_count, residue);
        };
        auto count = count_score(query_score);
        if (mode.key_mask != std::numeric_limits<uint16_t>::max()) {
            count += count_score(query_score ^ 0x8000U);
        }
        return count;
    };

    for (uint32_t bucket = 0; bucket < mode.indexed_bucket_count; ++bucket) {
        if (data.bucket_filled[bucket] == 0U) {
            continue;
        }
        metrics.nonzero_scores += settings.reference_count;
        metrics.top_bit_count += data.bucket_top_bit_counts[bucket];
        if (data.bucket_hot[bucket] != 0U) {
            ++metrics.populated_posting_lists;
        } else {
            metrics.populated_posting_lists += distinct_index_keys(
                settings.reference_count, data.bucket_shifts[bucket], mode.key_mask
            );
        }
    }
    metrics.posting_entries = metrics.nonzero_scores;

    for (uint32_t query_id = 0; query_id < settings.query_count; ++query_id) {
        auto const query_offset = static_cast<size_t>(query_id) * k_bucket_count;
        for (uint32_t bucket = 0; bucket < mode.indexed_bucket_count; ++bucket) {
            auto const query_score = data.queries[query_offset + bucket];
            if (query_score == 0U || data.bucket_filled[bucket] == 0U) {
                continue;
            }
            if (data.bucket_hot[bucket] != 0U) {
                if ((query_score & mode.key_mask) == (uint16_t{1U} & mode.key_mask)) {
                    metrics.posting_visits += settings.reference_count;
                }
                continue;
            }
            metrics.posting_visits += posting_count(query_score, data.bucket_shifts[bucket]);
        }
    }
    return metrics;
}

template <typename DeviceResult, typename Convert>
[[nodiscard]] std::vector<normalised_search_result> copy_normalised_results(
    thrust::device_vector<DeviceResult> const& device_results,
    uint32_t count,
    Convert&& convert
) {
    std::vector<DeviceResult> host_results(count);
    if (count != 0U) {
        CUDDL_CUDA_CALL(cudaMemcpy(
            host_results.data(),
            thrust::raw_pointer_cast(device_results.data()),
            host_results.size() * sizeof(DeviceResult),
            cudaMemcpyDeviceToHost
        ));
    }
    std::vector<normalised_search_result> results;
    results.reserve(host_results.size());
    std::transform(
        host_results.begin(),
        host_results.end(),
        std::back_inserter(results),
        std::forward<Convert>(convert)
    );
    return results;
}

[[nodiscard]] uint32_t read_result_count(thrust::device_vector<uint32_t> const& result_count) {
    uint32_t count = 0;
    CUDDL_CUDA_CALL(cudaMemcpy(
        &count, thrust::raw_pointer_cast(result_count.data()), sizeof(count), cudaMemcpyDeviceToHost
    ));
    return count;
}

template <typename Execute>
[[nodiscard]] search_output execute_exhaustive_search(
    database_type const& database,
    thrust::device_vector<uint16_t> const& queries,
    workload const& settings,
    cuddl::score_compatibility const& compatibility,
    uint32_t minimum_matches,
    Execute&& execute
) {
    if (settings.query_count == 1U) {
        auto const result_capacity = database.reference_count();
        auto const database_workspace_bytes = database.single_query_workspace_bytes();
        thrust::device_vector<uint8_t> database_workspace(database_workspace_bytes);
        thrust::device_vector<cuddl::reference_search_result> exhaustive_results(result_capacity);
        thrust::device_vector<cuddl::reference_search_result> selected_results(result_capacity);
        thrust::device_vector<uint32_t> selected_count(1U);
        size_t selection_workspace_bytes = 0;
        CUDDL_CUDA_CALL(
            cub::DeviceSelect::If(
                nullptr,
                selection_workspace_bytes,
                thrust::raw_pointer_cast(exhaustive_results.data()),
                thrust::raw_pointer_cast(selected_results.data()),
                thrust::raw_pointer_cast(selected_count.data()),
                static_cast<int64_t>(result_capacity),
                minimum_matches_predicate{minimum_matches}
            )
        );
        thrust::device_vector<uint8_t> selection_workspace(selection_workspace_bytes);
        execute([&](cudaStream_t stream) {
            CUDDL_UNWRAP(database.search_async(
                queries,
                compatibility,
                database_workspace,
                exhaustive_results,
                cuda::stream_ref{stream}
            ));
            CUDDL_CUDA_CALL(
                cub::DeviceSelect::If(
                    thrust::raw_pointer_cast(selection_workspace.data()),
                    selection_workspace_bytes,
                    thrust::raw_pointer_cast(exhaustive_results.data()),
                    thrust::raw_pointer_cast(selected_results.data()),
                    thrust::raw_pointer_cast(selected_count.data()),
                    static_cast<int64_t>(result_capacity),
                    minimum_matches_predicate{minimum_matches},
                    stream
                )
            );
        });
        auto const count = read_result_count(selected_count);
        return {
            .results = copy_normalised_results(
                selected_results,
                count,
                [](auto const& result) {
                    return normalised_search_result{0U, result.reference_id, result.summary};
                }
            ),
            .workspace_bytes = database_workspace_bytes + selection_workspace_bytes,
        };
    }

    auto const requirements =
        CUDDL_UNWRAP(database.batch_search_requirements(settings.query_count));
    auto const result_capacity = requirements.maximum_pair_count;
    thrust::device_vector<uint8_t> database_workspace(requirements.workspace_bytes);
    thrust::device_vector<cuddl::batch_search_result> exhaustive_results(result_capacity);
    thrust::device_vector<uint32_t> exhaustive_count(1U);
    thrust::device_vector<cuddl::batch_search_result> selected_results(result_capacity);
    thrust::device_vector<uint32_t> selected_count(1U);
    size_t selection_workspace_bytes = 0;
    CUDDL_CUDA_CALL(
        cub::DeviceSelect::If(
            nullptr,
            selection_workspace_bytes,
            thrust::raw_pointer_cast(exhaustive_results.data()),
            thrust::raw_pointer_cast(selected_results.data()),
            thrust::raw_pointer_cast(selected_count.data()),
            static_cast<int64_t>(result_capacity),
            minimum_matches_predicate{minimum_matches}
        )
    );
    thrust::device_vector<uint8_t> selection_workspace(selection_workspace_bytes);
    execute([&](cudaStream_t stream) {
        CUDDL_UNWRAP(database.search_batch_async(
            queries,
            compatibility,
            0U,
            database_workspace,
            exhaustive_results,
            exhaustive_count,
            cuda::stream_ref{stream}
        ));
        CUDDL_CUDA_CALL(
            cub::DeviceSelect::If(
                thrust::raw_pointer_cast(selection_workspace.data()),
                selection_workspace_bytes,
                thrust::raw_pointer_cast(exhaustive_results.data()),
                thrust::raw_pointer_cast(selected_results.data()),
                thrust::raw_pointer_cast(selected_count.data()),
                static_cast<int64_t>(result_capacity),
                minimum_matches_predicate{minimum_matches},
                stream
            )
        );
    });
    auto const count = read_result_count(selected_count);
    return {
        .results = copy_normalised_results(
            selected_results,
            count,
            [](auto const& result) {
                return normalised_search_result{
                    result.query_id, result.reference_id, result.summary
                };
            }
        ),
        .workspace_bytes = requirements.workspace_bytes + selection_workspace_bytes,
    };
}

template <typename Execute>
[[nodiscard]] search_output execute_indexed_search(
    database_type const& database,
    thrust::device_vector<uint16_t> const& queries,
    workload const& settings,
    cuddl::score_compatibility const& compatibility,
    uint32_t minimum_matches,
    Execute&& execute
) {
    if (settings.query_count == 1U) {
        auto const workspace_bytes = CUDDL_UNWRAP(database.indexed_single_query_workspace_bytes());
        thrust::device_vector<uint8_t> workspace(workspace_bytes);
        thrust::device_vector<cuddl::reference_search_result> results(database.reference_count());
        thrust::device_vector<uint32_t> result_count(1U);
        execute([&](cudaStream_t stream) {
            CUDDL_UNWRAP(database.search_indexed_async(
                queries,
                compatibility,
                workspace,
                results,
                result_count,
                {.minimum_matches = minimum_matches},
                cuda::stream_ref{stream}
            ));
        });
        auto const count = read_result_count(result_count);
        return {
            .results = copy_normalised_results(
                results,
                count,
                [](auto const& result) {
                    return normalised_search_result{0U, result.reference_id, result.summary};
                }
            ),
            .workspace_bytes = workspace_bytes,
        };
    }

    auto const requirements =
        CUDDL_UNWRAP(database.indexed_batch_search_requirements(settings.query_count));
    thrust::device_vector<uint8_t> workspace(requirements.workspace_bytes);
    thrust::device_vector<cuddl::batch_search_result> results(requirements.maximum_pair_count);
    thrust::device_vector<uint32_t> result_count(1U);
    execute([&](cudaStream_t stream) {
        CUDDL_UNWRAP(database.search_batch_indexed_async(
            queries,
            compatibility,
            0U,
            workspace,
            results,
            result_count,
            {.minimum_matches = minimum_matches},
            cuda::stream_ref{stream}
        ));
    });
    auto const count = read_result_count(result_count);
    return {
        .results = copy_normalised_results(
            results,
            count,
            [](auto const& result) {
                return normalised_search_result{
                    result.query_id, result.reference_id, result.summary
                };
            }
        ),
        .workspace_bytes = requirements.workspace_bytes,
    };
}

void validate_exhaustive_results(
    std::vector<normalised_search_result> const& results,
    workload const& settings,
    uint32_t minimum_matches
) {
    std::pair<uint32_t, uint32_t> previous_id{};
    bool first = true;
    for (auto const& result : results) {
        auto const id = std::pair{result.query_id, result.reference_id};
        if (result.query_id >= settings.query_count ||
            result.reference_id >= settings.reference_count ||
            result.summary.counts.equal < minimum_matches || (!first && id <= previous_id)) {
            throw std::runtime_error("exhaustive search returned invalid or unstable results");
        }
        previous_id = id;
        first = false;
    }
}

struct recall_metrics {
    uint64_t oracle_pairs{};
    uint64_t recalled_pairs{};
    double recall{};
};

[[nodiscard]] recall_metrics validate_indexed_results(
    std::vector<normalised_search_result> const& oracle,
    std::vector<normalised_search_result> indexed,
    workload const& settings,
    uint32_t minimum_matches
) {
    auto const by_id = [](auto const& left, auto const& right) {
        return std::pair{left.query_id, left.reference_id} <
               std::pair{right.query_id, right.reference_id};
    };
    auto const oracle_positive = static_cast<uint64_t>(
        std::count_if(oracle.begin(), oracle.end(), [minimum_matches](auto const& result) {
            return result.summary.counts.equal >= minimum_matches;
        })
    );
    std::sort(indexed.begin(), indexed.end(), by_id);
    uint64_t recalled_pairs = 0;
    std::pair<uint32_t, uint32_t> previous_id{};
    bool first = true;
    for (auto const& result : indexed) {
        auto const id = std::pair{result.query_id, result.reference_id};
        if (result.query_id >= settings.query_count ||
            result.reference_id >= settings.reference_count || (!first && id <= previous_id)) {
            throw std::runtime_error("indexed search returned invalid or unstable results");
        }
        auto const oracle_result = std::lower_bound(
            oracle.begin(), oracle.end(), id, [](auto const& candidate, auto const& target) {
                return std::pair{candidate.query_id, candidate.reference_id} < target;
            }
        );
        if (oracle_result == oracle.end() ||
            std::pair{oracle_result->query_id, oracle_result->reference_id} != id ||
            oracle_result->summary != result.summary) {
            throw std::runtime_error("indexed retained summary disagrees with exhaustive oracle");
        }
        recalled_pairs += result.summary.counts.equal >= minimum_matches;
        previous_id = id;
        first = false;
    }
    return {
        .oracle_pairs = oracle_positive,
        .recalled_pairs = recalled_pairs,
        .recall =
            oracle_positive == 0U ? 1.0 : static_cast<double>(recalled_pairs) / oracle_positive,
    };
}

[[nodiscard]] bool preflight(
    nvbench::state& state,
    workload const& settings,
    size_t required_device_bytes,
    cuddl::score_compatibility const& compatibility
) {
    size_t free_memory = 0;
    size_t total_memory = 0;
    CUDDL_CUDA_CALL(cudaMemGetInfo(&free_memory, &total_memory));
    add_common_metadata(state, settings, free_memory, compatibility);
    if (required_device_bytes > free_memory - free_memory / 10U) {
        state.skip(
            "estimated required device bytes " + std::to_string(required_device_bytes) +
            " exceed 90% of free bytes " + std::to_string(free_memory)
        );
        return false;
    }
    return true;
}

indexed_fixture_metrics add_fixture_metrics(
    nvbench::state& state,
    fixture const& data,
    workload const& settings,
    double generation_ms,
    double transfer_ms,
    index_mode const* mode = nullptr
) {
    add_value(state, "fixture_generation_ms", generation_ms);
    add_value(state, "host_to_device_ms", transfer_ms);
    indexed_fixture_metrics mode_metrics{};
    if (mode != nullptr) {
        mode_metrics = make_indexed_fixture_metrics(data, settings, *mode);
    }
    auto const top_bit_count =
        mode == nullptr
            ? std::accumulate(
                  data.bucket_top_bit_counts.begin(), data.bucket_top_bit_counts.end(), uint64_t{0}
              )
            : mode_metrics.top_bit_count;
    auto const nonzero_scores =
        mode == nullptr ? data.posting_entries : mode_metrics.nonzero_scores;
    add_value(
        state,
        "top_bit_frequency",
        nonzero_scores == 0U
            ? 0.0
            : static_cast<double>(top_bit_count) / static_cast<double>(nonzero_scores)
    );
    if (mode == nullptr) {
        add_value(
            state, "populated_posting_lists", static_cast<double>(data.populated_posting_lists)
        );
        add_value(state, "posting_entries", static_cast<double>(data.posting_entries));
        return mode_metrics;
    }
    auto const compatibility = indexed_compatibility(*mode);
    auto const storage = indexed_storage(settings.reference_count, compatibility);
    add_value(
        state, "populated_posting_lists", static_cast<double>(mode_metrics.populated_posting_lists)
    );
    add_value(state, "posting_entries", static_cast<double>(mode_metrics.posting_entries));
    add_value(state, "posting_visits", static_cast<double>(mode_metrics.posting_visits));
    add_value(state, "offset_bytes", static_cast<double>(storage.offset_bytes));
    add_value(state, "posting_bytes", static_cast<double>(storage.posting_bytes));
    return mode_metrics;
}

void compact_build(nvbench::state& state) {
    auto const settings = read_workload(state, false);
    auto const compatibility = cuddl::score_compatibility::current<k_kmer_length, k_bucket_count>();
    auto const row_bytes = database_type::persistent_row_bytes(settings.reference_count);
    if (!preflight(state, settings, 2U * row_bytes, compatibility)) {
        return;
    }

    auto started = clock_type::now();
    auto const data = make_fixture(settings);
    auto const generation_ms =
        std::chrono::duration<double, std::milli>(clock_type::now() - started).count();
    started = clock_type::now();
    thrust::device_vector<uint16_t> device_rows(data.rows);
    CUDDL_CUDA_CALL(cudaDeviceSynchronize());
    auto const transfer_ms =
        std::chrono::duration<double, std::milli>(clock_type::now() - started).count();
    add_fixture_metrics(state, data, settings, generation_ms, transfer_ms);
    add_value(state, "resident_bytes", static_cast<double>(row_bytes));
    add_value(state, "build_temporary_bytes", 0.0);
    state.add_element_count(data.rows.size(), "Scores Built");

    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
        auto database = CUDDL_UNWRAP(
            database_type::build_async(
                device_rows, compatibility, cuda::stream_ref{launch.get_stream()}
            )
        );
        do_not_optimise(database);
    });
}

void indexed_build(nvbench::state& state) {
    auto const settings = read_workload(state, false);
    auto const mode = read_index_mode(state);
    auto const compatibility = indexed_compatibility(mode);
    auto const row_bytes = database_type::persistent_row_bytes(settings.reference_count);
    auto const resident_bytes = indexed_resident_bytes(settings.reference_count, compatibility);
    auto const temporary_bytes = indexed_build_temporary_bytes(compatibility);
    if (!preflight(state, settings, row_bytes + resident_bytes + temporary_bytes, compatibility)) {
        return;
    }

    auto started = clock_type::now();
    auto const data = make_fixture(settings);
    auto const generation_ms =
        std::chrono::duration<double, std::milli>(clock_type::now() - started).count();
    started = clock_type::now();
    thrust::device_vector<uint16_t> device_rows(data.rows);
    CUDDL_CUDA_CALL(cudaDeviceSynchronize());
    auto const transfer_ms =
        std::chrono::duration<double, std::milli>(clock_type::now() - started).count();
    add_fixture_metrics(state, data, settings, generation_ms, transfer_ms, &mode);
    add_value(state, "resident_bytes", static_cast<double>(resident_bytes));
    add_value(state, "build_temporary_bytes", static_cast<double>(temporary_bytes));
    state.add_element_count(data.rows.size(), "Scores Indexed");

    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
        auto database = CUDDL_UNWRAP(
            database_type::build_indexed_async(
                device_rows, compatibility, cuda::stream_ref{launch.get_stream()}
            )
        );
        do_not_optimise(database);
    });
}

void exhaustive_search(nvbench::state& state) {
    auto const settings = read_workload(state, true);
    auto const compatibility = cuddl::score_compatibility::current<k_kmer_length, k_bucket_count>();
    auto const row_bytes = database_type::persistent_row_bytes(settings.reference_count);
    auto const query_bytes =
        static_cast<size_t>(settings.query_count) * k_bucket_count * sizeof(uint16_t);
    auto const result_bytes = static_cast<size_t>(settings.reference_count) * settings.query_count *
                              sizeof(cuddl::batch_search_result);
    auto const search_temporary_bytes = 2U * result_bytes + 32U * 1024U * 1024U;
    if (!preflight(
            state, settings, 2U * row_bytes + query_bytes + search_temporary_bytes, compatibility
        )) {
        return;
    }

    auto started = clock_type::now();
    auto const data = make_fixture(settings);
    auto const generation_ms =
        std::chrono::duration<double, std::milli>(clock_type::now() - started).count();
    started = clock_type::now();
    thrust::device_vector<uint16_t> device_rows(data.rows);
    thrust::device_vector<uint16_t> device_queries(data.queries);
    CUDDL_CUDA_CALL(cudaDeviceSynchronize());
    auto const transfer_ms =
        std::chrono::duration<double, std::milli>(clock_type::now() - started).count();
    add_fixture_metrics(state, data, settings, generation_ms, transfer_ms);

    auto database = CUDDL_UNWRAP(database_type::build_async(device_rows, compatibility));
    CUDDL_CUDA_CALL(cudaDeviceSynchronize());
    auto output = execute_exhaustive_search(
        database, device_queries, settings, compatibility, k_minimum_matches, [&](auto&& launch) {
            state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& nvbench_launch) {
                launch(nvbench_launch.get_stream());
            });
        }
    );

    started = clock_type::now();
    validate_exhaustive_results(output.results, settings, k_minimum_matches);
    auto const validation_ms =
        std::chrono::duration<double, std::milli>(clock_type::now() - started).count();
    add_value(state, "validation_ms", validation_ms);
    add_value(state, "resident_bytes", static_cast<double>(database.persistent_row_bytes()));
    add_value(state, "search_workspace_bytes", static_cast<double>(output.workspace_bytes));
    add_value(
        state,
        "exact_comparisons",
        static_cast<double>(settings.reference_count) * settings.query_count
    );
    add_value(state, "threshold_zero_recall", 1.0);
    state.add_element_count(settings.query_count, "Queries");
}

void indexed_search(nvbench::state& state) {
    auto const settings = read_workload(state, true);
    auto const mode = read_index_mode(state);
    auto const indexed_metadata = indexed_compatibility(mode);
    auto const exhaustive_metadata =
        cuddl::score_compatibility::current<k_kmer_length, k_bucket_count>();
    auto const row_bytes = database_type::persistent_row_bytes(settings.reference_count);
    auto const query_bytes =
        static_cast<size_t>(settings.query_count) * k_bucket_count * sizeof(uint16_t);
    auto const resident_bytes = indexed_resident_bytes(settings.reference_count, indexed_metadata);
    auto const build_temporary_bytes = indexed_build_temporary_bytes(indexed_metadata);
    auto const pair_count = static_cast<size_t>(settings.reference_count) * settings.query_count;
    auto const result_bytes = pair_count * sizeof(cuddl::batch_search_result);
    auto const indexed_search_temporary_bytes =
        result_bytes + pair_count * 2U * sizeof(uint32_t) + 32U * 1024U * 1024U;
    auto const exhaustive_search_temporary_bytes = 2U * result_bytes + 32U * 1024U * 1024U;
    auto const exhaustive_phase = 2U * row_bytes + query_bytes + exhaustive_search_temporary_bytes;
    auto const indexed_phase = row_bytes + query_bytes + resident_bytes +
                               std::max(build_temporary_bytes, indexed_search_temporary_bytes);
    if (!preflight(state, settings, std::max(exhaustive_phase, indexed_phase), indexed_metadata)) {
        return;
    }

    auto started = clock_type::now();
    auto const data = make_fixture(settings);
    auto const generation_ms =
        std::chrono::duration<double, std::milli>(clock_type::now() - started).count();
    started = clock_type::now();
    thrust::device_vector<uint16_t> device_rows(data.rows);
    thrust::device_vector<uint16_t> device_queries(data.queries);
    CUDDL_CUDA_CALL(cudaDeviceSynchronize());
    auto const transfer_ms =
        std::chrono::duration<double, std::milli>(clock_type::now() - started).count();
    auto const indexed_metrics =
        add_fixture_metrics(state, data, settings, generation_ms, transfer_ms, &mode);

    search_output exhaustive;
    search_output oracle;
    {
        auto database = CUDDL_UNWRAP(database_type::build_async(device_rows, exhaustive_metadata));
        CUDDL_CUDA_CALL(cudaDeviceSynchronize());
        auto const execute_once = [](auto&& launch) {
            launch(cudaStream_t{nullptr});
            CUDDL_CUDA_CALL(cudaDeviceSynchronize());
        };
        exhaustive = execute_exhaustive_search(
            database,
            device_queries,
            settings,
            exhaustive_metadata,
            k_minimum_matches,
            [&](auto&& launch) {
                state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& nvbench_launch) {
                    launch(nvbench_launch.get_stream());
                });
            }
        );
        oracle = execute_exhaustive_search(
            database, device_queries, settings, exhaustive_metadata, 0U, execute_once
        );
    }

    auto database = CUDDL_UNWRAP(database_type::build_indexed_async(device_rows, indexed_metadata));
    CUDDL_CUDA_CALL(cudaDeviceSynchronize());
    auto indexed = execute_indexed_search(
        database,
        device_queries,
        settings,
        indexed_metadata,
        k_minimum_matches,
        [&](auto&& launch) {
            state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& nvbench_launch) {
                launch(nvbench_launch.get_stream());
            });
        }
    );

    started = clock_type::now();
    validate_exhaustive_results(oracle.results, settings, 0U);
    validate_exhaustive_results(exhaustive.results, settings, k_minimum_matches);
    auto const recall =
        validate_indexed_results(oracle.results, indexed.results, settings, k_minimum_matches);
    auto const validation_ms =
        std::chrono::duration<double, std::milli>(clock_type::now() - started).count();
    add_value(state, "validation_ms", validation_ms);
    add_value(
        state,
        "resident_bytes",
        static_cast<double>(database.persistent_row_bytes() + database.persistent_index_bytes())
    );
    add_value(state, "search_workspace_bytes", static_cast<double>(indexed.workspace_bytes));
    add_value(state, "atomic_updates", static_cast<double>(indexed_metrics.posting_visits));
    auto const selected_candidates = static_cast<double>(indexed.results.size());
    add_value(state, "selected_candidates", selected_candidates);
    add_value(state, "exact_comparisons", selected_candidates);
    add_value(
        state,
        "candidate_inflation",
        recall.oracle_pairs == 0U
            ? 0.0
            : selected_candidates / static_cast<double>(recall.oracle_pairs)
    );
    add_value(state, "threshold_zero_recall", recall.recall);
    add_value(state, "exact_result_recall", recall.recall);
    add_value(state, "oracle_threshold_pairs", static_cast<double>(recall.oracle_pairs));
    add_value(state, "recalled_pairs", static_cast<double>(recall.recalled_pairs));
    state.add_element_count(settings.query_count, "Queries");
}

}  // namespace

NVBENCH_BENCH(compact_build)
    .add_int64_axis("References", reference_counts)
    .add_int64_axis("FillPermille", fill_permilles)
    .add_int64_axis("HotPercent", hot_percentages)
    .set_stopping_criterion("sample-count")
    .set_min_samples(20)
    .set_criterion_param_int64("target-samples", 20)
    .set_cold_warmup_runs(3)
    .set_skip_batched(true);

NVBENCH_BENCH(indexed_build)
    .add_int64_axis("References", reference_counts)
    .add_int64_axis("FillPermille", fill_permilles)
    .add_int64_axis("HotPercent", hot_percentages)
    .add_int64_axis("IndexedBuckets", indexed_bucket_counts)
    .add_int64_axis("KeyBits", key_bits)
    .set_stopping_criterion("sample-count")
    .set_min_samples(20)
    .set_criterion_param_int64("target-samples", 20)
    .set_cold_warmup_runs(3)
    .set_skip_batched(true);

NVBENCH_BENCH(exhaustive_search)
    .add_int64_axis("References", reference_counts)
    .add_int64_axis("FillPermille", fill_permilles)
    .add_int64_axis("HotPercent", hot_percentages)
    .add_int64_axis("Queries", query_counts)
    .add_string_axis("QueryProfile", query_profiles)
    .set_stopping_criterion("sample-count")
    .set_min_samples(20)
    .set_criterion_param_int64("target-samples", 20)
    .set_cold_warmup_runs(3)
    .set_skip_batched(true);

NVBENCH_BENCH(indexed_search)
    .add_int64_axis("References", reference_counts)
    .add_int64_axis("FillPermille", fill_permilles)
    .add_int64_axis("HotPercent", hot_percentages)
    .add_int64_axis("Queries", query_counts)
    .add_string_axis("QueryProfile", query_profiles)
    .add_int64_axis("IndexedBuckets", indexed_bucket_counts)
    .add_int64_axis("KeyBits", key_bits)
    .set_stopping_criterion("sample-count")
    .set_min_samples(20)
    .set_criterion_param_int64("target-samples", 20)
    .set_cold_warmup_runs(3)
    .set_skip_batched(true);
