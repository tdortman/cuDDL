#include <cuddl/cuddl.cuh>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>
#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
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
constexpr double k_hot_fraction = 0.5;
constexpr uint32_t k_minimum_matches = 5;
constexpr uint64_t k_score_count = uint64_t{1} << 16U;
constexpr uint64_t k_index_cell_count = k_bucket_count * k_score_count;
constexpr uint32_t k_score_period = std::numeric_limits<uint16_t>::max();

std::vector<nvbench::int64_t> const reference_counts{1024, 16384, 200687};
std::vector<nvbench::int64_t> const fill_permilles{250, 1000};
std::vector<std::string> const skews{"uniform", "hot"};
std::vector<nvbench::int64_t> const query_counts{1, 32};

struct workload {
    uint32_t reference_count{};
    double fill_ratio{};
    std::string skew;
    uint32_t query_count{};
};

struct fixture {
    std::vector<uint16_t> rows;
    std::vector<uint16_t> queries;
    uint64_t populated_posting_lists{};
    uint64_t posting_entries{};
    uint64_t posting_visits{};
};

struct normalised_search_result {
    uint32_t query_id{};
    uint32_t reference_id{};
    cuddl::pairwise_summary summary{};

    friend bool operator==(normalised_search_result const&, normalised_search_result const&) =
        default;
};

struct search_output {
    std::vector<normalised_search_result> results;
    size_t workspace_bytes{};
};

struct minimum_matches_predicate {
    template <typename Result>
    [[nodiscard]] __host__ __device__ bool operator()(Result const& result) const {
        return result.summary.counts.equal >= k_minimum_matches;
    }
};

[[nodiscard]] workload read_workload(nvbench::state& state, bool includes_queries) {
    auto const references = state.get_int64("References");
    auto const fill_permille = state.get_int64("FillPermille");
    auto const queries = includes_queries ? state.get_int64("Queries") : 1;
    if (references <= 0 || references > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("reference count exceeds the supported 32-bit range");
    }
    if (fill_permille <= 0 || fill_permille > 1000) {
        throw std::runtime_error("fill permille must be in [1, 1000]");
    }
    if (queries <= 0 || queries > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("query count exceeds the supported 32-bit range");
    }
    if (static_cast<uint64_t>(references) * static_cast<uint64_t>(queries) >
        std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("query/reference pairs exceed the supported 32-bit range");
    }
    return {
        .reference_count = static_cast<uint32_t>(references),
        .fill_ratio = static_cast<double>(fill_permille) / 1000.0,
        .skew = state.get_string("Skew"),
        .query_count = static_cast<uint32_t>(queries),
    };
}

void add_string(nvbench::state& state, char const* name, std::string value) {
    auto& summary = state.add_summary(name);
    summary.set_string("name", name);
    summary.set_string("value", std::move(value));
}

void add_common_metadata(nvbench::state& state, workload const& value, size_t free_memory_bytes) {
    auto const compatibility = cuddl::score_compatibility::current<k_kmer_length, k_bucket_count>();
    int runtime_version = 0;
    int driver_version = 0;
    CUDDL_CUDA_CALL(cudaRuntimeGetVersion(&runtime_version));
    CUDDL_CUDA_CALL(cudaDriverGetVersion(&driver_version));

    add_value(state, "fixture_seed", static_cast<double>(k_fixture_seed));
    add_value(state, "hot_fraction", k_hot_fraction);
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

[[nodiscard]] size_t indexed_build_temporary_bytes() {
    size_t scan_bytes = 0;
    uint32_t* offsets = nullptr;
    CUDDL_CUDA_CALL(
        cub::DeviceScan::ExclusiveSum(
            nullptr,
            scan_bytes,
            offsets,
            static_cast<int64_t>(k_index_cell_count + 1U),
            cudaStream_t{nullptr}
        )
    );
    return align_up(scan_bytes, alignof(uint32_t)) +
           static_cast<size_t>(k_index_cell_count) * sizeof(uint32_t);
}

[[nodiscard]] size_t indexed_resident_bytes(uint32_t reference_count) {
    auto const row_bytes = database_type::persistent_row_bytes(reference_count);
    auto const offset_bytes = static_cast<size_t>(k_index_cell_count + 1U) * sizeof(uint32_t);
    auto const posting_bytes = row_bytes / sizeof(uint16_t) * sizeof(uint32_t);
    return row_bytes + offset_bytes + posting_bytes;
}

[[nodiscard]] uint64_t congruent_reference_count(uint32_t reference_count, uint32_t reference_id) {
    auto const residue = reference_id % k_score_period;
    return 1U + (static_cast<uint64_t>(reference_count) - 1U - residue) / k_score_period;
}

[[nodiscard]] fixture make_fixture(workload const& settings) {
    auto const fill_count = std::clamp<size_t>(
        static_cast<size_t>(std::llround(settings.fill_ratio * k_bucket_count)), 1U, k_bucket_count
    );
    auto const hot_count =
        settings.skew == "hot"
            ? std::clamp<size_t>(
                  static_cast<size_t>(std::llround(k_hot_fraction * fill_count)), 1U, fill_count
              )
            : 0U;

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

    fixture value{
        .rows =
            std::vector<uint16_t>(static_cast<size_t>(settings.reference_count) * k_bucket_count),
        .queries =
            std::vector<uint16_t>(static_cast<size_t>(settings.query_count) * k_bucket_count),
        .populated_posting_lists =
            hot_count +
            (fill_count - hot_count) * std::min<uint64_t>(settings.reference_count, k_score_period),
        .posting_entries = static_cast<uint64_t>(settings.reference_count) * fill_count,
    };

    for (uint32_t reference_id = 0; reference_id < settings.reference_count; ++reference_id) {
        for (size_t index = 0; index < fill_count; ++index) {
            value.rows[static_cast<size_t>(reference_id) * k_bucket_count + buckets[index]] =
                index < hot_count ? 1U
                                  : static_cast<uint16_t>(
                                        1U + (static_cast<uint64_t>(reference_id) + shifts[index]) %
                                                 k_score_period
                                    );
        }
    }

    for (uint32_t query_id = 0; query_id < settings.query_count; ++query_id) {
        auto const reference_id = static_cast<uint32_t>(
            cuddl::detail::splitmix64(k_fixture_seed + 0x9e3779b97f4a7c15ULL + query_id) %
            settings.reference_count
        );
        std::memcpy(
            value.queries.data() + static_cast<size_t>(query_id) * k_bucket_count,
            value.rows.data() + static_cast<size_t>(reference_id) * k_bucket_count,
            k_bucket_count * sizeof(uint16_t)
        );
        value.posting_visits +=
            static_cast<uint64_t>(hot_count) * settings.reference_count +
            static_cast<uint64_t>(fill_count - hot_count) *
                congruent_reference_count(settings.reference_count, reference_id);
    }
    return value;
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
    Execute&& execute
) {
    auto const compatibility = cuddl::score_compatibility::current<k_kmer_length, k_bucket_count>();
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
                minimum_matches_predicate{}
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
                    minimum_matches_predicate{},
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
            minimum_matches_predicate{}
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
                minimum_matches_predicate{},
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
    Execute&& execute
) {
    auto const compatibility = cuddl::score_compatibility::current<k_kmer_length, k_bucket_count>();
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
                {.minimum_matches = k_minimum_matches},
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
            {.minimum_matches = k_minimum_matches},
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
    workload const& settings
) {
    std::pair<uint32_t, uint32_t> previous_id{};
    bool first = true;
    for (auto const& result : results) {
        auto const id = std::pair{result.query_id, result.reference_id};
        if (result.query_id >= settings.query_count ||
            result.reference_id >= settings.reference_count ||
            result.summary.counts.equal < k_minimum_matches || (!first && id <= previous_id)) {
            throw std::runtime_error(
                "thresholded exhaustive search returned invalid or unstable results"
            );
        }
        previous_id = id;
        first = false;
    }
}

void validate_indexed_results(
    std::vector<normalised_search_result> const& exhaustive,
    std::vector<normalised_search_result> indexed
) {
    auto const by_id = [](auto const& left, auto const& right) {
        return std::pair{left.query_id, left.reference_id} <
               std::pair{right.query_id, right.reference_id};
    };
    std::sort(indexed.begin(), indexed.end(), by_id);
    if (indexed != exhaustive) {
        throw std::runtime_error(
            "indexed candidate IDs or exact summaries disagree with exhaustive search"
        );
    }
}

[[nodiscard]] bool
preflight(nvbench::state& state, workload const& settings, size_t required_device_bytes) {
    size_t free_memory = 0;
    size_t total_memory = 0;
    CUDDL_CUDA_CALL(cudaMemGetInfo(&free_memory, &total_memory));
    add_common_metadata(state, settings, free_memory);
    if (required_device_bytes > free_memory - free_memory / 10U) {
        state.skip(
            "estimated required device bytes " + std::to_string(required_device_bytes) +
            " exceed 90% of free bytes " + std::to_string(free_memory)
        );
        return false;
    }
    return true;
}

void add_fixture_metrics(
    nvbench::state& state,
    fixture const& data,
    double generation_ms,
    double transfer_ms
) {
    add_value(state, "fixture_generation_ms", generation_ms);
    add_value(state, "host_to_device_ms", transfer_ms);
    add_value(state, "populated_posting_lists", static_cast<double>(data.populated_posting_lists));
    add_value(state, "posting_entries", static_cast<double>(data.posting_entries));
}

void compact_build(nvbench::state& state) {
    auto const settings = read_workload(state, false);
    auto const row_bytes = database_type::persistent_row_bytes(settings.reference_count);
    if (!preflight(state, settings, 2U * row_bytes)) {
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
    add_fixture_metrics(state, data, generation_ms, transfer_ms);
    add_value(state, "resident_bytes", static_cast<double>(row_bytes));
    add_value(state, "build_temporary_bytes", 0.0);
    state.add_element_count(data.rows.size(), "Scores Built");

    auto const compatibility = cuddl::score_compatibility::current<k_kmer_length, k_bucket_count>();
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
    auto const row_bytes = database_type::persistent_row_bytes(settings.reference_count);
    auto const resident_bytes = indexed_resident_bytes(settings.reference_count);
    auto const temporary_bytes = indexed_build_temporary_bytes();
    if (!preflight(state, settings, row_bytes + resident_bytes + temporary_bytes)) {
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
    add_fixture_metrics(state, data, generation_ms, transfer_ms);
    add_value(state, "resident_bytes", static_cast<double>(resident_bytes));
    add_value(state, "build_temporary_bytes", static_cast<double>(temporary_bytes));
    state.add_element_count(data.rows.size(), "Scores Indexed");

    auto const compatibility = cuddl::score_compatibility::current<k_kmer_length, k_bucket_count>();
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
    auto const row_bytes = database_type::persistent_row_bytes(settings.reference_count);
    auto const query_bytes =
        static_cast<size_t>(settings.query_count) * k_bucket_count * sizeof(uint16_t);
    auto const result_bytes = static_cast<size_t>(settings.reference_count) * settings.query_count *
                              sizeof(cuddl::batch_search_result);
    auto const search_temporary_bytes = 2U * result_bytes + 32U * 1024U * 1024U;
    if (!preflight(state, settings, 2U * row_bytes + query_bytes + search_temporary_bytes)) {
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
    add_fixture_metrics(state, data, generation_ms, transfer_ms);

    auto const compatibility = cuddl::score_compatibility::current<k_kmer_length, k_bucket_count>();
    auto database = CUDDL_UNWRAP(database_type::build_async(device_rows, compatibility));
    CUDDL_CUDA_CALL(cudaDeviceSynchronize());
    auto output = execute_exhaustive_search(database, device_queries, settings, [&](auto&& launch) {
        state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& nvbench_launch) {
            launch(nvbench_launch.get_stream());
        });
    });

    started = clock_type::now();
    validate_exhaustive_results(output.results, settings);
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
    state.add_element_count(settings.query_count, "Queries");
}

void indexed_search(nvbench::state& state) {
    auto const settings = read_workload(state, true);
    auto const row_bytes = database_type::persistent_row_bytes(settings.reference_count);
    auto const query_bytes =
        static_cast<size_t>(settings.query_count) * k_bucket_count * sizeof(uint16_t);
    auto const resident_bytes = indexed_resident_bytes(settings.reference_count);
    auto const build_temporary_bytes = indexed_build_temporary_bytes();
    auto const pair_count = static_cast<size_t>(settings.reference_count) * settings.query_count;
    auto const result_bytes = pair_count * sizeof(cuddl::batch_search_result);
    auto const indexed_search_temporary_bytes =
        result_bytes + pair_count * 2U * sizeof(uint32_t) + 32U * 1024U * 1024U;
    auto const exhaustive_search_temporary_bytes = 2U * result_bytes + 32U * 1024U * 1024U;
    auto const phase_temporary_bytes = std::max({
        build_temporary_bytes,
        indexed_search_temporary_bytes,
        exhaustive_search_temporary_bytes,
    });
    if (!preflight(
            state, settings, row_bytes + query_bytes + resident_bytes + phase_temporary_bytes
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
    add_fixture_metrics(state, data, generation_ms, transfer_ms);

    auto const compatibility = cuddl::score_compatibility::current<k_kmer_length, k_bucket_count>();
    auto database = CUDDL_UNWRAP(database_type::build_indexed_async(device_rows, compatibility));
    CUDDL_CUDA_CALL(cudaDeviceSynchronize());

    auto const execute_once = [](auto&& launch) {
        launch(cudaStream_t{nullptr});
        CUDDL_CUDA_CALL(cudaDeviceSynchronize());
    };
    auto const exhaustive =
        execute_exhaustive_search(database, device_queries, settings, execute_once);
    auto indexed = execute_indexed_search(database, device_queries, settings, [&](auto&& launch) {
        state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& nvbench_launch) {
            launch(nvbench_launch.get_stream());
        });
    });

    started = clock_type::now();
    validate_exhaustive_results(exhaustive.results, settings);
    validate_indexed_results(exhaustive.results, indexed.results);
    auto const validation_ms =
        std::chrono::duration<double, std::milli>(clock_type::now() - started).count();
    add_value(state, "validation_ms", validation_ms);
    add_value(
        state,
        "resident_bytes",
        static_cast<double>(database.persistent_row_bytes() + database.persistent_index_bytes())
    );
    add_value(state, "search_workspace_bytes", static_cast<double>(indexed.workspace_bytes));
    add_value(state, "posting_visits", static_cast<double>(data.posting_visits));
    add_value(state, "atomic_updates", static_cast<double>(data.posting_visits));
    add_value(state, "selected_candidates", static_cast<double>(indexed.results.size()));
    add_value(state, "exact_comparisons", static_cast<double>(indexed.results.size()));
    add_value(state, "exact_result_recall", 1.0);
    state.add_element_count(settings.query_count, "Queries");
}

}  // namespace

NVBENCH_BENCH(compact_build)
    .add_int64_axis("References", reference_counts)
    .add_int64_axis("FillPermille", fill_permilles)
    .add_string_axis("Skew", skews)
    .set_stopping_criterion("sample-count")
    .set_min_samples(20)
    .set_criterion_param_int64("target-samples", 20)
    .set_cold_warmup_runs(3)
    .set_skip_batched(true);

NVBENCH_BENCH(indexed_build)
    .add_int64_axis("References", reference_counts)
    .add_int64_axis("FillPermille", fill_permilles)
    .add_string_axis("Skew", skews)
    .set_stopping_criterion("sample-count")
    .set_min_samples(20)
    .set_criterion_param_int64("target-samples", 20)
    .set_cold_warmup_runs(3)
    .set_skip_batched(true);

NVBENCH_BENCH(exhaustive_search)
    .add_int64_axis("References", reference_counts)
    .add_int64_axis("FillPermille", fill_permilles)
    .add_string_axis("Skew", skews)
    .add_int64_axis("Queries", query_counts)
    .set_stopping_criterion("sample-count")
    .set_min_samples(20)
    .set_criterion_param_int64("target-samples", 20)
    .set_cold_warmup_runs(3)
    .set_skip_batched(true);

NVBENCH_BENCH(indexed_search)
    .add_int64_axis("References", reference_counts)
    .add_int64_axis("FillPermille", fill_permilles)
    .add_string_axis("Skew", skews)
    .add_int64_axis("Queries", query_counts)
    .set_stopping_criterion("sample-count")
    .set_min_samples(20)
    .set_criterion_param_int64("target-samples", 20)
    .set_cold_warmup_runs(3)
    .set_skip_batched(true);
