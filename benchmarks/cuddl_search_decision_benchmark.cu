#include <cuddl/a48.hpp>
#include <cuddl/cuddl.cuh>
#include <cuddl/refseq_parity.hpp>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>
#include <nvbench/main.cuh>
#include <nvbench/nvbench.cuh>
#include <nvbench/version.cuh>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "common.cuh"
#include "compressed_file.hpp"
#include "result_json.hpp"

namespace {

using clock_type = std::chrono::steady_clock;
using database_type = cuddl::reference_database<25, 2048>;
using refseq_register_layout = cuddl::register_layout<5, 11>;
using refseq_database_type = cuddl::reference_database<25, 4096, refseq_register_layout>;

constexpr uint32_t k_kmer_length = 25;
constexpr size_t k_bucket_count = 2048;
constexpr uint64_t k_fixture_seed = 4242;
constexpr nvbench::int64_t k_max_hot_percentage = 50;
constexpr uint32_t k_minimum_matches = 5;
// Fixture scores occupy the 15-bit space: the RefSeq DB's top register bit is
// empirically dead (paper Section 8.2: set in 461 of 383M registers), so the
// 15-bit key masking the paper recommends (Section 8.6) folds ~nothing.
constexpr uint32_t k_score_period = 0x7FFFU;
// Fraction of references that share the hot value inside a hot bucket. A hot
// register value is shared by thousands of unrelated sketches in real RefSeq
// data (paper, Section 9): ~1% of the 200,687 sketches, not all of them.
constexpr double k_hot_value_fraction = 0.01;

std::vector<nvbench::int64_t> const reference_counts{1024, 16384};

std::vector<nvbench::int64_t> const hot_percentages{0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50};
std::vector<nvbench::int64_t> const key_bits{15, 16};
std::vector<nvbench::int64_t> const
    refseq_reference_counts{1024, 2048, 4096, 8192, 16384, 32768, 49152, 65536, 98304, 148108};
constexpr uint32_t k_refseq_bucket_count = 4096U;
constexpr uint32_t k_refseq_reference_count = 148108U;
constexpr uint32_t k_refseq_query_count = 8192U;
constexpr uint64_t k_refseq_asset_size = 1270805218ULL;
std::string g_refseq_asset_path = "data/refseqSketchDDL_k25e5b4096.tsv.gz";

struct workload {
    uint32_t reference_count{};
    double fill_ratio{};
    double hot_fraction{};
    uint32_t query_count{};
};

struct index_mode {
    uint32_t indexed_bucket_count{};
    uint16_t key_mask{};
    uint32_t key_bits{};
};

struct fixture {
    std::vector<uint16_t> rows;
    std::vector<uint64_t> bucket_top_bit_counts;
    std::vector<uint8_t> bucket_filled;
    std::vector<uint8_t> bucket_hot;
    std::vector<uint32_t> bucket_shifts;
    uint64_t populated_posting_lists{};
    uint64_t posting_entries{};
    uint32_t hot_reference_count{};
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

std::string axes_key(workload const& settings, index_mode const* mode);
void stash_state(std::string const& benchmark, std::string const& key, nvbench::state& state);

[[nodiscard]] workload read_workload(nvbench::state& state) {
    auto const references = state.get_int64("References");
    auto const hot_percentage = state.get_int64("HotPercent");
    if (references <= 0 || references > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("reference count exceeds the supported 32-bit range");
    }
    if (hot_percentage < 0 || hot_percentage > k_max_hot_percentage) {
        throw std::runtime_error("hot skew must be in [0%, 50%]");
    }
    return {
        .reference_count = static_cast<uint32_t>(references),
        .fill_ratio = 1.0,
        .hot_fraction = static_cast<double>(hot_percentage) / 100.0,
        .query_count = static_cast<uint32_t>(references),
    };
}
[[nodiscard]] index_mode read_index_mode(nvbench::state& state) {
    auto const bits = state.get_int64("KeyBits");
    if (bits != 15 && bits != 16) {
        throw std::runtime_error("indexed key width must be 15 or 16 bits");
    }
    return {
        .indexed_bucket_count = k_bucket_count,
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

void add_environment_metadata(nvbench::state& state, size_t free_memory_bytes) {
    int runtime_version = 0;
    int driver_version = 0;
    CUDDL_CUDA_CALL(cudaRuntimeGetVersion(&runtime_version));
    CUDDL_CUDA_CALL(cudaDriverGetVersion(&driver_version));
    static json const host = benchmark_host_system();
    add_string(state, "system_os", host.at("os").get<std::string>());
    add_string(state, "system_kernel", host.at("kernel").get<std::string>());
    add_string(state, "system_architecture", host.at("architecture").get<std::string>());
    add_string(state, "system_cpu", host.at("cpu").get<std::string>());
    add_string(
        state,
        "system_logical_cpu_count",
        std::to_string(host.at("logical_cpu_count").get<uint64_t>())
    );
    add_string(state, "system_ram_bytes", std::to_string(host.at("ram_bytes").get<uint64_t>()));
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
}

void add_compatibility_metadata(
    nvbench::state& state,
    cuddl::score_compatibility const& compatibility
) {
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
}

void add_common_metadata(
    nvbench::state& state,
    workload const& value,
    size_t free_memory_bytes,
    cuddl::score_compatibility const& compatibility
) {
    add_environment_metadata(state, free_memory_bytes);
    add_compatibility_metadata(state, compatibility);
    add_value(state, "fixture_seed", static_cast<double>(k_fixture_seed));
    add_value(state, "hot_fraction", value.hot_fraction);
    add_value(state, "minimum_matches", k_minimum_matches);
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

[[nodiscard]] uint64_t congruent_reference_count(
    uint32_t reference_count,
    uint32_t reference_id,
    uint32_t period = k_score_period
) {
    auto const residue = reference_id % period;
    if (residue >= reference_count) {
        return 0U;
    }
    return 1U + (static_cast<uint64_t>(reference_count) - 1U - residue) / period;
}

[[nodiscard]] fixture make_fixture(workload const& settings) {
    auto const fill_count = std::clamp<size_t>(
        static_cast<size_t>(std::llround(settings.fill_ratio * k_bucket_count)), 1U, k_bucket_count
    );
    auto const hot_count = std::clamp<size_t>(
        static_cast<size_t>(std::llround(settings.hot_fraction * fill_count)), 0U, fill_count
    );
    auto const hot_reference_count =
        hot_count == 0U ? 0U
                        : std::clamp<size_t>(
                              static_cast<size_t>(
                                  std::llround(k_hot_value_fraction * settings.reference_count)
                              ),
                              1U,
                              settings.reference_count
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

    fixture value{
        .rows =
            std::vector<uint16_t>(static_cast<size_t>(settings.reference_count) * k_bucket_count),
        .bucket_top_bit_counts = std::vector<uint64_t>(k_bucket_count),
        .bucket_filled = std::vector<uint8_t>(k_bucket_count),
        .bucket_hot = std::vector<uint8_t>(k_bucket_count),
        .bucket_shifts = std::vector<uint32_t>(k_bucket_count),
        .populated_posting_lists =
            hot_count *
                (1U + std::min<uint64_t>(
                          settings.reference_count - hot_reference_count, k_score_period - 1U
                      )) +
            (fill_count - hot_count) * std::min<uint64_t>(settings.reference_count, k_score_period),
        .posting_entries = static_cast<uint64_t>(settings.reference_count) * fill_count,
        .hot_reference_count = static_cast<uint32_t>(hot_reference_count),
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
                index < hot_count
                    ? (reference_id < hot_reference_count
                           ? uint16_t{1U}
                           : static_cast<uint16_t>(
                                 2U + (static_cast<uint64_t>(reference_id) + shifts[index]) %
                                          (k_score_period - 1U)
                             ))
                    : static_cast<uint16_t>(
                          1U +
                          (static_cast<uint64_t>(reference_id) + shifts[index]) % k_score_period
                      );
            value.rows[static_cast<size_t>(reference_id) * k_bucket_count + bucket] = score;
            value.bucket_top_bit_counts[bucket] += (score & 0x8000U) != 0U;
        }
    }

    return value;
}

[[nodiscard]] uint64_t distinct_index_keys(
    uint32_t reference_count,
    uint32_t shift,
    uint16_t key_mask,
    uint32_t period,
    uint32_t base = 1U
) {
    if (key_mask == std::numeric_limits<uint16_t>::max()) {
        return std::min<uint64_t>(reference_count, period);
    }
    auto const key_count = static_cast<size_t>(key_mask) + 1U;
    if (reference_count >= period) {
        return std::min<uint64_t>(period, key_count);
    }
    std::vector<uint8_t> seen(key_count);
    for (uint32_t reference_id = 0; reference_id < reference_count; ++reference_id) {
        auto const score =
            static_cast<uint16_t>(base + (static_cast<uint64_t>(reference_id) + shift) % period);
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
            if (score == 0U || score > k_score_period) {
                // Scores above the fixture period cannot occur, so a 15-bit
                // fold pair outside the score space contributes nothing.
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
            metrics.populated_posting_lists +=
                1U +
                distinct_index_keys(
                    static_cast<uint32_t>(settings.reference_count - data.hot_reference_count),
                    static_cast<uint32_t>(
                        static_cast<uint64_t>(data.hot_reference_count) + data.bucket_shifts[bucket]
                    ),
                    mode.key_mask,
                    k_score_period - 1U,
                    2U
                );
        } else {
            metrics.populated_posting_lists += distinct_index_keys(
                settings.reference_count, data.bucket_shifts[bucket], mode.key_mask, k_score_period
            );
        }
    }
    metrics.posting_entries = metrics.nonzero_scores;

    for (uint32_t query_id = 0; query_id < settings.query_count; ++query_id) {
        auto const query_offset = static_cast<size_t>(query_id) * k_bucket_count;
        for (uint32_t bucket = 0; bucket < mode.indexed_bucket_count; ++bucket) {
            auto const query_score = data.rows[query_offset + bucket];
            if (query_score == 0U || data.bucket_filled[bucket] == 0U) {
                continue;
            }
            if (data.bucket_hot[bucket] != 0U) {
                if ((query_score & mode.key_mask) == (uint16_t{1U} & mode.key_mask)) {
                    auto visits = static_cast<uint64_t>(data.hot_reference_count);
                    if (mode.key_mask != std::numeric_limits<uint16_t>::max() &&
                        0x8001U <= k_score_period) {
                        // 15-bit keys fold score 0x8001 (32769) into the hot key; count
                        // the non-hot references whose spread score lands there. The
                        // folded score only exists when the fixture spans 16 bits.
                        constexpr uint32_t period = k_score_period - 1U;
                        auto const shift = data.bucket_shifts[bucket];
                        auto const folded =
                            static_cast<uint32_t>((32767ULL + period - (shift % period)) % period);
                        visits +=
                            congruent_reference_count(settings.reference_count, folded, period) -
                            congruent_reference_count(data.hot_reference_count, folded, period);
                    }
                    metrics.posting_visits += visits;
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

struct exhaustive_search_buffers {
    thrust::device_vector<uint8_t> database_workspace;
    thrust::device_vector<cuddl::batch_search_result> exhaustive_results;
    thrust::device_vector<uint32_t> exhaustive_count;
    thrust::device_vector<cuddl::batch_search_result> selected_results;
    thrust::device_vector<uint32_t> selected_count;
    thrust::device_vector<uint8_t> selection_workspace;
    size_t selection_workspace_bytes{};
};

[[nodiscard]] exhaustive_search_buffers
make_exhaustive_search_buffers(database_type const& database, uint32_t minimum_matches) {
    auto const requirements = CUDDL_UNWRAP(database.all_to_all_search_requirements());
    size_t selection_workspace_bytes = 0;
    cuddl::batch_search_result* input = nullptr;
    cuddl::batch_search_result* output = nullptr;
    uint32_t* count = nullptr;
    CUDDL_CUDA_CALL(
        cub::DeviceSelect::If(
            nullptr,
            selection_workspace_bytes,
            input,
            output,
            count,
            static_cast<int64_t>(requirements.maximum_pair_count),
            minimum_matches_predicate{minimum_matches}
        )
    );
    return {
        .database_workspace = thrust::device_vector<uint8_t>(requirements.workspace_bytes),
        .exhaustive_results =
            thrust::device_vector<cuddl::batch_search_result>(requirements.maximum_pair_count),
        .exhaustive_count = thrust::device_vector<uint32_t>(1U),
        .selected_results =
            thrust::device_vector<cuddl::batch_search_result>(requirements.maximum_pair_count),
        .selected_count = thrust::device_vector<uint32_t>(1U),
        .selection_workspace = thrust::device_vector<uint8_t>(selection_workspace_bytes),
        .selection_workspace_bytes = selection_workspace_bytes,
    };
}

template <typename Execute>
[[nodiscard]] search_output execute_exhaustive_search(
    database_type const& database,
    uint32_t minimum_matches,
    Execute&& execute
) {
    auto buffers = make_exhaustive_search_buffers(database, minimum_matches);
    execute([&](cudaStream_t stream) {
        CUDDL_UNWRAP(database.search_all_to_all_async(
            buffers.database_workspace,
            buffers.exhaustive_results,
            buffers.exhaustive_count,
            [&](uint32_t pair_count) {
                CUDDL_CUDA_CALL(
                    cub::DeviceSelect::If(
                        thrust::raw_pointer_cast(buffers.selection_workspace.data()),
                        buffers.selection_workspace_bytes,
                        thrust::raw_pointer_cast(buffers.exhaustive_results.data()),
                        thrust::raw_pointer_cast(buffers.selected_results.data()),
                        thrust::raw_pointer_cast(buffers.selected_count.data()),
                        static_cast<int64_t>(pair_count),
                        minimum_matches_predicate{minimum_matches},
                        stream
                    )
                );
            },
            cuda::stream_ref{stream}
        ));
    });
    return {
        .results = {},
        .workspace_bytes = buffers.database_workspace.size() + buffers.selection_workspace.size(),
    };
}

[[nodiscard]] search_output
collect_exhaustive_results(database_type const& database, uint32_t minimum_matches) {
    auto buffers = make_exhaustive_search_buffers(database, minimum_matches);
    search_output output{
        .results = {},
        .workspace_bytes = buffers.database_workspace.size() + buffers.selection_workspace.size(),
    };
    CUDDL_UNWRAP(database.search_all_to_all_async(
        buffers.database_workspace,
        buffers.exhaustive_results,
        buffers.exhaustive_count,
        [&](uint32_t pair_count) {
            CUDDL_CUDA_CALL(
                cub::DeviceSelect::If(
                    thrust::raw_pointer_cast(buffers.selection_workspace.data()),
                    buffers.selection_workspace_bytes,
                    thrust::raw_pointer_cast(buffers.exhaustive_results.data()),
                    thrust::raw_pointer_cast(buffers.selected_results.data()),
                    thrust::raw_pointer_cast(buffers.selected_count.data()),
                    static_cast<int64_t>(pair_count),
                    minimum_matches_predicate{minimum_matches}
                )
            );
            CUDDL_CUDA_CALL(cudaDeviceSynchronize());
            auto tile = copy_normalised_results(
                buffers.selected_results,
                read_result_count(buffers.selected_count),
                [](auto const& result) {
                    return normalised_search_result{
                        result.query_id, result.reference_id, result.summary
                    };
                }
            );
            output.results.insert(
                output.results.end(),
                std::make_move_iterator(tile.begin()),
                std::make_move_iterator(tile.end())
            );
        }
    ));
    return output;
}

struct indexed_search_buffers {
    thrust::device_vector<uint8_t> workspace;
    thrust::device_vector<cuddl::batch_search_result> results;
    thrust::device_vector<uint32_t> result_count;
};

[[nodiscard]] indexed_search_buffers make_indexed_search_buffers(database_type const& database) {
    auto const requirements = CUDDL_UNWRAP(database.indexed_all_to_all_search_requirements());
    return {
        .workspace = thrust::device_vector<uint8_t>(requirements.workspace_bytes),
        .results =
            thrust::device_vector<cuddl::batch_search_result>(requirements.maximum_pair_count),
        .result_count = thrust::device_vector<uint32_t>(1U),
    };
}

template <typename Execute>
[[nodiscard]] search_output
execute_indexed_search(database_type const& database, uint32_t minimum_matches, Execute&& execute) {
    auto buffers = make_indexed_search_buffers(database);
    execute([&](cudaStream_t stream) {
        CUDDL_UNWRAP(database.search_all_to_all_indexed_async(
            buffers.workspace,
            buffers.results,
            buffers.result_count,
            [](uint32_t) {},
            {.minimum_matches = minimum_matches},
            cuda::stream_ref{stream}
        ));
    });
    return {.results = {}, .workspace_bytes = buffers.workspace.size()};
}

[[nodiscard]] search_output
collect_indexed_results(database_type const& database, uint32_t minimum_matches) {
    auto buffers = make_indexed_search_buffers(database);
    search_output output{.results = {}, .workspace_bytes = buffers.workspace.size()};
    CUDDL_UNWRAP(database.search_all_to_all_indexed_async(
        buffers.workspace,
        buffers.results,
        buffers.result_count,
        [&](uint32_t) {
            CUDDL_CUDA_CALL(cudaDeviceSynchronize());
            auto tile = copy_normalised_results(
                buffers.results, read_result_count(buffers.result_count), [](auto const& result) {
                    return normalised_search_result{
                        result.query_id, result.reference_id, result.summary
                    };
                }
            );
            output.results.insert(
                output.results.end(),
                std::make_move_iterator(tile.begin()),
                std::make_move_iterator(tile.end())
            );
        },
        {.minimum_matches = minimum_matches}
    ));
    return output;
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
        if (result.query_id >= settings.query_count || result.query_id >= result.reference_id ||
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
    auto const oracle_positive = static_cast<uint64_t>(oracle.size());
    std::sort(indexed.begin(), indexed.end(), by_id);
    uint64_t recalled_pairs = 0;
    std::pair<uint32_t, uint32_t> previous_id{};
    bool first = true;
    for (auto const& result : indexed) {
        auto const id = std::pair{result.query_id, result.reference_id};
        if (result.query_id >= settings.query_count || result.query_id >= result.reference_id ||
            result.reference_id >= settings.reference_count || (!first && id <= previous_id)) {
            throw std::runtime_error("indexed search returned invalid or unstable results");
        }
        if (result.summary.counts.equal < minimum_matches) {
            previous_id = id;
            first = false;
            continue;
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
        ++recalled_pairs;
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
    auto const settings = read_workload(state);
    auto const compatibility = cuddl::score_compatibility::current<k_kmer_length, k_bucket_count>();
    auto const row_bytes = database_type::persistent_row_bytes(settings.reference_count);
    if (!preflight(state, settings, 2U * row_bytes, compatibility)) {
        stash_state("compact_build", axes_key(settings, nullptr), state);
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
    stash_state("compact_build", axes_key(settings, nullptr), state);
}

void indexed_build(nvbench::state& state) {
    auto const settings = read_workload(state);
    auto const mode = read_index_mode(state);
    auto const compatibility = indexed_compatibility(mode);
    auto const row_bytes = database_type::persistent_row_bytes(settings.reference_count);
    auto const resident_bytes = indexed_resident_bytes(settings.reference_count, compatibility);
    auto const temporary_bytes = indexed_build_temporary_bytes(compatibility);
    if (!preflight(state, settings, row_bytes + resident_bytes + temporary_bytes, compatibility)) {
        stash_state("indexed_build", axes_key(settings, &mode), state);
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
    stash_state("indexed_build", axes_key(settings, &mode), state);
}

void exhaustive_search(nvbench::state& state) {
    auto const settings = read_workload(state);
    auto const compatibility = cuddl::score_compatibility::current<k_kmer_length, k_bucket_count>();
    auto const row_bytes = database_type::persistent_row_bytes(settings.reference_count);
    auto const result_capacity =
        database_type::all_to_all_result_capacity(settings.reference_count);
    auto const result_bytes =
        static_cast<size_t>(result_capacity) * sizeof(cuddl::batch_search_result);
    auto const search_temporary_bytes = 2U * result_bytes + 32U * 1024U * 1024U;
    if (!preflight(state, settings, 2U * row_bytes + search_temporary_bytes, compatibility)) {
        stash_state("exhaustive_search", axes_key(settings, nullptr), state);
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

    auto database = CUDDL_UNWRAP(database_type::build_async(device_rows, compatibility));
    CUDDL_CUDA_CALL(cudaDeviceSynchronize());
    auto output = execute_exhaustive_search(database, k_minimum_matches, [&](auto&& launch) {
        state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& nvbench_launch) {
            launch(nvbench_launch.get_stream());
        });
    });

    started = clock_type::now();
    auto const validation = collect_exhaustive_results(database, k_minimum_matches);
    validate_exhaustive_results(validation.results, settings, k_minimum_matches);
    auto const validation_ms =
        std::chrono::duration<double, std::milli>(clock_type::now() - started).count();
    add_value(state, "validation_ms", validation_ms);
    add_value(state, "resident_bytes", static_cast<double>(database.persistent_row_bytes()));
    add_value(state, "search_workspace_bytes", static_cast<double>(output.workspace_bytes));
    add_value(
        state,
        "exact_comparisons",
        static_cast<double>(settings.reference_count) * (settings.reference_count - 1U) / 2.0
    );
    add_value(state, "threshold_zero_recall", 1.0);
    state.add_element_count(settings.query_count, "Queries");
    stash_state("exhaustive_search", axes_key(settings, nullptr), state);
}

void indexed_search(nvbench::state& state) {
    auto const settings = read_workload(state);
    auto const mode = read_index_mode(state);
    auto const indexed_metadata = indexed_compatibility(mode);
    auto const exhaustive_metadata =
        cuddl::score_compatibility::current<k_kmer_length, k_bucket_count>();
    auto const row_bytes = database_type::persistent_row_bytes(settings.reference_count);
    auto const resident_bytes = indexed_resident_bytes(settings.reference_count, indexed_metadata);
    auto const build_temporary_bytes = indexed_build_temporary_bytes(indexed_metadata);
    auto const result_capacity =
        database_type::all_to_all_result_capacity(settings.reference_count);
    auto const result_bytes =
        static_cast<size_t>(result_capacity) * sizeof(cuddl::batch_search_result);
    auto const indexed_search_temporary_bytes =
        result_bytes + static_cast<size_t>(result_capacity) * 2U * sizeof(uint32_t) +
        32U * 1024U * 1024U;
    auto const exhaustive_search_temporary_bytes = 2U * result_bytes + 32U * 1024U * 1024U;
    auto const exhaustive_phase = 2U * row_bytes + exhaustive_search_temporary_bytes;
    auto const indexed_phase = row_bytes + resident_bytes +
                               std::max(build_temporary_bytes, indexed_search_temporary_bytes);
    if (!preflight(state, settings, std::max(exhaustive_phase, indexed_phase), indexed_metadata)) {
        stash_state("indexed_search", axes_key(settings, &mode), state);
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
    auto const indexed_metrics =
        add_fixture_metrics(state, data, settings, generation_ms, transfer_ms, &mode);

    search_output oracle;
    {
        auto database = CUDDL_UNWRAP(database_type::build_async(device_rows, exhaustive_metadata));
        CUDDL_CUDA_CALL(cudaDeviceSynchronize());
        oracle = collect_exhaustive_results(database, k_minimum_matches);
    }

    auto database = CUDDL_UNWRAP(database_type::build_indexed_async(device_rows, indexed_metadata));
    CUDDL_CUDA_CALL(cudaDeviceSynchronize());
    auto indexed = execute_indexed_search(database, k_minimum_matches, [&](auto&& launch) {
        state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& nvbench_launch) {
            launch(nvbench_launch.get_stream());
        });
    });
    auto const indexed_validation = collect_indexed_results(database, k_minimum_matches);

    started = clock_type::now();
    validate_exhaustive_results(oracle.results, settings, k_minimum_matches);
    auto const recall = validate_indexed_results(
        oracle.results, indexed_validation.results, settings, k_minimum_matches
    );
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
    auto const selected_candidates = static_cast<double>(indexed_validation.results.size());
    add_value(state, "selected_candidates", selected_candidates);
    add_value(state, "exact_comparisons", selected_candidates);
    add_value(
        state,
        "candidate_inflation",
        recall.oracle_pairs == 0U ? 0.0
                                  : selected_candidates / static_cast<double>(recall.oracle_pairs)
    );
    add_value(state, "threshold_zero_recall", recall.recall);
    add_value(state, "exact_result_recall", recall.recall);
    add_value(state, "oracle_threshold_pairs", static_cast<double>(recall.oracle_pairs));
    add_value(state, "recalled_pairs", static_cast<double>(recall.recalled_pairs));
    state.add_element_count(settings.query_count, "Queries");
    stash_state("indexed_search", axes_key(settings, &mode), state);
}

struct refseq_fixture {
    std::vector<uint16_t> rows;
    cuddl::score_compatibility compatibility;
};

[[nodiscard]] refseq_fixture load_refseq_fixture() {
    if (std::filesystem::file_size(g_refseq_asset_path) != k_refseq_asset_size) {
        throw std::runtime_error(
            "RefSeq asset has the wrong compressed size: " + g_refseq_asset_path
        );
    }
    auto opaque = cuddl_bench::read_file_any(g_refseq_asset_path);
    auto decoded = CUDDL_UNWRAP(
        cuddl::a48::decode_a48_tsv_parallel(opaque, std::thread::hardware_concurrency())
    );
    if (!decoded.metadata.has_kmer_length || decoded.metadata.kmer_length != k_kmer_length ||
        !decoded.metadata.has_seed || !decoded.metadata.has_exponent ||
        decoded.metadata.exponent_bits != 5U ||
        decoded.records.size() != k_refseq_reference_count) {
        throw std::runtime_error("RefSeq asset metadata does not match the pinned v40.00 asset");
    }
    std::mt19937_64 generator{k_fixture_seed};
    std::shuffle(decoded.records.begin(), decoded.records.end(), generator);

    refseq_fixture fixture{
        .rows = std::vector<uint16_t>(
            static_cast<size_t>(decoded.records.size()) * k_refseq_bucket_count
        ),
        .compatibility = cuddl::decoded_compatibility(
            k_kmer_length,
            k_refseq_bucket_count,
            decoded.metadata.exponent_bits,
            decoded.metadata.seed
        ),
    };
    for (size_t row = 0; row < decoded.records.size(); ++row) {
        if (decoded.records[row].scores.size() != k_refseq_bucket_count) {
            throw std::runtime_error("RefSeq asset contains a row with the wrong bucket count");
        }
        std::copy(
            decoded.records[row].scores.begin(),
            decoded.records[row].scores.end(),
            fixture.rows.begin() + static_cast<std::ptrdiff_t>(row * k_refseq_bucket_count)
        );
    }
    return fixture;
}

struct refseq_context {
    cuddl::score_compatibility compatibility;
    thrust::device_vector<uint16_t> rows;
    std::optional<refseq_database_type> exhaustive_database;
    std::optional<refseq_database_type> indexed_database;
    uint32_t exhaustive_reference_count{};
    uint32_t indexed_reference_count{};
    uint32_t indexed_key_bits{};
};
std::unique_ptr<refseq_context> g_refseq_context;

[[nodiscard]] refseq_context& get_refseq_context() {
    if (!g_refseq_context) {
        auto fixture = load_refseq_fixture();
        g_refseq_context = std::make_unique<refseq_context>(refseq_context{
            .compatibility = fixture.compatibility,
            .rows = thrust::device_vector<uint16_t>(fixture.rows),
            .exhaustive_database = {},
            .indexed_database = {},
            .exhaustive_reference_count = 0U,
            .indexed_reference_count = 0U,
            .indexed_key_bits = 0U,
        });
    }
    return *g_refseq_context;
}

[[nodiscard]] uint32_t read_refseq_reference_count(nvbench::state& state) {
    auto const reference_count = state.get_int64("References");
    if (reference_count <= 0 || reference_count > k_refseq_reference_count) {
        throw std::runtime_error("RefSeq reference count is out of range");
    }
    return static_cast<uint32_t>(reference_count);
}

[[nodiscard]] index_mode read_refseq_index_mode(nvbench::state& state) {
    auto mode = read_index_mode(state);
    mode.indexed_bucket_count = k_refseq_bucket_count;
    return mode;
}

[[nodiscard]] std::string refseq_axes_key(uint32_t reference_count, index_mode const* mode) {
    auto key = std::to_string(reference_count);
    if (mode != nullptr) {
        key +=
            "," + std::to_string(mode->indexed_bucket_count) + "," + std::to_string(mode->key_bits);
    }
    return key;
}

void add_refseq_metadata(
    nvbench::state& state,
    uint32_t reference_count,
    cuddl::score_compatibility const& compatibility,
    size_t resident_bytes
) {
    size_t free_memory = 0U;
    size_t total_memory = 0U;
    CUDDL_CUDA_CALL(cudaMemGetInfo(&free_memory, &total_memory));
    static_cast<void>(total_memory);
    add_environment_metadata(state, free_memory);
    add_compatibility_metadata(state, compatibility);
    add_string(state, "dataset_path", g_refseq_asset_path);
    add_string(state, "query_profile", "refseq_reference_scaling");
    add_value(state, "reference_count", reference_count);
    add_value(state, "query_count", k_refseq_query_count);
    add_value(state, "fixture_seed", k_fixture_seed);
    add_value(state, "minimum_matches", k_minimum_matches);
    add_value(state, "resident_bytes", static_cast<double>(resident_bytes));
}

[[nodiscard]] cuddl::device_span<uint16_t const>
refseq_reference_rows(refseq_context const& context, uint32_t reference_count) {
    return {
        thrust::raw_pointer_cast(context.rows.data()),
        static_cast<size_t>(reference_count) * k_refseq_bucket_count,
    };
}

[[nodiscard]] cuddl::device_span<uint16_t const> refseq_queries(refseq_context const& context) {
    auto const offset = static_cast<size_t>(k_refseq_reference_count - k_refseq_query_count) *
                        k_refseq_bucket_count;
    return {
        thrust::raw_pointer_cast(context.rows.data()) + offset,
        static_cast<size_t>(k_refseq_query_count) * k_refseq_bucket_count,
    };
}

template <typename T>
[[nodiscard]] cuddl::device_span<T> device_vector_span(thrust::device_vector<T>& values) {
    return {thrust::raw_pointer_cast(values.data()), values.size()};
}

void refseq_exhaustive_search(nvbench::state& state) {
    auto const reference_count = read_refseq_reference_count(state);
    auto& context = get_refseq_context();
    if (!context.exhaustive_database || context.exhaustive_reference_count != reference_count) {
        context.exhaustive_database.reset();
        context.exhaustive_database.emplace(CUDDL_UNWRAP(
            refseq_database_type::build_async(
                refseq_reference_rows(context, reference_count), context.compatibility
            )
        ));
        context.exhaustive_reference_count = reference_count;
        CUDDL_CUDA_CALL(cudaDeviceSynchronize());
    }
    auto const& database = *context.exhaustive_database;
    auto const requirements =
        CUDDL_UNWRAP(database.batch_search_requirements(k_refseq_query_count));
    thrust::device_vector<uint8_t> workspace(requirements.workspace_bytes);
    thrust::device_vector<cuddl::batch_search_result> results(requirements.maximum_pair_count);
    thrust::device_vector<uint32_t> result_count(1U);
    add_refseq_metadata(
        state, reference_count, context.compatibility, database.persistent_row_bytes()
    );
    state.add_element_count(
        static_cast<size_t>(k_refseq_query_count) * reference_count, "Pair Comparisons"
    );

    CUDDL_UNWRAP(database.search_batch_async(
        refseq_queries(context),
        context.compatibility,
        0U,
        device_vector_span(workspace),
        device_vector_span(results),
        device_vector_span(result_count),
        [&](uint32_t expected_count) {
            uint32_t observed_count = 0U;
            CUDDL_CUDA_CALL(cudaMemcpy(
                &observed_count,
                thrust::raw_pointer_cast(result_count.data()),
                sizeof(observed_count),
                cudaMemcpyDeviceToHost
            ));
            if (observed_count != expected_count) {
                throw std::runtime_error("RefSeq exhaustive tile returned the wrong pair count");
            }
        }
    ));

    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
        CUDDL_UNWRAP(database.search_batch_async(
            refseq_queries(context),
            context.compatibility,
            0U,
            device_vector_span(workspace),
            device_vector_span(results),
            device_vector_span(result_count),
            [](uint32_t) {},
            {},
            cuda::stream_ref{launch.get_stream()}
        ));
    });
    stash_state("refseq_exhaustive_search", refseq_axes_key(reference_count, nullptr), state);
}

void refseq_indexed_search(nvbench::state& state) {
    auto const reference_count = read_refseq_reference_count(state);
    auto const mode = read_refseq_index_mode(state);
    auto compatibility = get_refseq_context().compatibility;
    compatibility.key_mask = mode.key_mask;
    auto& context = get_refseq_context();
    if (!context.indexed_database || context.indexed_reference_count != reference_count ||
        context.indexed_key_bits != mode.key_bits) {
        context.indexed_database.reset();
        auto database = CUDDL_UNWRAP(
            refseq_database_type::build_indexed_async(
                refseq_reference_rows(context, reference_count), compatibility
            )
        );
        context.indexed_database.emplace(std::move(database));
        context.indexed_reference_count = reference_count;
        context.indexed_key_bits = mode.key_bits;
        CUDDL_CUDA_CALL(cudaDeviceSynchronize());
    }
    auto const& database = *context.indexed_database;
    auto const requirements =
        CUDDL_UNWRAP(database.indexed_batch_search_requirements(k_refseq_query_count));
    thrust::device_vector<uint8_t> workspace(requirements.workspace_bytes);
    thrust::device_vector<cuddl::batch_search_result> results(requirements.maximum_pair_count);
    thrust::device_vector<uint32_t> result_count(1U);
    add_refseq_metadata(
        state,
        reference_count,
        compatibility,
        database.persistent_row_bytes() + database.persistent_index_bytes()
    );
    uint64_t selected_candidates = 0U;
    CUDDL_UNWRAP(database.search_batch_indexed_async(
        refseq_queries(context),
        compatibility,
        0U,
        device_vector_span(workspace),
        device_vector_span(results),
        device_vector_span(result_count),
        [&](uint32_t maximum_count) {
            uint32_t observed_count = 0U;
            CUDDL_CUDA_CALL(cudaMemcpy(
                &observed_count,
                thrust::raw_pointer_cast(result_count.data()),
                sizeof(observed_count),
                cudaMemcpyDeviceToHost
            ));
            if (observed_count > maximum_count) {
                throw std::runtime_error("RefSeq indexed tile exceeded its result capacity");
            }
            selected_candidates += observed_count;
        },
        {},
        {.minimum_matches = k_minimum_matches}
    ));
    add_value(state, "selected_candidates", static_cast<double>(selected_candidates));
    add_value(
        state,
        "candidate_fraction",
        static_cast<double>(selected_candidates) /
            (static_cast<double>(k_refseq_query_count) * reference_count)
    );
    state.add_element_count(
        static_cast<size_t>(k_refseq_query_count) * reference_count, "Database Pairs"
    );

    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
        CUDDL_UNWRAP(database.search_batch_indexed_async(
            refseq_queries(context),
            compatibility,
            0U,
            device_vector_span(workspace),
            device_vector_span(results),
            device_vector_span(result_count),
            [](uint32_t) {},
            {},
            {.minimum_matches = k_minimum_matches},
            cuda::stream_ref{launch.get_stream()}
        ));
    });
    stash_state("refseq_indexed_search", refseq_axes_key(reference_count, &mode), state);
}

// --- In-process summary collection -------------------------------------------
// The benchmark writes one summary CSV at the end of the run (see main), so the
// four benchmark functions stash their per-state results here instead of relying on
// the JSON + sidecar pipeline.

std::string g_summary_json_path = "results/cuddl-search-decision.json";

struct collected_state {
    std::map<std::string, std::string> values;  // summary tag -> stringified value
    bool skipped{};
    std::string skip_reason;
};

// Key: benchmark name + newline + comma-joined axes, mirroring the summariser.
std::map<std::string, collected_state> g_collected;

std::string format_number(double value) {
    std::array<char, 64> buffer{};
    auto const result = std::to_chars(
        buffer.data(), buffer.data() + buffer.size(), value, std::chars_format::general
    );
    return std::string(buffer.data(), result.ptr);
}

std::string axes_key(workload const& settings, index_mode const* mode) {
    auto key = std::to_string(settings.reference_count) + "," +
               std::to_string(static_cast<int>(std::llround(settings.fill_ratio * 1000.0))) + "," +
               std::to_string(static_cast<int>(std::llround(settings.hot_fraction * 100.0)));
    if (mode != nullptr) {
        key +=
            "," + std::to_string(mode->indexed_bucket_count) + "," + std::to_string(mode->key_bits);
    }
    return key;
}

void stash_state(std::string const& benchmark, std::string const& key, nvbench::state& state) {
    collected_state record;
    record.skipped = state.is_skipped();
    record.skip_reason = state.get_skip_reason();
    for (auto const& summary : state.get_summaries()) {
        if (!summary.has_value("value")) {
            continue;
        }
        switch (summary.get_type("value")) {
            case nvbench::named_values::type::float64:
                record.values[summary.get_tag()] = format_number(summary.get_float64("value"));
                break;
            case nvbench::named_values::type::int64:
                record.values[summary.get_tag()] = std::to_string(summary.get_int64("value"));
                break;
            case nvbench::named_values::type::string:
                record.values[summary.get_tag()] = summary.get_string("value");
                break;
        }
    }
    g_collected[benchmark + "\n" + key] = std::move(record);
}

}  // namespace

NVBENCH_BENCH(compact_build)
    .add_int64_axis("References", reference_counts)
    .add_int64_axis("HotPercent", hot_percentages)
    .set_stopping_criterion("sample-count")
    .set_min_samples(20)
    .set_criterion_param_int64("target-samples", 20)
    .set_cold_warmup_runs(3)
    .set_skip_batched(true);

NVBENCH_BENCH(indexed_build)
    .add_int64_axis("References", reference_counts)
    .add_int64_axis("HotPercent", hot_percentages)
    .add_int64_axis("KeyBits", key_bits)
    .set_stopping_criterion("sample-count")
    .set_min_samples(20)
    .set_criterion_param_int64("target-samples", 20)
    .set_cold_warmup_runs(3)
    .set_skip_batched(true);

NVBENCH_BENCH(exhaustive_search)
    .add_int64_axis("References", reference_counts)
    .add_int64_axis("HotPercent", hot_percentages)
    .set_stopping_criterion("sample-count")
    .set_min_samples(20)
    .set_criterion_param_int64("target-samples", 20)
    .set_cold_warmup_runs(3)
    .set_skip_batched(true);

NVBENCH_BENCH(indexed_search)
    .add_int64_axis("References", reference_counts)
    .add_int64_axis("HotPercent", hot_percentages)
    .add_int64_axis("KeyBits", key_bits)
    .set_stopping_criterion("sample-count")
    .set_min_samples(20)
    .set_criterion_param_int64("target-samples", 20)
    .set_cold_warmup_runs(3)
    .set_skip_batched(true);

NVBENCH_BENCH(refseq_exhaustive_search)
    .add_int64_axis("References", refseq_reference_counts)
    .set_stopping_criterion("sample-count")
    .set_min_samples(20)
    .set_criterion_param_int64("target-samples", 20)
    .set_cold_warmup_runs(3)
    .set_skip_batched(true);

NVBENCH_BENCH(refseq_indexed_search)
    .add_int64_axis("KeyBits", key_bits)
    .add_int64_axis("References", refseq_reference_counts)
    .set_stopping_criterion("sample-count")
    .set_min_samples(20)
    .set_criterion_param_int64("target-samples", 20)
    .set_cold_warmup_runs(3)
    .set_skip_batched(true);

namespace {

std::vector<std::string> split(std::string const& text, char separator) {
    std::vector<std::string> parts;
    std::stringstream stream(text);
    std::string part;
    while (std::getline(stream, part, separator)) {
        parts.push_back(part);
    }
    return parts;
}

// The decision result joins the four benchmark states per workload and index mode.
// Timing values come from nvbench's own cold-measurement summaries; the full
// per-state evidence remains available through nvbench's --csv output.
void write_summary_json() {
    if (g_collected.empty()) {
        return;
    }

    auto record_of = [](std::string const& benchmark) {
        auto prefix = benchmark + "\n";
        std::map<std::vector<std::string>, collected_state const*> states;
        for (auto const& [key, value] : g_collected) {
            if (key.starts_with(prefix)) {
                states[split(key.substr(prefix.size()), ',')] = &value;
            }
        }
        return states;
    };
    auto compact_states = record_of("compact_build");
    auto indexed_build_states = record_of("indexed_build");
    auto exhaustive_states = record_of("exhaustive_search");
    auto indexed_search_states = record_of("indexed_search");
    auto refseq_exhaustive_states = record_of("refseq_exhaustive_search");
    auto refseq_indexed_states = record_of("refseq_indexed_search");

    std::set<std::vector<std::string>> mode_keys;
    for (auto const& [key, state] : indexed_build_states) {
        mode_keys.insert({key.end() - 2, key.end()});
    }
    for (auto const& [key, state] : indexed_search_states) {
        mode_keys.insert({key.end() - 2, key.end()});
    }
    std::set<std::vector<std::string>> workload_keys;
    for (auto const& [key, state] : exhaustive_states) {
        workload_keys.insert(key);
    }
    for (auto const& [key, state] : indexed_search_states) {
        workload_keys.insert({key.begin(), key.end() - 2});
    }

    json measurements = json::array();

    auto value_of = [](collected_state const& state, std::string const& tag) {
        auto found = state.values.find(tag);
        return found == state.values.end() ? std::string{} : found->second;
    };
    auto timing_ms = [&](collected_state const& state, std::string const& statistic) {
        auto const text = value_of(state, "nv/cold/time/gpu/" + statistic);
        return text.empty() ? std::string{} : format_number(std::stod(text) * 1000.0);
    };
    auto integer_of = [](std::string const& text) {
        return static_cast<uint64_t>(std::llround(std::stod(text)));
    };

    for (auto const& workload : workload_keys) {
        for (auto const& mode : mode_keys) {
            std::vector<std::string> build_key(workload.begin(), workload.begin() + 3);
            auto indexed_build_key = build_key;
            indexed_build_key.insert(indexed_build_key.end(), mode.begin(), mode.end());
            auto indexed_search_key = workload;
            indexed_search_key.insert(indexed_search_key.end(), mode.begin(), mode.end());

            std::map<std::string, collected_state const*> selected{
                {"compact_build", nullptr},
                {"indexed_build", nullptr},
                {"exhaustive_search", nullptr},
                {"indexed_search", nullptr},
            };
            if (auto found = compact_states.find(build_key); found != compact_states.end()) {
                selected["compact_build"] = found->second;
            }
            if (auto found = indexed_build_states.find(indexed_build_key);
                found != indexed_build_states.end()) {
                selected["indexed_build"] = found->second;
            }
            if (auto found = exhaustive_states.find(workload); found != exhaustive_states.end()) {
                selected["exhaustive_search"] = found->second;
            }
            if (auto found = indexed_search_states.find(indexed_search_key);
                found != indexed_search_states.end()) {
                selected["indexed_search"] = found->second;
            }

            std::vector<std::string> missing;
            std::vector<std::string> skipped;
            for (auto const& [name, state] : selected) {
                if (state == nullptr) {
                    missing.push_back(name);
                } else if (state->skipped) {
                    skipped.push_back(name + ": " + state->skip_reason);
                }
            }
            auto const status = missing.empty() && skipped.empty() ? std::string_view("ok")
                                                                   : std::string_view("skipped");
            std::string error;
            for (auto const& name : missing) {
                error += (error.empty() ? "" : "; ") + ("missing " + name);
            }
            for (auto const& note : skipped) {
                error += (error.empty() ? "" : "; ") + note;
            }

            auto const& references = workload[0];
            auto const& fill_permille = workload[1];
            auto const& hot_percent = workload[2];
            auto const fill_ratio = std::stod(fill_permille) / 1000.0;
            auto const& query_count = references;
            auto const query_profile = "all_to_all";
            auto const& buckets = mode[0];
            auto const& bits = mode[1];

            std::map<std::string, std::string> row{
                {"status", std::string(status)},
                {"reference_count", references},
                {"fill_ratio", format_number(fill_ratio)},
                {"skew", hot_percent + "%"},
                {"query_count", query_count},
                {"query_profile", query_profile},
                {"index_mode", "b" + buckets + "_k" + bits},
                {"kill_gate_outcome", "inconclusive"},
                {"error", error},
            };

            if (status == "ok") {
                auto const& exhaustive = *selected["exhaustive_search"];
                auto const& indexed = *selected["indexed_search"];
                row["threshold_zero_recall"] = value_of(indexed, "threshold_zero_recall");
                row["exhaustive_min_ms"] = timing_ms(exhaustive, "min");
                row["exhaustive_mean_ms"] = timing_ms(exhaustive, "mean");
                row["exhaustive_p50_ms"] = timing_ms(exhaustive, "median");
                row["exhaustive_max_ms"] = timing_ms(exhaustive, "max");
                row["indexed_min_ms"] = timing_ms(indexed, "min");
                row["indexed_mean_ms"] = timing_ms(indexed, "mean");
                row["indexed_p50_ms"] = timing_ms(indexed, "median");
                row["indexed_max_ms"] = timing_ms(indexed, "max");
                row["atomic_updates"] = value_of(indexed, "atomic_updates");
                row["selected_candidates"] = value_of(indexed, "selected_candidates");
                row["candidate_inflation"] = value_of(indexed, "candidate_inflation");
                row["indexed_resident_bytes"] = value_of(indexed, "resident_bytes");

                auto const recall_text = row["threshold_zero_recall"];
                auto const recall = recall_text.empty() ? 0.0 : std::stod(recall_text);
                auto const exhaustive_median = std::stod(row["exhaustive_p50_ms"]);
                auto const indexed_median = std::stod(row["indexed_p50_ms"]);
                auto const recall_loss = !std::isfinite(recall) || recall < 1.0;
                auto const indexed_win = indexed_median < exhaustive_median;
                auto const exhaustive_win = exhaustive_median < indexed_median;
                if (recall_loss) {
                    row["kill_gate_outcome"] = "recall_loss";
                } else if (indexed_win) {
                    row["kill_gate_outcome"] = "indexed_win";
                } else if (exhaustive_win) {
                    row["kill_gate_outcome"] = "exhaustive_win";
                }
            }

            json metrics{{"kill_gate_outcome", row["kill_gate_outcome"]}};
            for (auto const* name : {
                     "threshold_zero_recall",
                     "exhaustive_min_ms",
                     "exhaustive_mean_ms",
                     "exhaustive_p50_ms",
                     "exhaustive_max_ms",
                     "indexed_min_ms",
                     "indexed_mean_ms",
                     "indexed_p50_ms",
                     "indexed_max_ms",
                     "candidate_inflation",
                 }) {
                if (!row[name].empty()) {
                    metrics[name] = std::stod(row[name]);
                }
            }
            for (auto const* name : {"atomic_updates", "selected_candidates"}) {
                if (!row[name].empty()) {
                    metrics[name] = integer_of(row[name]);
                }
            }
            json measurement{
                {"implementation", {{"name", "cuddl"}}},
                {"case",
                 {
                     {"status", row["status"]},
                     {"reference_count", std::stoull(references)},
                     {"fill_ratio", fill_ratio},
                     {"hot_fraction", std::stod(hot_percent) / 100.0},
                     {"skew", row["skew"]},
                     {"query_count", std::stoull(query_count)},
                     {"query_profile", query_profile},
                     {"index_mode", row["index_mode"]},
                     {"indexed_bucket_count", std::stoull(buckets)},
                     {"key_bits", std::stoull(bits)},
                     {"error", error},
                 }},
                {"metrics", std::move(metrics)},
            };
            if (!row["indexed_resident_bytes"].empty()) {
                measurement["memory_bytes"] = {
                    {"indexed_resident", integer_of(row["indexed_resident_bytes"])}
                };
            }
            measurements.push_back(std::move(measurement));
        }
    }

    for (auto const& [key, indexed] : refseq_indexed_states) {
        if (key.size() != 3U) {
            continue;
        }
        auto exhaustive = refseq_exhaustive_states.find({key[0]});
        auto const complete = exhaustive != refseq_exhaustive_states.end();
        auto const skipped = indexed->skipped || (complete && exhaustive->second->skipped);
        std::string error;
        if (!complete) {
            error = "missing refseq_exhaustive_search";
        } else if (indexed->skipped) {
            error = indexed->skip_reason;
        } else if (exhaustive->second->skipped) {
            error = exhaustive->second->skip_reason;
        }

        json metrics{{"kill_gate_outcome", "inconclusive"}};
        if (complete && !skipped) {
            auto const exhaustive_p50 = timing_ms(*exhaustive->second, "median");
            auto const indexed_p50 = timing_ms(*indexed, "median");
            auto const database_pairs =
                std::stoull(key[0]) * static_cast<uint64_t>(k_refseq_query_count);
            metrics.update({
                {"exhaustive_p50_ms", std::stod(exhaustive_p50)},
                {"indexed_p50_ms", std::stod(indexed_p50)},
                {"exhaustive_queries_per_second",
                 1000.0 * k_refseq_query_count / std::stod(exhaustive_p50)},
                {"indexed_queries_per_second",
                 1000.0 * k_refseq_query_count / std::stod(indexed_p50)},
                {"search_speedup", std::stod(exhaustive_p50) / std::stod(indexed_p50)},
                {"database_pairs", database_pairs},
                {"exhaustive_effective_pairs_per_second",
                 1000.0 * static_cast<double>(database_pairs) / std::stod(exhaustive_p50)},
                {"indexed_effective_pairs_per_second",
                 1000.0 * static_cast<double>(database_pairs) / std::stod(indexed_p50)},
            });
            auto const candidates = value_of(*indexed, "selected_candidates");
            if (!candidates.empty()) {
                auto const candidate_count = integer_of(candidates);
                metrics["selected_candidates"] = candidate_count;
                metrics["candidate_fraction"] =
                    static_cast<double>(candidate_count) / static_cast<double>(database_pairs);
            }
        }

        json measurement{
            {"implementation", {{"name", "cuddl"}}},
            {"case",
             {
                 {"status", complete && !skipped ? "ok" : "skipped"},
                 {"dataset", "BBTools RefSeq DDL v40.00"},
                 {"dataset_path", g_refseq_asset_path},
                 {"reference_count", std::stoull(key[0])},
                 {"query_count", k_refseq_query_count},
                 {"query_profile", "refseq_reference_scaling"},
                 {"fixture_seed", k_fixture_seed},
                 {"index_mode", "b" + key[1] + "_k" + key[2]},
                 {"indexed_bucket_count", std::stoull(key[1])},
                 {"key_bits", std::stoull(key[2])},
                 {"error", error},
             }},
            {"metrics", std::move(metrics)},
        };
        auto const resident_bytes = value_of(*indexed, "resident_bytes");
        if (!resident_bytes.empty()) {
            measurement["memory_bytes"] = {{"indexed_resident", integer_of(resident_bytes)}};
        }
        measurements.push_back(std::move(measurement));
    }
    write_benchmark_result(
        g_summary_json_path,
        make_benchmark_result(
            "cuDDL search decision", "search_decision", "kernel", std::move(measurements)
        )
    );
}

}  // namespace

int main(int argc, char** argv) try {
    nvbench::detail::main_initialize(argc, argv);
    {
        std::vector<char*> remaining{argv[0]};
        for (int i = 1; i < argc; ++i) {
            std::string_view const argument = argv[i];
            if (argument == "--summary-json") {
                if (i + 1 < argc) {
                    g_summary_json_path = argv[++i];
                }
                continue;
            }
            if (argument.starts_with("--summary-json=")) {
                g_summary_json_path = std::string(argument.substr(15));
                continue;
            }
            if (argument == "--refseq-asset") {
                if (i + 1 >= argc) {
                    throw std::runtime_error("--refseq-asset requires a path");
                }
                g_refseq_asset_path = argv[++i];
                continue;
            }
            if (argument.starts_with("--refseq-asset=")) {
                g_refseq_asset_path = std::string(argument.substr(15));
                continue;
            }
            remaining.push_back(argv[i]);
        }
        auto const args = nvbench::detail::main_convert_args(
            static_cast<int>(remaining.size()), remaining.data()
        );
        nvbench::option_parser parser;
        parser.parse(args);
        nvbench::detail::main_print_preamble(parser);
        nvbench::detail::main_run_benchmarks(parser);
        nvbench::detail::main_print_epilogue(parser);
        nvbench::detail::main_print_results(parser);
    }
    g_refseq_context.reset();
    nvbench::detail::main_finalize();
    write_summary_json();
    return 0;
} catch (std::exception& error) {
    g_refseq_context.reset();
    std::cerr << "\nNVBench encountered an error:\n\n" << error.what() << "\n";
    return 1;
} catch (...) {
    g_refseq_context.reset();
    std::cerr << "\nNVBench encountered an unknown error.\n";
    return 1;
}
