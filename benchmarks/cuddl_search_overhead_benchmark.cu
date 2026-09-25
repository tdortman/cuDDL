#include <cuda/algorithm>
#include <cuda/buffer>
#include <cuddl/cuddl.cuh>
#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

// Inputs, allocation, index construction, and validation are outside the timed region.
template <bool exhaustive>
void run_search_overhead(nvbench::state& state) {
    constexpr size_t buckets = 2048U;
    using database_type = cuddl::reference_database<25U, buckets>;
    using index_type = cuddl::reference_index<25U, buckets>;
    auto const stream = cuda::stream_ref{state.get_cuda_stream()};
    auto const references = static_cast<uint32_t>(state.get_int64("References"));
    auto const queries = static_cast<uint32_t>(state.get_int64("Queries"));
    auto const index = state.get_string("Index");
    auto const storage =
        index == "sparse" ? cuddl::index_storage::sparse : cuddl::index_storage::dense;
    std::vector<uint16_t> rows(size_t{references} * buckets);
    std::vector<uint16_t> query_rows(size_t{queries} * buckets);
    for (uint32_t r = 0; r < references; ++r) {
        std::fill_n(
            rows.begin() + size_t{r} * buckets, buckets, static_cast<uint16_t>(1U + r % 16U)
        );
    }
    uint32_t expected = 0U;
    for (uint32_t q = 0; q < queries; ++q) {
        std::fill_n(
            query_rows.begin() + size_t{q} * buckets, buckets, static_cast<uint16_t>(1U + q % 16U)
        );
        expected += exhaustive ? references : references / 16U + (q % 16U < references % 16U);
    }
    auto input = cuda::make_device_buffer<uint16_t>(stream, stream.device(), rows);
    auto query_input = cuda::make_device_buffer<uint16_t>(stream, stream.device(), query_rows);
    auto const compatibility = cuddl::score_compatibility::current<25U, buckets>();
    auto database = CUDDL_UNWRAP(database_type::build_async(input, compatibility, stream));
    std::optional<index_type> acceleration;
    if constexpr (!exhaustive) {
        acceleration.emplace(CUDDL_UNWRAP(index_type::build_async(database, stream, storage)));
    }
    auto requirements = CUDDL_UNWRAP(
        database.batch_search_requirements(queries, stream, acceleration ? &*acceleration : nullptr)
    );
    uint32_t const capacity = requirements.maximum_pair_count;
    auto workspace = cuda::make_device_buffer<uint8_t>(
        stream, stream.device(), requirements.workspace_bytes, cuda::no_init
    );
    auto results = cuda::make_device_buffer<cuddl::packed_pairwise_counts>(
        stream, stream.device(), capacity, cuda::no_init
    );
    auto matches = cuda::make_device_buffer<uint32_t>(
        stream, stream.device(), capacity, uint32_t{0xffffffffU}
    );
    auto search = [&](cuda::stream_ref execution_stream, auto&& on_tile) {
        if constexpr (exhaustive) {
            CUDDL_UNWRAP(database.search_batch_async(
                query_input,
                compatibility,
                0U,
                workspace,
                results,
                on_tile,
                matches,
                execution_stream
            ));
        } else {
            CUDDL_UNWRAP(database.search_batch_async(
                query_input,
                compatibility,
                0U,
                workspace,
                results,
                on_tile,
                matches,
                {.minimum_matches = 5U},
                execution_stream,
                &*acceleration
            ));
        }
    };
    std::vector<cuddl::batch_search_result> host_results;
    std::vector<uint32_t> host_matches;
    search(stream, [&](cuddl::batch_result_tile const& tile) {
        auto const host = CUDDL_UNWRAP(cuddl::download(tile, stream));
        auto const passing = host.passing();
        auto const counts = host.passing_match_counts();
        host_results.insert(host_results.end(), passing.begin(), passing.end());
        host_matches.insert(host_matches.end(), counts.begin(), counts.end());
    });
    if (host_results.size() != expected) {
        throw std::runtime_error("incorrect candidate count");
    }
    uint32_t position = 0U;
    for (uint32_t q = 0U; q < queries; ++q) {
        for (uint32_t r = 0U; r < references; ++r) {
            if (!exhaustive && q % 16U != r % 16U) {
                continue;
            }
            auto const& result = host_results[position];
            bool const equal = q % 16U == r % 16U;
            if (result.query_id != q || result.reference_id != r ||
                result.counts.equal != (equal ? buckets : 0U) ||
                host_matches[position] != (equal ? buckets : 0U)) {
                throw std::runtime_error("incorrect query/reference result");
            }
            ++position;
        }
    }
    state.add_element_count(size_t{references} * queries, "Pairs");
    state.exec([&](nvbench::launch& launch) {
        search(cuda::stream_ref{launch.get_stream()}, [](cuddl::batch_result_tile const&) {});
    });
}

void search_overhead(nvbench::state& state) {
    if (state.get_string("Mode") == "exhaustive") {
        run_search_overhead<true>(state);
    } else {
        run_search_overhead<false>(state);
    }
}

NVBENCH_BENCH(search_overhead)
    .add_int64_axis("References", {33, 4096})
    .add_int64_axis("Queries", {1, 8, 64})
    .add_string_axis("Mode", {"exhaustive", "full"})
    .add_string_axis("Index", {"dense", "sparse"})
    .set_stopping_criterion("sample-count")
    .set_min_samples(30)
    .set_criterion_param_int64("target-samples", 30);
