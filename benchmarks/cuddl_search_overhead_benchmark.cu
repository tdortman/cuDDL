#include <cuda/algorithm>
#include <cuda/buffer>
#include <cuddl/cuddl.cuh>
#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

// Inputs, allocation, index construction, and validation are outside the timed region.
void search_overhead(nvbench::state& state) {
    constexpr size_t buckets = 2048U;
    using database_type = cuddl::reference_database<25U, buckets>;
    auto const stream = cuda::stream_ref{state.get_cuda_stream()};
    auto const references = static_cast<uint32_t>(state.get_int64("References"));
    auto const queries = static_cast<uint32_t>(state.get_int64("Queries"));
    auto const mode = state.get_string("Mode");
    auto const index = state.get_string("Index");
    auto const storage =
        index == "sparse" ? cuddl::index_storage::sparse : cuddl::index_storage::dense;
    bool const exhaustive = mode == "exhaustive";
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
    auto database = CUDDL_UNWRAP(
        exhaustive ? database_type::build_async(input, compatibility, stream)
                   : database_type::build_indexed_async(input, compatibility, stream, storage)
    );
    auto requirements = CUDDL_UNWRAP(
        exhaustive ? database.batch_search_requirements(queries, stream)
                   : database.indexed_batch_search_requirements(queries, stream)
    );
    uint32_t const capacity = mode == "bounded-fit"        ? expected
                              : mode == "bounded-overflow" ? expected - 1U
                                                           : requirements.maximum_pair_count;
    auto workspace = cuda::make_device_buffer<uint8_t>(
        stream, stream.device(), requirements.workspace_bytes, cuda::no_init
    );
    cuddl::batch_search_result const sentinel{.query_id = 0xffffffffU, .reference_id = 0xffffffffU};
    auto results = cuda::make_device_buffer<cuddl::batch_search_result>(
        stream, stream.device(), capacity, sentinel
    );
    auto count = cuda::make_device_buffer<uint32_t>(stream, stream.device(), 1U, cuda::no_init);
    auto matches = cuda::make_device_buffer<uint32_t>(
        stream, stream.device(), capacity, uint32_t{0xffffffffU}
    );
    auto search = [&](cuda::stream_ref execution_stream) {
        if (exhaustive) {
            CUDDL_UNWRAP(database.search_batch_async(
                query_input,
                compatibility,
                0U,
                workspace,
                results,
                count,
                [](uint32_t) {},
                matches,
                execution_stream
            ));
        } else {
            CUDDL_UNWRAP(database.search_batch_indexed_async(
                query_input,
                compatibility,
                0U,
                workspace,
                results,
                count,
                [](uint32_t) {},
                matches,
                {.minimum_matches = 5U},
                execution_stream
            ));
        }
    };
    search(stream);
    uint32_t actual = 0U;
    std::vector<cuddl::batch_search_result> host_results(capacity);
    std::vector<uint32_t> host_matches(capacity);
    cuda::copy_bytes(stream, count, cuda::std::span{&actual, size_t{1}});
    cuda::copy_bytes(stream, results, host_results);
    cuda::copy_bytes(stream, matches, host_matches);
    stream.sync();
    if (actual != expected) {
        throw std::runtime_error("incorrect candidate count");
    }
    if (mode == "bounded-overflow") {
        for (uint32_t i = 0U; i < capacity; ++i) {
            if (host_results[i] != sentinel || host_matches[i] != 0xffffffffU) {
                throw std::runtime_error("overflow changed output storage");
            }
        }
    } else {
        uint32_t index = 0U;
        for (uint32_t q = 0U; q < queries; ++q) {
            for (uint32_t r = 0U; r < references; ++r) {
                if (!exhaustive && q % 16U != r % 16U) {
                    continue;
                }
                auto const& result = host_results[index];
                bool const equal = q % 16U == r % 16U;
                if (result.query_id != q || result.reference_id != r ||
                    result.summary.counts.equal != (equal ? buckets : 0U) ||
                    host_matches[index] != (equal ? buckets : 0U)) {
                    throw std::runtime_error("incorrect query/reference result");
                }
                ++index;
            }
        }
    }
    state.add_element_count(size_t{references} * queries, "Pairs");
    state.exec([&](nvbench::launch& launch) { search(cuda::stream_ref{launch.get_stream()}); });
}

NVBENCH_BENCH(search_overhead)
    .add_int64_axis("References", {33, 4096})
    .add_int64_axis("Queries", {1, 8, 64})
    .add_string_axis("Mode", {"exhaustive", "full", "bounded-fit", "bounded-overflow"})
    .add_string_axis("Index", {"dense", "sparse"})
    .set_stopping_criterion("sample-count")
    .set_min_samples(30)
    .set_criterion_param_int64("target-samples", 30);
