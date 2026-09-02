#include <cuddl/cuddl.cuh>
#include <cuddl/detail/comparison.cuh>
#include <cuddl/detail/fasta_parser.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/warp/warp_reduce.cuh>
#include <nvbench/main.cuh>
#include <nvbench/nvbench.cuh>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>
#include "common.cuh"
#include "result_json.hpp"

namespace {
namespace cg = cooperative_groups;

constexpr uint32_t k_kmer_length = 25;
constexpr size_t k_bucket_count = 2048;
constexpr uint32_t k_launch_block_size = 256;
std::vector<nvbench::int64_t> const reference_powers{8, 10, 12, 14, 16, 18, 20};
std::vector<nvbench::int64_t> const indexed_reference_powers{8, 10, 12, 14, 16, 18};
std::vector<nvbench::int64_t> const
    launch_reference_counts{1, 8, 64, 256, 512, 1024, 2048, 4096, 8192, 16384, 65536, 200687};
std::vector<std::string> const launch_shapes{
    "current_cub_warp",
    "raw_cub_warp",
    "span_cub_warp",
    "cg_1_warp",
    "cg_2_warps",
    "cg_4_warps",
    "cg_8_warps",
};

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
__device__ cuddl::pairwise_counts
reduce_warp(cg::thread_block_tile<32> const warp, cuddl::pairwise_counts counts) {
    counts.lower = cg::reduce(warp, counts.lower, cg::plus<uint32_t>{});
    counts.equal = cg::reduce(warp, counts.equal, cg::plus<uint32_t>{});
    counts.higher = cg::reduce(warp, counts.higher, cg::plus<uint32_t>{});
    counts.both_empty = cg::reduce(warp, counts.both_empty, cg::plus<uint32_t>{});
    return counts;
}

__device__ void write_search_result(
    cuddl::reference_search_result* results,
    uint32_t reference_id,
    cuddl::pairwise_counts counts
) {
    results[reference_id].reference_id = reference_id;
    results[reference_id].summary.counts = counts;
    results[reference_id].summary.cardinality = 0.0;
}
template <typename Rows, typename Query>
__global__ __launch_bounds__(k_launch_block_size) void cub_warp_exhaustive_search_kernel(
    Rows rows,
    uint32_t reference_count,
    Query query,
    cuddl::reference_search_result* results
) {
    constexpr uint32_t warp_width = 32;
    constexpr uint32_t warps_per_block = k_launch_block_size / warp_width;
    using warp_reduce = cub::WarpReduce<cuddl::pairwise_counts>;
    __shared__ typename warp_reduce::TempStorage storage[warps_per_block];

    auto const warp = static_cast<uint32_t>(threadIdx.x) / warp_width;
    auto const lane = static_cast<uint32_t>(threadIdx.x) % warp_width;
    auto const reference_id = static_cast<uint32_t>(blockIdx.x) * warps_per_block + warp;
    if (reference_id >= reference_count) {
        return;
    }

    cuddl::pairwise_counts local{};
    auto const row_offset = static_cast<size_t>(reference_id) * k_bucket_count;
    for (auto bucket = static_cast<size_t>(lane); bucket < k_bucket_count; bucket += warp_width) {
        cuddl::detail::classify(local, query[bucket], rows[row_offset + bucket]);
    }
    auto const total = warp_reduce(storage[warp]).Sum(local);
    if (lane == 0U) {
        write_search_result(results, reference_id, total);
    }
}

void launch_raw_cub_exhaustive(
    thrust::device_vector<uint16_t> const& rows,
    uint32_t reference_count,
    thrust::device_vector<uint16_t> const& query,
    thrust::device_vector<cuddl::reference_search_result>& results,
    cudaStream_t stream
) {
    constexpr uint32_t references_per_block = k_launch_block_size / 32U;
    auto const blocks = (reference_count + references_per_block - 1U) / references_per_block;
    cub_warp_exhaustive_search_kernel<<<blocks, k_launch_block_size, 0, stream>>>(
        thrust::raw_pointer_cast(rows.data()),
        reference_count,
        thrust::raw_pointer_cast(query.data()),
        thrust::raw_pointer_cast(results.data())
    );
    CUDDL_CUDA_CALL(cudaGetLastError());
}

void launch_span_cub_exhaustive(
    thrust::device_vector<uint16_t> const& rows,
    uint32_t reference_count,
    thrust::device_vector<uint16_t> const& query,
    thrust::device_vector<cuddl::reference_search_result>& results,
    cudaStream_t stream
) {
    constexpr uint32_t references_per_block = k_launch_block_size / 32U;
    auto const blocks = (reference_count + references_per_block - 1U) / references_per_block;
    cub_warp_exhaustive_search_kernel<<<blocks, k_launch_block_size, 0, stream>>>(
        cuddl::device_span<uint16_t const>{thrust::raw_pointer_cast(rows.data()), rows.size()},
        reference_count,
        cuddl::device_span<uint16_t const>{thrust::raw_pointer_cast(query.data()), query.size()},
        thrust::raw_pointer_cast(results.data())
    );
    CUDDL_CUDA_CALL(cudaGetLastError());
}

template <uint32_t WarpsPerReference>
__global__ __launch_bounds__(k_launch_block_size) void cooperative_exhaustive_search_kernel(
    uint16_t const* rows,
    uint32_t reference_count,
    uint16_t const* query,
    cuddl::reference_search_result* results
) {
    static_assert(
        WarpsPerReference == 1U || WarpsPerReference == 2U || WarpsPerReference == 4U ||
        WarpsPerReference == 8U
    );
    constexpr uint32_t warp_width = 32;
    constexpr uint32_t warps_per_block = k_launch_block_size / warp_width;
    constexpr uint32_t threads_per_reference = warp_width * WarpsPerReference;
    constexpr uint32_t references_per_block = warps_per_block / WarpsPerReference;

    __shared__ cuddl::pairwise_counts warp_summaries[warps_per_block];
    auto const block = cg::this_thread_block();
    auto const warp = cg::tiled_partition<warp_width>(block);
    auto const reference_in_block = static_cast<uint32_t>(threadIdx.x) / threads_per_reference;
    auto const thread_in_reference = static_cast<uint32_t>(threadIdx.x) % threads_per_reference;
    auto const warp_in_reference = thread_in_reference / warp_width;
    auto const reference_id =
        static_cast<uint32_t>(blockIdx.x) * references_per_block + reference_in_block;
    auto const valid_reference = reference_id < reference_count;

    cuddl::pairwise_counts local{};
    if (valid_reference) {
        auto const row_offset = static_cast<size_t>(reference_id) * k_bucket_count;
        for (auto bucket = static_cast<size_t>(thread_in_reference); bucket < k_bucket_count;
             bucket += threads_per_reference) {
            cuddl::detail::classify(local, query[bucket], rows[row_offset + bucket]);
        }
    }
    auto const warp_total = reduce_warp(warp, local);

    if constexpr (WarpsPerReference == 1U) {
        if (warp.thread_rank() == 0U && valid_reference) {
            write_search_result(results, reference_id, warp_total);
        }
        return;
    }

    if (warp.thread_rank() == 0U) {
        warp_summaries[static_cast<uint32_t>(threadIdx.x) / warp_width] = warp_total;
    }
    block.sync();

    if (warp_in_reference == 0U) {
        cuddl::pairwise_counts partial{};
        if (warp.thread_rank() < WarpsPerReference) {
            partial = warp_summaries[reference_in_block * WarpsPerReference + warp.thread_rank()];
        }
        auto const reference_total = reduce_warp(warp, partial);
        if (warp.thread_rank() == 0U && valid_reference) {
            write_search_result(results, reference_id, reference_total);
        }
    }
}

template <uint32_t WarpsPerReference>
void launch_cooperative_exhaustive(
    thrust::device_vector<uint16_t> const& rows,
    uint32_t reference_count,
    thrust::device_vector<uint16_t> const& query,
    thrust::device_vector<cuddl::reference_search_result>& results,
    cudaStream_t stream
) {
    constexpr uint32_t references_per_block = k_launch_block_size / (32U * WarpsPerReference);
    auto const blocks = (reference_count + references_per_block - 1U) / references_per_block;
    cooperative_exhaustive_search_kernel<WarpsPerReference>
        <<<blocks, k_launch_block_size, 0, stream>>>(
            thrust::raw_pointer_cast(rows.data()),
            reference_count,
            thrust::raw_pointer_cast(query.data()),
            thrust::raw_pointer_cast(results.data())
        );
    CUDDL_CUDA_CALL(cudaGetLastError());
}

std::vector<nvbench::int64_t> const
    parameter_reference_counts{1, 64, 1024, 2048, 2560, 3072, 4096, 8192, 16384, 65536, 200687};
std::vector<std::string> const parameter_row_types{"compact", "packed"};
std::vector<std::string> const parameter_launches{
    "cub_b32_w1", "cub_b64_w1", "cub_b128_w1", "cub_b256_w1", "cub_b512_w1", "cub_b1024_w1",
    "cg_b32_w1",  "cg_b64_w1",  "cg_b128_w1",  "cg_b256_w1",  "cg_b512_w1",  "cg_b1024_w1",
    "cg_b64_w2",  "cg_b128_w2", "cg_b256_w2",  "cg_b512_w2",  "cg_b1024_w2", "cg_b128_w4",
    "cg_b256_w4", "cg_b512_w4", "cg_b1024_w4", "cg_b256_w8",  "cg_b512_w8",  "cg_b1024_w8",
};

std::vector<uint32_t> pack_rows(std::vector<uint16_t> const& rows) {
    std::vector<uint32_t> packed(rows.size());
    for (size_t index = 0; index < rows.size(); ++index) {
        packed[index] = cuddl::detail::pack(rows[index], rows[index] == 0U ? 0U : 1U);
    }
    return packed;
}

template <uint32_t BlockSize, typename ReferenceRow>
__global__ __launch_bounds__(BlockSize) void parameterised_cub_exhaustive_search_kernel(
    ReferenceRow const* rows,
    uint32_t reference_count,
    uint16_t const* query,
    cuddl::reference_search_result* results
) {
    static_assert(BlockSize % 32U == 0U);
    constexpr uint32_t warp_width = 32U;
    constexpr uint32_t warps_per_block = BlockSize / warp_width;
    using warp_reduce = cub::WarpReduce<cuddl::pairwise_counts>;
    __shared__ typename warp_reduce::TempStorage storage[warps_per_block];

    auto const warp = static_cast<uint32_t>(threadIdx.x) / warp_width;
    auto const lane = static_cast<uint32_t>(threadIdx.x) % warp_width;
    auto const reference_id = static_cast<uint32_t>(blockIdx.x) * warps_per_block + warp;
    if (reference_id >= reference_count) {
        return;
    }

    cuddl::pairwise_counts local{};
    auto const row_offset = static_cast<size_t>(reference_id) * k_bucket_count;
    for (auto bucket = static_cast<size_t>(lane); bucket < k_bucket_count; bucket += warp_width) {
        cuddl::detail::classify(
            local, query[bucket], cuddl::detail::reference_score(rows[row_offset + bucket])
        );
    }
    auto const total = warp_reduce(storage[warp]).Sum(local);
    if (lane == 0U) {
        write_search_result(results, reference_id, total);
    }
}

template <uint32_t BlockSize, uint32_t WarpsPerReference, typename ReferenceRow>
__global__ __launch_bounds__(BlockSize) void parameterised_cooperative_exhaustive_search_kernel(
    ReferenceRow const* rows,
    uint32_t reference_count,
    uint16_t const* query,
    cuddl::reference_search_result* results
) {
    static_assert(BlockSize % 32U == 0U);
    static_assert(
        WarpsPerReference == 1U || WarpsPerReference == 2U || WarpsPerReference == 4U ||
        WarpsPerReference == 8U
    );
    static_assert(BlockSize >= 32U * WarpsPerReference);
    static_assert(BlockSize % (32U * WarpsPerReference) == 0U);
    constexpr uint32_t warp_width = 32U;
    constexpr uint32_t warps_per_block = BlockSize / warp_width;
    constexpr uint32_t threads_per_reference = warp_width * WarpsPerReference;
    constexpr uint32_t references_per_block = warps_per_block / WarpsPerReference;

    __shared__ cuddl::pairwise_counts warp_summaries[warps_per_block];
    auto const block = cg::this_thread_block();
    auto const warp = cg::tiled_partition<warp_width>(block);
    auto const reference_in_block = static_cast<uint32_t>(threadIdx.x) / threads_per_reference;
    auto const thread_in_reference = static_cast<uint32_t>(threadIdx.x) % threads_per_reference;
    auto const warp_in_reference = thread_in_reference / warp_width;
    auto const reference_id =
        static_cast<uint32_t>(blockIdx.x) * references_per_block + reference_in_block;
    auto const valid_reference = reference_id < reference_count;

    cuddl::pairwise_counts local{};
    if (valid_reference) {
        auto const row_offset = static_cast<size_t>(reference_id) * k_bucket_count;
        for (auto bucket = static_cast<size_t>(thread_in_reference); bucket < k_bucket_count;
             bucket += threads_per_reference) {
            cuddl::detail::classify(
                local, query[bucket], cuddl::detail::reference_score(rows[row_offset + bucket])
            );
        }
    }
    auto const warp_total = reduce_warp(warp, local);

    if constexpr (WarpsPerReference == 1U) {
        if (warp.thread_rank() == 0U && valid_reference) {
            write_search_result(results, reference_id, warp_total);
        }
        return;
    }

    if (warp.thread_rank() == 0U) {
        warp_summaries[static_cast<uint32_t>(threadIdx.x) / warp_width] = warp_total;
    }
    block.sync();

    if (warp_in_reference == 0U) {
        cuddl::pairwise_counts partial{};
        if (warp.thread_rank() < WarpsPerReference) {
            partial = warp_summaries[reference_in_block * WarpsPerReference + warp.thread_rank()];
        }
        auto const reference_total = reduce_warp(warp, partial);
        if (warp.thread_rank() == 0U && valid_reference) {
            write_search_result(results, reference_id, reference_total);
        }
    }
}

template <uint32_t BlockSize, typename ReferenceRow>
void launch_parameterised_cub(
    thrust::device_vector<ReferenceRow> const& rows,
    uint32_t reference_count,
    thrust::device_vector<uint16_t> const& query,
    thrust::device_vector<cuddl::reference_search_result>& results,
    cudaStream_t stream
) {
    constexpr uint32_t references_per_block = BlockSize / 32U;
    auto const blocks = (reference_count + references_per_block - 1U) / references_per_block;
    parameterised_cub_exhaustive_search_kernel<BlockSize><<<blocks, BlockSize, 0, stream>>>(
        thrust::raw_pointer_cast(rows.data()),
        reference_count,
        thrust::raw_pointer_cast(query.data()),
        thrust::raw_pointer_cast(results.data())
    );
    CUDDL_CUDA_CALL(cudaGetLastError());
}

template <uint32_t BlockSize, uint32_t WarpsPerReference, typename ReferenceRow>
void launch_parameterised_cooperative(
    thrust::device_vector<ReferenceRow> const& rows,
    uint32_t reference_count,
    thrust::device_vector<uint16_t> const& query,
    thrust::device_vector<cuddl::reference_search_result>& results,
    cudaStream_t stream
) {
    constexpr uint32_t references_per_block = BlockSize / (32U * WarpsPerReference);
    auto const blocks = (reference_count + references_per_block - 1U) / references_per_block;
    parameterised_cooperative_exhaustive_search_kernel<BlockSize, WarpsPerReference>
        <<<blocks, BlockSize, 0, stream>>>(
            thrust::raw_pointer_cast(rows.data()),
            reference_count,
            thrust::raw_pointer_cast(query.data()),
            thrust::raw_pointer_cast(results.data())
        );
    CUDDL_CUDA_CALL(cudaGetLastError());
}

template <typename ReferenceRow>
void launch_parameterised(
    std::string const& launch,
    thrust::device_vector<ReferenceRow> const& rows,
    uint32_t reference_count,
    thrust::device_vector<uint16_t> const& query,
    thrust::device_vector<cuddl::reference_search_result>& results,
    cudaStream_t stream
) {
    auto const block_begin = launch.find("_b") + 2U;
    auto const warp_begin = launch.find("_w") + 2U;
    auto const block_size = static_cast<uint32_t>(
        std::stoul(launch.substr(block_begin, warp_begin - 2U - block_begin))
    );
    auto const warps = static_cast<uint32_t>(std::stoul(launch.substr(warp_begin)));
    auto const is_cub = launch.starts_with("cub_");

    if (is_cub) {
        if (warps != 1U) {
            throw std::runtime_error("CUB parameterisation requires one warp per reference");
        }
        switch (block_size) {
            case 32U:
                launch_parameterised_cub<32U>(rows, reference_count, query, results, stream);
                return;
            case 64U:
                launch_parameterised_cub<64U>(rows, reference_count, query, results, stream);
                return;
            case 128U:
                launch_parameterised_cub<128U>(rows, reference_count, query, results, stream);
                return;
            case 256U:
                launch_parameterised_cub<256U>(rows, reference_count, query, results, stream);
                return;
            case 512U:
                launch_parameterised_cub<512U>(rows, reference_count, query, results, stream);
                return;
            case 1024U:
                launch_parameterised_cub<1024U>(rows, reference_count, query, results, stream);
                return;
            default:
                throw std::runtime_error("unknown CUB parameterised block size");
        }
    }

    switch (block_size) {
        case 32U:
            if (warps == 1U) {
                launch_parameterised_cooperative<32U, 1U>(
                    rows, reference_count, query, results, stream
                );
                return;
            }
            break;
        case 64U:
            if (warps == 1U) {
                launch_parameterised_cooperative<64U, 1U>(
                    rows, reference_count, query, results, stream
                );
                return;
            }
            if (warps == 2U) {
                launch_parameterised_cooperative<64U, 2U>(
                    rows, reference_count, query, results, stream
                );
                return;
            }
            break;
        case 128U:
            if (warps == 1U) {
                launch_parameterised_cooperative<128U, 1U>(
                    rows, reference_count, query, results, stream
                );
                return;
            }
            if (warps == 2U) {
                launch_parameterised_cooperative<128U, 2U>(
                    rows, reference_count, query, results, stream
                );
                return;
            }
            if (warps == 4U) {
                launch_parameterised_cooperative<128U, 4U>(
                    rows, reference_count, query, results, stream
                );
                return;
            }
            break;
        case 256U:
            if (warps == 1U) {
                launch_parameterised_cooperative<256U, 1U>(
                    rows, reference_count, query, results, stream
                );
                return;
            }
            if (warps == 2U) {
                launch_parameterised_cooperative<256U, 2U>(
                    rows, reference_count, query, results, stream
                );
                return;
            }
            if (warps == 4U) {
                launch_parameterised_cooperative<256U, 4U>(
                    rows, reference_count, query, results, stream
                );
                return;
            }
            if (warps == 8U) {
                launch_parameterised_cooperative<256U, 8U>(
                    rows, reference_count, query, results, stream
                );
                return;
            }
            break;
        case 512U:
            if (warps == 1U) {
                launch_parameterised_cooperative<512U, 1U>(
                    rows, reference_count, query, results, stream
                );
                return;
            }
            if (warps == 2U) {
                launch_parameterised_cooperative<512U, 2U>(
                    rows, reference_count, query, results, stream
                );
                return;
            }
            if (warps == 4U) {
                launch_parameterised_cooperative<512U, 4U>(
                    rows, reference_count, query, results, stream
                );
                return;
            }
            if (warps == 8U) {
                launch_parameterised_cooperative<512U, 8U>(
                    rows, reference_count, query, results, stream
                );
                return;
            }
            break;
        case 1024U:
            if (warps == 1U) {
                launch_parameterised_cooperative<1024U, 1U>(
                    rows, reference_count, query, results, stream
                );
                return;
            }
            if (warps == 2U) {
                launch_parameterised_cooperative<1024U, 2U>(
                    rows, reference_count, query, results, stream
                );
                return;
            }
            if (warps == 4U) {
                launch_parameterised_cooperative<1024U, 4U>(
                    rows, reference_count, query, results, stream
                );
                return;
            }
            if (warps == 8U) {
                launch_parameterised_cooperative<1024U, 8U>(
                    rows, reference_count, query, results, stream
                );
                return;
            }
            break;
        default:
            break;
    }
    throw std::runtime_error("unknown cooperative parameterised launch");
}

template <typename ReferenceRow>
void run_parameterised_exhaustive(
    nvbench::state& state,
    std::vector<ReferenceRow> const& rows,
    std::vector<uint16_t> const& query,
    thrust::host_vector<cuddl::reference_search_result> const& expected,
    uint32_t reference_count,
    std::string const& launch
) {
    thrust::device_vector<ReferenceRow> device_rows(rows);
    thrust::device_vector<uint16_t> device_query(query);
    thrust::device_vector<cuddl::reference_search_result> results(reference_count);
    auto const execute = [&](cudaStream_t stream) {
        launch_parameterised(launch, device_rows, reference_count, device_query, results, stream);
    };

    execute(cudaStream_t{nullptr});
    CUDDL_CUDA_CALL(cudaDeviceSynchronize());
    thrust::host_vector<cuddl::reference_search_result> observed(results);
    if (!std::equal(expected.begin(), expected.end(), observed.begin())) {
        throw std::runtime_error("parameterised exhaustive search disagrees with scalar oracle");
    }

    state.exec([&](nvbench::launch& nvbench_launch) { execute(nvbench_launch.get_stream()); });
}

void compact_exhaustive_parameter_sweep(nvbench::state& state) {
    auto const reference_count = static_cast<uint32_t>(state.get_int64("References"));
    auto const row_type = state.get_string("Row");
    auto const launch = state.get_string("Launch");
    auto const fixture = make_indexed_fixture(reference_count);
    thrust::host_vector<cuddl::reference_search_result> expected_host(reference_count);
    for (uint32_t reference_id = 0; reference_id < reference_count; ++reference_id) {
        expected_host[reference_id] = {
            .reference_id = reference_id,
            .summary = score_row_oracle(fixture.query, fixture.rows, reference_id),
        };
    }

    state.add_element_count(reference_count, "Exact Comparisons");
    if (row_type == "compact") {
        run_parameterised_exhaustive(
            state, fixture.rows, fixture.query, expected_host, reference_count, launch
        );
    } else if (row_type == "packed") {
        run_parameterised_exhaustive(
            state, pack_rows(fixture.rows), fixture.query, expected_host, reference_count, launch
        );
    } else {
        throw std::runtime_error("unknown parameter sweep row type");
    }
    add_median_time(state);
    add_median_throughput(state, reference_count);
}

/// @brief Sequence-level FASTA end-to-end pipeline mode (activated with
///        `--reference <fasta> --query <fasta>` instead of nvbench axes).
using pipeline_database = cuddl::reference_database<25U, 4096U>;
constexpr uint32_t k_pipeline_k = 25U;
constexpr size_t k_pipeline_buckets = 4096U;
constexpr nvbench::int64_t k_pipeline_samples = 20;
constexpr nvbench::int64_t k_pipeline_warmup_runs = 3;
class pipeline_failure : public std::runtime_error {
   public:
    using std::runtime_error::runtime_error;
};
__global__ void
extract_winner_scores_kernel(uint32_t const* registers, uint16_t* scores, size_t count) {
    for (auto i = static_cast<size_t>(blockIdx.x * blockDim.x + threadIdx.x); i < count;
         i += static_cast<size_t>(blockDim.x * gridDim.x)) {
        scores[i] = cuddl::detail::winner(registers[i]);
    }
}

/// @brief Aborts into a pipeline failure carrying the wrapped error message.
template <typename T>
T pipeline_unwrap(cuddl::Result<T> result) {
    if (!result) {
        throw pipeline_failure(result.error().message());
    }
    return std::move(*result);
}

void pipeline_check(cuddl::Result<void> result) {
    if (!result) {
        throw pipeline_failure(result.error().message());
    }
}

int pipeline_run(std::string const& command, std::string& output) {
    FILE* pipe = popen((command + " 2>&1").c_str(), "r");
    if (pipe == nullptr) {
        throw pipeline_failure("cannot invoke: " + command);
    }
    output.clear();
    char buffer[4096];
    while (std::fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        output += buffer;
    }
    return pclose(pipe);
}

std::string pipeline_quote(std::string_view value) {
    std::string quoted = "'";
    for (char const character : value) {
        if (character == '\'') {
            quoted += "'\\''";
        } else {
            quoted.push_back(character);
        }
    }
    quoted.push_back('\'');
    return quoted;
}

std::string pipeline_sha256(std::string const& path) {
    std::string output;
    if (pipeline_run("sha256sum " + pipeline_quote(path), output) != 0) {
        throw pipeline_failure("sha256sum failed for " + path + ": " + output);
    }
    std::string hex;
    for (char const character : output) {
        if (hex.size() == 64U) {
            break;
        }
        if (std::isxdigit(static_cast<unsigned char>(character))) {
            hex.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(character))));
        }
    }
    if (hex.size() != 64U) {
        throw pipeline_failure("unexpected sha256sum output for " + path);
    }
    return hex;
}

std::string pipeline_revision() {
    std::string output;
    if (pipeline_run("git rev-parse HEAD", output) != 0) {
        return "(no revision)";
    }
    while (!output.empty() && (output.back() == '\n' || output.back() == '\r')) {
        output.pop_back();
    }
    return output;
}

uint64_t pipeline_host_peak_bytes() {
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.rfind("VmHWM:", 0U) == 0U) {
            return std::stoull(line.substr(line.find_first_of("0123456789"))) * 1024U;
        }
    }
    return 0U;
}

uint64_t pipeline_device_used_bytes() {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    CUDDL_CUDA_CALL(cudaMemGetInfo(&free_bytes, &total_bytes));
    return total_bytes - free_bytes;
}

struct pipeline_options {
    std::string name;
    std::string reference;
    std::string query;
    std::string output;
    uint32_t minimum_matches = 5U;
};

template <typename Function>
struct pipeline_callable {
    Function function;

    void operator()(nvbench::state& state, nvbench::type_list<>) {
        function(state);
    }
};

template <typename Function>
json pipeline_measure(
    std::string const& name,
    std::vector<std::string> stages,
    bool cpu_only,
    Function&& function
) {
    using function_type = std::decay_t<Function>;
    using callable_type = pipeline_callable<function_type>;
    nvbench::benchmark<callable_type> benchmark(callable_type{std::forward<Function>(function)});
    benchmark.set_name(name)
        .add_string_axis("Stage", std::move(stages))
        .set_stopping_criterion("sample-count")
        .set_min_samples(k_pipeline_samples)
        .set_criterion_param_int64("target-samples", k_pipeline_samples)
        .set_cold_warmup_runs(k_pipeline_warmup_runs)
        .set_skip_batched(true);
    if (cpu_only) {
        benchmark.set_is_cpu_only(true);
    } else {
        int device = 0;
        CUDDL_CUDA_CALL(cudaGetDevice(&device));
        benchmark.add_device(device);
    }
    benchmark.run();

    auto const clock = cpu_only ? "cpu" : "gpu";
    auto const measurement = cpu_only ? "nv/cpu_only" : "nv/cold";
    json measurements = json::object();
    for (auto const& state : benchmark.get_states()) {
        if (state.is_skipped()) {
            throw pipeline_failure(state.get_skip_reason());
        }
        auto const prefix = std::string{measurement} + "/time/" + clock;
        measurements[state.get_string("Stage")] = {
            {"samples",
             state.get_summary(std::string{measurement} + "/sample_size").get_int64("value")},
            {"median_ms", state.get_summary(prefix + "/median").get_float64("value") * 1000.0},
            {"min_ms", state.get_summary(prefix + "/min").get_float64("value") * 1000.0},
            {"max_ms", state.get_summary(prefix + "/max").get_float64("value") * 1000.0},
            {"relative_stddev_percent",
             state.get_summary(prefix + "/stdev/relative").get_float64("value") * 100.0},
        };
    }
    return measurements;
}

json run_pipeline(pipeline_options const& options) {
    auto const benchmark_name =
        options.name.empty() ? std::filesystem::path(options.reference).stem().string() + " vs " +
                                   std::filesystem::path(options.query).stem().string()
                             : options.name;
    auto const checksum_reference = pipeline_sha256(options.reference);
    auto const checksum_query = pipeline_sha256(options.query);

    auto const device_baseline = pipeline_device_used_bytes();
    uint64_t device_peak = 0;
    auto sample_device = [&]() {
        auto const used = pipeline_device_used_bytes();
        device_peak = std::max(device_peak, used - device_baseline);
    };

    auto const reference_kmers =
        pipeline_unwrap(cuddl::detail::parse_fasta(options.reference, k_pipeline_k));
    auto const query_kmers =
        pipeline_unwrap(cuddl::detail::parse_fasta(options.query, k_pipeline_k));

    thrust::device_vector<uint64_t> device_reference_kmers(reference_kmers.kmers);
    thrust::device_vector<uint64_t> device_query_kmers(query_kmers.kmers);

    cuddl::sketch<k_pipeline_k, k_pipeline_buckets> reference_sketch;
    pipeline_check(reference_sketch.add_async(device_reference_kmers));

    cuddl::sketch<k_pipeline_k, k_pipeline_buckets> query_sketch;
    pipeline_check(query_sketch.add_async(device_query_kmers));

    thrust::device_vector<uint16_t> device_reference_rows(k_pipeline_buckets);
    thrust::device_vector<uint16_t> device_query_rows(k_pipeline_buckets);
    constexpr uint32_t extract_block = 256;
    constexpr uint32_t extract_grid =
        static_cast<uint32_t>((k_pipeline_buckets + extract_block - 1) / extract_block);
    extract_winner_scores_kernel<<<extract_grid, extract_block>>>(
        thrust::raw_pointer_cast(reference_sketch.data().data()),
        thrust::raw_pointer_cast(device_reference_rows.data()),
        k_pipeline_buckets
    );
    extract_winner_scores_kernel<<<extract_grid, extract_block>>>(
        thrust::raw_pointer_cast(query_sketch.data().data()),
        thrust::raw_pointer_cast(device_query_rows.data()),
        k_pipeline_buckets
    );
    CUDDL_CUDA_CALL(cudaGetLastError());
    sample_device();

    auto const compatibility =
        cuddl::score_compatibility::current<k_pipeline_k, k_pipeline_buckets>();

    auto bare_database =
        pipeline_unwrap(pipeline_database::build_async(device_reference_rows, compatibility));

    auto database = pipeline_unwrap(
        pipeline_database::build_indexed_async(device_reference_rows, compatibility)
    );

    auto const workspace_bytes = pipeline_unwrap(database.indexed_single_query_workspace_bytes());
    thrust::device_vector<cuddl::reference_search_result> exhaustive_results(
        database.reference_count()
    );
    thrust::device_vector<cuddl::reference_search_result> indexed_results(
        database.reference_count()
    );
    thrust::device_vector<uint8_t> workspace(workspace_bytes);
    thrust::device_vector<uint32_t> result_count(1U);

    auto const query_span = cuddl::device_span<uint16_t const>{
        thrust::raw_pointer_cast(device_query_rows.data()), device_query_rows.size()
    };
    auto exhaustive_span = cuddl::device_span<cuddl::reference_search_result>{
        thrust::raw_pointer_cast(exhaustive_results.data()), exhaustive_results.size()
    };
    auto indexed_span = cuddl::device_span<cuddl::reference_search_result>{
        thrust::raw_pointer_cast(indexed_results.data()), indexed_results.size()
    };
    auto workspace_span =
        cuddl::device_span<uint8_t>{thrust::raw_pointer_cast(workspace.data()), workspace.size()};
    auto count_span = cuddl::device_span<uint32_t>{
        thrust::raw_pointer_cast(result_count.data()), result_count.size()
    };

    pipeline_check(database.search_async(query_span, compatibility, {}, exhaustive_span));

    pipeline_check(database.search_indexed_async(
        query_span,
        compatibility,
        workspace_span,
        indexed_span,
        count_span,
        {.minimum_matches = options.minimum_matches}
    ));
    CUDDL_CUDA_CALL(cudaDeviceSynchronize());

    thrust::host_vector<cuddl::reference_search_result> exhaustive_host(exhaustive_results);
    thrust::host_vector<cuddl::reference_search_result> indexed_host(indexed_results);
    thrust::host_vector<uint32_t> count_host(result_count);
    auto const host_peak = pipeline_host_peak_bytes();

    auto timing_statistics = pipeline_measure(
        "pipeline_parse", {"parse_reference", "parse_query"}, true, [&](nvbench::state& state) {
            auto const& path =
                state.get_string("Stage") == "parse_reference" ? options.reference : options.query;
            state.exec(nvbench::exec_tag::timer, [&](nvbench::launch&, auto& timer) {
                timer.start();
                auto parsed = cuddl::detail::parse_fasta(path, k_pipeline_k);
                timer.stop();
                if (!parsed) {
                    throw pipeline_failure(parsed.error().message());
                }
                do_not_optimise(*parsed);
            });
        }
    );

    auto gpu_statistics = pipeline_measure(
        "pipeline_device",
        {
            "host_to_device_transfer",
            "sketch_reference",
            "sketch_query",
            "sketch_extract",
            "database_build",
            "index_build",
            "search_exhaustive",
            "search_indexed",
        },
        false,
        [&](nvbench::state& state) {
            auto const stage = state.get_string("Stage");
            state.exec(
                nvbench::exec_tag::sync | nvbench::exec_tag::timer,
                [&](nvbench::launch& launch, auto& timer) {
                    auto const stream = cuda::stream_ref{launch.get_stream()};
                    if (stage == "host_to_device_transfer") {
                        timer.start();
                        CUDDL_CUDA_CALL(cudaMemcpyAsync(
                            thrust::raw_pointer_cast(device_reference_kmers.data()),
                            reference_kmers.kmers.data(),
                            reference_kmers.kmers.size() * sizeof(uint64_t),
                            cudaMemcpyHostToDevice,
                            launch.get_stream()
                        ));
                        CUDDL_CUDA_CALL(cudaMemcpyAsync(
                            thrust::raw_pointer_cast(device_query_kmers.data()),
                            query_kmers.kmers.data(),
                            query_kmers.kmers.size() * sizeof(uint64_t),
                            cudaMemcpyHostToDevice,
                            launch.get_stream()
                        ));
                        timer.stop();
                    } else if (stage == "sketch_reference") {
                        pipeline_check(reference_sketch.clear_async(stream));
                        timer.start();
                        pipeline_check(reference_sketch.add_async(device_reference_kmers, stream));
                        timer.stop();
                    } else if (stage == "sketch_query") {
                        pipeline_check(query_sketch.clear_async(stream));
                        timer.start();
                        pipeline_check(query_sketch.add_async(device_query_kmers, stream));
                        timer.stop();
                    } else if (stage == "sketch_extract") {
                        timer.start();
                        extract_winner_scores_kernel<<<
                            extract_grid,
                            extract_block,
                            0,
                            launch.get_stream()>>>(
                            thrust::raw_pointer_cast(reference_sketch.data().data()),
                            thrust::raw_pointer_cast(device_reference_rows.data()),
                            k_pipeline_buckets
                        );
                        extract_winner_scores_kernel<<<
                            extract_grid,
                            extract_block,
                            0,
                            launch.get_stream()>>>(
                            thrust::raw_pointer_cast(query_sketch.data().data()),
                            thrust::raw_pointer_cast(device_query_rows.data()),
                            k_pipeline_buckets
                        );
                        CUDDL_CUDA_CALL(cudaGetLastError());
                        timer.stop();
                    } else if (stage == "database_build") {
                        timer.start();
                        auto measured = pipeline_database::build_async(
                            device_reference_rows, compatibility, stream
                        );
                        timer.stop();
                        auto measured_database = pipeline_unwrap(std::move(measured));
                        do_not_optimise(measured_database);
                    } else if (stage == "index_build") {
                        timer.start();
                        auto measured = pipeline_database::build_indexed_async(
                            device_reference_rows, compatibility, stream
                        );
                        timer.stop();
                        auto measured_database = pipeline_unwrap(std::move(measured));
                        do_not_optimise(measured_database);
                    } else if (stage == "search_exhaustive") {
                        timer.start();
                        pipeline_check(database.search_async(
                            query_span, compatibility, {}, exhaustive_span, stream
                        ));
                        timer.stop();
                    } else if (stage == "search_indexed") {
                        timer.start();
                        pipeline_check(database.search_indexed_async(
                            query_span,
                            compatibility,
                            workspace_span,
                            indexed_span,
                            count_span,
                            {.minimum_matches = options.minimum_matches},
                            stream
                        ));
                        timer.stop();
                    } else {
                        throw pipeline_failure("unknown pipeline timing stage: " + stage);
                    }
                }
            );
        }
    );
    timing_statistics.update(gpu_statistics);
    sample_device();

    auto const candidate_count = count_host.front();
    bool agreement = true;
    json matches = json::array();
    for (uint32_t candidate = 0U; candidate < candidate_count; ++candidate) {
        auto const indexed = indexed_host[candidate];
        auto const exhaustive = exhaustive_host[indexed.reference_id];
        if (!(indexed.summary == exhaustive.summary)) {
            agreement = false;
        }
        matches.push_back({
            {"implementation", {{"name", "cuddl"}, {"revision", pipeline_revision()}}},
            {"case", {{"measurement", "match"}, {"reference_id", indexed.reference_id}}},
            {"metrics",
             {
                 {"exhaustive_lower", exhaustive.summary.counts.lower},
                 {"exhaustive_equal", exhaustive.summary.counts.equal},
                 {"exhaustive_higher", exhaustive.summary.counts.higher},
                 {"exhaustive_both_empty", exhaustive.summary.counts.both_empty},
                 {"indexed_lower", indexed.summary.counts.lower},
                 {"indexed_equal", indexed.summary.counts.equal},
                 {"indexed_higher", indexed.summary.counts.higher},
                 {"indexed_both_empty", indexed.summary.counts.both_empty},
             }},
        });
    }

    uint64_t posting_visits = 0;
    for (auto const& result : exhaustive_host) {
        posting_visits += result.summary.counts.equal;
    }

    json measurements = json::array({
        {
            {"implementation", {{"name", "cuddl"}, {"revision", pipeline_revision()}}},
            {"case",
             {
                 {"measurement", "pipeline"},
                 {"k", k_pipeline_k},
                 {"buckets", k_pipeline_buckets},
                 {"exponent_bits", compatibility.exponent_bits},
                 {"mantissa_bits", compatibility.mantissa_bits},
                 {"hash_identity", compatibility.hash_identity},
                 {"hash_seed", compatibility.hash_seed},
                 {"canonicalisation_policy", compatibility.canonicalisation_policy},
                 {"minimum_matches", options.minimum_matches},
                 {"references", 1U},
                 {"queries", 1U},
                 {"reference_kmers", reference_kmers.valid_kmers},
                 {"query_kmers", query_kmers.valid_kmers},
                 {"reference_bases", reference_kmers.bases},
                 {"query_bases", query_kmers.bases},
                 {"reference_invalid_windows", reference_kmers.invalid_windows},
                 {"query_invalid_windows", query_kmers.invalid_windows},
                 {"warmup_runs", k_pipeline_warmup_runs},
                 {"target_samples", k_pipeline_samples},
             }},
            {"metrics",
             {
                 {"posting_visits", posting_visits},
                 {"candidates", candidate_count},
                 {"indexed_vs_exhaustive", agreement},
                 {"checked_candidates", candidate_count},
             }},
            {"timings", timing_statistics},
            {"memory_bytes", {{"host_peak", host_peak}, {"device_peak", device_peak}}},
        },
    });
    for (auto& match : matches) {
        measurements.push_back(std::move(match));
    }

    return make_benchmark_result(
        benchmark_name,
        "indexed_search",
        "end_to_end",
        std::move(measurements),
        {
            {"reference", {{"path", options.reference}, {"sha256", checksum_reference}}},
            {"query", {{"path", options.query}, {"sha256", checksum_query}}},
        }
    );
}

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
void compact_exhaustive_launch_shape(nvbench::state& state) {
    auto const reference_count = static_cast<uint32_t>(state.get_int64("References"));
    auto const launch_shape = state.get_string("Launch");
    auto const fixture = make_indexed_fixture(reference_count);
    thrust::device_vector<uint16_t> device_rows(fixture.rows);
    thrust::device_vector<uint16_t> device_query(fixture.query);
    thrust::device_vector<cuddl::reference_search_result> expected(reference_count);
    thrust::device_vector<cuddl::reference_search_result> results(reference_count);
    thrust::device_vector<uint8_t> workspace;
    auto const compatibility = cuddl::score_compatibility::current<k_kmer_length, k_bucket_count>();
    auto database =
        CUDDL_UNWRAP((cuddl::reference_database<k_kmer_length, k_bucket_count>::build_async(
            device_rows, compatibility
        )));

    auto const launch = [&](cudaStream_t stream) {
        if (launch_shape == "current_cub_warp") {
            CUDDL_UNWRAP(database.search_async(
                device_query, compatibility, workspace, results, cuda::stream_ref{stream}
            ));
        } else if (launch_shape == "raw_cub_warp") {
            launch_raw_cub_exhaustive(device_rows, reference_count, device_query, results, stream);
        } else if (launch_shape == "span_cub_warp") {
            launch_span_cub_exhaustive(device_rows, reference_count, device_query, results, stream);
        } else if (launch_shape == "cg_1_warp") {
            launch_cooperative_exhaustive<1U>(
                device_rows, reference_count, device_query, results, stream
            );
        } else if (launch_shape == "cg_2_warps") {
            launch_cooperative_exhaustive<2U>(
                device_rows, reference_count, device_query, results, stream
            );
        } else if (launch_shape == "cg_4_warps") {
            launch_cooperative_exhaustive<4U>(
                device_rows, reference_count, device_query, results, stream
            );
        } else if (launch_shape == "cg_8_warps") {
            launch_cooperative_exhaustive<8U>(
                device_rows, reference_count, device_query, results, stream
            );
        } else {
            throw std::runtime_error("unknown exhaustive launch shape");
        }
    };

    CUDDL_UNWRAP(database.search_async(device_query, compatibility, workspace, expected));
    launch(cudaStream_t{nullptr});
    CUDDL_CUDA_CALL(cudaDeviceSynchronize());
    thrust::host_vector<cuddl::reference_search_result> expected_host(expected);
    thrust::host_vector<cuddl::reference_search_result> results_host(results);
    if (!std::equal(expected_host.begin(), expected_host.end(), results_host.begin())) {
        throw std::runtime_error("cooperative exhaustive search disagrees with current search");
    }

    state.add_element_count(reference_count, "Exact Comparisons");
    state.add_global_memory_reads<uint16_t>(
        2ULL * static_cast<uint64_t>(reference_count) * k_bucket_count
    );
    state.exec([&](nvbench::launch& nvbench_launch) { launch(nvbench_launch.get_stream()); });
    add_median_time(state);
    add_median_throughput(state, reference_count);
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

}  // namespace

NVBENCH_BENCH(compact_exhaustive_search)
    .add_int64_power_of_two_axis("References", reference_powers);
NVBENCH_BENCH(compact_exhaustive_launch_shape)
    .add_int64_axis("References", launch_reference_counts)
    .add_string_axis("Launch", launch_shapes)
    .set_stopping_criterion("sample-count")
    .set_min_samples(20)
    .set_criterion_param_int64("target-samples", 20);
NVBENCH_BENCH(compact_exhaustive_parameter_sweep)
    .add_int64_axis("References", parameter_reference_counts)
    .add_string_axis("Row", parameter_row_types)
    .add_string_axis("Launch", parameter_launches)
    .set_stopping_criterion("sample-count")
    .set_min_samples(20)
    .set_criterion_param_int64("target-samples", 20);
NVBENCH_BENCH(compact_indexed_build)
    .add_int64_power_of_two_axis("References", indexed_reference_powers);
NVBENCH_BENCH(compact_indexed_search)
    .add_int64_power_of_two_axis("References", indexed_reference_powers);
NVBENCH_BENCH(compact_indexed_zero_threshold_search)
    .add_int64_power_of_two_axis("References", indexed_reference_powers);
NVBENCH_BENCH(compact_batch_and_all_to_all_search);

/// @brief Intercepts FASTA pipeline options; the rest reaches nvbench.
int main(int argc, char** argv) try {
    ::pipeline_options options;
    bool pipeline_mode = false;
    std::vector<char*> remaining{argv[0]};
    for (int argument = 1; argument < argc; ++argument) {
        std::string_view const value = argv[argument];
        auto const split = value.find('=');
        auto const name = value.substr(0U, split);
        if (name == "--name" || name == "--reference" || name == "--query" || name == "--output" ||
            name == "--minimum-matches") {
            std::string_view take = value.substr(split + 1U);
            if (split == std::string_view::npos) {
                if (argument + 1 >= argc) {
                    throw pipeline_failure(std::string(name) + " needs a value");
                }
                take = argv[++argument];
            }
            if (name == "--minimum-matches") {
                options.minimum_matches = static_cast<uint32_t>(std::stoul(std::string(take)));
            } else if (name == "--name") {
                options.name = take;
            } else if (name == "--reference") {
                options.reference = take;
            } else if (name == "--query") {
                options.query = take;
            } else {
                options.output = take;
            }
            pipeline_mode = true;
            continue;
        }
        remaining.push_back(argv[argument]);
    }

    if (!pipeline_mode) {
        nvbench::detail::main_initialize(argc, argv);
        {
            auto args = nvbench::detail::main_convert_args(
                static_cast<int>(remaining.size()), remaining.data()
            );
            nvbench::option_parser parser;
            parser.parse(args);
            nvbench::detail::main_print_preamble(parser);
            nvbench::detail::main_run_benchmarks(parser);
            nvbench::detail::main_print_epilogue(parser);
            nvbench::detail::main_print_results(parser);
        }
        nvbench::detail::main_finalize();
        return 0;
    }

    if (options.reference.empty() || options.query.empty()) {
        throw pipeline_failure("--reference and --query both need a FASTA path");
    }
    auto const report = ::run_pipeline(options);
    auto const text = report.dump(2);
    if (options.output.empty()) {
        std::cout << text << '\n';
    } else {
        std::ofstream target(options.output);
        if (!target) {
            throw pipeline_failure("cannot write " + options.output);
        }
        target << text << '\n';
    }
    return 0;
} catch (std::exception& error) {
    std::cerr << "\ncuddl-compact-search error: " << error.what() << "\n";
    return 1;
}
