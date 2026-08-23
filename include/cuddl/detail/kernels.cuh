#pragma once

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>

#include <cuddl/detail/cardinality.cuh>
#include <cuddl/detail/comparison.cuh>
#include <cuddl/detail/hash.cuh>
#include <cuddl/detail/hybrid_cardinality.cuh>
#include <cuddl/detail/register.cuh>
#include <cuddl/pairwise_counts.cuh>

namespace cg = cooperative_groups;

namespace cuddl::detail {

/// @brief Threads per CTA for the single-pair and cardinality reduction kernels.
constexpr uint32_t block_size = 256;

/// @brief Per-thread accumulator feeding a CUB block reduction for fused summaries.
struct summary_payload {
    pairwise_counts counts{};
    uint64_t empty{};
    double restored_sum{};

    __device__ summary_payload& operator+=(summary_payload const& other) noexcept {
        counts += other.counts;
        empty += other.empty;
        restored_sum += other.restored_sum;
        return *this;
    }

    friend __device__ summary_payload
    operator+(summary_payload left, summary_payload const& right) noexcept {
        return left += right;
    }
};

namespace {

/// @brief Combines two per-thread payloads at compile time.
template <bool IncludeCardinality>
__device__ summary_payload combine_payloads(summary_payload a, summary_payload const& b) noexcept {
    a.counts += b.counts;
    if constexpr (IncludeCardinality) {
        a.empty += b.empty;
        a.restored_sum += b.restored_sum;
    }
    return a;
}

}  // namespace

/// @brief Per-thread accumulator for a standalone cardinality reduction.
struct cardinality_payload {
    uint64_t empty{};
    double restored_sum{};

    __device__ cardinality_payload& operator+=(cardinality_payload const& other) noexcept {
        empty += other.empty;
        restored_sum += other.restored_sum;
        return *this;
    }

    friend __device__ cardinality_payload
    operator+(cardinality_payload left, cardinality_payload const& right) noexcept {
        return left += right;
    }
};

inline __device__ cardinality_payload
combine_cardinality(cardinality_payload const& a, cardinality_payload const& b) noexcept {
    return a + b;
}

/**
 * @brief Constructs a sketch from packed k-mers using direct global packed CAS.
 *
 * @p saturation records whether any register's winner count saturated.
 */
template <size_t BucketCount>
__global__ __launch_bounds__(block_size, 4) void add_kernel(
    uint64_t const* input,
    size_t input_size,
    uint32_t* registers,
    uint32_t& saturation
) {
    auto const index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    auto const stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (auto offset = index; offset < input_size; offset += stride * 2U) {
        _Pragma("unroll")
        for (uint32_t item = 0; item < 2U; ++item) {
            auto const current = offset + stride * item;
            if (current < input_size) {
                auto const hash = hash_kmer(input[current]);
                update(&registers[bucket_of<BucketCount>(hash)], score(hash), saturation);
            }
        }
    }
}

/**
 * @brief Computes pairwise counts and optionally cardinality for two constructed sketches.
 */
template <size_t BucketCount, bool IncludeCardinality>
__global__ void summary_kernel(
    uint32_t const* left, uint32_t const* right, pairwise_summary& output
) {
    using block_reduce = cub::BlockReduce<summary_payload, block_size>;
    __shared__ typename block_reduce::TempStorage storage;

    summary_payload local{};
    for (auto bucket = static_cast<size_t>(threadIdx.x); bucket < BucketCount;
         bucket += block_size) {
        auto const left_reg = left[bucket];
        auto const right_reg = right[bucket];
        classify(local.counts, left_reg, right_reg);
        if constexpr (IncludeCardinality) {
            if (winner(left_reg) == 0U) {
                ++local.empty;
            } else {
                local.restored_sum += restore(winner(left_reg));
            }
        }
    }
    auto const total = block_reduce(storage).Reduce(local, combine_payloads<IncludeCardinality>);
    if (threadIdx.x == 0) {
        output.counts = total.counts;
        if constexpr (IncludeCardinality) {
            output.cardinality = cardinality(
                static_cast<double>(BucketCount),
                static_cast<double>(total.empty),
                total.restored_sum
            );
        }
    }
}

/// @brief Reads the search score from either supported exact row backing.
__host__ __device__ constexpr uint16_t reference_score(uint16_t score) noexcept {
    return score;
}

__host__ __device__ constexpr uint16_t reference_score(uint32_t packed_register) noexcept {
    return winner(packed_register);
}

/**
 * @brief Reduces four pairwise counters within one warp.
 */
inline __device__ pairwise_counts
reduce_warp(cg::thread_block_tile<32> const warp, pairwise_counts counts) {
    counts.lower = cg::reduce(warp, counts.lower, cg::plus<uint32_t>{});
    counts.equal = cg::reduce(warp, counts.equal, cg::plus<uint32_t>{});
    counts.higher = cg::reduce(warp, counts.higher, cg::plus<uint32_t>{});
    counts.both_empty = cg::reduce(warp, counts.both_empty, cg::plus<uint32_t>{});
    return counts;
}

/**
 * @brief Compares one compact query row with every row in a reference database.
 *
 * Each reference owns a compile-time number of warps and writes exactly one stable-ID result.
 */
template <
    size_t BucketCount,
    uint32_t BlockSize,
    uint32_t WarpsPerReference,
    typename ReferenceRow,
    typename SearchResult>
__global__ __launch_bounds__(BlockSize) void exhaustive_search_kernel(
    ReferenceRow const* rows,
    uint32_t reference_count,
    uint16_t const* query,
    SearchResult* results
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

    __shared__ pairwise_counts warp_summaries[warps_per_block];
    auto const block = cg::this_thread_block();
    auto const warp = cg::tiled_partition<warp_width>(block);
    auto const reference_in_block = static_cast<uint32_t>(threadIdx.x) / threads_per_reference;
    auto const thread_in_reference = static_cast<uint32_t>(threadIdx.x) % threads_per_reference;
    auto const warp_in_reference = thread_in_reference / warp_width;
    auto const reference_id =
        static_cast<uint32_t>(blockIdx.x) * references_per_block + reference_in_block;
    auto const valid_reference = reference_id < reference_count;

    pairwise_counts local{};
    if (valid_reference) {
        auto const row_offset = static_cast<size_t>(reference_id) * BucketCount;
        for (auto bucket = static_cast<size_t>(thread_in_reference); bucket < BucketCount;
             bucket += threads_per_reference) {
            classify(local, query[bucket], reference_score(rows[row_offset + bucket]));
        }
    }
    auto const warp_total = reduce_warp(warp, local);

    if constexpr (WarpsPerReference == 1U) {
        if (warp.thread_rank() == 0U && valid_reference) {
            results[reference_id].reference_id = reference_id;
            results[reference_id].summary.counts = warp_total;
            results[reference_id].summary.cardinality = 0.0;
        }
        return;
    }

    if (warp.thread_rank() == 0U) {
        warp_summaries[static_cast<uint32_t>(threadIdx.x) / warp_width] = warp_total;
    }
    block.sync();

    if (warp_in_reference == 0U) {
        pairwise_counts partial{};
        if (warp.thread_rank() < WarpsPerReference) {
            partial = warp_summaries[reference_in_block * WarpsPerReference + warp.thread_rank()];
        }
        auto const reference_total = reduce_warp(warp, partial);
        if (warp.thread_rank() == 0U && valid_reference) {
            results[reference_id].reference_id = reference_id;
            results[reference_id].summary.counts = reference_total;
            results[reference_id].summary.cardinality = 0.0;
        }
    }
}

/// @brief Counts every non-empty indexed row entry in its dense index cell.
///
/// Raw score zero is discarded before masking so masked key zero remains valid.
template <size_t BucketCount, typename ReferenceRow>
__global__ void count_index_cells_kernel(
    ReferenceRow const* rows,
    size_t indexed_row_count,
    uint32_t indexed_bucket_count,
    uint16_t key_mask,
    uint32_t* cell_counts
) {
    auto offset = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    auto const stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    auto const key_count = static_cast<uint32_t>(key_mask) + 1U;
    for (; offset < indexed_row_count; offset += stride) {
        auto const reference_id = offset / indexed_bucket_count;
        auto const bucket = offset % indexed_bucket_count;
        auto const score = reference_score(rows[reference_id * BucketCount + bucket]);
        if (score == 0U) {
            continue;
        }
        auto const key = static_cast<uint32_t>(score & key_mask);
        auto const cell = static_cast<uint64_t>(bucket) * key_count + key;
        atomicAdd(&cell_counts[cell], 1U);
    }
}

/// @brief Scatters every non-empty indexed row entry into its dense CSR posting range.
template <size_t BucketCount, typename ReferenceRow>
__global__ void scatter_index_postings_kernel(
    ReferenceRow const* rows,
    size_t indexed_row_count,
    uint32_t indexed_bucket_count,
    uint16_t key_mask,
    uint32_t* cursors,
    uint32_t* postings
) {
    auto offset = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    auto const stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    auto const key_count = static_cast<uint32_t>(key_mask) + 1U;
    for (; offset < indexed_row_count; offset += stride) {
        auto const reference_id = offset / indexed_bucket_count;
        auto const bucket = offset % indexed_bucket_count;
        auto const score = reference_score(rows[reference_id * BucketCount + bucket]);
        if (score == 0U) {
            continue;
        }
        auto const key = static_cast<uint32_t>(score & key_mask);
        auto const cell = static_cast<uint64_t>(bucket) * key_count + key;
        auto const posting = atomicAdd(&cursors[cell], 1U);
        postings[posting] = static_cast<uint32_t>(reference_id);
    }
}

/// @brief Counts the query's non-empty posting matches for every reference.
template <size_t BucketCount>
__global__ void count_index_matches_kernel(
    uint16_t const* query,
    uint32_t const* offsets,
    uint32_t const* postings,
    uint32_t indexed_bucket_count,
    uint16_t key_mask,
    uint32_t* match_counts
) {
    auto const bucket = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (bucket >= indexed_bucket_count) {
        return;
    }
    auto const score = query[bucket];
    if (score == 0U) {
        return;
    }
    auto const key_count = static_cast<uint32_t>(key_mask) + 1U;
    auto const key = static_cast<uint32_t>(score & key_mask);
    auto const cell = static_cast<uint64_t>(bucket) * key_count + key;
    auto const begin = offsets[cell];
    auto const end = offsets[cell + 1U];
    for (auto posting = begin; posting < end; ++posting) {
        atomicAdd(&match_counts[postings[posting]], 1U);
    }
}

/// @brief Stable CUB selection predicate over per-reference match counts.
struct minimum_match_predicate {
    uint32_t const* match_counts;
    uint32_t minimum_matches;

    [[nodiscard]] __host__ __device__ bool operator()(uint32_t reference_id) const noexcept {
        return match_counts[reference_id] >= minimum_matches;
    }
};

/// @brief Writes a known exhaustive result count without host-lifetime coupling.
__global__ void write_result_count_kernel(uint32_t value, uint32_t* result_count) {
    if (threadIdx.x == 0) {
        *result_count = value;
    }
}

/// @brief Exactly refines every selected reference over its full winner-score row.
template <size_t BucketCount, typename ReferenceRow, typename SearchResult>
__global__ __launch_bounds__(block_size) void refine_index_candidates_kernel(
    ReferenceRow const* rows,
    uint16_t const* query,
    uint32_t const* candidate_ids,
    uint32_t const* candidate_count,
    SearchResult* results
) {
    constexpr uint32_t warp_width = 32;
    constexpr uint32_t warps_per_block = block_size / warp_width;
    using warp_reduce = cub::WarpReduce<pairwise_counts>;
    __shared__ typename warp_reduce::TempStorage storage[warps_per_block];

    auto const warp = static_cast<uint32_t>(threadIdx.x) / warp_width;
    auto const lane = static_cast<uint32_t>(threadIdx.x) % warp_width;
    auto const candidate_index = static_cast<uint32_t>(blockIdx.x) * warps_per_block + warp;
    if (candidate_index >= *candidate_count) {
        return;
    }

    auto const reference_id = candidate_ids[candidate_index];
    pairwise_counts local{};
    auto const row_offset = static_cast<size_t>(reference_id) * BucketCount;
    for (auto bucket = static_cast<size_t>(lane); bucket < BucketCount; bucket += warp_width) {
        classify(local, query[bucket], reference_score(rows[row_offset + bucket]));
    }
    auto const total = warp_reduce(storage[warp]).Sum(local);
    if (lane == 0U) {
        results[candidate_index].reference_id = reference_id;
        results[candidate_index].summary.counts = total;
        results[candidate_index].summary.cardinality = 0.0;
    }
}

/**
 * @brief Exactly compares a query tile with a reference database.
 *
 * In all-to-all mode the query rows come from the database backing and only pairs with
 * `query_id < reference_id` are emitted.
 */
template <
    size_t BucketCount,
    bool AllToAll,
    typename QueryRow,
    typename ReferenceRow,
    typename SearchResult>
__global__ __launch_bounds__(block_size) void batch_exhaustive_search_kernel(
    QueryRow const* queries,
    size_t query_row_offset,
    uint32_t query_count,
    uint32_t query_id_offset,
    ReferenceRow const* rows,
    uint32_t reference_count,
    SearchResult* results,
    uint32_t* result_match_counts
) {
    constexpr uint32_t warp_width = 32;
    constexpr uint32_t warps_per_block = block_size / warp_width;
    using count_reduce = cub::WarpReduce<pairwise_counts>;
    using match_reduce = cub::WarpReduce<uint32_t>;
    __shared__ typename count_reduce::TempStorage count_storage[warps_per_block];
    __shared__ typename match_reduce::TempStorage match_storage[warps_per_block];

    auto const warp = static_cast<uint32_t>(threadIdx.x) / warp_width;
    auto const lane = static_cast<uint32_t>(threadIdx.x) % warp_width;
    for (auto query = static_cast<uint64_t>(blockIdx.y); query < query_count;
         query += static_cast<uint64_t>(gridDim.y)) {
        auto const query_index = static_cast<uint32_t>(query);
        auto const query_id = query_id_offset + query_index;
        auto const first_reference = AllToAll ? static_cast<uint64_t>(query_id) + 1U : uint64_t{0};
        auto const earlier_queries = query_index == 0U ? 0U : query_index - 1U;
        auto const preceding_pairs =
            AllToAll ? static_cast<size_t>(query_index) * (reference_count - query_id_offset - 1U) -
                           static_cast<size_t>(query_index) * earlier_queries / 2U
                     : static_cast<size_t>(query_index) * reference_count;
        auto const query_offset =
            (query_row_offset + static_cast<size_t>(query_index)) * BucketCount;

        auto reference =
            first_reference + static_cast<uint64_t>(blockIdx.x) * warps_per_block + warp;
        auto const reference_stride = static_cast<uint64_t>(gridDim.x) * warps_per_block;
        for (; reference < reference_count; reference += reference_stride) {
            auto const reference_id = static_cast<uint32_t>(reference);
            pairwise_counts local{};
            uint32_t local_matches = 0;
            auto const reference_offset = static_cast<size_t>(reference_id) * BucketCount;
            for (auto bucket = static_cast<size_t>(lane); bucket < BucketCount;
                 bucket += warp_width) {
                auto const query_score = reference_score(queries[query_offset + bucket]);
                auto const stored_score = reference_score(rows[reference_offset + bucket]);
                classify(local, query_score, stored_score);
                local_matches += query_score != 0U && query_score == stored_score;
            }
            auto const total = count_reduce(count_storage[warp]).Sum(local);
            auto const matches = match_reduce(match_storage[warp]).Sum(local_matches);
            if (lane == 0U) {
                auto const result_index = preceding_pairs + reference_id - first_reference;
                results[result_index].query_id = query_id;
                results[result_index].reference_id = reference_id;
                results[result_index].summary.counts = total;
                results[result_index].summary.cardinality = 0.0;
                if (result_match_counts != nullptr) {
                    result_match_counts[result_index] = matches;
                }
            }
            __syncwarp();
        }
    }
}

/// @brief Counts dense index matches for every query/reference pair in one tile.
template <size_t BucketCount, typename QueryRow>
__global__ void count_batch_index_matches_kernel(
    QueryRow const* queries,
    size_t query_row_offset,
    uint32_t query_count,
    uint32_t const* offsets,
    uint32_t const* postings,
    uint32_t reference_count,
    uint32_t indexed_bucket_count,
    uint16_t key_mask,
    uint32_t* match_counts
) {
    auto query_bucket = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    auto const stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    auto const query_buckets = static_cast<size_t>(query_count) * indexed_bucket_count;
    auto const key_count = static_cast<uint32_t>(key_mask) + 1U;
    for (; query_bucket < query_buckets; query_bucket += stride) {
        auto const query_index = query_bucket / indexed_bucket_count;
        auto const bucket = query_bucket % indexed_bucket_count;
        auto const score =
            reference_score(queries[(query_row_offset + query_index) * BucketCount + bucket]);
        if (score == 0U) {
            continue;
        }
        auto const key = static_cast<uint32_t>(score & key_mask);
        auto const cell = static_cast<uint64_t>(bucket) * key_count + key;
        auto const begin = offsets[cell];
        auto const end = offsets[cell + 1U];
        for (auto posting = begin; posting < end; ++posting) {
            atomicAdd(&match_counts[query_index * reference_count + postings[posting]], 1U);
        }
    }
}

/// @brief Stable query-major selection predicate for external or all-to-all batch search.
struct batch_minimum_match_predicate {
    uint32_t const* match_counts;
    uint32_t minimum_matches;
    uint32_t reference_count;
    uint32_t query_id_offset;
    bool all_to_all;

    [[nodiscard]] __host__ __device__ bool operator()(uint32_t pair_id) const noexcept {
        auto const query_index = pair_id / reference_count;
        auto const reference_id = pair_id % reference_count;
        return (!all_to_all || query_id_offset + query_index < reference_id) &&
               match_counts[pair_id] >= minimum_matches;
    }
};

/// @brief Exactly refines a bounded stable query-major candidate list.
template <size_t BucketCount, typename QueryRow, typename ReferenceRow, typename SearchResult>
__global__ __launch_bounds__(block_size) void refine_batch_index_candidates_kernel(
    QueryRow const* queries,
    size_t query_row_offset,
    uint32_t query_id_offset,
    ReferenceRow const* rows,
    uint32_t reference_count,
    uint32_t const* match_counts,
    uint32_t const* candidate_ids,
    uint32_t const* candidate_count,
    uint32_t result_capacity,
    SearchResult* results,
    uint32_t* result_match_counts
) {
    constexpr uint32_t warp_width = 32;
    constexpr uint32_t warps_per_block = block_size / warp_width;
    using warp_reduce = cub::WarpReduce<pairwise_counts>;
    __shared__ typename warp_reduce::TempStorage storage[warps_per_block];

    if (*candidate_count > result_capacity) {
        return;
    }
    auto const warp = static_cast<uint32_t>(threadIdx.x) / warp_width;
    auto const lane = static_cast<uint32_t>(threadIdx.x) % warp_width;
    auto const candidate_index = static_cast<uint32_t>(blockIdx.x) * warps_per_block + warp;
    if (candidate_index >= *candidate_count) {
        return;
    }

    auto const pair_id = candidate_ids[candidate_index];
    auto const query_index = pair_id / reference_count;
    auto const reference_id = pair_id % reference_count;
    pairwise_counts local{};
    auto const query_offset = (query_row_offset + static_cast<size_t>(query_index)) * BucketCount;
    auto const reference_offset = static_cast<size_t>(reference_id) * BucketCount;
    for (auto bucket = static_cast<size_t>(lane); bucket < BucketCount; bucket += warp_width) {
        classify(
            local,
            reference_score(queries[query_offset + bucket]),
            reference_score(rows[reference_offset + bucket])
        );
    }
    auto const total = warp_reduce(storage[warp]).Sum(local);
    if (lane == 0U) {
        results[candidate_index].query_id = query_id_offset + query_index;
        results[candidate_index].reference_id = reference_id;
        results[candidate_index].summary.counts = total;
        results[candidate_index].summary.cardinality = 0.0;
        if (result_match_counts != nullptr) {
            result_match_counts[candidate_index] = match_counts[pair_id];
        }
    }
}

/**
 * @brief Computes the cardinality of a single constructed sketch.
 *
 * @p empty_out receives the empty-register count; @p estimate_out receives the estimate.
 */
template <size_t BucketCount>
__global__ void cardinality_kernel(
    uint32_t const* const registers,
    uint64_t* const empty_out,
    double* const estimate_out
) {
    constexpr uint32_t cardinality_block_size = 64;
    using warp_reduce = cub::WarpReduce<cardinality_payload>;
    __shared__ typename warp_reduce::TempStorage warp_storage[2];
    __shared__ cardinality_payload warp_totals[2];

    cardinality_payload local{};
    for (auto bucket = static_cast<size_t>(threadIdx.x); bucket < BucketCount;
         bucket += cardinality_block_size) {
        auto const stored = winner(registers[bucket]);
        if (stored == 0U) {
            ++local.empty;
        } else {
            local.restored_sum += restore(stored);
        }
    }
    auto const warp = threadIdx.x >> 5U;
    local = warp_reduce(warp_storage[warp]).Reduce(local, combine_cardinality);
    if ((threadIdx.x & 31U) == 0U) {
        warp_totals[threadIdx.x >> 5U] = local;
    }
    __syncthreads();
    auto const total = warp_totals[0] + warp_totals[1];
    if (threadIdx.x == 0) {
        *empty_out = total.empty;
        *estimate_out = cardinality(
            static_cast<double>(BucketCount), static_cast<double>(total.empty), total.restored_sum
        );
    }
}

template <size_t BucketCount>
__global__ void hybrid_cardinality_kernel(
    uint32_t const* const registers,
    hybrid_cardinality_estimates* const estimates
) {
    __shared__ uint32_t bins[nlz_bins];
    __shared__ double restored[block_size];
    for (auto bin = static_cast<uint32_t>(threadIdx.x); bin < nlz_bins; bin += blockDim.x) {
        bins[bin] = 0U;
    }
    double local_restored = 0.0;
    __syncthreads();

    for (auto bucket = static_cast<size_t>(threadIdx.x); bucket < BucketCount;
         bucket += blockDim.x) {
        auto const stored = winner(registers[bucket]);
        if (stored == 0U) {
            atomicAdd(bins, 1U);
        } else {
            atomicAdd(bins + static_cast<uint32_t>(stored >> mantissa_bits) + 1U, 1U);
            local_restored += restore(stored);
        }
    }
    restored[threadIdx.x] = local_restored;
    __syncthreads();

    for (auto stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            restored[threadIdx.x] += restored[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        *estimates = hybrid_estimates(bins, static_cast<double>(BucketCount), restored[0]);
    }
}

template <size_t BucketCount, hybrid_variant Variant>
__global__ void
hybrid_cardinality_variant_kernel(uint32_t const* const registers, double* const estimate) {
    __shared__ uint32_t bins[nlz_bins];
    __shared__ double restored[block_size];
    for (auto bin = static_cast<uint32_t>(threadIdx.x); bin < nlz_bins; bin += blockDim.x) {
        bins[bin] = 0U;
    }
    double local_restored = 0.0;
    __syncthreads();

    for (auto bucket = static_cast<size_t>(threadIdx.x); bucket < BucketCount;
         bucket += blockDim.x) {
        auto const stored = winner(registers[bucket]);
        if (stored == 0U) {
            atomicAdd(bins, 1U);
        } else {
            atomicAdd(bins + static_cast<uint32_t>(stored >> mantissa_bits) + 1U, 1U);
            local_restored += restore(stored);
        }
    }
    restored[threadIdx.x] = local_restored;
    __syncthreads();

    for (auto stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            restored[threadIdx.x] += restored[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        auto const estimates =
            hybrid_estimates(bins, static_cast<double>(BucketCount), restored[0]);
        if constexpr (Variant == hybrid_variant::bbtools) {
            *estimate = estimates.bbtools;
        } else {
            *estimate = estimates.paper;
        }
    }
}

/// @brief Extracts per-register winner counts and the sketch-level saturation flag.
template <size_t BucketCount>
__global__ void winner_counts_kernel(
    uint32_t const* const registers,
    uint32_t const* const saturation_in,
    uint16_t* const counts_out,
    uint32_t* const saturation_out
) {
    for (auto bucket = static_cast<size_t>(threadIdx.x); bucket < BucketCount;
         bucket += block_size) {
        counts_out[bucket] = count(registers[bucket]);
    }
    if (threadIdx.x == 0) {
        *saturation_out = *saturation_in;
    }
}

}  // namespace cuddl::detail
