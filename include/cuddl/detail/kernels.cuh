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
#include <cuddl/helpers.cuh>
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

/// @brief Largest sketch (in registers) whose per-CTA staging fits in default static shared memory.
constexpr size_t shared_construction_max_buckets = (size_t{1} << 13);

/// @brief Threads per CTA for the CTA-local construction kernel.
constexpr uint32_t shared_construction_block_size = 768;

/**
 * @brief Constructs a sketch through a CTA-local shared-memory winner array with a deferred
 * tie-count fix-up.
 *
 * Each shared word stores `(winner << 16) | count`, so the winner dominates the packed value and
 * a plain fire-and-forget 32-bit atomic max applies the DDL winner rule. Every item keeps its
 * atomic's return value for one slot; the deferred settle then compares the returned winner
 * against the item's score and, on the rare install (`old < score`) or tie (`old == score`),
 * increments the count through a CAS loop that re-validates the winner, so a stale increment can
 * never land on a replaced winner. A saturated per-block counter records the sketch-level
 * saturation flag, matching the sequential @ref update semantics. The per-item path is hash,
 * score, one atomic max, and one deferred compare: no dependent load, no branch on the common
 * path, no pair cache, and no separate count phase. The merge then applies the DDL winner/count
 * rule to the global registers, and the result is bit-identical to @ref add_kernel.
 *
 * The host launches two CTAs per SM and the kernel walks the input with a runtime grid-stride
 * loop over 256-bit chunks, so the grid is a single balanced wave for every input size and the
 * per-CTA merge traffic stays minimal.
 */
template <size_t BucketCount>
__global__ __launch_bounds__(shared_construction_block_size) void add_shared_kernel(
    uint64_t const* input,
    size_t input_size,
    uint32_t* registers,
    uint32_t& saturation,
    bool vector_input
) {
    static_assert(BucketCount <= shared_construction_max_buckets);
    __shared__ uint32_t state[BucketCount];
    for (auto i = threadIdx.x; i < BucketCount; i += blockDim.x) {
        state[i] = 0U;
    }
    __syncthreads();

    uint32_t prev_bucket = 0U;
    uint32_t prev_score = 0xffffU;  // primes the first settle off without a `have` flag
    uint32_t prev_old = 0U;

    auto const settle = [&] {
        // A strict install already carries count 1 in the atomic max's replacement value,
        // so only ties (old == score: count + 1) need the deferred increment. Re-validate
        // the winner under CAS so an increment for a replaced winner is dropped, and
        // saturate the per-block count exactly like the sequential update rule.
        if ((prev_old >> 16U) == prev_score) {
            auto expected = state[prev_bucket];
            while ((expected >> 16U) == prev_score) {
                if ((expected & 0xffffU) == max_winner_count) {
                    atomicExch(&saturation, 1U);
                    break;
                }
                auto const actual = atomicCAS(&state[prev_bucket], expected, expected + 1U);
                if (actual == expected) {
                    break;
                }
                expected = actual;
            }
        }
    };
    auto const process = [&](uint64_t value) {
        auto const hash = hash_kmer(value);
        auto const incoming = static_cast<uint32_t>(score(hash));
        auto const bucket = static_cast<uint32_t>(bucket_of<BucketCount>(hash));
        auto const old = atomicMax(&state[bucket], (incoming << 16U) | 1U);
        settle();
        prev_bucket = bucket;
        prev_score = incoming;
        prev_old = old;
    };

    auto const stride = static_cast<size_t>(gridDim.x) * blockDim.x * 4U;
    auto const index = (static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 4U;
    for (auto offset = index; offset < input_size; offset += stride) {
        if (vector_input && offset + 4U <= input_size) {
            uint64_t values[4];
            load_256_global_nc(input + offset, values);
            _Pragma("unroll")
            for (uint32_t item = 0; item < 4U; ++item) {
                process(values[item]);
            }
        } else {
            // Scalar path: unaligned input keeps striding exactly like the vector path; only
            // the aligned input's final partial chunk (fewer than four items) ends the loop.
            _Pragma("unroll")
            for (uint32_t item = 0; item < 4U; ++item) {
                if (offset + item < input_size) {
                    process(input[offset + item]);
                }
            }
            if (vector_input) {
                break;
            }
        }
    }
    settle();
    __syncthreads();

    // Each CTA starts its merge at a rotated bucket offset (odd multiplier, coprime with
    // the power-of-two bucket count), so the CTAs' atomic merges interleave across different
    // addresses instead of all colliding on the same bucket at once. The per-address CAS
    // serialization then sees a smooth stream of arrivals rather than a synchronized burst.
    auto const bucket_offset = (static_cast<size_t>(blockIdx.x) * 139U) & (BucketCount - 1U);
    for (auto j = threadIdx.x; j < BucketCount; j += blockDim.x) {
        auto const i = (static_cast<size_t>(j) + bucket_offset) & (BucketCount - 1U);
        auto const stored = state[i];
        if ((stored >> 16U) != 0U) {
            merge_register(
                &registers[i],
                pack(static_cast<uint16_t>(stored >> 16U), static_cast<uint16_t>(stored & 0xffffU)),
                saturation
            );
        }
    }
}

/**
 * @brief Computes pairwise counts and optionally cardinality for two constructed sketches.
 */
template <size_t BucketCount, bool IncludeCardinality>
__global__ void
summary_kernel(uint32_t const* left, uint32_t const* right, pairwise_summary& output) {
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
                local.restored_sum += restore_midpoint(winner(left_reg));
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

/**
 * @brief Compares corresponding rows from two contiguous packed-register batches.
 *
 * One warp owns one pair. Grid-stride traversal keeps the launch bounded for very large batches.
 */
template <size_t BucketCount>
__global__ __launch_bounds__(block_size) void batch_summary_kernel(
    uint32_t const* left_rows,
    uint32_t const* right_rows,
    size_t pair_count,
    pairwise_summary* outputs
) {
    constexpr uint32_t warp_width = 32U;
    constexpr uint32_t warps_per_block = block_size / warp_width;
    using warp_reduce = cub::WarpReduce<pairwise_counts>;
    __shared__ typename warp_reduce::TempStorage storage[warps_per_block];

    auto const warp = static_cast<uint32_t>(threadIdx.x) / warp_width;
    auto const lane = static_cast<uint32_t>(threadIdx.x) % warp_width;
    auto pair = static_cast<size_t>(blockIdx.x) * warps_per_block + warp;
    auto const pair_stride = static_cast<size_t>(gridDim.x) * warps_per_block;
    for (; pair < pair_count; pair += pair_stride) {
        auto const row_offset = pair * BucketCount;
        pairwise_counts local{};
        for (auto bucket = static_cast<size_t>(lane); bucket < BucketCount; bucket += warp_width) {
            classify(local, left_rows[row_offset + bucket], right_rows[row_offset + bucket]);
        }
        auto const total = warp_reduce(storage[warp]).Sum(local);
        if (lane == 0U) {
            outputs[pair].counts = total;
            outputs[pair].cardinality = 0.0;
        }
        __syncwarp();
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

/// @brief Transposes the indexed-bucket slice of a row-major score matrix into bucket-major
/// order.
///
/// The per-bucket build passes read contiguous references from the transposed layout instead of
/// striding across rows, and each bucket's dense key range stays L2-resident while its atomics
/// accumulate. Tiled through shared memory so both the reads and the writes are coalesced.
template <typename Row>
__global__ void transpose_indexed_scores_kernel(
    Row const* rows,
    uint32_t reference_count,
    uint32_t indexed_bucket_count,
    uint32_t full_bucket_count,
    Row* transposed
) {
    constexpr uint32_t tile = 32U;
    __shared__ Row staging[tile][tile + 1U];
    auto const bucket = static_cast<uint32_t>(blockIdx.x) * tile + threadIdx.x;
    auto const reference = static_cast<uint32_t>(blockIdx.y) * tile + threadIdx.y;
    if (bucket < indexed_bucket_count && reference < reference_count) {
        staging[threadIdx.y][threadIdx.x] =
            rows[static_cast<size_t>(reference) * full_bucket_count + bucket];
    }
    __syncthreads();
    // Output (bucket, reference) = input (reference, bucket). Threads with consecutive
    // threadIdx.x must write consecutive references of one bucket, so the output bucket index
    // comes from threadIdx.y and the output reference from threadIdx.x; the shared-memory read
    // is the transposed (column) access instead.
    auto const out_bucket = static_cast<uint32_t>(blockIdx.x) * tile + threadIdx.y;
    auto const out_reference = static_cast<uint32_t>(blockIdx.y) * tile + threadIdx.x;
    if (out_bucket < indexed_bucket_count && out_reference < reference_count) {
        transposed[static_cast<size_t>(out_bucket) * reference_count + out_reference] =
            staging[threadIdx.x][threadIdx.y];
    }
}

/// @brief Counts every non-empty indexed row entry in its dense index cell, over the
/// transposed scores.
///
/// Each wave-blocked block owns one bucket and counts into a quarter-key shared-memory table
/// (shared-memory atomics only), then flushes the table with plain coalesced stores. Every
/// bucket slice is written by exactly one block, so the cell array needs no zeroing pass and
/// the count contributes no global atomics at all.
template <typename Row>
__global__ void count_index_cells_bucket_kernel(
    Row const* transposed,
    uint32_t indexed_bucket_count,
    uint32_t reference_count,
    uint16_t key_mask,
    uint32_t* cell_counts
) {
    extern __shared__ uint32_t key_counts[];
    auto const key_count = static_cast<uint32_t>(key_mask) + 1U;
    auto const quarter_keys = key_count / 4U;
    for (auto i = threadIdx.x; i < quarter_keys; i += blockDim.x) {
        key_counts[i] = 0U;
    }
    __syncthreads();

    for (auto bucket = blockIdx.x; bucket < indexed_bucket_count; bucket += gridDim.x) {
        auto const* bucket_scores = transposed + static_cast<size_t>(bucket) * reference_count;
        auto* bucket_counts = cell_counts + static_cast<size_t>(bucket) * key_count;
        for (uint32_t quarter = 0U; quarter < 4U; ++quarter) {
            auto const key_base = quarter * quarter_keys;
            for (uint32_t reference = threadIdx.x; reference < reference_count;
                 reference += blockDim.x) {
                auto const score = reference_score(__ldcs(&bucket_scores[reference]));
                if (score == 0U) {
                    continue;
                }
                auto const key = static_cast<uint32_t>(score & key_mask);
                if (key < key_base || key >= key_base + quarter_keys) {
                    continue;
                }
                atomicAdd(&key_counts[key - key_base], 1U);
            }
            __syncthreads();
            for (auto i = threadIdx.x; i < quarter_keys; i += blockDim.x) {
                bucket_counts[key_base + i] = key_counts[i];
                key_counts[i] = 0U;
            }
            __syncthreads();
        }
    }
}

/// @brief Scatters every non-empty indexed row entry into its dense CSR posting range without
/// any global atomics.
///
/// Each wave-blocked block owns one bucket and keeps the per-key running rank for a quarter of
/// the key space in dynamic shared memory (64 KiB for 16-bit keys, under this device's 99 KiB
/// opt-in shared-memory cap). The posting position is computed directly as `offsets[cell] +
/// local_rank`: the block's shared-memory rank is the only per-cell cursor state, so no global
/// cursor atomics and no per-cell cursor scratch are needed. Posting order within a cell is
/// unspecified (the interface never depended on it).
template <typename Row>
__global__ void scatter_index_postings_bucket_kernel(
    Row const* transposed,
    uint32_t indexed_bucket_count,
    uint32_t reference_count,
    uint16_t key_mask,
    uint32_t const* offsets,
    uint32_t* postings
) {
    extern __shared__ uint32_t key_ranks[];
    auto const key_count = static_cast<uint32_t>(key_mask) + 1U;
    auto const quarter_keys = key_count / 4U;
    for (auto i = threadIdx.x; i < quarter_keys; i += blockDim.x) {
        key_ranks[i] = 0U;
    }
    __syncthreads();

    for (auto bucket = blockIdx.x; bucket < indexed_bucket_count; bucket += gridDim.x) {
        auto const* bucket_scores = transposed + static_cast<size_t>(bucket) * reference_count;
        auto const* bucket_offsets = offsets + static_cast<size_t>(bucket) * key_count;
        for (uint32_t quarter = 0U; quarter < 4U; ++quarter) {
            auto const key_base = quarter * quarter_keys;
            for (uint32_t reference = threadIdx.x; reference < reference_count;
                 reference += blockDim.x) {
                auto const score = reference_score(__ldcs(&bucket_scores[reference]));
                if (score == 0U) {
                    continue;
                }
                auto const key = static_cast<uint32_t>(score & key_mask);
                if (key < key_base || key >= key_base + quarter_keys) {
                    continue;
                }
                auto const rank = atomicAdd(&key_ranks[key - key_base], 1U);
                __stcs(&postings[bucket_offsets[key] + rank], reference);
            }
            __syncthreads();
            for (auto i = threadIdx.x; i < quarter_keys; i += blockDim.x) {
                key_ranks[i] = 0U;
            }
            __syncthreads();
        }
    }
}

/// @brief Counts the query's non-empty posting matches for every reference.
///
/// One warp owns each bucket cell and walks its posting list with a lane stride,
/// so hot keys with long lists (the dominant cost on skewed rows) are consumed 32
/// postings at a time instead of serially by a single thread.
template <size_t BucketCount>
__global__ void count_index_matches_kernel(
    uint16_t const* query,
    uint32_t const* offsets,
    uint32_t const* postings,
    uint32_t indexed_bucket_count,
    uint16_t key_mask,
    uint32_t* match_counts
) {
    constexpr uint32_t warp_width = 32;
    constexpr uint32_t warps_per_block = block_size / warp_width;
    auto const lane = static_cast<uint32_t>(threadIdx.x) % warp_width;
    auto const warp = static_cast<uint32_t>(threadIdx.x) / warp_width;
    auto const bucket = static_cast<size_t>(blockIdx.x) * warps_per_block + warp;
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
    for (auto posting = begin + lane; posting < end; posting += warp_width) {
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
    if constexpr (!AllToAll) {
        // Reference-major traversal: each warp owns one reference and compares it against
        // every query in the tile, so each reference row is loaded once (and reused from L1
        // for the remaining queries) instead of once per (query, reference) pair. The query
        // rows themselves fit in L2 and are shared by every warp; staging them in shared
        // memory was profiled and rejected because the larger shared footprint caps the
        // occupancy at one block per SM. Results keep their dense query-major positions, so
        // the output layout is unchanged.
        auto reference = static_cast<uint64_t>(blockIdx.x) * warps_per_block + warp;
        auto const reference_stride = static_cast<uint64_t>(gridDim.x) * warps_per_block;
        for (; reference < reference_count; reference += reference_stride) {
            auto const reference_id = static_cast<uint32_t>(reference);
            auto const reference_offset = static_cast<size_t>(reference_id) * BucketCount;
            for (uint32_t query_index = 0U; query_index < query_count; ++query_index) {
                auto const query_id = query_id_offset + query_index;
                auto const query_offset =
                    (query_row_offset + static_cast<size_t>(query_index)) * BucketCount;
                pairwise_counts local{};
                uint32_t local_matches = 0U;
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
                    auto const result_index =
                        static_cast<size_t>(query_index) * reference_count + reference_id;
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
    } else {
        for (auto query = static_cast<uint64_t>(blockIdx.y); query < query_count;
             query += static_cast<uint64_t>(gridDim.y)) {
            auto const query_index = static_cast<uint32_t>(query);
            auto const query_id = query_id_offset + query_index;
            auto const first_reference =
                AllToAll ? static_cast<uint64_t>(query_id) + 1U : uint64_t{0};
            auto const earlier_queries = query_index == 0U ? 0U : query_index - 1U;
            auto const preceding_pairs =
                AllToAll
                    ? static_cast<size_t>(query_index) * (reference_count - query_id_offset - 1U) -
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
}

/// @brief Counts dense index matches for every query/reference pair in one tile.
///
/// One warp owns each (query, bucket) cell. The cell's posting list is walked with a lane
/// stride, so hot keys with long lists (the dominant cost on skewed rows) are consumed 32
/// postings at a time instead of serially by a single thread, and the two per-cell offset
/// loads collapse into warp-uniform broadcasts.
template <size_t BucketCount, typename QueryRow>
__global__ __launch_bounds__(block_size) void count_batch_index_matches_kernel(
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
    constexpr uint32_t warp_width = 32;
    constexpr uint32_t warps_per_block = block_size / warp_width;
    auto const lane = static_cast<uint32_t>(threadIdx.x) % warp_width;
    auto const warp = static_cast<uint32_t>(threadIdx.x) / warp_width;
    auto const cell_index = static_cast<uint64_t>(blockIdx.x) * warps_per_block + warp;
    auto const cell_stride = static_cast<uint64_t>(gridDim.x) * warps_per_block;
    auto const total_cells = static_cast<uint64_t>(query_count) * indexed_bucket_count;
    auto const key_count = static_cast<uint32_t>(key_mask) + 1U;
    for (auto cell = cell_index; cell < total_cells; cell += cell_stride) {
        auto const query_index = static_cast<uint32_t>(cell / indexed_bucket_count);
        auto const bucket = static_cast<uint32_t>(cell % indexed_bucket_count);
        auto const score =
            reference_score(queries[(query_row_offset + query_index) * BucketCount + bucket]);
        if (score == 0U) {
            continue;
        }
        auto const key = static_cast<uint32_t>(score & key_mask);
        auto const index_cell = static_cast<uint64_t>(bucket) * key_count + key;
        auto const begin = offsets[index_cell];
        auto const end = offsets[index_cell + 1U];
        auto* const counts = match_counts + static_cast<size_t>(query_index) * reference_count;
        for (auto posting = begin + lane; posting < end; posting += warp_width) {
            atomicAdd(&counts[postings[posting]], 1U);
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

/// @brief Histograms selected batch candidates per reference id.
__global__ void histogram_batch_candidates_kernel(
    uint32_t const* candidate_ids,
    uint32_t const* candidate_count,
    uint32_t reference_count,
    uint32_t* reference_histogram
) {
    auto const count = *candidate_count;
    auto index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    auto const stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    for (; index < count; index += stride) {
        auto const reference_id = candidate_ids[index] % reference_count;
        atomicAdd(&reference_histogram[reference_id], 1U);
    }
}

/// @brief Scatters batch candidate indices into reference-major runs.
///
/// @p reference_cursor holds the exclusive per-reference scan of the histogram; each atomic
/// bump assigns the next slot of the owning reference's contiguous run, so the refinement can
/// walk one run per reference row load.
__global__ void scatter_batch_candidates_kernel(
    uint32_t const* candidate_ids,
    uint32_t const* candidate_count,
    uint32_t reference_count,
    uint32_t* reference_cursor,
    uint32_t* reference_candidates
) {
    auto const count = *candidate_count;
    auto index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    auto const stride = static_cast<uint64_t>(blockDim.x) * gridDim.x;
    for (; index < count; index += stride) {
        auto const reference_id = candidate_ids[index] % reference_count;
        auto const slot = atomicAdd(&reference_cursor[reference_id], 1U);
        reference_candidates[slot] = static_cast<uint32_t>(index);
    }
}

/// @brief Classifies one contiguous per-lane chunk of both rows.
///
/// With @p Use256 the chunk covers 16 buckets for 16-bit scores (one 256-bit load per row, via
/// `ld.global.nc.v8.u32` on sm_100+) and 8 buckets for packed 32-bit registers; that path
/// requires 32-byte-aligned rows. Without it the chunk covers 8 or 4 buckets with per-element
/// scalar loads, which are safe for rows aligned only to their score type (2 or 4 bytes).
template <bool Use256, typename QueryScore, typename ReferenceScore>
__device__ void classify_wide_chunk(
    pairwise_counts& target,
    QueryScore const* query,
    ReferenceScore const* reference
) noexcept {
    constexpr uint32_t chunk_buckets =
        Use256 ? ((sizeof(QueryScore) == 2U || sizeof(ReferenceScore) == 2U) ? 16U : 8U)
               : ((sizeof(QueryScore) == 2U || sizeof(ReferenceScore) == 2U) ? 8U : 4U);
    if constexpr (Use256) {
        if constexpr (sizeof(QueryScore) == 2U && sizeof(ReferenceScore) == 2U) {
            uint32_t q[8];
            uint32_t r[8];
            load_256_global_nc(reinterpret_cast<uint32_t const*>(query), q);
            load_256_global_nc(reinterpret_cast<uint32_t const*>(reference), r);
            auto const* qs = reinterpret_cast<uint16_t const*>(q);
            auto const* rs = reinterpret_cast<uint16_t const*>(r);
            _Pragma("unroll")
            for (uint32_t i = 0; i < chunk_buckets; ++i) {
                classify(target, qs[i], rs[i]);
            }
        } else if constexpr (sizeof(QueryScore) == 2U) {
            uint32_t q[8];
            uint32_t r0[8];
            uint32_t r1[8];
            load_256_global_nc(reinterpret_cast<uint32_t const*>(query), q);
            load_256_global_nc(reinterpret_cast<uint32_t const*>(reference), r0);
            load_256_global_nc(reinterpret_cast<uint32_t const*>(reference) + 8U, r1);
            auto const* qs = reinterpret_cast<uint16_t const*>(q);
            _Pragma("unroll")
            for (uint32_t i = 0; i < 8U; ++i) {
                classify(target, qs[i], reference_score(r0[i]));
            }
            _Pragma("unroll")
            for (uint32_t i = 0; i < 8U; ++i) {
                classify(target, qs[8U + i], reference_score(r1[i]));
            }
        } else if constexpr (sizeof(ReferenceScore) == 2U) {
            uint32_t q0[8];
            uint32_t q1[8];
            uint32_t r[8];
            load_256_global_nc(reinterpret_cast<uint32_t const*>(query), q0);
            load_256_global_nc(reinterpret_cast<uint32_t const*>(query) + 8U, q1);
            load_256_global_nc(reinterpret_cast<uint32_t const*>(reference), r);
            auto const* rs = reinterpret_cast<uint16_t const*>(r);
            _Pragma("unroll")
            for (uint32_t i = 0; i < 8U; ++i) {
                classify(target, reference_score(q0[i]), rs[i]);
            }
            _Pragma("unroll")
            for (uint32_t i = 0; i < 8U; ++i) {
                classify(target, reference_score(q1[i]), rs[8U + i]);
            }
        } else {
            uint32_t q[8];
            uint32_t r[8];
            load_256_global_nc(reinterpret_cast<uint32_t const*>(query), q);
            load_256_global_nc(reinterpret_cast<uint32_t const*>(reference), r);
            _Pragma("unroll")
            for (uint32_t i = 0; i < chunk_buckets; ++i) {
                classify(target, q[i], r[i]);
            }
        }
    } else {
        // Scalar fallback: callers only guarantee the score type's alignment (2 or 4 bytes),
        // so no vector load is safe here.
        _Pragma("unroll")
        for (uint32_t i = 0; i < chunk_buckets; ++i) {
            classify(target, reference_score(query[i]), reference_score(reference[i]));
        }
    }
}

/// @brief Warp-reduces one refined candidate and writes its stable result slot.
template <typename SearchResult>
__device__ void refine_write_result(
    uint32_t index,
    uint32_t query_index,
    uint32_t reference_id,
    uint32_t query_id_offset,
    uint32_t const* match_counts,
    uint32_t pair_id,
    pairwise_counts local,
    cub::WarpReduce<pairwise_counts>::TempStorage& storage,
    uint32_t lane,
    SearchResult* results,
    uint32_t* result_match_counts
) {
    auto const total = cub::WarpReduce<pairwise_counts>(storage).Sum(local);
    if (lane == 0U) {
        results[index].query_id = query_id_offset + query_index;
        results[index].reference_id = reference_id;
        results[index].summary.counts = total;
        results[index].summary.cardinality = 0.0;
        if (result_match_counts != nullptr) {
            result_match_counts[index] = match_counts[pair_id];
        }
    }
}

/// @brief Exactly refines a bounded stable candidate list, one reference row load per reference.
///
/// The candidates arrive reference-major (one contiguous run per reference), so a warp loads
/// its reference row once and reuses it from the cache for every query in the run instead of
/// re-reading the full row per (query, reference) pair. Results keep their stable query-major
/// positions from the selection.
template <size_t BucketCount, typename QueryRow, typename ReferenceRow, typename SearchResult>
__global__ __launch_bounds__(block_size) void refine_batch_index_candidates_kernel(
    QueryRow const* queries,
    size_t query_row_offset,
    uint32_t query_id_offset,
    ReferenceRow const* rows,
    uint32_t reference_count,
    uint32_t const* match_counts,
    uint32_t const* candidate_ids,
    uint32_t const* reference_candidates,
    uint32_t const* reference_histogram,
    uint32_t const* reference_cursor,
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
    auto const reference_id = static_cast<uint32_t>(blockIdx.x) * warps_per_block + warp;
    if (reference_id >= reference_count) {
        return;
    }

    auto const run_end = reference_cursor[reference_id];
    auto const run_begin = run_end - reference_histogram[reference_id];
    if (run_begin == run_end) {
        return;
    }

    // The 256-bit load path needs 32-byte-aligned rows; callers may hand over spans whose base
    // breaks that, in which case the same chunking falls back to 128-bit loads.
    auto const wide256 = (reinterpret_cast<uintptr_t>(queries) & 31U) == 0U &&
                         (reinterpret_cast<uintptr_t>(rows) & 31U) == 0U;
    auto const reference_offset = static_cast<size_t>(reference_id) * BucketCount;
    if (wide256) {
        constexpr uint32_t chunk_buckets =
            (sizeof(QueryRow) == 2U || sizeof(ReferenceRow) == 2U) ? 16U : 8U;
        static_assert(BucketCount % chunk_buckets == 0U);
        constexpr uint32_t chunks_per_row = BucketCount / chunk_buckets;
        for (auto slot = run_begin; slot < run_end; ++slot) {
            auto const index = reference_candidates[slot];
            auto const pair_id = candidate_ids[index];
            auto const query_index = pair_id / reference_count;
            pairwise_counts local{};
            auto const query_offset =
                (query_row_offset + static_cast<size_t>(query_index)) * BucketCount;
            for (auto chunk = static_cast<uint32_t>(lane); chunk < chunks_per_row;
                 chunk += warp_width) {
                classify_wide_chunk<true>(
                    local,
                    queries + query_offset + static_cast<size_t>(chunk) * chunk_buckets,
                    rows + reference_offset + static_cast<size_t>(chunk) * chunk_buckets
                );
            }
            refine_write_result(
                index,
                query_index,
                reference_id,
                query_id_offset,
                match_counts,
                pair_id,
                local,
                storage[warp],
                lane,
                results,
                result_match_counts
            );
        }
    } else {
        constexpr uint32_t chunk_buckets =
            (sizeof(QueryRow) == 2U || sizeof(ReferenceRow) == 2U) ? 8U : 4U;
        static_assert(BucketCount % chunk_buckets == 0U);
        constexpr uint32_t chunks_per_row = BucketCount / chunk_buckets;
        for (auto slot = run_begin; slot < run_end; ++slot) {
            auto const index = reference_candidates[slot];
            auto const pair_id = candidate_ids[index];
            auto const query_index = pair_id / reference_count;
            pairwise_counts local{};
            auto const query_offset =
                (query_row_offset + static_cast<size_t>(query_index)) * BucketCount;
            for (auto chunk = static_cast<uint32_t>(lane); chunk < chunks_per_row;
                 chunk += warp_width) {
                classify_wide_chunk<false>(
                    local,
                    queries + query_offset + static_cast<size_t>(chunk) * chunk_buckets,
                    rows + reference_offset + static_cast<size_t>(chunk) * chunk_buckets
                );
            }
            refine_write_result(
                index,
                query_index,
                reference_id,
                query_id_offset,
                match_counts,
                pair_id,
                local,
                storage[warp],
                lane,
                results,
                result_match_counts
            );
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
    __shared__ uint32_t empty[block_size];
    __shared__ float restored[block_size];

    uint32_t local_empty = 0U;
    float local_restored = 0.0f;
    for (auto bucket = static_cast<size_t>(threadIdx.x); bucket < BucketCount;
         bucket += blockDim.x) {
        auto const stored = winner(__ldcs(&registers[bucket]));
        if (stored == 0U) {
            ++local_empty;
        } else {
            local_restored += static_cast<float>(restore_midpoint(stored));
        }
    }
    empty[threadIdx.x] = local_empty;
    restored[threadIdx.x] = local_restored;
    __syncthreads();

    for (auto stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            empty[threadIdx.x] += empty[threadIdx.x + stride];
            restored[threadIdx.x] += restored[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        *empty_out = empty[0];
        *estimate_out = static_cast<double>(cardinality_f32(
            static_cast<float>(BucketCount), static_cast<float>(empty[0]), restored[0]
        ));
    }
}

template <size_t BucketCount>
__global__ void hybrid_cardinality_kernel(
    uint32_t const* const registers,
    hybrid_cardinality_estimates* const estimates
) {
    __shared__ uint32_t bins[nlz_bins];
    __shared__ float restored[block_size];
    for (auto bin = static_cast<uint32_t>(threadIdx.x); bin < nlz_bins; bin += blockDim.x) {
        bins[bin] = 0U;
    }
    float local_restored = 0.0f;
    __syncthreads();

    for (auto bucket = static_cast<size_t>(threadIdx.x); bucket < BucketCount;
         bucket += blockDim.x) {
        auto const stored = winner(__ldcs(&registers[bucket]));
        if (stored == 0U) {
            atomicAdd(bins, 1U);
        } else {
            atomicAdd(bins + static_cast<uint32_t>(stored >> mantissa_bits) + 1U, 1U);
            local_restored += static_cast<float>(restore(stored));
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
        *estimates = hybrid_estimates_f32(bins, static_cast<float>(BucketCount), restored[0]);
    }
}

template <size_t BucketCount, hybrid_variant Variant>
__global__ void
hybrid_cardinality_variant_kernel(uint32_t const* const registers, double* const estimate) {
    __shared__ uint32_t bins[nlz_bins];
    __shared__ float restored[block_size];
    for (auto bin = static_cast<uint32_t>(threadIdx.x); bin < nlz_bins; bin += blockDim.x) {
        bins[bin] = 0U;
    }
    float local_restored = 0.0f;
    __syncthreads();

    for (auto bucket = static_cast<size_t>(threadIdx.x); bucket < BucketCount;
         bucket += blockDim.x) {
        auto const stored = winner(__ldcs(&registers[bucket]));
        if (stored == 0U) {
            atomicAdd(bins, 1U);
        } else {
            atomicAdd(bins + static_cast<uint32_t>(stored >> mantissa_bits) + 1U, 1U);
            local_restored += static_cast<float>(restore(stored));
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
            hybrid_estimates_f32(bins, static_cast<float>(BucketCount), restored[0]);
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
