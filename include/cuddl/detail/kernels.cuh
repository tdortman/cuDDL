#pragma once

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>
#include <cub/block/block_histogram.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cuda/std/algorithm>
#include <cuda/std/bit>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>

#include <cuddl/blacklist.cuh>
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

/// @brief Stored sketches one batch launch may index, bounded by the CUDA grid dimension.
constexpr size_t maximum_batch_rows = (size_t{1} << 31) - 1U;

/// @brief Per-thread accumulator feeding a CUB block reduction for fused summaries.
struct summary_payload {
    pairwise_counts counts{};
    uint64_t empty{};
    double restored_sum{};

    /// @brief Accumulates @p other into this instance for block reductions.
    __device__ summary_payload& operator+=(summary_payload const& other) noexcept {
        counts += other.counts;
        empty += other.empty;
        restored_sum += other.restored_sum;
        return *this;
    }

    /// @brief Sums two accumulators for CUB block reductions.
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

/// @brief Buckets one contiguous chunk covers, for a query/reference score pair.
///
/// One wide load covers 16 buckets whenever either row stores 16-bit scores, and 8 buckets
/// when both store 32-bit registers. The scalar fallback covers half as many.
template <typename QueryScore, typename ReferenceScore>
constexpr uint32_t wide_chunk_buckets =
    (sizeof(QueryScore) == 2U || sizeof(ReferenceScore) == 2U) ? 16U : 8U;

/// @brief Threads per CTA for the CTA-local construction kernel.
constexpr uint32_t shared_construction_block_size = 768;

/**
 * @brief Constructs a sketch through a CTA-local shared-memory register array.
 *
 * Each item costs a hash, a score, and one fire-and-forget shared atomic max; the CTA then merges
 * its registers into the global sketch with one global atomic max per nonempty bucket.
 *
 * The host launches two CTAs per SM and the kernel walks the input with a runtime grid-stride
 * loop over 256-bit chunks, so the grid is a single balanced wave for every input size and the
 * per-CTA merge traffic stays minimal.
 *
 * With FloorRounds > 0, the CTA reduces its minimum local register after that many uniform input
 * epochs. Later scores at or below that bound cannot change any register and skip the atomic.
 */
template <
    size_t BucketCount,
    typename Layout = default_register_layout,
    uint32_t FloorRounds = 0,
    bool HasBlacklist = false>
__global__ __launch_bounds__(shared_construction_block_size) void add_shared_kernel(
    uint64_t const* input,
    size_t input_size,
    uint32_t* registers,
    bool vector_input,
    blacklist_view blacklist = {}
) {
    __shared__ uint32_t state[BucketCount];
    for (auto i = threadIdx.x; i < BucketCount; i += blockDim.x) {
        state[i] = 0U;
    }
    __syncthreads();

    uint32_t floor = 0U;
    auto const process = [&](uint64_t value) {
        auto const hash = hash_kmer(value);
        auto const incoming = static_cast<uint32_t>(score<Layout>(hash));
        if constexpr (FloorRounds != 0U) {
            if (incoming <= floor) {
                return;
            }
        }
        auto const bucket = bucket_of<BucketCount>(hash);
        if constexpr (HasBlacklist) {
            if (blacklist.contains(value, bucket)) return;
        }
        atomicMax(&state[bucket], incoming);
    };

    auto const stride = static_cast<size_t>(gridDim.x) * blockDim.x * 4U;
    auto const index = (static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 4U;
    if constexpr (FloorRounds != 0U) {
        // All lanes complete the same warm-up epochs before reducing the local registers.
        using reduce = cub::BlockReduce<uint32_t, shared_construction_block_size>;
        __shared__ typename reduce::TempStorage floor_storage;
        __shared__ uint32_t shared_floor;
        for (uint32_t round = 0; round < FloorRounds; ++round) {
            auto const offset = index + stride * round;
            if (vector_input && offset + 4U <= input_size) {
                uint64_t values[4];
                load_256_global_nc(input + offset, values);
                _Pragma("unroll")
                for (auto& value : values) process(value);
            } else {
                _Pragma("unroll")
                for (uint32_t item = 0; item < 4U; ++item) {
                    if (offset + item < input_size) process(input[offset + item]);
                }
            }
        }
        __syncthreads();
        uint32_t minimum = 0xffffU;
        for (auto bucket = threadIdx.x; bucket < BucketCount; bucket += blockDim.x) {
            minimum = cuda::std::min(minimum, state[bucket]);
        }
        auto const reduced = reduce(floor_storage).Reduce(minimum, cuda::minimum<>{});
        if (threadIdx.x == 0U) shared_floor = reduced;
        __syncthreads();
        floor = shared_floor;
    }
    // Keep a partial final warp together; individual loads remain bounds-checked.
    auto const lane_offset = FloorRounds != 0U ? (threadIdx.x % warpSize) * 4U : 0U;
    for (auto offset = index + stride * FloorRounds; offset - lane_offset < input_size;
         offset += stride) {
        // Reconverge after score filtering before issuing the next global load.
        if constexpr (FloorRounds != 0U) __syncwarp();
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
    __syncthreads();

    // Each CTA starts its merge at a rotated bucket offset (odd multiplier, coprime with
    // the power-of-two bucket count), so the CTAs' atomic merges interleave across different
    // addresses instead of all colliding on the same bucket at once. The per-address atomic
    // serialization then sees a smooth stream of arrivals rather than a synchronized burst.
    auto const bucket_offset = (static_cast<size_t>(blockIdx.x) * 139U) & (BucketCount - 1U);
    for (auto j = threadIdx.x; j < BucketCount; j += blockDim.x) {
        auto const i = (static_cast<size_t>(j) + bucket_offset) & (BucketCount - 1U);
        auto const stored = state[i];
        if (stored != 0U) atomicMax(&registers[i], stored);
    }
}

/**
 * @brief Computes pairwise counts and optionally cardinality for two constructed sketches.
 */
template <size_t BucketCount, bool IncludeCardinality, typename Layout = default_register_layout>
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
            if (left_reg == 0U) {
                ++local.empty;
            } else {
                local.restored_sum += restore_midpoint<Layout>(static_cast<uint16_t>(left_reg));
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
 * @brief Compares corresponding rows from two contiguous register batches.
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
    // The exclusive scan also reads the trailing element used for the CSR end offset.
    if (blockIdx.x == 0U && threadIdx.x == 0U) {
        cell_counts[static_cast<size_t>(indexed_bucket_count) * key_count] = 0U;
    }
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
                auto const score = static_cast<uint16_t>(__ldcs(&bucket_scores[reference]));
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
/// arbitrary here; the index build sorts each cell afterwards.
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
                auto const score = static_cast<uint16_t>(__ldcs(&bucket_scores[reference]));
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

/// @brief Stable CUB selection predicate over per-reference match counts.
struct minimum_match_predicate {
    uint32_t const* match_counts;
    uint32_t minimum_matches;

    [[nodiscard]] __host__ __device__ bool operator()(uint32_t reference_id) const noexcept {
        return match_counts[reference_id] >= minimum_matches;
    }
};

/// @brief Cells one warp owns per iteration of @ref count_batch_index_matches_kernel: one per
/// lane.
///
/// The host divides the grid by this so every launched warp stays busy.
constexpr uint32_t index_match_cells_per_warp = 32U;

/// @brief One thread's posting range for a bucket/key cell of a dense or sparse index.
__device__ inline uint2 lane_posting_range(
    uint32_t const* offsets,
    uint16_t const* sorted_keys,
    uint32_t reference_count,
    uint32_t bucket,
    uint32_t key,
    uint32_t key_count
) {
    if (sorted_keys == nullptr) {
        auto const cell = static_cast<size_t>(bucket) * key_count + key;
        return {offsets[cell], offsets[cell + 1U]};
    }
    auto const* first = sorted_keys + static_cast<size_t>(bucket) * reference_count;
    auto const sparse_key = static_cast<uint16_t>(key + (key_count == 32768U));
    auto const range = cuda::std::equal_range(first, first + reference_count, sparse_key);
    return {
        static_cast<uint32_t>(range.first - sorted_keys),
        static_cast<uint32_t>(range.second - sorted_keys)
    };
}

/// @brief Low key bits a sparse key directory cell spans for @p reference_count references.
///
/// Cells average about eight keys per bucket for uniformly spread keys, so a lookup reads the
/// cell's two bounds and then searches within about one memory sector.
__host__ __device__ constexpr uint32_t sparse_directory_shift(uint32_t reference_count) noexcept {
    uint32_t cell_bits = 0U;
    while (cell_bits < 16U && (uint64_t{1} << (cell_bits + 1U)) * 8U <= reference_count) {
        ++cell_bits;
    }
    return 16U - cell_bits;
}

/// @brief Bounds per bucket in a sparse key directory.
__host__ __device__ constexpr uint32_t sparse_directory_entries(uint32_t reference_count) noexcept {
    return (1U << (16U - sparse_directory_shift(reference_count))) + 1U;
}

/// @brief Builds a sparse key directory: entry c of bucket b is the first position in b's
/// sorted keys whose key is at least `c << shift`.
__global__ __launch_bounds__(block_size) static void build_sparse_directory_kernel(
    uint16_t const* sorted_keys,
    uint32_t reference_count,
    uint32_t indexed_bucket_count,
    uint32_t* directory
) {
    auto const entries = sparse_directory_entries(reference_count);
    auto const shift = sparse_directory_shift(reference_count);
    auto const total = static_cast<size_t>(indexed_bucket_count) * entries;
    for (auto i = static_cast<size_t>(blockIdx.x) * block_size + threadIdx.x; i < total;
         i += static_cast<size_t>(gridDim.x) * block_size) {
        auto const bucket = i / entries;
        auto const bound = static_cast<uint32_t>(i % entries) << shift;
        auto const* first = sorted_keys + bucket * reference_count;
        directory[i] = static_cast<uint32_t>(
            cuda::std::lower_bound(
                first,
                first + reference_count,
                bound,
                [](uint16_t key, uint32_t value) { return key < value; }
            ) -
            first
        );
    }
}

/// @brief Adds the squared length of every nonzero-key posting list to @p work.
///
/// Dense indexes pass their cell offsets (@p key_count cells per bucket); sparse indexes pass
/// their per-bucket sorted keys instead, whose runs are the posting lists. The sum is the
/// number of (reference, reference, bucket) triples sharing an index key: an index walks about
/// `work / reference_count` postings per database-like query.
__global__ __launch_bounds__(block_size) static void index_pair_work_kernel(
    uint32_t const* offsets,
    uint16_t const* sorted_keys,
    uint32_t reference_count,
    uint32_t indexed_bucket_count,
    uint32_t key_count,
    unsigned long long* work
) {
    auto const total = offsets != nullptr
                           ? static_cast<size_t>(indexed_bucket_count) * key_count
                           : static_cast<size_t>(indexed_bucket_count) * reference_count;
    unsigned long long local = 0U;
    for (auto i = static_cast<size_t>(blockIdx.x) * block_size + threadIdx.x; i < total;
         i += static_cast<size_t>(gridDim.x) * block_size) {
        uint64_t length = 0U;
        if (offsets != nullptr) {
            if (i % key_count != 0U) {
                length = offsets[i + 1U] - offsets[i];
            }
        } else {
            auto const position = static_cast<uint32_t>(i % reference_count);
            auto const* first = sorted_keys + (i - position);
            auto const key = first[position];
            if (key != 0U && (position == 0U || first[position - 1U] != key)) {
                length = static_cast<uint64_t>(
                    cuda::std::upper_bound(first + position, first + reference_count, key) -
                    (first + position)
                );
            }
        }
        local += length * length;
    }
    if (local != 0U) {
        atomicAdd(work, local);
    }
}

/// @brief One sparse bucket/key posting range, narrowed through the bucket's key directory.
__device__ inline uint2 directory_posting_range(
    uint16_t const* sorted_keys,
    uint32_t const* directory,
    uint32_t reference_count,
    uint32_t bucket,
    uint32_t sparse_key
) {
    auto const* first = sorted_keys + static_cast<size_t>(bucket) * reference_count;
    auto const* cells =
        directory + static_cast<size_t>(bucket) * sparse_directory_entries(reference_count);
    auto const cell = sparse_key >> sparse_directory_shift(reference_count);
    auto const* low = first + cells[cell];
    auto const* high = first + cells[cell + 1U];
    auto const less = [](uint16_t key, uint32_t value) {
        return key < value;
    };
    auto const* begin = cuda::std::lower_bound(low, high, sparse_key, less);
    auto const* end = cuda::std::lower_bound(begin, high, sparse_key + 1U, less);
    return {static_cast<uint32_t>(begin - sorted_keys), static_cast<uint32_t>(end - sorted_keys)};
}

/// @brief Threads of the single @ref index_posting_ranges_kernel block.
constexpr uint32_t posting_range_block_size = 1024U;

/// @brief Resolves one query's posting range in every indexed bucket.
///
/// Writes each bucket's first posting and the inclusive prefix sum of the range lengths, so
/// @ref count_balanced_index_matches_kernel can split all postings evenly across its grid
/// however skewed the individual lists are.
static __global__ __launch_bounds__(posting_range_block_size) void index_posting_ranges_kernel(
    uint16_t const* query,
    uint32_t const* offsets,
    uint16_t const* sorted_keys,
    uint32_t reference_count,
    uint32_t indexed_bucket_count,
    uint16_t key_mask,
    uint32_t* range_begin,
    uint32_t* range_end_scan
) {
    using block_scan = cub::BlockScan<uint32_t, posting_range_block_size>;
    __shared__ typename block_scan::TempStorage storage;
    auto const key_count = static_cast<uint32_t>(key_mask) + 1U;
    uint32_t carry = 0U;
    for (uint32_t base = 0U; base < indexed_bucket_count; base += posting_range_block_size) {
        auto const bucket = base + static_cast<uint32_t>(threadIdx.x);
        auto const score = bucket < indexed_bucket_count ? query[bucket] : uint16_t{0};
        uint2 range{0U, 0U};
        if (score != 0U) {
            range = lane_posting_range(
                offsets, sorted_keys, reference_count, bucket, score & key_mask, key_count
            );
        }
        uint32_t end = 0U;
        uint32_t aggregate = 0U;
        block_scan(storage).InclusiveSum(range.y - range.x, end, aggregate);
        if (bucket < indexed_bucket_count) {
            range_begin[bucket] = range.x;
            range_end_scan[bucket] = carry + end;
        }
        carry += aggregate;
        __syncthreads();
    }
}

/// @brief Counts the query's posting matches for every reference, one posting per thread.
static __global__ __launch_bounds__(block_size) void count_balanced_index_matches_kernel(
    uint32_t const* range_begin,
    uint32_t const* range_end_scan,
    uint32_t indexed_bucket_count,
    uint32_t const* postings,
    uint32_t* match_counts
) {
    auto const total = range_end_scan[indexed_bucket_count - 1U];
    auto const stride = static_cast<uint32_t>(gridDim.x) * block_size;
    for (auto item = static_cast<uint32_t>(blockIdx.x) * block_size + threadIdx.x; item < total;
         item += stride) {
        auto const bucket = static_cast<uint32_t>(
            cuda::std::upper_bound(range_end_scan, range_end_scan + indexed_bucket_count, item) -
            range_end_scan
        );
        auto const first_item = bucket == 0U ? 0U : range_end_scan[bucket - 1U];
        atomicAdd(&match_counts[postings[range_begin[bucket] + (item - first_item)]], 1U);
    }
}

/// @brief Counts index matches for every query/reference pair in one tile.
///
/// Each lane of a warp resolves the posting range of its own (query, bucket) cell, so a warp
/// keeps 32 independent range lookups in flight. The warp then walks every non-empty range with
/// a lane stride, so hot keys with long lists (the dominant cost on skewed rows) are consumed 32
/// postings at a time.
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
    uint32_t* match_counts,
    uint16_t const* sorted_keys = nullptr
) {
    constexpr uint32_t warp_width = 32;
    constexpr uint32_t warps_per_block = block_size / warp_width;
    auto const lane = static_cast<uint32_t>(threadIdx.x) % warp_width;
    auto const warp = static_cast<uint32_t>(threadIdx.x) / warp_width;
    auto const total_cells = static_cast<uint64_t>(query_count) * indexed_bucket_count;
    auto const key_count = static_cast<uint32_t>(key_mask) + 1U;
    // The indexed bucket count is always a power of two (BucketCount or BucketCount / 2), so
    // splitting the linear cell id costs a shift and a mask instead of a 64-bit division.
    auto const bucket_shift = static_cast<uint32_t>(cuda::std::countr_zero(indexed_bucket_count));
    auto const bucket_mask = indexed_bucket_count - 1U;
    auto const first_cell =
        (static_cast<uint64_t>(blockIdx.x) * warps_per_block + warp) * index_match_cells_per_warp;
    auto const warp_cell_stride =
        static_cast<uint64_t>(gridDim.x) * warps_per_block * index_match_cells_per_warp;
    for (auto cell_base = first_cell; cell_base < total_cells; cell_base += warp_cell_stride) {
        auto const cell = cell_base + lane;
        auto const query_index = static_cast<uint32_t>(cell >> bucket_shift);
        auto const bucket = static_cast<uint32_t>(cell) & bucket_mask;
        auto const score =
            cell < total_cells
                ? static_cast<uint16_t>(
                      queries[(query_row_offset + query_index) * BucketCount + bucket]
                  )
                : 0U;
        uint2 range{0U, 0U};
        if (score != 0U) {
            range = lane_posting_range(
                offsets,
                sorted_keys,
                reference_count,
                bucket,
                static_cast<uint32_t>(score & key_mask),
                key_count
            );
        }
        auto pending = __ballot_sync(0xffffffffU, range.x < range.y);
        while (pending != 0U) {
            auto const source = static_cast<uint32_t>(__ffs(pending) - 1);
            pending &= pending - 1U;
            auto const end = __shfl_sync(0xffffffffU, range.y, source);
            auto posting = __shfl_sync(0xffffffffU, range.x, source) + lane;
            auto* const counts =
                match_counts + static_cast<size_t>(__shfl_sync(0xffffffffU, query_index, source)) *
                                   reference_count;
            // Four independent posting loads in flight per lane keep the atomic stream fed
            // while the following loads are still outstanding.
            for (; posting + 3U * warp_width < end; posting += 4U * warp_width) {
                uint32_t ids[4];
                _Pragma("unroll")
                for (uint32_t j = 0U; j < 4U; ++j) {
                    ids[j] = postings[posting + j * warp_width];
                }
                _Pragma("unroll")
                for (uint32_t j = 0U; j < 4U; ++j) {
                    atomicAdd(&counts[ids[j]], 1U);
                }
            }
            for (; posting < end; posting += warp_width) {
                atomicAdd(&counts[postings[posting]], 1U);
            }
        }
    }
}

/// @brief Candidate bitmap words per query row, one bit per reference.
__host__ __device__ constexpr uint32_t candidate_bit_words(uint32_t reference_count) noexcept {
    return (reference_count + 31U) / 32U;
}

/// @brief Threads per @ref count_batch_index_tile_kernel block.
constexpr uint32_t index_tile_block_size = 1024U;

/// @brief Longest posting range one @ref count_batch_index_tile_kernel lane walks alone.
constexpr uint32_t index_tile_lane_postings = 16U;

/// @brief Long posting ranges one @ref count_batch_index_tile_kernel block walks together.
constexpr uint32_t index_tile_long_capacity = 32U;

/// @brief Whether a batch counts through @ref count_batch_index_tile_kernel.
///
/// Its 16-bit counters hold at most 65535 matches. Tiny batches finish sooner on the global
/// kernel, which needs no per-block counter initialization or flush.
[[nodiscard]] inline bool uses_tiled_index_counts(
    uint32_t reference_count,
    uint32_t query_count,
    uint32_t indexed_bucket_count
) noexcept {
    return reference_count != 0U && indexed_bucket_count <= 0xffffU &&
           (reference_count >= 8192U || query_count >= 512U);
}

/// @brief References per tile, bounded by the device's shared memory and small enough that
/// the batch launches about four blocks per SM. Tiles align to 64 references.
[[nodiscard]] inline uint32_t index_tile_references(
    uint32_t reference_count,
    uint32_t query_count,
    uint32_t multiprocessors,
    uint32_t max_shared_bytes
) noexcept {
    constexpr uint32_t static_bytes = index_tile_long_capacity * sizeof(uint2) + sizeof(uint32_t);
    auto const max_references = (max_shared_bytes - static_bytes) / (64U * sizeof(uint16_t)) * 64U;
    auto const needed = (reference_count + max_references - 1U) / max_references;
    auto const filling = (4U * multiprocessors + query_count - 1U) / query_count;
    auto const tiles = needed > filling ? needed : filling;
    return ((reference_count + tiles - 1U) / tiles + 63U) / 64U * 64U;
}

/// @brief Selects posting lists whose bitmap uses no more storage than their reference IDs.
struct posting_bitmap_cell {
    uint32_t const* offsets;
    uint16_t const* sorted_keys;
    uint32_t reference_count;
    uint32_t words;
    __device__ bool operator()(uint32_t cell) const {
        if (sorted_keys == nullptr) {
            return offsets[cell + 1U] - offsets[cell] >= words;
        }
        auto const position = cell % reference_count;
        auto const key = sorted_keys[cell];
        return key != 0U && (position == 0U || sorted_keys[cell - 1U] != key) &&
               words <= reference_count - position && sorted_keys[cell + words - 1U] == key;
    }
};

/// @brief Materializes selected posting lists into zero-initialized reference bitmaps.
/// Sparse selections arrive as run starts; the kernel replaces them with bucket/key cell IDs.
static __global__ void build_posting_bitmaps_kernel(
    uint32_t* cells,
    uint32_t const* offsets,
    uint16_t const* sorted_keys,
    uint32_t reference_count,
    uint32_t key_count,
    uint32_t const* postings,
    uint32_t words,
    uint32_t* bitmaps
) {
    __shared__ uint2 range;
    if (threadIdx.x == 0U) {
        auto const cell = cells[blockIdx.x];
        if (sorted_keys == nullptr) {
            range = {offsets[cell], offsets[cell + 1U]};
        } else {
            auto const bucket = cell / reference_count;
            auto const key = sorted_keys[cell];
            auto const* end = sorted_keys + static_cast<size_t>(bucket + 1U) * reference_count;
            range = {
                cell,
                static_cast<uint32_t>(
                    cuda::std::upper_bound(sorted_keys + cell, end, key) - sorted_keys
                )
            };
            cells[blockIdx.x] = bucket * key_count + key - (key_count == 32768U);
        }
    }
    __syncthreads();
    auto* bitmap = bitmaps + static_cast<size_t>(blockIdx.x) * words;
    for (auto i = static_cast<size_t>(range.x) + threadIdx.x; i < range.y; i += blockDim.x) {
        auto const reference = postings[i];
        atomicOr(bitmap + reference / 32U, 1U << (reference % 32U));
    }
}

/// @brief Splits each query's posting ranges into short lists and bitmap offsets.
/// With no dense @p offsets, consumes sparse ranges already staged in @p ranges.
/// Packs bitmap offsets in bucket order so nearby queries walk their shared bitmaps together.
template <size_t BucketCount>
__global__ void hybrid_posting_ranges_kernel(
    uint16_t const* queries,
    uint32_t const* offsets,
    uint32_t const* bitmap_cells,
    uint32_t bitmap_count,
    uint32_t words,
    uint32_t indexed_buckets,
    uint16_t key_mask,
    uint2* ranges,
    uint32_t* bitmap_starts,
    uint32_t* bitmap_counts
) {
    using block_scan = cub::BlockScan<uint32_t, block_size>;
    __shared__ typename block_scan::TempStorage storage;
    uint32_t count = 0U;
    auto const query = static_cast<uint32_t>(blockIdx.x);
    auto const base = static_cast<size_t>(query) * indexed_buckets;
    for (uint32_t first = 0U; first < indexed_buckets; first += block_size) {
        auto const bucket = first + threadIdx.x;
        uint2 range{0U, 0U};
        uint32_t bitmap = 0U;
        bool selected = false;
        auto const score = bucket < indexed_buckets
                               ? queries[static_cast<size_t>(query) * BucketCount + bucket]
                               : uint16_t{0};
        if (score != 0U) {
            auto const cell = bucket * (static_cast<uint32_t>(key_mask) + 1U) + (score & key_mask);
            if (offsets != nullptr) {
                range = {offsets[cell], offsets[cell + 1U]};
            } else {
                range = ranges[base + bucket];
            }
            if (range.y - range.x >= words) {
                uint32_t low = 0U, high = bitmap_count;
                while (low < high) {
                    auto const middle = low + (high - low) / 2U;
                    if (bitmap_cells[middle] < cell) {
                        low = middle + 1U;
                    } else {
                        high = middle;
                    }
                }
                bitmap = low * words;
                selected = true;
                range = {0U, 0U};
            }
        }
        uint32_t position = 0U, aggregate = 0U;
        block_scan(storage).ExclusiveSum(static_cast<uint32_t>(selected), position, aggregate);
        if (selected) {
            bitmap_starts[base + count + position] = bitmap;
        }
        if (bucket < indexed_buckets) {
            ranges[base + bucket] = range;
        }
        count += aggregate;
        __syncthreads();
    }
    if (threadIdx.x == 0U) {
        bitmap_counts[query] = count;
    }
}

constexpr uint32_t bitmap_count_words_per_query = 64U;

/// @brief Records threshold crossings while accumulating long posting lists in bit-sliced counters.
/// Each block handles adjacent queries over a shared reference interval for bitmap cache reuse.
/// Independent bitmap loads overlap in batches. All-to-all searches exclude the diagonal and
/// lower triangle from @p passes. The short-list stage consumes @p planes.
/// Once a reference passes, its count is unused and may wrap; the short-list stage seeds it
/// with @p minimum_matches instead.
template <uint32_t CounterBits>
__global__ void count_posting_bitmaps_kernel(
    uint32_t const* bitmaps,
    uint32_t const* starts,
    uint32_t const* counts,
    uint32_t query_count,
    uint32_t reference_count,
    uint32_t indexed_buckets,
    uint32_t minimum_matches,
    uint32_t query_id_offset,
    bool all_to_all,
    uint32_t* planes,
    uint32_t* passes
) {
    auto const words = candidate_bit_words(reference_count);
    auto const queries_per_block = blockDim.x / bitmap_count_words_per_query;
    auto const query_tiles = (query_count + queries_per_block - 1U) / queries_per_block;
    auto const word_tiles =
        (words + bitmap_count_words_per_query - 1U) / bitmap_count_words_per_query;
    for (size_t tile = blockIdx.x; tile < static_cast<size_t>(query_tiles) * word_tiles;
         tile += gridDim.x) {
        auto const query =
            tile % query_tiles * queries_per_block + threadIdx.x / bitmap_count_words_per_query;
        auto const word = tile / query_tiles * bitmap_count_words_per_query +
                          threadIdx.x % bitmap_count_words_per_query;
        if (query >= query_count || word >= words) continue;
        uint32_t excluded = 0U;
        if (all_to_all && word * 32U <= query_id_offset + query) {
            auto const skipped = query_id_offset + query - word * 32U + 1U;
            excluded = skipped >= 32U ? ~0U : (1U << skipped) - 1U;
        }
        uint32_t counters[CounterBits]{};
        uint32_t passed = minimum_matches == 0U ? ~0U : excluded;
        auto const count = counts[query];
        constexpr uint32_t batch = 8U;
        for (uint32_t i = 0U; i < count && passed != ~0U; i += batch) {
            // Keep independent bitmap reads in flight before consuming their values.
            uint32_t loaded[batch]{};
            _Pragma("unroll")
            for (uint32_t j = 0U; j < batch; ++j) {
                if (i + j < count) {
                    loaded[j] = bitmaps[starts[query * indexed_buckets + i + j] + word];
                }
            }
            // Reduce eight bitmaps to four input count planes.
            auto const sum0 = loaded[0] ^ loaded[1] ^ loaded[2];
            auto const carry0 = (loaded[0] & loaded[1]) | ((loaded[0] | loaded[1]) & loaded[2]);
            auto const sum1 = loaded[3] ^ loaded[4] ^ loaded[5];
            auto const carry1 = (loaded[3] & loaded[4]) | ((loaded[3] | loaded[4]) & loaded[5]);
            auto const ones0 = sum0 ^ sum1 ^ loaded[6];
            auto const carry2 = (sum0 & sum1) | ((sum0 | sum1) & loaded[6]);
            auto const twos0 = carry0 ^ carry1 ^ carry2;
            auto const fours0 = (carry0 & carry1) | ((carry0 | carry1) & carry2);
            auto const carry3 = ones0 & loaded[7];
            auto const carry4 = twos0 & carry3;
            uint32_t const additions[]{
                ones0 ^ loaded[7], twos0 ^ carry3, fours0 ^ carry4, fours0 & carry4
            };
            uint32_t carry = 0U;
            uint32_t at_least = ~0U;
            _Pragma("unroll")
            for (uint32_t bit = 0U; bit < CounterBits; ++bit) {
                auto const extra = bit < 4U ? additions[bit] : 0U;
                auto const previous = counters[bit];
                counters[bit] = previous ^ carry ^ extra;
                carry = (previous & carry) | ((previous | carry) & extra);
                auto const key = 0U - ((minimum_matches >> bit) & 1U);
                at_least = (counters[bit] & ~key) | ((counters[bit] | ~key) & at_least);
            }
            // Overflow is above every threshold representable by these counter planes.
            passed |= at_least | carry;
            _Pragma("unroll")
            for (uint32_t bit = CounterBits; bit < 4U; ++bit) {
                passed |= additions[bit];
            }
        }
        _Pragma("unroll")
        for (uint32_t bit = 0U; bit < CounterBits; ++bit) {
            planes[(query * CounterBits + bit) * words + word] = counters[bit];
        }
        auto const valid = reference_count - word * 32U;
        passes[query * words + word] =
            passed & ~excluded & (valid >= 32U ? ~0U : (1U << valid) - 1U);
    }
}

/// @brief Counts index matches for one query against one reference tile in shared memory.
///
/// Block b owns query b / tiles and references [t * tile, (t + 1) * tile) for t = b % tiles,
/// with two 16-bit counters per shared word. A query's tiles run as neighbouring blocks, so they
/// share its posting-range lookups through the caches. Postings ascend by reference within each
/// key, so each posting list is narrowed to the tile with two binary searches. The flush writes
/// every pair of the tile, so @p match_counts needs no zeroing.
/// Seeded counting starts with saturated bitmap counts and adds the remaining posting lists.
///
/// Launched with @ref index_tile_block_size threads: the shared counters, not the thread count,
/// bound how many blocks fit on an SM, so larger blocks keep more warps resident to hide the
/// posting loads' latency.
template <size_t BucketCount, bool Seeded, typename QueryRow>
__global__ __launch_bounds__(index_tile_block_size) void count_batch_index_tile_kernel(
    QueryRow const* queries,
    size_t query_row_offset,
    uint32_t const* offsets,
    uint32_t const* postings,
    uint32_t reference_count,
    uint32_t indexed_bucket_count,
    uint16_t key_mask,
    uint32_t tile,
    uint32_t* match_counts,
    uint16_t const* sorted_keys,
    uint2 const* ranges,
    uint32_t minimum_matches,
    uint32_t query_id_offset,
    bool all_to_all,
    uint32_t* candidate_bits,
    uint32_t const* initial_count_planes,
    uint32_t initial_count_bits
) {
    constexpr uint32_t warp_width = 32;
    constexpr uint32_t warps_per_block = index_tile_block_size / warp_width;
    extern __shared__ uint32_t packed_counts[];
    auto const tiles = (reference_count + tile - 1U) / tile;
    auto const query_index = static_cast<uint32_t>(blockIdx.x) / tiles;
    auto low = static_cast<uint32_t>(blockIdx.x) % tiles * tile;
    auto const high = min(low + tile, reference_count);
    // All-to-all tiles below the diagonal contain no selected pairs.
    if (candidate_bits != nullptr && all_to_all && high <= query_id_offset + query_index + 1U) {
        auto* bits = candidate_bits +
                     static_cast<size_t>(query_index) * candidate_bit_words(reference_count) +
                     low / warp_width;
        for (auto i = threadIdx.x; i < (high - low + warp_width - 1U) / warp_width;
             i += blockDim.x) {
            bits[i] = 0U;
        }
        return;
    }
    if (candidate_bits != nullptr && all_to_all) {
        // Keep counters aligned while excluding complete words below the diagonal.
        auto const first = max(low, (query_id_offset + query_index + 1U) / 64U * 64U);
        auto* bits = candidate_bits +
                     static_cast<size_t>(query_index) * candidate_bit_words(reference_count) +
                     low / warp_width;
        for (auto i = threadIdx.x; i < (first - low) / warp_width; i += blockDim.x) {
            bits[i] = 0U;
        }
        low = first;
    }
    auto const width = high - low;
    __shared__ uint2 long_ranges[index_tile_long_capacity];
    __shared__ uint32_t long_count;
    auto const lane = static_cast<uint32_t>(threadIdx.x) % warp_width;
    auto const warp = static_cast<uint32_t>(threadIdx.x) / warp_width;
    for (auto base = warp * warp_width; base < (width + 1U) / 2U; base += blockDim.x) {
        auto const i = base + lane;
        uint32_t packed = 0U;
        if constexpr (Seeded) {
            auto const words = candidate_bit_words(reference_count);
            auto const word = (low + 2U * base) / warp_width + lane / 16U;
            auto const shift = (lane % 16U) * 2U;
            // Each half warp loads the planes of one word concurrently, then broadcasts them.
            uint32_t seed = 0U;
            if (lane % 16U < initial_count_bits && word < words) {
                seed = initial_count_planes
                    [(static_cast<size_t>(query_index) * initial_count_bits + lane % 16U) * words +
                     word];
            }
            for (uint32_t plane = 0U; plane < initial_count_bits; ++plane) {
                auto const bits =
                    (__shfl_sync(0xffffffffU, seed, plane + (lane / 16U) * 16U) >> shift) & 3U;
                packed |= ((bits & 1U) | ((bits & 2U) << 15U)) << plane;
            }
            auto const passed =
                word < words
                    ? candidate_bits[static_cast<size_t>(query_index) * words + word] >> shift
                    : 0U;
            if (passed & 1U) {
                packed = (packed & 0xffff0000U) | minimum_matches;
            }
            if (passed & 2U) {
                packed = (packed & 0xffffU) | (minimum_matches << 16U);
            }
        }
        if (i < (width + 1U) / 2U) {
            packed_counts[i] = packed;
        }
    }
    if (threadIdx.x == 0U) {
        long_count = 0U;
    }
    __syncthreads();
    auto const key_count = static_cast<uint32_t>(key_mask) + 1U;
    auto const* query = queries + (query_row_offset + query_index) * BucketCount;
    auto const count = [&](uint32_t reference) {
        auto const slot = reference - low;
        atomicAdd(&packed_counts[slot / 2U], 1U << ((slot % 2U) * 16U));
    };
    // Each lane resolves one bucket's posting range, so a warp keeps 32 independent offset
    // loads and binary searches in flight; the warp then walks the ranges together.
    for (auto base = warp * warp_width; base < indexed_bucket_count;
         base += warps_per_block * warp_width) {
        auto const bucket = base + lane;
        auto const score =
            bucket < indexed_bucket_count ? static_cast<uint16_t>(query[bucket]) : 0U;
        uint32_t begin = 0U;
        uint32_t end = 0U;
        if (score != 0U) {
            auto const range =
                ranges != nullptr
                    ? ranges[static_cast<size_t>(query_index) * indexed_bucket_count + bucket]
                    : lane_posting_range(
                          offsets,
                          sorted_keys,
                          reference_count,
                          bucket,
                          static_cast<uint32_t>(score & key_mask),
                          key_count
                      );
            begin = range.x;
            end = range.y;
            // Long lists are narrowed to the tile by binary search; short ones are cheaper to
            // filter while walking them below.
            if (end - begin > index_tile_lane_postings) {
                if (low != 0U) {
                    begin = static_cast<uint32_t>(
                        cuda::std::lower_bound(postings + begin, postings + end, low) - postings
                    );
                }
                if (high != reference_count) {
                    end = static_cast<uint32_t>(
                        cuda::std::lower_bound(postings + begin, postings + end, high) - postings
                    );
                }
            }
        }
        // Typical ranges hold a few postings, so each lane walks its own short range, keeping
        // the postings inside the tile; only long ranges are walked by the whole warp, which
        // would otherwise leave most lanes idle.
        auto const lane_walk = end - begin <= index_tile_lane_postings;
        if (lane_walk) {
            auto posting = begin;
            // Preload full groups before issuing dependent shared-memory atomics.
            for (; end - posting >= 4U; posting += 4U) {
                uint32_t ids[4];
                _Pragma("unroll")
                for (uint32_t j = 0U; j < 4U; ++j) {
                    ids[j] = postings[posting + j];
                }
                _Pragma("unroll")
                for (uint32_t j = 0U; j < 4U; ++j) {
                    if (ids[j] >= low && ids[j] < high) {
                        count(ids[j]);
                    }
                }
                if (ids[3] >= high) {
                    posting = end;
                    break;
                }
            }
            for (; posting < end; ++posting) {
                auto const reference = postings[posting];
                if (reference >= high) {
                    break;
                }
                if (reference >= low) {
                    count(reference);
                }
            }
        }
        // Only ranges large enough to fill the block are queued; the owning warp walks smaller
        // ranges and any that do not fit in the queue.
        uint32_t slot = index_tile_long_capacity;
        if (end - begin >= index_tile_block_size) {
            slot = atomicAdd(&long_count, 1U);
            if (slot < index_tile_long_capacity) {
                long_ranges[slot] = make_uint2(begin, end);
            }
        }
        auto pending = __ballot_sync(0xffffffffU, !lane_walk && slot >= index_tile_long_capacity);
        while (pending != 0U) {
            auto const source = static_cast<uint32_t>(__ffs(pending) - 1);
            pending &= pending - 1U;
            auto const range_end = __shfl_sync(0xffffffffU, end, source);
            auto posting = __shfl_sync(0xffffffffU, begin, source) + lane;
            for (; posting + 3U * warp_width < range_end; posting += 4U * warp_width) {
                uint32_t ids[4];
                _Pragma("unroll")
                for (uint32_t j = 0U; j < 4U; ++j) {
                    ids[j] = postings[posting + j * warp_width];
                }
                _Pragma("unroll")
                for (uint32_t j = 0U; j < 4U; ++j) {
                    count(ids[j]);
                }
            }
            for (; posting < range_end; posting += warp_width) {
                count(postings[posting]);
            }
        }
    }
    __syncthreads();
    // Every thread takes part in each long range, four loads in flight per thread.
    auto const long_ranges_used =
        long_count < index_tile_long_capacity ? long_count : index_tile_long_capacity;
    for (uint32_t i = 0U; i < long_ranges_used; ++i) {
        auto const range = long_ranges[i];
        auto posting = range.x + static_cast<uint32_t>(threadIdx.x);
        for (; posting + 3U * index_tile_block_size < range.y;
             posting += 4U * index_tile_block_size) {
            uint32_t ids[4];
            _Pragma("unroll")
            for (uint32_t j = 0U; j < 4U; ++j) {
                ids[j] = postings[posting + j * index_tile_block_size];
            }
            _Pragma("unroll")
            for (uint32_t j = 0U; j < 4U; ++j) {
                count(ids[j]);
            }
        }
        for (; posting < range.y; posting += index_tile_block_size) {
            count(postings[posting]);
        }
    }
    __syncthreads();
    if (candidate_bits == nullptr) {
        auto* const counts =
            match_counts + static_cast<size_t>(query_index) * reference_count + low;
        for (auto i = threadIdx.x; i < width; i += blockDim.x) {
            counts[i] = (packed_counts[i / 2U] >> ((i % 2U) * 16U)) & 0xffffU;
        }
        return;
    }
    // Apply the selection predicate here and flush one bit per pair; tiles start on 64-reference
    // boundaries, so every word belongs to this block.
    auto* const bits = candidate_bits +
                       static_cast<size_t>(query_index) * candidate_bit_words(reference_count) +
                       low / 32U;
    for (auto base = warp * warp_width; base < width; base += index_tile_block_size) {
        auto const i = base + lane;
        auto selected = false;
        if (i < width) {
            auto const count = (packed_counts[i / 2U] >> ((i % 2U) * 16U)) & 0xffffU;
            selected = count >= minimum_matches &&
                       (!all_to_all || query_id_offset + query_index < low + i);
        }
        auto const word = __ballot_sync(0xffffffffU, selected);
        if (lane == 0U) {
            bits[base / warp_width] = word;
        }
    }
}

/// @brief Resolves every query's sparse posting range for one indexed bucket per block.
///
/// Block b owns bucket b, so its lookups share the bucket's directory and keys through L1.
/// @p ranges is query-major, one entry per (query, indexed bucket).
template <size_t BucketCount, typename QueryRow>
__global__ __launch_bounds__(block_size) void sparse_batch_posting_ranges_kernel(
    QueryRow const* queries,
    size_t query_row_offset,
    uint32_t query_count,
    uint16_t const* sorted_keys,
    uint32_t const* directory,
    uint32_t reference_count,
    uint32_t indexed_bucket_count,
    uint16_t key_mask,
    uint2* ranges
) {
    auto const bucket = static_cast<uint32_t>(blockIdx.x);
    auto const key_count = static_cast<uint32_t>(key_mask) + 1U;
    for (auto query = static_cast<uint32_t>(threadIdx.x); query < query_count;
         query += block_size) {
        auto const score =
            static_cast<uint16_t>(queries[(query_row_offset + query) * BucketCount + bucket]);
        uint2 range{0U, 0U};
        if (score != 0U) {
            range = directory_posting_range(
                sorted_keys,
                directory,
                reference_count,
                bucket,
                static_cast<uint32_t>(score & key_mask) + (key_count == 32768U ? 1U : 0U)
            );
        }
        ranges[static_cast<size_t>(query) * indexed_bucket_count + bucket] = range;
    }
}

/// @brief Classifies 16 buckets of an already loaded 16-bit query chunk against one
/// @ref load_256_alignment aligned reference chunk.
template <typename ReferenceScore>
__device__ __forceinline__ void classify_query_words(
    pairwise_counts& target,
    uint32_t const (&q)[8],
    ReferenceScore const* reference
) noexcept {
    auto const* qs = reinterpret_cast<uint16_t const*>(q);
    if constexpr (sizeof(ReferenceScore) == 2U) {
        uint32_t r[8];
        load_256_global_nc(reinterpret_cast<uint32_t const*>(reference), r);
        // Differences of zero-extended 16-bit scores carry the comparison in their sign bit,
        // and a sum of two scores minus one is negative only when both are empty. Equal is
        // whatever remains of the chunk's 16 buckets.
        uint32_t lower = 0U;
        uint32_t higher = 0U;
        uint32_t empty = 0U;
        _Pragma("unroll")
        for (uint32_t i = 0; i < 8U; ++i) {
            auto const q_low = q[i] & 0xffffU;
            auto const q_high = q[i] >> 16U;
            auto const r_low = r[i] & 0xffffU;
            auto const r_high = r[i] >> 16U;
            lower += ((q_low - r_low) >> 31U) + ((q_high - r_high) >> 31U);
            higher += ((r_low - q_low) >> 31U) + ((r_high - q_high) >> 31U);
            empty += ((q_low + r_low - 1U) >> 31U) + ((q_high + r_high - 1U) >> 31U);
        }
        target.lower += lower;
        target.higher += higher;
        target.both_empty += empty;
        target.equal += 16U - lower - higher - empty;
    } else {
        uint32_t r0[8];
        uint32_t r1[8];
        load_256_global_nc(reinterpret_cast<uint32_t const*>(reference), r0);
        load_256_global_nc(reinterpret_cast<uint32_t const*>(reference) + 8U, r1);
        _Pragma("unroll")
        for (uint32_t i = 0; i < 8U; ++i) {
            classify(target, qs[i], r0[i]);
        }
        _Pragma("unroll")
        for (uint32_t i = 0; i < 8U; ++i) {
            classify(target, qs[8U + i], r1[i]);
        }
    }
}

/// @brief Warp-reduces one refined candidate and writes its stable result slot.
///
/// Each counter reduces with one `redux.sync` instruction instead of a shuffle tree.
template <typename SearchResult>
__device__ void refine_write_result(
    uint32_t index,
    uint32_t query_index,
    uint32_t reference_id,
    uint32_t query_id_offset,
    uint32_t const* match_counts,
    uint32_t pair_id,
    pairwise_counts local,
    uint32_t lane,
    SearchResult* results,
    uint32_t* result_match_counts
) {
    pairwise_counts const total{
        .lower = __reduce_add_sync(0xffffffffU, local.lower),
        .equal = __reduce_add_sync(0xffffffffU, local.equal),
        .higher = __reduce_add_sync(0xffffffffU, local.higher),
        .both_empty = __reduce_add_sync(0xffffffffU, local.both_empty),
    };
    if (lane == 0U) {
        results[index].query_id = query_id_offset + query_index;
        results[index].reference_id = reference_id;
        results[index].counts = total;
        if (result_match_counts != nullptr) {
            result_match_counts[index] = match_counts[pair_id];
        }
    }
}

/// @brief Reference-row bytes one @ref refine_batch_index_candidates_kernel block spans.
///
/// Candidates are refined one (reference block, query) cell at a time, reference-block major,
/// so the resident warps share a few blocks' rows through L2 instead of every query
/// streaming every candidate reference row from DRAM.
constexpr size_t refine_block_row_bytes = size_t{8} << 20U;

/// @brief Bits per score, and so bit-planes per 32-bucket group.
constexpr uint32_t score_planes = 16U;

/// @brief Position of plane @p plane of 32-bucket group @p group within a row's bit-planes.
///
/// A row of `BucketCount` 16-bit scores becomes `BucketCount / 2` plane words: word p of group g
/// holds bit p of the group's 32 scores. Lane `g % 32` of a warp owns group g, and its 16 plane
/// words form four consecutive uint4s at lane-contiguous positions, so a warp reads a row with
/// coalesced 128-bit loads and shared copies with conflict-free 128-bit loads.
__host__ __device__ constexpr uint32_t score_plane_index(uint32_t group, uint32_t plane) noexcept {
    return (((group / 32U) * 4U + plane / 4U) * 32U + group % 32U) * 4U + plane % 4U;
}

/// @brief Writes the bit-planes of one 32-bucket group whose scores the warp holds, one per lane.
__device__ __forceinline__ void
write_score_planes(uint32_t score, uint32_t group, uint32_t lane, uint32_t* planes) {
    uint32_t mine = 0U;
    _Pragma("unroll")
    for (uint32_t plane = 0; plane < score_planes; ++plane) {
        auto const word = __ballot_sync(0xffffffffU, ((score >> plane) & 1U) != 0U);
        if (lane == plane) {
            mine = word;
        }
    }
    if (lane < score_planes) {
        planes[score_plane_index(group, lane)] = mine;
    }
}

/// @brief Transposes compact score rows into the bit-plane layout of @ref score_plane_index.
template <size_t BucketCount>
__global__ __launch_bounds__(
    block_size
) void build_score_planes_kernel(uint16_t const* rows, uint32_t row_count, uint32_t* planes) {
    constexpr uint32_t warp_width = 32;
    constexpr uint32_t groups_per_row = static_cast<uint32_t>(BucketCount / warp_width);
    auto const lane = static_cast<uint32_t>(threadIdx.x) % warp_width;
    auto const total = static_cast<size_t>(row_count) * groups_per_row;
    auto const stride = static_cast<size_t>(gridDim.x) * (block_size / warp_width);
    for (auto item = (static_cast<size_t>(blockIdx.x) * block_size + threadIdx.x) / warp_width;
         item < total;
         item += stride) {
        auto const row = item / groups_per_row;
        auto const group = static_cast<uint32_t>(item % groups_per_row);
        write_score_planes(
            rows[row * BucketCount + group * warp_width + lane],
            group,
            lane,
            planes + row * (BucketCount / 2U)
        );
    }
}

/// @brief Loads group @p group of one bit-plane row (@ref score_plane_index layout).
__device__ __forceinline__ void
load_plane_group(uint4 const* row, uint32_t group, uint32_t (&words)[score_planes]) {
    _Pragma("unroll")
    for (uint32_t c = 0; c < score_planes / 4U; ++c) {
        auto const v = row[((group / 32U) * 4U + c) * 32U + group % 32U];
        words[4U * c] = v.x;
        words[4U * c + 1U] = v.y;
        words[4U * c + 2U] = v.z;
        words[4U * c + 3U] = v.w;
    }
}

/// @brief Bit-sliced pairwise counts of one 32-bucket group, accumulated per lane.
///
/// Differing and occupied buckets are counted rather than their equal and empty complements,
/// so no complement is computed per plane group; the reduction recovers the complements.
struct plane_counts {
    uint32_t lower{};
    /// Buckets whose scores differ.
    uint32_t differ{};
    /// Buckets whose scores differ or whose reference score is nonzero: all but both-empty.
    uint32_t occupied{};
    uint32_t matches{};

    /// @brief Warp-reduces the counts into an exact pairwise summary.
    template <size_t BucketCount, bool ReferenceFull = false>
    [[nodiscard]] __device__ pairwise_counts reduce() const noexcept {
        auto const total_lower = __reduce_add_sync(0xffffffffU, lower);
        auto const total_differ = __reduce_add_sync(0xffffffffU, differ);
        auto const total_occupied = ReferenceFull ? static_cast<uint32_t>(BucketCount)
                                                  : __reduce_add_sync(0xffffffffU, occupied);
        return {
            .lower = total_lower,
            .equal = total_occupied - total_differ,
            .higher = total_differ - total_lower,
            .both_empty = static_cast<uint32_t>(BucketCount) - total_occupied,
        };
    }
};

/// @brief Buckets of a plane group holding a nonzero score.
[[nodiscard]] __device__ __forceinline__ uint32_t
plane_any(uint32_t const (&planes)[score_planes]) noexcept {
    uint32_t any = 0U;
    _Pragma("unroll")
    for (uint32_t p = 0; p < score_planes; ++p) {
        any |= planes[p];
    }
    return any;
}

/// @brief Buckets of a plane group whose score is nonzero below the top plane.
[[nodiscard]] __device__ __forceinline__ uint32_t
plane_low_any(uint32_t const (&planes)[score_planes]) noexcept {
    uint32_t any = 0U;
    _Pragma("unroll")
    for (uint32_t p = 0; p + 1U < score_planes; ++p) {
        any |= planes[p];
    }
    return any;
}

/// @brief Adds one group's comparison of query planes @p q with reference planes @p r, whose
/// nonzero buckets @p reference_any the caller hoists out of per-query loops.
///
/// From the least significant plane up, each plane's verdict overrides the lower planes' unless
/// its bits are equal: one 3-input logic operation per plane for `lt`, one for `ne`. With
/// @p count_matches, buckets whose masked keys agree and whose scores are both nonzero are
/// counted too, as the index counts them.
///
/// With @p CountMatches, matches are counted in the buckets of @p match_lanes; the two stored
/// key masks come from the plane chain: a full key
/// matches exactly the equal nonempty buckets, and the folded 15-bit key those whose low planes
/// agree and that are nonempty in both rows, which needs the reference's low-plane nonzero mask
/// @p reference_low_any. Only those two masks pass score_compatibility validation.
///
/// @p ReferenceFull asserts every reference score is nonzero, so no bucket is both empty.
template <bool CountMatches, bool ReferenceFull = false>
__device__ __forceinline__ void compare_plane_group(
    uint32_t const (&q)[score_planes],
    uint32_t const (&r)[score_planes],
    uint32_t reference_any,
    uint32_t reference_low_any,
    uint16_t key_mask,
    uint32_t match_lanes,
    plane_counts& counts
) noexcept {
    uint32_t lt = 0U;
    uint32_t ne = 0U;
    uint32_t low_ne = 0U;
    _Pragma("unroll")
    for (uint32_t p = 0; p < score_planes; ++p) {
        if (p == score_planes - 1U) {
            low_ne = ne;
        }
        lt = (~q[p] & r[p]) | (~(q[p] ^ r[p]) & lt);
        ne |= q[p] ^ r[p];
    }
    counts.lower += static_cast<uint32_t>(__popc(lt));
    counts.differ += static_cast<uint32_t>(__popc(ne));
    if constexpr (!ReferenceFull) {
        counts.occupied += static_cast<uint32_t>(__popc(ne | reference_any));
    }
    if constexpr (CountMatches) {
        // Validated key masks cover the low planes and, for a full key, the top one. Where the
        // compared planes agree, the low planes agree, so both rows are nonzero exactly when the
        // shared low planes are or both top bits are set.
        constexpr auto top = score_planes - 1U;
        auto const top_keyed = 0U - (static_cast<uint32_t>(key_mask) >> top);
        auto const key_differs = (ne & top_keyed) | (low_ne & ~top_keyed);
        counts.matches += static_cast<uint32_t>(
            __popc(~key_differs & (reference_low_any | (q[top] & r[top])) & match_lanes)
        );
    }
}

/// @brief Recovers row-major scores from bit-plane rows, one warp per 32-bucket group.
template <size_t BucketCount>
__global__ __launch_bounds__(
    block_size
) void decode_score_planes_kernel(uint32_t const* planes, size_t row_count, uint16_t* rows) {
    constexpr uint32_t warp_width = 32;
    constexpr uint32_t groups_per_row = static_cast<uint32_t>(BucketCount / warp_width);
    auto const lane = static_cast<uint32_t>(threadIdx.x) % warp_width;
    auto const total = row_count * groups_per_row;
    auto const stride = static_cast<size_t>(gridDim.x) * (block_size / warp_width);
    for (auto item = (static_cast<size_t>(blockIdx.x) * block_size + threadIdx.x) / warp_width;
         item < total;
         item += stride) {
        auto const row = item / groups_per_row;
        auto const group = static_cast<uint32_t>(item % groups_per_row);
        uint32_t words[score_planes];
        load_plane_group(
            reinterpret_cast<uint4 const*>(planes + row * (BucketCount / 2U)), group, words
        );
        uint32_t score = 0U;
        _Pragma("unroll")
        for (uint32_t p = 0; p < score_planes; ++p) {
            score |= ((words[p] >> lane) & 1U) << p;
        }
        rows[row * BucketCount + group * warp_width + lane] = static_cast<uint16_t>(score);
    }
}

/// @brief Blocks for a grid-stride launch covering @p warps warps of @ref block_size threads.
[[nodiscard]] inline uint32_t warp_grid_blocks(size_t warps) noexcept {
    constexpr size_t warps_per_block = block_size / 32U;
    return static_cast<uint32_t>(
        cuda::std::min<size_t>((warps + warps_per_block - 1U) / warps_per_block, 65535U)
    );
}

/// @brief Compares one query's bit-planes with every reference, or with the selected candidates.
///
/// One warp compares one reference; lane l takes groups l, l + 32, ... With @p candidate_ids,
/// item i refines reference `candidate_ids[i]` for `*candidate_count` items; otherwise item i is
/// reference i.
template <size_t BucketCount, typename SearchResult>
__global__ __launch_bounds__(block_size) void single_query_planes_kernel(
    uint32_t const* query_planes,
    uint32_t const* reference_planes,
    uint32_t reference_count,
    uint32_t const* candidate_ids,
    uint32_t const* candidate_count,
    SearchResult* results
) {
    constexpr uint32_t warp_width = 32;
    constexpr uint32_t row_words = static_cast<uint32_t>(BucketCount / 2U);
    constexpr uint32_t groups_per_row = static_cast<uint32_t>(BucketCount / warp_width);
    auto const lane = static_cast<uint32_t>(threadIdx.x) % warp_width;
    auto const items = candidate_ids == nullptr ? reference_count : *candidate_count;
    auto const stride = static_cast<uint32_t>(gridDim.x) * (block_size / warp_width);
    auto const* query = reinterpret_cast<uint4 const*>(query_planes);
    for (auto item = (static_cast<uint32_t>(blockIdx.x) * block_size + threadIdx.x) / warp_width;
         item < items;
         item += stride) {
        auto const reference_id = candidate_ids == nullptr ? item : candidate_ids[item];
        auto const* reference = reinterpret_cast<uint4 const*>(
            reference_planes + static_cast<size_t>(reference_id) * row_words
        );
        plane_counts counts{};
        for (auto group = lane; group < groups_per_row; group += warp_width) {
            uint32_t q[score_planes];
            uint32_t r[score_planes];
            load_plane_group(query, group, q);
            load_plane_group(reference, group, r);
            compare_plane_group<false>(q, r, plane_any(r), 0U, 0U, 0U, counts);
        }
        auto const total = counts.reduce<BucketCount>();
        if (lane == 0U) {
            results[item].reference_id = reference_id;
            results[item].counts = total;
        }
    }
}

/// @brief Counts every query/reference pair's index matches from bit-planes, one warp per pair.
///
/// Used when a search asks for index semantics without an index: the counts equal the ones the
/// index postings would produce.
template <size_t BucketCount>
__global__ __launch_bounds__(block_size) void count_plane_matches_kernel(
    uint32_t const* query_planes,
    uint32_t query_count,
    uint32_t const* reference_planes,
    uint32_t reference_count,
    uint32_t indexed_bucket_count,
    uint16_t key_mask,
    uint32_t* match_counts
) {
    constexpr uint32_t warp_width = 32;
    constexpr uint32_t row_words = static_cast<uint32_t>(BucketCount / 2U);
    auto const lane = static_cast<uint32_t>(threadIdx.x) % warp_width;
    auto const pairs = static_cast<uint64_t>(query_count) * reference_count;
    auto const stride = static_cast<uint64_t>(gridDim.x) * (block_size / warp_width);
    auto const indexed_groups = indexed_bucket_count / warp_width;
    for (auto pair = (static_cast<uint64_t>(blockIdx.x) * block_size + threadIdx.x) / warp_width;
         pair < pairs;
         pair += stride) {
        auto const query_index = pair / reference_count;
        auto const reference_id = pair % reference_count;
        auto const* query = reinterpret_cast<uint4 const*>(query_planes + query_index * row_words);
        auto const* reference =
            reinterpret_cast<uint4 const*>(reference_planes + reference_id * row_words);
        plane_counts counts{};
        for (auto group = lane; group < indexed_groups; group += warp_width) {
            uint32_t q[score_planes];
            uint32_t r[score_planes];
            load_plane_group(query, group, q);
            load_plane_group(reference, group, r);
            compare_plane_group<true>(q, r, plane_any(r), plane_low_any(r), key_mask, ~0U, counts);
        }
        auto const total = __reduce_add_sync(0xffffffffU, counts.matches);
        if (lane == 0U) {
            match_counts[pair] = total;
        }
    }
}

/// @brief Turns per-pair match counts into the query-major candidate bitmap.
static __global__ __launch_bounds__(block_size) void candidate_bits_from_counts_kernel(
    uint32_t const* match_counts,
    uint32_t query_count,
    uint32_t reference_count,
    uint32_t minimum_matches,
    uint32_t query_id_offset,
    bool all_to_all,
    uint32_t* candidate_bits
) {
    auto const words = candidate_bit_words(reference_count);
    auto const total = static_cast<size_t>(query_count) * words;
    for (auto item = static_cast<size_t>(blockIdx.x) * block_size + threadIdx.x; item < total;
         item += static_cast<size_t>(gridDim.x) * block_size) {
        auto const query_index = static_cast<uint32_t>(item / words);
        auto const first = static_cast<uint32_t>(item % words) * 32U;
        uint32_t bits = 0U;
        for (uint32_t bit = 0U; bit < 32U && first + bit < reference_count; ++bit) {
            auto const reference_id = first + bit;
            auto const selected =
                match_counts[static_cast<size_t>(query_index) * reference_count + reference_id] >=
                    minimum_matches &&
                (!all_to_all || query_id_offset + query_index < reference_id);
            bits |= static_cast<uint32_t>(selected) << bit;
        }
        candidate_bits[item] = bits;
    }
}

/// @brief Threads per @ref refine_batch_bitmap_kernel block.
/// Leaves register space for query-plane loads alongside cached reference planes.
constexpr uint32_t bitmap_refine_block_size = 768U;

/// @brief Shared bytes @ref refine_batch_bitmap_kernel may stage query planes in; one block
/// per SM fits on every supported architecture.
constexpr size_t bitmap_refine_shared_bytes = 96U * 1024U;

/// @brief Whether a query group's planes fit @ref bitmap_refine_shared_bytes.
template <size_t BucketCount>
constexpr bool bitmap_refine_staged = bitmap_refine_shared_bytes >= BucketCount * sizeof(uint16_t);

/// @brief Queries one @ref refine_batch_bitmap_kernel block refines together.
///
/// Each reference row is read once per group, so the group size divides the kernel's L2
/// traffic. Rows too large to stage in shared memory are read from global memory in groups of 8.
template <size_t BucketCount>
constexpr uint32_t bitmap_refine_group =
    bitmap_refine_staged<BucketCount>
        ? static_cast<uint32_t>(cuda::std::min<size_t>(
              24U,
              bitmap_refine_shared_bytes / (BucketCount * sizeof(uint16_t))
          ))
        : 8U;

/// @brief Shared bytes one @ref refine_batch_bitmap_kernel block stages.
template <size_t BucketCount>
constexpr size_t bitmap_refine_dynamic_bytes =
    bitmap_refine_staged<BucketCount>
        ? static_cast<size_t>(bitmap_refine_group<BucketCount>) * BucketCount * sizeof(uint16_t)
        : 0U;

/// @brief The cell a refinement-style block processes after @p cell: claimed from
/// @p cell_counter, a zeroed per-launch counter, so blocks balance uneven cells; without one,
/// grid-stride. Called by every thread of the block; @p claimed is the block's shared slot.
/// The first cell is `next_cell(cell_counter, blockIdx.x - gridDim.x, claimed)`.
__device__ __forceinline__ uint32_t
next_cell(uint32_t* cell_counter, uint32_t cell, uint32_t& claimed) {
    if (cell_counter == nullptr) {
        return cell + gridDim.x;
    }
    __syncthreads();
    if (threadIdx.x == 0U) {
        claimed = atomicAdd(cell_counter, 1U);
    }
    __syncthreads();
    return claimed;
}

/// @brief Pairs @ref refine_batch_bitmap_kernel compares.
enum class refine_candidates {
    /// Exactly the pairs set in a pass bitmap.
    bitmap,
    /// Every pair of the layout, or with pass bits requested, those meeting the threshold.
    all,
    /// Every pair of the layout whose reference is set in one reference-wide bitmap.
    reference_mask,
};

/// @brief Position of the pair (@p query_index of the tile, @p reference_id) in a tile's
/// results: rows of `reference_count` references per query, or for all-to-all tiles the packed
/// strict upper triangle of pairs with `query_id < reference_id`, where query IDs start at
/// @p query_id_offset.
template <bool UpperTriangle>
[[nodiscard]] __host__ __device__ inline uint64_t batch_result_slot(
    uint32_t query_index,
    uint32_t reference_id,
    uint32_t query_id_offset,
    uint32_t reference_count
) noexcept {
    if constexpr (UpperTriangle) {
        auto const query = static_cast<uint64_t>(query_index);
        auto const preceding = query * (reference_count - query_id_offset - 1U) -
                               query * (query == 0U ? 0U : query - 1U) / 2U;
        return preceding + reference_id - (query_id_offset + query_index) - 1U;
    } else {
        return static_cast<uint64_t>(query_index) * reference_count + reference_id;
    }
}

/// @brief Popcount of one pass-bit word.
struct word_popcount {
    __host__ __device__ uint32_t operator()(uint32_t word) const noexcept {
        return static_cast<uint32_t>(cuda::std::popcount(word));
    }
};

/// @brief Packs a threshold tile's passing results in order.
///
/// One warp per pass-bit word in `[word_begin, word_end)`; lane l handles reference
/// `32 * word + l`, so consecutive lanes read consecutive slots and write consecutive packed
/// positions. @p word_offsets is the exclusive scan of the word popcounts; results land at their
/// packed position minus @p packed_base.
template <bool UpperTriangle, typename SearchResult>
__global__ __launch_bounds__(block_size) void gather_passing_kernel(
    SearchResult const* results,
    uint32_t const* match_counts,
    uint32_t const* pass_bits,
    uint32_t const* word_offsets,
    uint32_t first_query_id,
    uint32_t reference_count,
    size_t word_begin,
    size_t word_end,
    uint32_t packed_base,
    SearchResult* packed,
    uint32_t* packed_match_counts
) {
    constexpr uint32_t warp_width = 32;
    auto const words_per_query = candidate_bit_words(reference_count);
    auto const lane = static_cast<uint32_t>(threadIdx.x) % warp_width;
    for (auto w =
             word_begin + (static_cast<size_t>(blockIdx.x) * block_size + threadIdx.x) / warp_width;
         w < word_end;
         w += static_cast<size_t>(gridDim.x) * (block_size / warp_width)) {
        auto const bits = pass_bits[w];
        if (((bits >> lane) & 1U) == 0U) {
            continue;
        }
        auto const query = static_cast<uint32_t>(w / words_per_query);
        auto const reference = static_cast<uint32_t>(w % words_per_query) * warp_width + lane;
        auto const slot =
            batch_result_slot<UpperTriangle>(query, reference, first_query_id, reference_count);
        auto const position = word_offsets[w] - packed_base +
                              static_cast<uint32_t>(__popc(bits & ((1U << lane) - 1U)));
        packed[position] = results[slot];
        if (packed_match_counts != nullptr) {
            packed_match_counts[position] = match_counts[slot];
        }
    }
}

/// @brief Writes each query's selection signature for @ref refine_batch_bitmap_kernel groups:
/// key q is the MinHash of query q's selected references (a hash of their IDs, minimized) in
/// the high half and q in the low half. Queries selecting similar reference sets get equal
/// signatures with probability equal to the sets' Jaccard similarity, so sorting the keys by
/// their high halves clusters them. One warp per query.
__global__ __launch_bounds__(block_size) static void selection_signature_kernel(
    uint32_t const* candidate_bits,
    uint32_t query_count,
    uint32_t reference_count,
    uint64_t* keys
) {
    constexpr uint32_t warp_width = 32;
    auto const lane = static_cast<uint32_t>(threadIdx.x) % warp_width;
    auto const words = candidate_bit_words(reference_count);
    for (auto query = (static_cast<uint32_t>(blockIdx.x) * block_size + threadIdx.x) / warp_width;
         query < query_count;
         query += gridDim.x * (block_size / warp_width)) {
        auto signature = ~0U;
        for (auto word = lane; word < words; word += warp_width) {
            for (auto bits = candidate_bits[static_cast<size_t>(query) * words + word]; bits != 0U;
                 bits &= bits - 1U) {
                auto const reference = word * warp_width + static_cast<uint32_t>(__ffs(bits) - 1);
                signature =
                    cuda::std::min(signature, static_cast<uint32_t>(splitmix64(reference) >> 32U));
            }
        }
        signature = __reduce_min_sync(0xffffffffU, signature);
        if (lane == 0U) {
            keys[query] = static_cast<uint64_t>(signature) << 32U | query;
        }
    }
}

/// @brief Exactly compares query bit-planes with references, sharing each reference row across
/// a query group.
///
/// Block cells are (reference block, group of @ref bitmap_refine_group queries), reference-block
/// major so resident blocks share reference rows through L2. A block stages its group's query
/// bit-planes (@ref score_plane_index) in shared memory when they fit; warps then claim
/// 32-reference words and, for every reference any group query selects, compare the reference's
/// bit-planes with each selecting query. A row thus crosses L2 once per group rather than once
/// per pair.
///
/// Group g holds tile queries `g * QueryGroup ...` in order, or with @p query_order (staged
/// planes only) the queries in the low halves of those entries of @p query_order, so that
/// queries selecting the same references share a group and each row load serves more pairs.
///
/// Each written result lands at its @ref batch_result_slot. @p Candidates selects which pairs are
/// written: those in @p candidate_bits (one word per query and 32 references), or every pair of
/// the layout. With @p pass_bits (counting kernels only), only the selected pairs with at least
/// @p minimum_matches index matches are written and recorded in @p pass_bits, in the
/// candidate-bitmap layout. In bitmap mode @p pass_bits must be @p candidate_bits, whose failing
/// pairs' bits are cleared in place. With @p result_match_counts, results also get their index
/// match counts, recomputed from the planes.
template <
    size_t BucketCount,
    bool CountMatches,
    typename SearchResult,
    refine_candidates Candidates,
    bool UpperTriangle,
    uint32_t QueryGroup>
__global__ __launch_bounds__(bitmap_refine_block_size) void refine_batch_bitmap_kernel(
    uint32_t const* query_planes,
    uint32_t query_id_offset,
    uint32_t query_count,
    uint32_t const* reference_planes,
    uint32_t reference_count,
    uint32_t indexed_bucket_count,
    uint16_t key_mask,
    uint32_t const* candidate_bits,
    SearchResult* results,
    uint32_t* result_match_counts,
    uint32_t minimum_matches,
    uint32_t* pass_bits,
    uint32_t* cell_counter,
    uint64_t const* query_order
) {
    constexpr uint32_t warp_width = 32;
    constexpr uint32_t group = QueryGroup;
    static_assert(group >= 1U && group <= warp_width);
    constexpr bool staged = bitmap_refine_staged<BucketCount>;
    constexpr uint32_t row_words = static_cast<uint32_t>(BucketCount / 2U);
    constexpr uint32_t groups_per_row = static_cast<uint32_t>(BucketCount / warp_width);
    static_assert(groups_per_row % warp_width == 0U);
    constexpr uint32_t lane_groups = groups_per_row / warp_width;
    // Reference groups one lane keeps in registers across the selecting queries.
    constexpr uint32_t cached_groups = lane_groups < 2U ? lane_groups : 2U;
    constexpr uint32_t max_block_words = static_cast<uint32_t>(cuda::std::max<size_t>(
        1U, refine_block_row_bytes / (BucketCount * sizeof(uint16_t)) / warp_width
    ));
    // Bitmap selections cluster in a few words, so their warps claim quarter words.
    constexpr uint32_t word_slices = Candidates == refine_candidates::all ? 1U : 4U;
    extern __shared__ uint4 query_plane_storage[];
    __shared__ uint32_t next_word;
    __shared__ uint32_t claimed_cell;

    auto const lane = static_cast<uint32_t>(threadIdx.x) % warp_width;
    auto const words_per_query = candidate_bit_words(reference_count);
    auto const groups = (query_count + group - 1U) / group;
    // Fill the grid for small batches without spilling whole query groups into a second wave.
    auto const reference_blocks_per_wave = cuda::std::max(1U, gridDim.x / groups);
    auto const block_words = cuda::std::min(
        max_block_words,
        (words_per_query + reference_blocks_per_wave - 1U) / reference_blocks_per_wave
    );
    auto const reference_blocks = (words_per_query + block_words - 1U) / block_words;
    auto const cell_count = groups * reference_blocks;
    auto const indexed_groups = indexed_bucket_count / warp_width;
    auto const thresholded = CountMatches && pass_bits != nullptr;

    for (auto cell = next_cell(cell_counter, blockIdx.x - gridDim.x, claimed_cell);
         cell < cell_count;
         cell = next_cell(cell_counter, cell, claimed_cell)) {
        auto const first_query = (cell % groups) * group;
        auto const group_size = cuda::std::min(group, query_count - first_query);
        auto const first_word = (cell / groups) * block_words;
        auto const last_word = cuda::std::min(first_word + block_words, words_per_query);
        // Tile query index of group member m: with @p query_order, the low half of its entry.
        auto const member_query = [&](uint32_t m) {
            return query_order == nullptr ? first_query + m
                                          : static_cast<uint32_t>(query_order[first_query + m]);
        };
        auto const my_query = lane < group_size ? member_query(lane) : first_query;
        auto const* group_planes = reinterpret_cast<uint4 const*>(
            query_planes + static_cast<size_t>(first_query) * row_words
        );

        __syncthreads();
        if (threadIdx.x == 0U) {
            next_word = first_word * word_slices;
        }
        if constexpr (staged) {
            constexpr uint32_t member_vectors = row_words / 4U;
            for (auto i = static_cast<uint32_t>(threadIdx.x); i < group_size * member_vectors;
                 i += bitmap_refine_block_size) {
                query_plane_storage[i] = reinterpret_cast<uint4 const*>(
                    query_planes + static_cast<size_t>(member_query(i / member_vectors)) * row_words
                )[i % member_vectors];
            }
        }
        __syncthreads();
        uint4 const* const member_planes = staged ? query_plane_storage : group_planes;

        // Words carry uneven candidate counts, so warps claim them one slice at a time.
        for (;;) {
            uint32_t claim = 0U;
            if (lane == 0U) {
                claim = atomicAdd(&next_word, 1U);
            }
            claim = __shfl_sync(0xffffffffU, claim, 0);
            auto const word = claim / word_slices;
            if (word >= last_word) {
                break;
            }
            // Lane m holds query m's selected references in this slice of the word.
            uint32_t bits = 0U;
            if (lane < group_size) {
                constexpr uint32_t slice_width = warp_width / word_slices;
                constexpr uint32_t slice_mask =
                    slice_width == warp_width ? ~0U : (1U << slice_width) - 1U;
                if constexpr (Candidates == refine_candidates::bitmap) {
                    bits = candidate_bits[static_cast<size_t>(my_query) * words_per_query + word] &
                           (slice_mask << (claim % word_slices * slice_width));
                } else {
                    auto const first_reference = word * warp_width;
                    auto const left = reference_count - first_reference;
                    bits = left >= warp_width ? ~0U : (1U << left) - 1U;
                    if constexpr (UpperTriangle) {
                        auto const query_id = query_id_offset + my_query;
                        if (first_reference <= query_id) {
                            auto const skipped = query_id - first_reference + 1U;
                            bits &= skipped >= warp_width ? 0U : ~0U << skipped;
                        }
                    }
                    if constexpr (Candidates == refine_candidates::reference_mask) {
                        bits &= candidate_bits[word] &
                                (slice_mask << (claim % word_slices * slice_width));
                    }
                }
            }
            // Lane m collects query m's passing references in this word.
            uint32_t passed = 0U;
            auto remaining = __reduce_or_sync(0xffffffffU, bits);
            while (remaining != 0U) {
                auto const bit = static_cast<uint32_t>(__ffs(remaining) - 1);
                remaining &= remaining - 1U;
                auto const reference_id = word * warp_width + bit;
                auto const selected = __ballot_sync(0xffffffffU, ((bits >> bit) & 1U) != 0U);
                auto const* reference = reinterpret_cast<uint4 const*>(
                    reference_planes + static_cast<size_t>(reference_id) * row_words
                );
                uint32_t r[cached_groups][score_planes];
                uint32_t r_any[cached_groups];
                uint32_t r_low_any[cached_groups];
                _Pragma("unroll")
                for (uint32_t j = 0; j < cached_groups; ++j) {
                    load_plane_group(reference, j * warp_width + lane, r[j]);
                    r_any[j] = plane_any(r[j]);
                    r_low_any[j] = CountMatches ? plane_low_any(r[j]) : 0U;
                }
                uint32_t reference_nonzero = ~0U;
                _Pragma("unroll")
                for (uint32_t j = 0; j < cached_groups; ++j) {
                    reference_nonzero &= r_any[j];
                }
                auto const reference_full = cached_groups == lane_groups &&
                                            __all_sync(0xffffffffU, reference_nonzero == ~0U);
                // Lane m keeps member m's totals, so the selecting lanes store their records
                // together once the reference is done.
                uint32_t mine_lower = 0U;
                uint32_t mine_equal = 0U;
                uint32_t mine_empty = 0U;
                uint32_t mine_matches = 0U;
                auto const compare_members = [&](auto full) {
                    auto const compare_member = [&](uint32_t m) {
                        auto const* member =
                            member_planes + static_cast<size_t>(m) * (row_words / 4U);
                        plane_counts counts{};
                        auto const compare = [&](uint32_t bucket_group,
                                                 uint32_t const(&rg)[score_planes],
                                                 uint32_t rg_any,
                                                 uint32_t rg_low_any) {
                            uint32_t q[score_planes];
                            load_plane_group(member, bucket_group, q);
                            compare_plane_group<CountMatches, decltype(full)::value>(
                                q,
                                rg,
                                rg_any,
                                rg_low_any,
                                key_mask,
                                bucket_group < indexed_groups ? ~0U : 0U,
                                counts
                            );
                        };
                        _Pragma("unroll")
                        for (uint32_t j = 0; j < cached_groups; ++j) {
                            compare(j * warp_width + lane, r[j], r_any[j], r_low_any[j]);
                        }
                        for (uint32_t j = cached_groups; j < lane_groups; ++j) {
                            uint32_t rg[score_planes];
                            load_plane_group(reference, j * warp_width + lane, rg);
                            compare(
                                j * warp_width + lane,
                                rg,
                                plane_any(rg),
                                CountMatches ? plane_low_any(rg) : 0U
                            );
                        }
                        auto const total = counts.reduce<BucketCount, decltype(full)::value>();
                        uint32_t counts_matches_total = 0U;
                        if constexpr (CountMatches) {
                            counts_matches_total = __reduce_add_sync(0xffffffffU, counts.matches);
                        }
                        if (lane == m) {
                            mine_lower = total.lower;
                            mine_equal = total.equal;
                            mine_empty = total.both_empty;
                            mine_matches = counts_matches_total;
                        }
                    };
                    if constexpr (Candidates != refine_candidates::bitmap) {
                        // Layout and reference-mask selections are prefixes of the query group.
                        auto const member_count = static_cast<uint32_t>(__popc(selected));
                        for (uint32_t m = 0U; m < member_count; ++m) {
                            compare_member(m);
                        }
                    } else {
                        for (auto members = selected; members != 0U; members &= members - 1U) {
                            compare_member(static_cast<uint32_t>(__ffs(members) - 1));
                        }
                    }
                };
                if (reference_full) {
                    compare_members(cuda::std::true_type{});
                } else {
                    compare_members(cuda::std::false_type{});
                }
                auto const keep = !thresholded || mine_matches >= minimum_matches;
                if (((selected >> lane) & 1U) != 0U && keep) {
                    auto const slot = batch_result_slot<UpperTriangle>(
                        my_query, reference_id, query_id_offset, reference_count
                    );
                    // The slot implies the pair; only its counts are stored.
                    results[slot] = SearchResult::pack(
                        mine_lower,
                        mine_equal,
                        static_cast<uint32_t>(BucketCount) - mine_lower - mine_equal - mine_empty
                    );
                    if constexpr (CountMatches) {
                        if (result_match_counts != nullptr) {
                            result_match_counts[slot] = mine_matches;
                        }
                    }
                    passed |= 1U << bit;
                }
            }
            if constexpr (Candidates == refine_candidates::bitmap) {
                // Slices of a word may run concurrently, so failing pairs are cleared in place.
                if (thresholded && (bits & ~passed) != 0U) {
                    atomicAnd(
                        &pass_bits[static_cast<size_t>(my_query) * words_per_query + word],
                        ~(bits & ~passed)
                    );
                }
            } else if constexpr (Candidates == refine_candidates::all) {
                if (thresholded && lane < group_size) {
                    pass_bits[static_cast<size_t>(my_query) * words_per_query + word] = passed;
                }
            }
        }
    }
}

/// @brief Threads per @ref match_filter_kernel block. Four references' low planes stay in
/// registers, so a block holds fewer threads than a refine block.
constexpr uint32_t match_filter_block_size = 512U;

/// @brief Whether @ref match_filter_kernel supports @p BucketCount: a lane caches at most two
/// plane groups of each of its references.
template <size_t BucketCount>
constexpr bool match_filter_supported =
    bitmap_refine_staged<BucketCount> && BucketCount % 1024U == 0U && BucketCount <= 2048U;

/// @brief Loads plane chunk @p chunk (planes `4 chunk ... 4 chunk + 3`) of group @p group of a
/// bit-plane row (@ref score_plane_index layout).
__device__ __forceinline__ void
load_plane_chunk(uint4 const* row, uint32_t group, uint32_t chunk, uint32_t (&words)[4]) {
    auto const value = row[((group / 32U) * 4U + chunk) * 32U + group % 32U];
    words[0] = value.x;
    words[1] = value.y;
    words[2] = value.z;
    words[3] = value.w;
}

/// @brief Low planes @ref match_filter_kernel compares.
constexpr uint32_t match_filter_planes = 12U;

/// @brief Records in @p candidate_bits (candidate-bitmap layout) the pairs of the layout whose
/// scores agree in their low @ref match_filter_planes planes in at least @p minimum_matches
/// buckets.
///
/// Every validated key mask covers those planes, so the count bounds a pair's index matches from
/// above and the candidates include every pair meeting the threshold. On RefSeq sketches the
/// bound passes barely more pairs than the exact count; a thresholded bitmap refinement then
/// counts matches exactly.
///
/// Cells, query staging, and word claiming follow @ref refine_batch_bitmap_kernel. A warp takes
/// a claimed word's references four at a time and keeps their low planes in registers, so each
/// query plane chunk read from shared memory serves four pairs.
template <size_t BucketCount, bool UpperTriangle, uint32_t QueryGroup>
__global__ __launch_bounds__(match_filter_block_size, 1) void match_filter_kernel(
    uint32_t const* query_planes,
    uint32_t query_id_offset,
    uint32_t query_count,
    uint32_t const* reference_planes,
    uint32_t reference_count,
    uint32_t indexed_bucket_count,
    uint32_t minimum_matches,
    uint32_t* candidate_bits,
    uint32_t* cell_counter
) {
    static_assert(match_filter_supported<BucketCount>);
    constexpr uint32_t warp_width = 32;
    constexpr uint32_t group = QueryGroup;
    static_assert(group >= 1U && group <= warp_width);
    constexpr uint32_t row_words = static_cast<uint32_t>(BucketCount / 2U);
    constexpr uint32_t lane_groups = static_cast<uint32_t>(BucketCount / warp_width / warp_width);
    constexpr uint32_t max_block_words = static_cast<uint32_t>(cuda::std::max<size_t>(
        1U, refine_block_row_bytes / (BucketCount * sizeof(uint16_t)) / warp_width
    ));
    extern __shared__ uint4 filter_query_planes[];
    __shared__ uint32_t next_word;
    __shared__ uint32_t claimed_cell;

    auto const lane = static_cast<uint32_t>(threadIdx.x) % warp_width;
    auto const words_per_query = candidate_bit_words(reference_count);
    auto const groups = (query_count + group - 1U) / group;
    auto const reference_blocks_per_wave = cuda::std::max(1U, gridDim.x / groups);
    auto const block_words = cuda::std::min(
        max_block_words,
        (words_per_query + reference_blocks_per_wave - 1U) / reference_blocks_per_wave
    );
    auto const reference_blocks = (words_per_query + block_words - 1U) / block_words;
    auto const cell_count = groups * reference_blocks;
    auto const indexed_groups = indexed_bucket_count / warp_width;
    // Unindexed buckets are seeded as differing, so they never count toward the bound.
    uint32_t excluded_lanes[lane_groups];
    _Pragma("unroll")
    for (uint32_t j = 0; j < lane_groups; ++j) {
        excluded_lanes[j] = j * warp_width + lane < indexed_groups ? 0U : ~0U;
    }

    for (auto cell = next_cell(cell_counter, blockIdx.x - gridDim.x, claimed_cell);
         cell < cell_count;
         cell = next_cell(cell_counter, cell, claimed_cell)) {
        auto const first_query = (cell % groups) * group;
        auto const group_size = cuda::std::min(group, query_count - first_query);
        auto const first_word = (cell / groups) * block_words;
        auto const last_word = cuda::std::min(first_word + block_words, words_per_query);
        auto const* group_planes = reinterpret_cast<uint4 const*>(
            query_planes + static_cast<size_t>(first_query) * row_words
        );

        __syncthreads();
        if (threadIdx.x == 0U) {
            next_word = first_word;
        }
        for (auto i = static_cast<uint32_t>(threadIdx.x); i < group_size * (row_words / 4U);
             i += match_filter_block_size) {
            filter_query_planes[i] = group_planes[i];
        }
        __syncthreads();

        for (;;) {
            uint32_t word = 0U;
            if (lane == 0U) {
                word = atomicAdd(&next_word, 1U);
            }
            word = __shfl_sync(0xffffffffU, word, 0);
            if (word >= last_word) {
                break;
            }
            // Lane m holds query m's selected references in this word.
            uint32_t bits = 0U;
            if (lane < group_size) {
                auto const first_reference = word * warp_width;
                auto const left = reference_count - first_reference;
                bits = left >= warp_width ? ~0U : (1U << left) - 1U;
                if constexpr (UpperTriangle) {
                    auto const query_id = query_id_offset + first_query + lane;
                    if (first_reference <= query_id) {
                        auto const skipped = query_id - first_reference + 1U;
                        bits &= skipped >= warp_width ? 0U : ~0U << skipped;
                    }
                }
            }
            uint32_t passed = 0U;
            auto remaining = __reduce_or_sync(0xffffffffU, bits);
            constexpr uint32_t refs = 4U;
            constexpr uint32_t bound_chunks = match_filter_planes / 4U;
            constexpr uint32_t bound_planes = match_filter_planes;
            while (remaining != 0U) {
                uint32_t bit[refs];
                bool has[refs];
                uint32_t last_bit = 0U;
                _Pragma("unroll")
                for (uint32_t i = 0; i < refs; ++i) {
                    has[i] = remaining != 0U;
                    bit[i] = has[i] ? static_cast<uint32_t>(__ffs(remaining) - 1) : bit[0];
                    last_bit = has[i] ? bit[i] : last_bit;
                    remaining &= remaining - 1U;
                }
                uint4 const* reference[refs];
                uint32_t selecting[refs];
                uint32_t r[refs][lane_groups][bound_planes];
                _Pragma("unroll")
                for (uint32_t i = 0; i < refs; ++i) {
                    reference[i] = reinterpret_cast<uint4 const*>(
                        reference_planes +
                        static_cast<size_t>(word * warp_width + bit[i]) * row_words
                    );
                    selecting[i] = has[i] ? __ballot_sync(0xffffffffU, (bits >> bit[i]) & 1U) : 0U;
                    _Pragma("unroll")
                    for (uint32_t j = 0; j < lane_groups; ++j) {
                        _Pragma("unroll")
                        for (uint32_t c = 0; c < bound_chunks; ++c) {
                            uint32_t words[4];
                            load_plane_chunk(reference[i], j * warp_width + lane, c, words);
                            _Pragma("unroll")
                            for (uint32_t w = 0; w < 4U; ++w) {
                                r[i][j][4U * c + w] = words[w];
                            }
                        }
                    }
                }
                // Selections are prefixes of the query group, growing with the reference ID.
                auto const members = static_cast<uint32_t>(
                    __popc(__ballot_sync(0xffffffffU, (bits >> last_bit) & 1U))
                );
                for (uint32_t m = 0U; m < members; ++m) {
                    auto const* member =
                        filter_query_planes + static_cast<size_t>(m) * (row_words / 4U);
                    uint32_t mismatches[refs] = {};
                    _Pragma("unroll")
                    for (uint32_t j = 0; j < lane_groups; ++j) {
                        uint32_t differ[refs];
                        _Pragma("unroll")
                        for (uint32_t i = 0; i < refs; ++i) {
                            differ[i] = excluded_lanes[j];
                        }
                        _Pragma("unroll")
                        for (uint32_t c = 0; c < bound_chunks; ++c) {
                            uint32_t q[4];
                            load_plane_chunk(member, j * warp_width + lane, c, q);
                            _Pragma("unroll")
                            for (uint32_t i = 0; i < refs; ++i) {
                                _Pragma("unroll")
                                for (uint32_t w = 0; w < 4U; ++w) {
                                    differ[i] |= q[w] ^ r[i][j][4U * c + w];
                                }
                            }
                        }
                        _Pragma("unroll")
                        for (uint32_t i = 0; i < refs; ++i) {
                            mismatches[i] += static_cast<uint32_t>(__popc(differ[i]));
                        }
                    }
                    _Pragma("unroll")
                    for (uint32_t i = 0; i < refs; ++i) {
                        auto const total_bound = static_cast<uint32_t>(BucketCount) -
                                                 __reduce_add_sync(0xffffffffU, mismatches[i]);
                        if (lane == m && ((selecting[i] >> m) & 1U) != 0U &&
                            total_bound >= minimum_matches) {
                            passed |= 1U << bit[i];
                        }
                    }
                }
            }
            if (lane < group_size) {
                candidate_bits[static_cast<size_t>(first_query + lane) * words_per_query + word] =
                    passed;
            }
        }
    }
}

/// @brief Threads per @ref refine_all_pairs_kernel block. Two references' planes stay in
/// registers, so a block holds fewer threads than a refine block.
constexpr uint32_t all_pairs_block_size = 512U;

/// @brief Exactly compares every pair of the layout, as @ref refine_batch_bitmap_kernel does for
/// @ref refine_candidates::all without match counts.
///
/// Cells, query staging, and word claiming follow @ref refine_batch_bitmap_kernel. A warp takes
/// a claimed word's references two at a time and keeps both references' planes in registers, so
/// each query plane group read from shared memory serves two pairs, and the two comparisons give
/// every warp independent logic chains to interleave.
template <size_t BucketCount, typename SearchResult, bool UpperTriangle, uint32_t QueryGroup>
__global__ __launch_bounds__(all_pairs_block_size, 1) void refine_all_pairs_kernel(
    uint32_t const* query_planes,
    uint32_t query_id_offset,
    uint32_t query_count,
    uint32_t const* reference_planes,
    uint32_t reference_count,
    SearchResult* results,
    uint32_t* cell_counter
) {
    static_assert(match_filter_supported<BucketCount>);
    constexpr uint32_t warp_width = 32;
    constexpr uint32_t group = QueryGroup;
    static_assert(group >= 1U && group <= warp_width);
    constexpr uint32_t row_words = static_cast<uint32_t>(BucketCount / 2U);
    constexpr uint32_t lane_groups = static_cast<uint32_t>(BucketCount / warp_width / warp_width);
    constexpr uint32_t max_block_words = static_cast<uint32_t>(cuda::std::max<size_t>(
        1U, refine_block_row_bytes / (BucketCount * sizeof(uint16_t)) / warp_width
    ));
    extern __shared__ uint4 all_pairs_query_planes[];
    __shared__ uint32_t next_word;
    __shared__ uint32_t claimed_cell;

    auto const lane = static_cast<uint32_t>(threadIdx.x) % warp_width;
    auto const words_per_query = candidate_bit_words(reference_count);
    auto const groups = (query_count + group - 1U) / group;
    auto const reference_blocks_per_wave = cuda::std::max(1U, gridDim.x / groups);
    auto const block_words = cuda::std::min(
        max_block_words,
        (words_per_query + reference_blocks_per_wave - 1U) / reference_blocks_per_wave
    );
    auto const reference_blocks = (words_per_query + block_words - 1U) / block_words;
    auto const cell_count = groups * reference_blocks;

    for (auto cell = next_cell(cell_counter, blockIdx.x - gridDim.x, claimed_cell);
         cell < cell_count;
         cell = next_cell(cell_counter, cell, claimed_cell)) {
        auto const first_query = (cell % groups) * group;
        auto const group_size = cuda::std::min(group, query_count - first_query);
        auto const first_word = (cell / groups) * block_words;
        auto const last_word = cuda::std::min(first_word + block_words, words_per_query);
        auto const* group_planes = reinterpret_cast<uint4 const*>(
            query_planes + static_cast<size_t>(first_query) * row_words
        );

        __syncthreads();
        if (threadIdx.x == 0U) {
            next_word = first_word;
        }
        for (auto i = static_cast<uint32_t>(threadIdx.x); i < group_size * (row_words / 4U);
             i += all_pairs_block_size) {
            all_pairs_query_planes[i] = group_planes[i];
        }
        __syncthreads();

        for (;;) {
            uint32_t word = 0U;
            if (lane == 0U) {
                word = atomicAdd(&next_word, 1U);
            }
            word = __shfl_sync(0xffffffffU, word, 0);
            if (word >= last_word) {
                break;
            }
            // Lane m holds query m's selected references in this word.
            uint32_t bits = 0U;
            if (lane < group_size) {
                auto const first_reference = word * warp_width;
                auto const left = reference_count - first_reference;
                bits = left >= warp_width ? ~0U : (1U << left) - 1U;
                if constexpr (UpperTriangle) {
                    auto const query_id = query_id_offset + first_query + lane;
                    if (first_reference <= query_id) {
                        auto const skipped = query_id - first_reference + 1U;
                        bits &= skipped >= warp_width ? 0U : ~0U << skipped;
                    }
                }
            }
            auto remaining = __reduce_or_sync(0xffffffffU, bits);
            while (remaining != 0U) {
                auto const bit_a = static_cast<uint32_t>(__ffs(remaining) - 1);
                remaining &= remaining - 1U;
                auto const has_b = remaining != 0U;
                auto const bit_b = has_b ? static_cast<uint32_t>(__ffs(remaining) - 1) : bit_a;
                remaining &= remaining - 1U;
                auto const reference_a = word * warp_width + bit_a;
                auto const reference_b = word * warp_width + bit_b;
                auto const* planes_a = reinterpret_cast<uint4 const*>(
                    reference_planes + static_cast<size_t>(reference_a) * row_words
                );
                auto const* planes_b = reinterpret_cast<uint4 const*>(
                    reference_planes + static_cast<size_t>(reference_b) * row_words
                );
                uint32_t a[lane_groups][score_planes];
                uint32_t b[lane_groups][score_planes];
                uint32_t a_any[lane_groups];
                uint32_t b_any[lane_groups];
                uint32_t nonzero = ~0U;
                _Pragma("unroll")
                for (uint32_t j = 0; j < lane_groups; ++j) {
                    load_plane_group(planes_a, j * warp_width + lane, a[j]);
                    load_plane_group(planes_b, j * warp_width + lane, b[j]);
                    a_any[j] = plane_any(a[j]);
                    b_any[j] = plane_any(b[j]);
                    nonzero &= a_any[j] & b_any[j];
                }
                // Selections are prefixes of the query group, and B's includes A's.
                auto const selected_a = __ballot_sync(0xffffffffU, (bits >> bit_a) & 1U);
                auto const selected_b = __ballot_sync(0xffffffffU, (bits >> bit_b) & 1U);
                auto const members = static_cast<uint32_t>(__popc(selected_b));
                // Lane m keeps member m's totals for both references.
                pairwise_counts mine_a{};
                pairwise_counts mine_b{};
                auto const compare = [&](auto full) {
                    for (uint32_t m = 0U; m < members; ++m) {
                        auto const* member =
                            all_pairs_query_planes + static_cast<size_t>(m) * (row_words / 4U);
                        plane_counts counts_a{};
                        plane_counts counts_b{};
                        _Pragma("unroll")
                        for (uint32_t j = 0; j < lane_groups; ++j) {
                            uint32_t q[score_planes];
                            load_plane_group(member, j * warp_width + lane, q);
                            compare_plane_group<false, decltype(full)::value>(
                                q, a[j], a_any[j], 0U, 0U, 0U, counts_a
                            );
                            compare_plane_group<false, decltype(full)::value>(
                                q, b[j], b_any[j], 0U, 0U, 0U, counts_b
                            );
                        }
                        auto const total_a = counts_a.reduce<BucketCount, decltype(full)::value>();
                        auto const total_b = counts_b.reduce<BucketCount, decltype(full)::value>();
                        if (lane == m) {
                            mine_a = total_a;
                            mine_b = total_b;
                        }
                    }
                };
                if (__all_sync(0xffffffffU, nonzero == ~0U)) {
                    compare(cuda::std::true_type{});
                } else {
                    compare(cuda::std::false_type{});
                }
                // The slot implies the pair; only its counts are stored.
                if (((selected_a >> lane) & 1U) != 0U) {
                    results[batch_result_slot<UpperTriangle>(
                        first_query + lane, reference_a, query_id_offset, reference_count
                    )] = SearchResult::pack(mine_a.lower, mine_a.equal, mine_a.higher);
                }
                if (has_b && ((selected_b >> lane) & 1U) != 0U) {
                    results[batch_result_slot<UpperTriangle>(
                        first_query + lane, reference_b, query_id_offset, reference_count
                    )] = SearchResult::pack(mine_b.lower, mine_b.equal, mine_b.higher);
                }
            }
        }
    }
}

/// @brief Per-thread accumulator feeding the CUB block reduction for cardinality.
struct cardinality_payload {
    uint32_t empty{};
    float restored{};

    /// @brief Sums two accumulators for the block reduction.
    friend __device__ cardinality_payload
    operator+(cardinality_payload left, cardinality_payload right) noexcept {
        return {left.empty + right.empty, left.restored + right.restored};
    }
};

/**
 * @brief Computes the cardinality of a single constructed sketch.
 *
 * @p empty_out receives the empty-register count; @p estimate_out receives the estimate.
 */
template <size_t BucketCount, typename Layout = default_register_layout>
__global__ void
cardinality_kernel(uint32_t const* registers, uint64_t* empty_out, double* estimate_out) {
    using block_reduce = cub::BlockReduce<cardinality_payload, block_size>;
    __shared__ typename block_reduce::TempStorage storage;
    cardinality_payload local{};
    for (auto bucket = static_cast<size_t>(threadIdx.x); bucket < BucketCount;
         bucket += blockDim.x) {
        auto const stored = static_cast<uint16_t>(__ldcs(&registers[bucket]));
        if (stored == 0U) {
            ++local.empty;
        } else {
            local.restored += static_cast<float>(restore_midpoint<Layout>(stored));
        }
    }
    auto const total = block_reduce(storage).Sum(local);
    if (threadIdx.x == 0) {
        *empty_out = total.empty;
        *estimate_out = static_cast<double>(cardinality_f32(
            static_cast<float>(BucketCount), static_cast<float>(total.empty), total.restored
        ));
    }
}

/// @brief Computes BBTools and paper-style HybridDDL estimates in one register scan.
template <size_t BucketCount, typename Layout = default_register_layout>
__global__ void hybrid_cardinality_kernel(
    uint32_t const* const registers,
    hybrid_cardinality_estimates* const estimates
) {
    __shared__ uint32_t bins[nlz_bins];
    using block_histogram =
        cub::BlockHistogram<uint32_t, block_size, 1, nlz_bins, cub::BLOCK_HISTO_ATOMIC>;
    using block_reduce = cub::BlockReduce<float, block_size>;
    __shared__ typename block_reduce::TempStorage storage;
    block_histogram histogram;
    histogram.InitHistogram(bins);
    float local_restored = 0.0f;
    __syncthreads();

    for (auto bucket = static_cast<size_t>(threadIdx.x); bucket < BucketCount;
         bucket += blockDim.x) {
        auto const stored = static_cast<uint16_t>(__ldcs(&registers[bucket]));
        uint32_t bin[1]{
            stored == 0U ? 0U : static_cast<uint32_t>(stored >> Layout::mantissa_bits) + 1U
        };
        histogram.Composite(bin, bins);
        if (stored != 0U) {
            local_restored += static_cast<float>(restore<Layout>(stored));
        }
    }
    __syncthreads();
    auto const restored = block_reduce(storage).Sum(local_restored);
    if (threadIdx.x == 0) {
        *estimates = hybrid_estimates_f32(bins, static_cast<float>(BucketCount), restored);
    }
}

/// @brief Computes one HybridDDL variant estimate for estimator comparisons.
template <size_t BucketCount, hybrid_variant Variant, typename Layout = default_register_layout>
__global__ void
hybrid_cardinality_variant_kernel(uint32_t const* const registers, double* const estimate) {
    __shared__ uint32_t bins[nlz_bins];
    using block_histogram =
        cub::BlockHistogram<uint32_t, block_size, 1, nlz_bins, cub::BLOCK_HISTO_ATOMIC>;
    using block_reduce = cub::BlockReduce<float, block_size>;
    __shared__ typename block_reduce::TempStorage storage;
    block_histogram histogram;
    histogram.InitHistogram(bins);
    float local_restored = 0.0f;
    __syncthreads();

    for (auto bucket = static_cast<size_t>(threadIdx.x); bucket < BucketCount;
         bucket += blockDim.x) {
        auto const stored = static_cast<uint16_t>(__ldcs(&registers[bucket]));
        uint32_t bin[1]{
            stored == 0U ? 0U : static_cast<uint32_t>(stored >> Layout::mantissa_bits) + 1U
        };
        histogram.Composite(bin, bins);
        if (stored != 0U) {
            local_restored += static_cast<float>(restore<Layout>(stored));
        }
    }
    __syncthreads();
    auto const restored = block_reduce(storage).Sum(local_restored);
    if (threadIdx.x == 0) {
        auto const estimates =
            hybrid_estimates_f32(bins, static_cast<float>(BucketCount), restored);
        if constexpr (Variant == hybrid_variant::bbtools) {
            *estimate = estimates.bbtools;
        } else {
            *estimate = estimates.paper;
        }
    }
}

/// @brief Computes the cardinality of every stored sketch in one launch.
///
/// One block reduces one row of the row-major store of `BucketCount` registers per sketch.
template <size_t BucketCount, typename Layout = default_register_layout>
__global__ void batch_cardinality_kernel(
    uint32_t const* const registers,
    uint32_t const row_count,
    uint64_t* const empty_out,
    double* const estimates_out
) {
    auto const row = static_cast<size_t>(blockIdx.x);
    if (row >= row_count) {
        return;
    }
    using block_reduce = cub::BlockReduce<cardinality_payload, block_size>;
    __shared__ typename block_reduce::TempStorage storage;
    auto const* const mine = registers + row * BucketCount;
    cardinality_payload local{};
    for (auto bucket = static_cast<size_t>(threadIdx.x); bucket < BucketCount;
         bucket += blockDim.x) {
        auto const stored = static_cast<uint16_t>(__ldcs(&mine[bucket]));
        if (stored == 0U) {
            ++local.empty;
        } else {
            local.restored += static_cast<float>(restore_midpoint<Layout>(stored));
        }
    }
    auto const total = block_reduce(storage).Sum(local);
    if (threadIdx.x == 0) {
        empty_out[row] = total.empty;
        estimates_out[row] = static_cast<double>(cardinality_f32(
            static_cast<float>(BucketCount), static_cast<float>(total.empty), total.restored
        ));
    }
}

/// @brief Copies the winning score of every register into compact row-major scores.
template <size_t BucketCount>
__global__ void batch_scores_kernel(
    uint32_t const* const registers,
    uint32_t const row_count,
    uint16_t* const scores_out
) {
    auto const row = static_cast<size_t>(blockIdx.x);
    if (row >= row_count) {
        return;
    }
    auto const* const mine = registers + row * BucketCount;
    auto* const scores = scores_out + row * BucketCount;
    for (auto bucket = static_cast<size_t>(threadIdx.x); bucket < BucketCount;
         bucket += blockDim.x) {
        scores[bucket] = static_cast<uint16_t>(__ldcs(&mine[bucket]));
    }
}

}  // namespace cuddl::detail
