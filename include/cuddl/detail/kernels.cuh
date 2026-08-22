#pragma once

#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>

#include <cuddl/detail/cardinality.cuh>
#include <cuddl/detail/comparison.cuh>
#include <cuddl/detail/hash.cuh>
#include <cuddl/detail/hybrid_cardinality.cuh>
#include <cuddl/detail/register.cuh>
#include <cuddl/device_span.cuh>
#include <cuddl/pairwise_counts.cuh>

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
    device_span<uint64_t const> input,
    device_span<uint32_t> registers,
    uint32_t& saturation
) {
    auto const index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    auto const stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (auto offset = index; offset < input.size(); offset += stride * 2U) {
        _Pragma("unroll")
        for (uint32_t item = 0; item < 2U; ++item) {
            auto const current = offset + stride * item;
            if (current < input.size()) {
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
    device_span<uint32_t const> left,
    device_span<uint32_t const> right,
    pairwise_summary& output
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
 * @brief Compares one compact query row with every row in a reference database.
 *
 * Each warp owns one reference row and writes exactly one stable-ID result.
 */
template <size_t BucketCount, typename ReferenceRow, typename SearchResult>
__global__ __launch_bounds__(block_size) void exhaustive_search_kernel(
    device_span<ReferenceRow const> rows,
    uint32_t reference_count,
    device_span<uint16_t const> query,
    SearchResult* results
) {
    constexpr uint32_t warp_width = 32;
    constexpr uint32_t warps_per_block = block_size / warp_width;
    using warp_reduce = cub::WarpReduce<pairwise_counts>;
    __shared__ typename warp_reduce::TempStorage storage[warps_per_block];

    auto const warp = static_cast<uint32_t>(threadIdx.x) / warp_width;
    auto const lane = static_cast<uint32_t>(threadIdx.x) % warp_width;
    auto const reference_id =
        static_cast<uint32_t>(blockIdx.x) * warps_per_block + warp;
    if (reference_id >= reference_count) {
        return;
    }

    pairwise_counts local{};
    auto const row_offset = static_cast<size_t>(reference_id) * BucketCount;
    for (auto bucket = static_cast<size_t>(lane); bucket < BucketCount; bucket += warp_width) {
        classify(local, query[bucket], reference_score(rows[row_offset + bucket]));
    }
    auto const total = warp_reduce(storage[warp]).Sum(local);
    if (lane == 0U) {
        results[reference_id].reference_id = reference_id;
        results[reference_id].summary.counts = total;
        results[reference_id].summary.cardinality = 0.0;
    }
}

/// @brief Counts every non-empty reference row entry in its dense index cell.
template <size_t BucketCount, typename ReferenceRow>
__global__ void
count_index_cells_kernel(device_span<ReferenceRow const> rows, uint32_t* cell_counts) {
    constexpr uint64_t score_count = uint64_t{1} << 16U;
    auto offset = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    auto const stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (; offset < rows.size(); offset += stride) {
        auto const score = reference_score(rows[offset]);
        if (score == 0U) {
            continue;
        }
        auto const bucket = offset % BucketCount;
        auto const cell = bucket * score_count + score;
        atomicAdd(&cell_counts[cell], 1U);
    }
}

/// @brief Scatters every non-empty reference row entry into its dense CSR posting range.
template <size_t BucketCount, typename ReferenceRow>
__global__ void scatter_index_postings_kernel(
    device_span<ReferenceRow const> rows,
    uint32_t* cursors,
    uint32_t* postings
) {
    constexpr uint64_t score_count = uint64_t{1} << 16U;
    auto offset = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    auto const stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (; offset < rows.size(); offset += stride) {
        auto const score = reference_score(rows[offset]);
        if (score == 0U) {
            continue;
        }
        auto const bucket = offset % BucketCount;
        auto const cell = bucket * score_count + score;
        auto const posting = atomicAdd(&cursors[cell], 1U);
        postings[posting] = static_cast<uint32_t>(offset / BucketCount);
    }
}

/// @brief Counts the query's non-empty posting matches for every reference.
template <size_t BucketCount>
__global__ void count_index_matches_kernel(
    device_span<uint16_t const> query,
    device_span<uint32_t const> offsets,
    device_span<uint32_t const> postings,
    uint32_t* match_counts
) {
    constexpr uint64_t score_count = uint64_t{1} << 16U;
    auto const bucket = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (bucket >= BucketCount) {
        return;
    }
    auto const score = query[bucket];
    if (score == 0U) {
        return;
    }
    auto const cell = bucket * score_count + score;
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
    device_span<ReferenceRow const> rows,
    device_span<uint16_t const> query,
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
