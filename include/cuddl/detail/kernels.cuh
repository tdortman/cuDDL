#pragma once

#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>

#include <cuddl/detail/cardinality.cuh>
#include <cuddl/detail/comparison.cuh>
#include <cuddl/detail/hash.cuh>
#include <cuddl/detail/register.cuh>
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

/// @brief Combines two per-thread payloads, summing only the fields selected by @p Mask.
template <uint32_t Mask>
__device__ summary_payload combine_payloads(summary_payload a, summary_payload const& b) noexcept {
    if constexpr ((Mask & summary_mask::pairwise) != 0U) {
        a.counts += b.counts;
    }
    if constexpr ((Mask & summary_mask::cardinality) != 0U) {
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
__global__ void add_kernel(
    uint64_t const* const first,
    size_t const count,
    uint32_t* const registers,
    uint32_t* const saturation
) {
    auto const stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (auto index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; index < count;
         index += stride) {
        auto const hash = hash_kmer(first[index]);
        update(registers + bucket_of<BucketCount>(hash), score(hash), saturation);
    }
}

/**
 * @brief Computes the raw pairwise summary of two constructed sketches.
 *
 * One CTA per pair. When the @ref summary_mask::pairwise field is selected it classifies every
 * bucket as lower/equal/higher/both-empty; when @ref summary_mask::cardinality is selected it
 * accumulates the query sketch's empty-register count and restored hash magnitudes and reduces
 * them through hybridDDL. Unselected fields are inert and reduce to zero.
 */
template <size_t BucketCount, uint32_t Mask>
__global__ void summary_kernel(
    uint32_t const* const left,
    uint32_t const* const right,
    pairwise_summary* const output
) {
    using block_reduce = cub::BlockReduce<summary_payload, block_size>;
    __shared__ typename block_reduce::TempStorage storage;

    summary_payload local{};
    for (auto bucket = static_cast<size_t>(threadIdx.x); bucket < BucketCount;
         bucket += block_size) {
        auto const left_reg = left[bucket];
        auto const right_reg = right[bucket];
        if constexpr ((Mask & summary_mask::pairwise) != 0U) {
            classify(local.counts, left_reg, right_reg);
        }
        if constexpr ((Mask & summary_mask::cardinality) != 0U) {
            if (winner(left_reg) == 0U) {
                ++local.empty;
            } else {
                local.restored_sum += restore(winner(left_reg));
            }
        }
    }
    auto const total = block_reduce(storage).Reduce(local, combine_payloads<Mask>);
    if (threadIdx.x == 0) {
        output->counts = total.counts;
        if constexpr ((Mask & summary_mask::cardinality) != 0U) {
            output->cardinality = hybrid_ddl(
                static_cast<double>(BucketCount),
                static_cast<double>(total.empty),
                total.restored_sum
            );
        }
    }
}

/**
 * @brief Computes the hybridDDL cardinality of a single constructed sketch.
 *
 * @p empty_out receives the empty-register count; @p estimate_out receives the hybridDDL estimate.
 */
template <size_t BucketCount>
__global__ void cardinality_kernel(
    uint32_t const* const registers,
    uint64_t* const empty_out,
    double* const estimate_out
) {
    using block_reduce = cub::BlockReduce<cardinality_payload, block_size>;
    __shared__ typename block_reduce::TempStorage storage;

    cardinality_payload local{};
    for (auto bucket = static_cast<size_t>(threadIdx.x); bucket < BucketCount;
         bucket += block_size) {
        auto const reg = registers[bucket];
        if (winner(reg) == 0U) {
            ++local.empty;
        } else {
            local.restored_sum += restore(winner(reg));
        }
    }
    auto const total = block_reduce(storage).Reduce(local, combine_cardinality);
    if (threadIdx.x == 0) {
        *empty_out = total.empty;
        *estimate_out = hybrid_ddl(
            static_cast<double>(BucketCount), static_cast<double>(total.empty), total.restored_sum
        );
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
