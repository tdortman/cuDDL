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
    restored[threadIdx.x] = 0.0;
    __syncthreads();

    for (auto bucket = static_cast<size_t>(threadIdx.x); bucket < BucketCount;
         bucket += blockDim.x) {
        auto const stored = winner(registers[bucket]);
        if (stored == 0U) {
            atomicAdd(bins, 1U);
        } else {
            atomicAdd(bins + static_cast<uint32_t>(stored >> mantissa_bits) + 1U, 1U);
            restored[threadIdx.x] += restore(stored);
        }
    }
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
    restored[threadIdx.x] = 0.0;
    __syncthreads();

    for (auto bucket = static_cast<size_t>(threadIdx.x); bucket < BucketCount;
         bucket += blockDim.x) {
        auto const stored = winner(registers[bucket]);
        if (stored == 0U) {
            atomicAdd(bins, 1U);
        } else {
            atomicAdd(bins + static_cast<uint32_t>(stored >> mantissa_bits) + 1U, 1U);
            restored[threadIdx.x] += restore(stored);
        }
    }
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
