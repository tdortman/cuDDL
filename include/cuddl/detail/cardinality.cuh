#pragma once

#include <cuda/std/cmath>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

namespace cuddl::detail {

/// @brief `2^64`, the range of the 64-bit hash used by the sketch.
constexpr double hash_range = 0x1p64;

/**
 * @brief Estimates the number of distinct hashes from the minimum hash in each bucket.
 *
 * Each distinct hash lands in one of @p bucket_count buckets. A bucket therefore receives about
 * `cardinality / bucket_count` hashes. More hashes make its minimum smaller, so the observed
 * minima tell us how many hashes must have produced them.
 *
 * The denominator combines all observations after scaling hash values to `[0, 1]`:
 * - each empty bucket contributes `1`, meaning that its minimum lies beyond the hash range;
 * - each filled bucket contributes its restored interval midpoint divided by @ref hash_range.
 *
 * Dividing `bucket_count * filled_bucket_count` by that total gives the maximum-likelihood
 * cardinality estimate. If no bucket is empty, this reduces to the usual MeanM estimate.
 *
 * @param bucket_count Total number of sketch buckets.
 * @param empty_count Number of buckets that received no hashes.
 * @param sum_restored Sum of the restored interval midpoint from every filled bucket.
 *
 * @return Estimated distinct hash count, or zero when every bucket is empty.
 */
__host__ __device__ inline double
minimum_mle(double bucket_count, double empty_count, double sum_restored) noexcept {
    auto const filled = bucket_count - empty_count;
    if (filled <= 0.0) {
        return 0.0;
    }
    return bucket_count * filled / (empty_count + sum_restored / hash_range);
}

/**
 * @brief MeanM estimate retained as the full-sketch form of @ref minimum_mle.
 */
__host__ __device__ inline double
mean_m(double bucket_count, double filled, double sum_restored) noexcept {
    if (filled <= 0.0 || sum_restored <= 0.0) {
        return 0.0;
    }
    return bucket_count * filled * hash_range / sum_restored;
}

/**
 * @brief Public DDL cardinality estimate.
 *
 * Minimum MLE is the current production estimator. Keeping this dispatch point makes changing the
 * estimator explicit without exposing implementation-specific cardinality APIs.
 */
__host__ __device__ inline double
cardinality(double bucket_count, double empty_count, double sum_restored) noexcept {
    return minimum_mle(bucket_count, empty_count, sum_restored);
}

/**
 * @brief Single-precision variants of the cardinality estimators.
 *
 * The reduction dominates the cardinality kernel. Keeping the restored-sum reduction and the
 * scalar estimate in FP32 avoids the 1/64 FP32:FP64 throughput ratio on consumer Blackwell while
 * preserving far more precision than the ~2% standard error of a 2,048-register sketch.
 */
__host__ __device__ inline float
minimum_mle_f32(float bucket_count, float empty_count, float sum_restored) noexcept {
    auto const filled = bucket_count - empty_count;
    if (filled <= 0.0f) {
        return 0.0f;
    }
    return bucket_count * filled /
           (empty_count + sum_restored / static_cast<float>(hash_range));
}

__host__ __device__ inline float
mean_m_f32(float bucket_count, float filled, float sum_restored) noexcept {
    if (filled <= 0.0f || sum_restored <= 0.0f) {
        return 0.0f;
    }
    return bucket_count * filled * static_cast<float>(hash_range) / sum_restored;
}

__host__ __device__ inline float
cardinality_f32(float bucket_count, float empty_count, float sum_restored) noexcept {
    return minimum_mle_f32(bucket_count, empty_count, sum_restored);
}

}  // namespace cuddl::detail
