#pragma once

#include <cuda/std/cmath>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

namespace cuddl::detail {

/// @brief `2^64`, the range of the 64-bit hash used by the sketch.
constexpr double hash_range = 0x1p64;

/**
 * @brief Poisson maximum-likelihood estimate from empty and observed bucket minima.
 *
 * Empty buckets contribute a censored observation at the hash-range boundary. Filled buckets
 * contribute their restored minimum directly. The resulting closed-form MLE is valid from sparse
 * through saturated sketches and converges to MeanM when no bucket is empty.
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


}  // namespace cuddl::detail
