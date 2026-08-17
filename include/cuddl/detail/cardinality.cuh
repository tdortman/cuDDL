#pragma once

#include <cuda/std/cmath>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

namespace cuddl::detail {

/// @brief Lower boundary of the hybridDDL blend zone, as a multiple of the bucket count.
constexpr double blend_low_ratio = 0.2;

/// @brief Upper boundary of the hybridDDL blend zone, as a multiple of the bucket count.
constexpr double blend_high_ratio = 7.5;

/// @brief Pure linear counting over @p V empty buckets of a size @p B sketch.
///
/// Returns zero for an empty sketch and infinity for a full sketch (no empty buckets).
__host__ __device__ inline double
linear_counting(double bucket_count, double empty_count) noexcept {
    if (empty_count <= 0.0) {
        return cuda::std::numeric_limits<double>::infinity();
    }
    return bucket_count * cuda::std::log(bucket_count / empty_count);
}

/**
 * @brief MeanM estimate from the arithmetic mean of restored hash magnitudes.
 *
 * Reconstructs the approximate hash magnitude per filled bucket, estimates the count that beats
 * each winning register as `2^64 / dbar`, then scales by the bucket count and applies the
 * empty-bucket linear correction.
 */
__host__ __device__ inline double
mean_m(double bucket_count, double filled, double sum_restored) noexcept {
    if (filled <= 0.0 || sum_restored <= 0.0) {
        return 0.0;
    }
    auto const dbar = sum_restored / filled;
    // `2^64 / dbar` is the expected count of hashes that beat the winning register magnitude in
    // one bucket; scaling by `bucket_count` recovers the total distinct estimate across buckets,
    // then the empty-bucket linear correction applies as `(filled + bucket_count) / 2`.
    auto const per_bucket = cuda::std::ldexp(1.0, 64) / dbar;
    return per_bucket * ((filled + bucket_count) / 2.0);
}

/**
 * @brief DDL's production hybridDDL cardinality estimator.
 *
 * Uses pure linear counting below `blend_low_ratio` buckets, a log-linear blend from linear
 * counting to MeanM across the transition region, and pure MeanM above `blend_high_ratio`
 * buckets. Zone selection uses a mantissa-free cardinality estimate (linear counting when any
 * bucket is empty, otherwise MeanM).
 *
 * @param bucket_count  Number of buckets in the sketch.
 * @param empty_count   Number of empty registers.
 * @param sum_restored  Sum of restored hash magnitudes over the filled registers.
 * @return The estimated distinct-element cardinality.
 */
__host__ __device__ inline double
hybrid_ddl(double bucket_count, double empty_count, double sum_restored) noexcept {
    auto const filled = bucket_count - empty_count;
    if (filled <= 0.0) {
        return 0.0;
    }
    auto const low = blend_low_ratio * bucket_count;
    auto const high = blend_high_ratio * bucket_count;
    // Linear counting is undefined (infinite) when no bucket is empty; MeanM alone governs.
    if (empty_count <= 0.0) {
        return mean_m(bucket_count, filled, sum_restored);
    }
    // Zone selection mirrors the mantissa-free estimator: linear counting up to low occupancy.
    auto const lc = linear_counting(bucket_count, empty_count);
    if (lc <= low) {
        return lc;
    }
    if (lc >= high) {
        return mean_m(bucket_count, filled, sum_restored);
    }
    auto const mean = mean_m(bucket_count, filled, sum_restored);

    using cuda::std::exp;
    using cuda::std::log;

    auto const t = (log(lc) - log(low)) / (log(high) - log(low));
    return exp((1.0 - t) * log(lc) + t * log(mean));
}

}  // namespace cuddl::detail
