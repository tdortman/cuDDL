#pragma once

#include <cuda/std/cmath>
#include <cuda/std/cstdint>

#include <cuddl/detail/cardinality.cuh>
#include <cuddl/hybrid_cardinality.cuh>

namespace cuddl::detail {

constexpr int32_t nlz_bins = 66;
constexpr double dlc_blend_low = 0.088;
constexpr double dlc_blend_high = 0.952;
constexpr double dlc_min_fraction = 0.002;
constexpr double dlc_info_power = 4.5;
constexpr double hybrid_low = 0.2;
constexpr double hybrid_high = 5.0;

enum class hybrid_variant : uint8_t {
    bbtools,
    paper,
};
constexpr uint32_t ddl_cf_size = 13;

static __device__ constexpr float ddl_cf_keys[] = {
    256.0f,
    512.0f,
    1024.0f,
    2048.0f,
    4096.0f,
    8192.0f,
    16384.0f,
    32768.0f,
    65536.0f,
    131072.0f,
    262144.0f,
    524288.0f,
    1048576.0f,
};
static __device__ constexpr float ddl_mean_m_cf[] = {
    0.92533006f,
    0.887325502f,
    0.833883596f,
    0.810002197f,
    0.850952315f,
    0.948926187f,
    0.998481533f,
    0.999045759f,
    0.996665452f,
    1.00076597f,
    0.999102624f,
    0.99457023f,
    1.00041684f,
};

__device__ inline double interpolate_mean_m_cf(double estimate) noexcept {
    if (estimate <= ddl_cf_keys[0]) {
        return ddl_mean_m_cf[0];
    }
    if (estimate >= ddl_cf_keys[ddl_cf_size - 1]) {
        return ddl_mean_m_cf[ddl_cf_size - 1];
    }
    uint32_t low = 0;
    uint32_t high = ddl_cf_size - 1;
    while (low + 1 < high) {
        auto const middle = (low + high) / 2;
        if (ddl_cf_keys[middle] <= estimate) {
            low = middle;
        } else {
            high = middle;
        }
    }
    auto const fraction = (estimate - ddl_cf_keys[low]) / (ddl_cf_keys[high] - ddl_cf_keys[low]);
    return ddl_mean_m_cf[low] + fraction * (ddl_mean_m_cf[high] - ddl_mean_m_cf[low]);
}

__device__ inline double lc_min(uint32_t const* bins, double bucket_count) noexcept {
    auto cumulative = bins[0];
    if (cumulative > 0U) {
        return bucket_count * cuda::std::log(bucket_count / cumulative);
    }
    for (int32_t tier = 0; tier + 1 < nlz_bins; ++tier) {
        cumulative += bins[tier + 1];
        if (cumulative > 0U) {
            return cuda::std::ldexp(
                bucket_count * cuda::std::log(bucket_count / cumulative), tier + 1
            );
        }
    }
    return 0.0;
}

__device__ inline double dlc(uint32_t const* bins, double bucket_count, double lc) noexcept {
    auto const empty = bins[0];
    auto const minimum = max(1.0, bucket_count * dlc_min_fraction);
    auto const maximum = bucket_count - minimum;
    auto cumulative = empty;
    double weight_sum = 0.0;
    double weighted_log_sum = 0.0;
    for (int32_t tier = 0; tier + 1 < nlz_bins; ++tier) {
        if (cumulative >= minimum && cumulative <= maximum) {
            auto const occupied = bucket_count - cumulative;
            auto const tier_estimate =
                cuda::std::ldexp(bucket_count * cuda::std::log(bucket_count / cumulative), tier);
            
            auto const error = cuda::std::sqrt(2.0 / 3.14159265358979323846) *
                               cuda::std::sqrt(occupied / (bucket_count * cumulative)) /
                               cuda::std::log(bucket_count / cumulative);

            auto const weight = cuda::std::pow(1.0 / error, dlc_info_power);
            weight_sum += weight;
            weighted_log_sum += weight * cuda::std::log(tier_estimate);
        }
        cumulative += bins[tier + 1];
        if (cumulative >= bucket_count) {
            break;
        }
    }
    if (weight_sum == 0.0) {
        return lc;
    }
    auto const pure = cuda::std::exp(weighted_log_sum / weight_sum);
    if (empty >= dlc_blend_high * bucket_count) {
        return lc;
    }
    if (empty > dlc_blend_low * bucket_count) {
        auto const t = (empty - dlc_blend_low * bucket_count) /
                       ((dlc_blend_high - dlc_blend_low) * bucket_count);
        return t * lc + (1.0 - t) * pure;
    }
    return pure;
}

__device__ inline double hybrid(
    double low_estimate,
    double zone_estimate,
    double corrected_mean,
    double bucket_count
) noexcept {
    auto const low = hybrid_low * bucket_count;
    auto const high = hybrid_high * bucket_count;
    if (zone_estimate <= low) {
        return low_estimate;
    }
    if (zone_estimate >= high) {
        return corrected_mean;
    }
    auto const t = cuda::std::log(zone_estimate / low) / cuda::std::log(high / low);
    return (1.0 - t) * low_estimate + t * corrected_mean;
}

__device__ inline double
bbtools_mean_m(double bucket_count, double filled, double restored_sum) noexcept {
    if (filled <= 0.0 || restored_sum <= 0.0) {
        return 0.0;
    }
    auto const mean = restored_sum / filled;
    auto const occupancy_correction = (filled + bucket_count) / (2.0 * bucket_count);
    return hash_range / mean * filled * occupancy_correction;
}

__device__ inline hybrid_cardinality_estimates
hybrid_estimates(uint32_t const* bins, double bucket_count, double restored_sum) noexcept {
    auto const empty = static_cast<double>(bins[0]);
    auto const filled = bucket_count - empty;
    auto const raw_mean = bbtools_mean_m(bucket_count, filled, restored_sum);
    auto const lc = lc_min(bins, bucket_count);
    auto const dlc_estimate = dlc(bins, bucket_count, lc);
    auto const corrected_mean = raw_mean * interpolate_mean_m_cf(dlc_estimate);
    return {
        hybrid(lc, lc, corrected_mean, bucket_count),
        hybrid(lc, dlc_estimate, corrected_mean, bucket_count),
        lc,
        dlc_estimate,
        raw_mean,
    };
}

}  // namespace cuddl::detail
