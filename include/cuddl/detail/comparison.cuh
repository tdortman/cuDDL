#pragma once

#include <cuda/std/cstdint>

#include <cuddl/detail/register.cuh>
#include <cuddl/pairwise_counts.cuh>

namespace cuddl::detail {

/// @brief Classifies one compact score relative to `this` into @p target.
__device__ inline void
classify(pairwise_counts& target, uint16_t this_score, uint16_t other_score) noexcept {
    if (this_score == 0U && other_score == 0U) {
        ++target.both_empty;
    } else if (this_score < other_score) {
        ++target.lower;
    } else if (this_score > other_score) {
        ++target.higher;
    } else {
        ++target.equal;
    }
}

/// @brief Classifies one packed register relative to `this` into @p target.
__device__ inline void
classify(pairwise_counts& target, uint32_t this_reg, uint32_t other_reg) noexcept {
    classify(target, winner(this_reg), winner(other_reg));
}

}  // namespace cuddl::detail
