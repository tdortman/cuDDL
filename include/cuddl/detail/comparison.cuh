#pragma once

#include <cuda/std/cstdint>

#include <cuddl/detail/register.cuh>
#include <cuddl/pairwise_counts.cuh>

namespace cuddl::detail {

/// @brief Classifies one compact score relative to `this` into @p target.
///
/// Branchless: all four outcomes are computed as predicated arithmetic so a warp with mixed
/// comparisons (random sketch data) never pays branch-reconvergence costs. Both-empty is the
/// equality outcome restricted to zero scores.
__device__ inline void
classify(pairwise_counts& target, uint16_t this_score, uint16_t other_score) noexcept {
    auto const this_zero = static_cast<uint32_t>(this_score == 0U);
    auto const lt = static_cast<uint32_t>(this_score < other_score);
    auto const gt = static_cast<uint32_t>(this_score > other_score);
    auto const eq = 1U - lt - gt;
    target.both_empty += eq & this_zero;
    target.lower += lt;
    target.higher += gt;
    target.equal += eq & (1U - this_zero);
}

/// @brief Classifies one packed register relative to `this` into @p target.
__device__ inline void
classify(pairwise_counts& target, uint32_t this_reg, uint32_t other_reg) noexcept {
    classify(target, winner(this_reg), winner(other_reg));
}

}  // namespace cuddl::detail
