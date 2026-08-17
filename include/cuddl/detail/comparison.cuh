#pragma once

#include <cuda/std/cstdint>

#include <cuddl/detail/register.cuh>
#include <cuddl/pairwise_counts.cuh>

namespace cuddl::detail {

/// @brief Classifies one bucket relative to `this` into @p target.
__device__ inline void
classify(pairwise_counts& target, uint32_t this_reg, uint32_t other_reg) noexcept {
    auto const this_winner = winner(this_reg);
    auto const other_winner = winner(other_reg);
    if (this_winner == 0U && other_winner == 0U) {
        ++target.both_empty;
    } else if (this_winner < other_winner) {
        ++target.lower;
    } else if (this_winner > other_winner) {
        ++target.higher;
    } else {
        ++target.equal;
    }
}

}  // namespace cuddl::detail
