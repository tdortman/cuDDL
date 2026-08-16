#pragma once

#include <cuda/std/cstdint>

namespace cuddl {

/// @brief Raw per-bucket classification counts for one sketch pair comparison.
struct pairwise_counts {
    /// `this` register is strictly lower than the other's in that bucket.
    uint32_t lower{};
    /// `this` register equals the other's in that bucket.
    uint32_t equal{};
    /// `this` register is strictly higher than the other's in that bucket.
    uint32_t higher{};
    /// Both registers are empty in that bucket.
    uint32_t both_empty{};

    /// @brief Accumulates @p other into this instance (used for device reductions).
    __host__ __device__ pairwise_counts& operator+=(pairwise_counts const other) noexcept {
        lower += other.lower;
        equal += other.equal;
        higher += other.higher;
        both_empty += other.both_empty;
        return *this;
    }

    friend __host__ __device__ pairwise_counts
    operator+(pairwise_counts left, pairwise_counts const right) noexcept {
        return left += right;
    }

    friend bool operator==(pairwise_counts const&, pairwise_counts const&) = default;
};

/// @brief Raw result payload of a pairwise summary reduction.
///
/// @p counts is populated when the pairwise field is selected; @p cardinality is the estimate for
/// the query sketch when the cardinality field is selected.
struct pairwise_summary {
    pairwise_counts counts{};
    double cardinality{};

    friend bool operator==(pairwise_summary const&, pairwise_summary const&) = default;
};

/// @brief Compile-time field mask selecting which reductions a fused summary computes.
struct summary_mask {
    /// Selects the raw @ref pairwise_counts reduction.
    static constexpr uint32_t pairwise = 1U << 0;
    /// Selects the query sketch's hybridDDL cardinality reduction.
    static constexpr uint32_t cardinality = 1U << 1;
};

}  // namespace cuddl
