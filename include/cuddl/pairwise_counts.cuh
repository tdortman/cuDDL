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

    /// @brief Sums two count sets for device reductions.
    friend __host__ __device__ pairwise_counts
    operator+(pairwise_counts left, pairwise_counts const right) noexcept {
        return left += right;
    }

    friend bool operator==(pairwise_counts const&, pairwise_counts const&) = default;
};

/**
 * @brief One pair's exact bucket comparison counts in 64 bits, the storage and transfer form of
 * batch-search results.
 *
 * Lower occupies bits 0-20, equal bits 21-41 and higher bits 42-63; both-empty is the bucket
 * count minus the three, so every supported bucket count (at most 2^17) fits.
 */
struct packed_pairwise_counts {
    uint64_t bits{};

    [[nodiscard]] static __host__ __device__ constexpr packed_pairwise_counts
    pack(uint32_t lower, uint32_t equal, uint32_t higher) noexcept {
        return {
            static_cast<uint64_t>(lower) | (static_cast<uint64_t>(equal) << 21U) |
            (static_cast<uint64_t>(higher) << 42U)
        };
    }

    [[nodiscard]] __host__ __device__ constexpr pairwise_counts unpack(
        uint32_t bucket_count
    ) const noexcept {
        constexpr uint64_t field = (uint64_t{1} << 21U) - 1U;
        auto const lower = static_cast<uint32_t>(bits & field);
        auto const equal = static_cast<uint32_t>((bits >> 21U) & field);
        auto const higher = static_cast<uint32_t>(bits >> 42U);
        return {
            .lower = lower,
            .equal = equal,
            .higher = higher,
            .both_empty = bucket_count - lower - equal - higher,
        };
    }

    friend bool operator==(packed_pairwise_counts const&, packed_pairwise_counts const&) = default;
};

/// @brief Pairwise counts with an optional compile-time cardinality field.
struct pairwise_summary {
    pairwise_counts counts{};
    double cardinality{};

    friend bool operator==(pairwise_summary const&, pairwise_summary const&) = default;
};

}  // namespace cuddl
