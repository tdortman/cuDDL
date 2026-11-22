#pragma once

#include <cuda/std/cstdint>

#include <cmath>

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

    /// @brief The same comparison seen from the other sketch: lower and higher swap.
    [[nodiscard]] __host__ __device__ constexpr pairwise_counts mirrored() const noexcept {
        return {.lower = higher, .equal = equal, .higher = lower, .both_empty = both_empty};
    }

    friend bool operator==(pairwise_counts const&, pairwise_counts const&) = default;
};

/**
 * @brief One pair's exact bucket comparison counts in 64 bits, the storage and transfer form of
 * batch-search results.
 *
 * Lower occupies bits 0-20, equal bits 21-41 and higher bits 42-63; both-empty is the bucket
 * count minus the three, so every supported bucket count (at most 2^13) fits.
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

/**
 * @brief One pair's exact bucket comparison counts in 32 bits using combinatorial ranking.
 *
 * Because `lower + equal + higher + both_empty == B`, the three separator positions
 * `p1 = lower`, `p2 = lower + equal + 1`, `p3 = lower + equal + higher + 2` form a strict
 * 3-subset of `{0, ..., B + 2}` with unique combinatorial rank
 * `R = p1 + C(p2, 2) + C(p3, 3) < C(B + 3, 3)`.
 * For `B = 2048`, `C(2051, 3) = 1,435,853,825 < 2^31`; valid for any `B <= 2951`.
 */
struct ranked_pairwise_counts {
    static constexpr uint32_t max_bucket_count = 2951U;
    uint32_t rank{};

    [[nodiscard]] static __host__ __device__ constexpr uint64_t binom2(uint32_t n) noexcept {
        return (static_cast<uint64_t>(n) * static_cast<uint64_t>(n - 1U)) >> 1U;
    }

    [[nodiscard]] static __host__ __device__ constexpr uint64_t binom3(uint32_t n) noexcept {
        return (static_cast<uint64_t>(n) * static_cast<uint64_t>(n - 1U) *
                static_cast<uint64_t>(n - 2U)) /
               6U;
    }

    [[nodiscard]] static __host__ __device__ constexpr ranked_pairwise_counts
    pack(uint32_t lower, uint32_t equal, uint32_t higher) noexcept {
        uint32_t const p1 = lower;
        uint32_t const p2 = p1 + equal + 1U;
        uint32_t const p3 = p2 + higher + 1U;
        return {static_cast<uint32_t>(uint64_t{p1} + binom2(p2) + binom3(p3))};
    }

    [[nodiscard]] static __host__ __device__ constexpr ranked_pairwise_counts
    from_packed(packed_pairwise_counts packed) noexcept {
        constexpr uint64_t field = (uint64_t{1} << 21U) - 1U;
        auto const lower = static_cast<uint32_t>(packed.bits & field);
        auto const equal = static_cast<uint32_t>((packed.bits >> 21U) & field);
        auto const higher = static_cast<uint32_t>(packed.bits >> 42U);
        return pack(lower, equal, higher);
    }

    [[nodiscard]] __host__ __device__ pairwise_counts unpack(uint32_t bucket_count) const noexcept {
        uint64_t const r = rank;
        // Invert C(p3, 3) <= r < C(p3 + 1, 3).
        uint32_t p3 = 2U;
        if (r > 0U) {
            auto approx = static_cast<uint32_t>(cbrt(6.0 * static_cast<double>(r))) + 1U;
            if (approx < 2U) approx = 2U;
            if (approx > bucket_count + 2U) approx = bucket_count + 2U;
            p3 = approx;
            while (p3 > 2U && binom3(p3) > r) {
                --p3;
            }
            while (p3 < bucket_count + 2U && binom3(p3 + 1U) <= r) {
                ++p3;
            }
        }
        uint64_t const rem = r - binom3(p3);
        // Invert C(p2, 2) <= rem < C(p2 + 1, 2).
        uint32_t p2 = 1U;
        if (rem > 0U) {
            auto approx =
                static_cast<uint32_t>((1.0 + sqrt(1.0 + 8.0 * static_cast<double>(rem))) * 0.5);
            if (approx < 1U) approx = 1U;
            if (approx >= p3) approx = p3 - 1U;
            p2 = approx;
            while (p2 > 1U && binom2(p2) > rem) {
                --p2;
            }
            while (p2 + 1U < p3 && binom2(p2 + 1U) <= rem) {
                ++p2;
            }
        }
        auto const p1 = static_cast<uint32_t>(rem - binom2(p2));
        auto const lower = p1;
        auto const equal = p2 - p1 - 1U;
        auto const higher = p3 - p2 - 1U;
        return {
            .lower = lower,
            .equal = equal,
            .higher = higher,
            .both_empty = bucket_count - lower - equal - higher,
        };
    }

    friend bool operator==(ranked_pairwise_counts const&, ranked_pairwise_counts const&) = default;
};

static_assert(sizeof(ranked_pairwise_counts) == 4);
static_assert(
    ranked_pairwise_counts::binom3(ranked_pairwise_counts::max_bucket_count + 3U) <=
    (uint64_t{1} << 32U)
);
static_assert(
    ranked_pairwise_counts::binom3(ranked_pairwise_counts::max_bucket_count + 4U) >
    (uint64_t{1} << 32U)
);

/**
 * @brief Weighted k-mer identity of a pair, `equal / max(equal + min(lower, higher), 6)`, as
 * sketch::wkid computes it, or 0 when no bucket is comparable.
 *
 * The divisor's floor of six suppresses similarity from sparse random collisions.
 */
[[nodiscard]] __host__ __device__ constexpr float weighted_kmer_identity(
    pairwise_counts const& counts
) noexcept {
    auto const divisor =
        counts.equal + (counts.lower < counts.higher ? counts.lower : counts.higher);
    if (divisor == 0U) return 0.0F;
    return static_cast<float>(counts.equal) / static_cast<float>(divisor < 6U ? 6U : divisor);
}

/// @brief Pairwise counts with an optional compile-time cardinality field.
struct pairwise_summary {
    pairwise_counts counts{};
    double cardinality{};

    friend bool operator==(pairwise_summary const&, pairwise_summary const&) = default;
};

}  // namespace cuddl
