#pragma once

#include <cuda/std/algorithm>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

#include <cuddl/detail/hash.cuh>

namespace cuddl {

/// @brief Compile-time split of a 16-bit DDL score into exponent and mantissa fields.
template <uint32_t ExponentBits, uint32_t MantissaBits>
struct register_layout {
    static_assert(ExponentBits > 0U);
    static_assert(MantissaBits > 0U);
    static_assert(ExponentBits + MantissaBits == 16U);

    static constexpr uint32_t exponent_bits = ExponentBits;
    static constexpr uint32_t mantissa_bits = MantissaBits;
};

using default_register_layout = register_layout<6, 10>;

namespace detail {

/// @brief Default score layout retained for source compatibility.
constexpr uint32_t mantissa_bits = default_register_layout::mantissa_bits;
constexpr uint32_t mantissa_mask = (1U << mantissa_bits) - 1U;

/// @brief Maximum winner multiplicity representable in the low 16 bits of a packed register.
constexpr uint32_t max_winner_count = 0xffffU;

/// @brief Default restoration constants retained for source compatibility.
constexpr uint32_t restore_shift_base = 63U - mantissa_bits;
constexpr uint32_t max_nlz = restore_shift_base;

/**
 * @brief Computes the 16-bit DDL rarity score for @p hash.
 *
 * The exponent field stores the number of leading zeros (NLZ); the mantissa field stores the
 * @ref mantissa_bits bits immediately following the leading one, bitwise inverted. A larger score
 * always encodes a rarer hash. Zero is reserved for the empty register, so a hash whose raw
 * exponent/mantissa encoding is zero collapses to one.
 */
template <typename Layout = default_register_layout>
__host__ __device__ inline uint16_t score(uint64_t hash) noexcept {
    constexpr auto layout_mantissa_mask = (1U << Layout::mantissa_bits) - 1U;
    constexpr auto layout_restore_shift_base = 63U - Layout::mantissa_bits;
    constexpr auto layout_max_nlz =
        cuda::std::min(layout_restore_shift_base, (1U << Layout::exponent_bits) - 1U);
#ifdef __CUDA_ARCH__
    auto const nlz = static_cast<uint32_t>(__clzll(hash | 1ULL));
#else
    auto const nlz = static_cast<uint32_t>(__builtin_clzll(hash | 1ULL));
#endif
    auto const capped = nlz > layout_max_nlz ? layout_max_nlz : nlz;
    auto const shift = layout_restore_shift_base - capped;
    auto const mantissa = static_cast<uint32_t>((hash >> shift) & layout_mantissa_mask);
    auto const inverted = static_cast<uint32_t>(mantissa ^ layout_mantissa_mask);
    auto const raw = static_cast<uint16_t>((capped << Layout::mantissa_bits) | inverted);
    return raw == 0U ? static_cast<uint16_t>(1U) : raw;
}

/// @brief Packs a 16-bit winner score and its observation count into one register.
__host__ __device__ __forceinline__ constexpr uint32_t
pack(uint16_t winner, uint16_t count) noexcept {
    return static_cast<uint32_t>(winner) << 16U | count;
}

/// @brief Extracts the winning score from a packed register.
__host__ __device__ __forceinline__ constexpr uint16_t winner(uint32_t state) noexcept {
    return static_cast<uint16_t>(state >> 16U);
}

/// @brief Extracts the winning score's observation count from a packed register.
__host__ __device__ __forceinline__ constexpr uint16_t count(uint32_t state) noexcept {
    return static_cast<uint16_t>(state & 0xffffU);
}

/// @brief Adds two winner counts, saturating at @ref max_winner_count.
__host__ __device__ constexpr uint16_t saturated_add(uint16_t left, uint16_t right) noexcept {
    auto const sum = static_cast<uint32_t>(left) + right;
    return sum < max_winner_count ? static_cast<uint16_t>(sum)
                                  : static_cast<uint16_t>(max_winner_count);
}

/**
 * @brief Reconstructs the approximate original hash magnitude from a stored score.
 *
 * Inverts the @ref score encoding: recovers NLZ from the exponent field, uninverts the mantissa
 * bits and prepends the implicit leading one, then shifts the result back into 64-bit space.
 */
template <typename Layout = default_register_layout>
__host__ __device__ constexpr uint64_t restore(uint16_t stored) noexcept {
    constexpr auto layout_mantissa_mask = (1U << Layout::mantissa_bits) - 1U;
    constexpr auto layout_restore_shift_base = 63U - Layout::mantissa_bits;
    auto const nlz = static_cast<uint32_t>(stored >> Layout::mantissa_bits);
    auto const lowbits = static_cast<uint32_t>((~stored) & layout_mantissa_mask);
    auto const mantissa = (1U << Layout::mantissa_bits) | lowbits;
    auto const shift = layout_restore_shift_base - nlz;
    return static_cast<uint64_t>(mantissa) << shift;
}

/**
 * @brief Reconstructs the interval midpoint for @ref restore.
 *
 * @ref restore returns the lower bound of the hash interval represented by @p stored. The actual
 * hash is uniformly distributed across that interval, so the midpoint is the minimum-variance
 * scalar summary and reduces the small positive bias that lower-bound restoration introduces
 * into hash-magnitude cardinality sums.
 */
template <typename Layout = default_register_layout>
__host__ __device__ constexpr uint64_t restore_midpoint(uint16_t stored) noexcept {
    constexpr auto layout_mantissa_mask = (1U << Layout::mantissa_bits) - 1U;
    constexpr auto layout_restore_shift_base = 63U - Layout::mantissa_bits;
    auto const nlz = static_cast<uint32_t>(stored >> Layout::mantissa_bits);
    auto const lowbits = static_cast<uint32_t>((~stored) & layout_mantissa_mask);
    auto const mantissa = (1U << Layout::mantissa_bits) | lowbits;
    auto const shift = layout_restore_shift_base - nlz;
    auto const lower = static_cast<uint64_t>(mantissa) << shift;
    // `(1 << shift) >> 1` is 2^(shift-1) for ordinary tiers and exactly zero for the
    // clamped top tier (`shift == 0`), so the midpoint needs no branch.
    return lower + ((1ULL << shift) >> 1U);
}

/**
 * @brief Atomically applies the DDL winner-update rule to @p address.
 *
 * A better score replaces the winner and resets its count to one; an equal score increments the
 * count, saturating at @ref max_winner_count; a worse score is ignored. The 65,536th equal
 * observation (an increment attempted on an already-saturated counter) sets the sketch-level
 * saturation flag through @p saturation using an idempotent store.
 *
 */
__device__ inline void update(uint32_t* address, uint16_t incoming, uint32_t& saturation) noexcept {
    auto observed = *address;
    while (incoming >= winner(observed)) {
        uint32_t replacement;
        if (incoming > winner(observed)) {
            replacement = pack(incoming, 1U);
        } else if (count(observed) == max_winner_count) {
            atomicExch(&saturation, 1U);
            return;
        } else {
            replacement = pack(incoming, static_cast<uint16_t>(count(observed) + 1U));
        }
        auto const previous = atomicCAS(address, observed, replacement);
        if (previous == observed) {
            return;
        }
        observed = previous;
    }
}

/**
 * @brief Merges one CTA-local partial register into a global register.
 *
 * A higher partial winner replaces the global winner with its count; an equal winner adds the
 * counts, saturating at @ref max_winner_count. A saturated sum records the sketch-level
 * saturation flag through @p saturation, matching the sequential @ref update semantics.
 */
__device__ inline void
merge_register(uint32_t* address, uint32_t partial, uint32_t& saturation) noexcept {
    if (partial == 0U) {
        return;
    }
    auto const partial_winner = winner(partial);
    auto observed = *address;
    while (true) {
        auto const observed_winner = winner(observed);
        uint32_t replacement;
        if (partial_winner > observed_winner) {
            replacement = partial;
        } else if (partial_winner < observed_winner) {
            return;
        } else {
            auto const sum = static_cast<uint32_t>(count(observed)) + count(partial);
            if (sum > max_winner_count) {
                atomicExch(&saturation, 1U);
                replacement = pack(partial_winner, max_winner_count);
            } else {
                replacement = pack(partial_winner, static_cast<uint16_t>(sum));
            }
        }
        auto const previous = atomicCAS(address, observed, replacement);
        if (previous == observed) {
            return;
        }
        observed = previous;
    }
}

}  // namespace detail
}  // namespace cuddl
