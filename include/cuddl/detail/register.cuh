#pragma once

#include <cuda/std/algorithm>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

#include <cuddl/detail/hash.cuh>

namespace cuddl {

/// @brief Compile-time split of a 16-bit DDL score into exponent and mantissa fields.
template <uint8_t ExponentBits, uint8_t MantissaBits>
struct register_layout {
    static_assert(ExponentBits > 0U);
    static_assert(MantissaBits > 0U);
    static_assert(ExponentBits + MantissaBits == 16U);

    static constexpr uint8_t exponent_bits = ExponentBits;
    static constexpr uint8_t mantissa_bits = MantissaBits;
};

using default_register_layout = register_layout<6, 10>;

namespace detail {

/// @brief Default score layout retained for source compatibility.
constexpr uint8_t mantissa_bits = default_register_layout::mantissa_bits;
constexpr uint32_t mantissa_mask = (1U << mantissa_bits) - 1U;

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

}  // namespace detail
}  // namespace cuddl
